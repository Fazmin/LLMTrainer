//! A small HTTP/1.1 server that behaves like Hugging Face's `resolve` endpoint plus its CDN, with switches for the
//! ways real networks fail. Everything runs on loopback; no real network is touched.
//!
//! * `GET|HEAD /hf/<name>` answers `302` to `/cdn/<token>/<name>` with `x-linked-size` and `x-linked-etag` (the file's
//!   SHA-256), and hands out a *new* one-use token every time, like a signed URL.
//! * `GET /cdn/<token>/<name>` with `Range: bytes=a-b` answers `206` with `Content-Range`, `ETag` and the bytes. A
//!   token works once; reusing it answers `403`, the way an expired signature does.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub const BOUNDARY: &[u8] = b"\n<|endoftext|>\n";

/// Ways to make the server misbehave. Each applies until reset.
#[derive(Default, Clone)]
pub struct Faults {
    /// Each ranged CDN `GET` takes one entry: send only this many body bytes, then close the connection.
    pub drop_after: VecDeque<usize>,
    /// The next this-many CDN requests answer `403`.
    pub forbid_next_cdn: usize,
    /// Answer `200` with the whole file even to a ranged request.
    pub ignore_range: bool,
    /// Flip one byte in every body sent.
    pub tamper: bool,
    /// Put this total in `Content-Range`.
    pub wrong_total: Option<u64>,
    /// Report this in `x-linked-etag`.
    pub linked_etag: Option<String>,
    /// Report this in `x-linked-size`.
    pub linked_size: Option<u64>,
    /// Never answer `206`; reply with this status to CDN requests.
    pub cdn_status: Option<u16>,
}

#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub range: Option<(u64, u64)>,
    pub accept_encoding: Option<String>,
    pub status: u16,
    /// The request used a CDN token that had been used before.
    pub stale_token: bool,
}

struct Entry {
    body: Vec<u8>,
    sha256: String,
    cdn_etag: Mutex<String>,
}

struct Shared {
    files: HashMap<String, Entry>,
    log: Mutex<Vec<Req>>,
    tokens: Mutex<(u64, HashSet<String>)>,
    faults: Mutex<Faults>,
}

pub struct Server {
    pub addr: SocketAddr,
    shared: Arc<Shared>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Server {
    pub async fn start(files: Vec<(&str, Vec<u8>)>) -> Server {
        let files = files
            .into_iter()
            .map(|(name, body)| {
                let sha256 = hex(&Sha256::digest(&body));
                let etag = format!("cdn-{}", &sha256[..16]);
                (name.to_string(), Entry { body, sha256, cdn_etag: Mutex::new(etag) })
            })
            .collect();
        let shared = Arc::new(Shared {
            files,
            log: Mutex::new(Vec::new()),
            tokens: Mutex::new((0, HashSet::new())),
            faults: Mutex::new(Faults::default()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = shared.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else { return };
                let state = state.clone();
                tokio::spawn(async move {
                    let _ = handle(stream, addr, state).await;
                });
            }
        });
        Server { addr, shared, task }
    }

    /// The base URL a [`Starter`](minagi_data::Starter) can be pointed at.
    pub fn base_url(&self) -> String {
        format!("http://{}/hf", self.addr)
    }

    pub fn url(&self, name: &str) -> String {
        format!("{}/{name}", self.base_url())
    }

    pub fn sha256_of(&self, name: &str) -> String {
        self.shared.files[name].sha256.clone()
    }

    pub fn set_faults(&self, f: impl FnOnce(&mut Faults)) {
        f(&mut self.shared.faults.lock().unwrap());
    }

    pub fn reset_faults(&self) {
        *self.shared.faults.lock().unwrap() = Faults::default();
    }

    pub fn set_cdn_etag(&self, name: &str, etag: &str) {
        *self.shared.files[name].cdn_etag.lock().unwrap() = etag.to_string();
    }

    pub fn requests(&self) -> Vec<Req> {
        self.shared.log.lock().unwrap().clone()
    }

    pub fn clear_log(&self) {
        self.shared.log.lock().unwrap().clear();
    }

    pub fn heads(&self) -> Vec<Req> {
        self.requests().into_iter().filter(|r| r.method == "HEAD").collect()
    }

    /// Ranged `GET`s that reached the CDN (whatever the answer).
    pub fn cdn_gets(&self) -> Vec<Req> {
        self.requests().into_iter().filter(|r| r.method == "GET" && r.path.starts_with("/cdn/")).collect()
    }

    pub fn stale_token_requests(&self) -> usize {
        self.requests().iter().filter(|r| r.stale_token).count()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

async fn handle(mut stream: TcpStream, addr: SocketAddr, shared: Arc<Shared>) -> std::io::Result<()> {
    // Read the request head.
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).await? == 0 {
            return Ok(());
        }
        head.push(byte[0]);
        if head.len() > 16 * 1024 {
            return Ok(());
        }
    }
    let text = String::from_utf8_lossy(&head).into_owned();
    let mut lines = text.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
    let header = |name: &str| -> Option<String> {
        text.split("\r\n").skip(1).find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    let range = header("range").and_then(|r| {
        let r = r.strip_prefix("bytes=")?;
        let (a, b) = r.split_once('-')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let accept_encoding = header("accept-encoding");

    let mut req =
        Req { method: method.clone(), path: path.clone(), range, accept_encoding, status: 0, stale_token: false };
    let faults = shared.faults.lock().unwrap().clone();
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    let result = match segments.as_slice() {
        ["hf", name] if shared.files.contains_key(*name) => {
            let entry = &shared.files[*name];
            let token = {
                let mut t = shared.tokens.lock().unwrap();
                t.0 += 1;
                format!("tok{}", t.0)
            };
            let size = faults.linked_size.unwrap_or(entry.body.len() as u64);
            let etag = faults.linked_etag.clone().unwrap_or_else(|| entry.sha256.clone());
            req.status = 302;
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: http://{addr}/cdn/{token}/{name}\r\nx-linked-size: {size}\r\nx-linked-etag: \"{etag}\"\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            shared.log.lock().unwrap().push(req);
            stream.write_all(response.as_bytes()).await
        }
        ["cdn", token, name] if shared.files.contains_key(*name) => {
            let entry = &shared.files[*name];
            let stale = {
                let mut t = shared.tokens.lock().unwrap();
                !t.1.insert(token.to_string())
            };
            req.stale_token = stale;
            let forbid = stale || {
                let mut f = shared.faults.lock().unwrap();
                if f.forbid_next_cdn > 0 {
                    f.forbid_next_cdn -= 1;
                    true
                } else {
                    false
                }
            };
            if forbid {
                req.status = 403;
                shared.log.lock().unwrap().push(req);
                return stream
                    .write_all(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await;
            }
            if let Some(status) = faults.cdn_status {
                req.status = status;
                shared.log.lock().unwrap().push(req);
                return stream
                    .write_all(
                        format!("HTTP/1.1 {status} Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").as_bytes(),
                    )
                    .await;
            }
            serve_cdn(&mut stream, &shared, entry, &method, range, &faults, req).await
        }
        _ => {
            req.status = 404;
            shared.log.lock().unwrap().push(req);
            stream.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").await
        }
    };
    let _ = stream.shutdown().await;
    result
}

async fn serve_cdn(
    stream: &mut TcpStream,
    shared: &Arc<Shared>,
    entry: &Entry,
    method: &str,
    range: Option<(u64, u64)>,
    faults: &Faults,
    mut req: Req,
) -> std::io::Result<()> {
    let total = entry.body.len() as u64;
    let etag = entry.cdn_etag.lock().unwrap().clone();
    let mut body: Vec<u8> = entry.body.clone();
    if faults.tamper && !body.is_empty() {
        let at = body.len() / 2;
        body[at] ^= 0x01;
    }
    if method == "HEAD" {
        req.status = 200;
        shared.log.lock().unwrap().push(req);
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {total}\r\naccept-ranges: bytes\r\netag: \"{etag}\"\r\nconnection: close\r\n\r\n"
        );
        return stream.write_all(head.as_bytes()).await;
    }
    let (head, payload): (String, Vec<u8>) = match range {
        Some((start, end)) if !faults.ignore_range => {
            let end = end.min(total.saturating_sub(1));
            if start >= total {
                req.status = 416;
                shared.log.lock().unwrap().push(req);
                return stream
                    .write_all(format!("HTTP/1.1 416 Range Not Satisfiable\r\ncontent-range: bytes */{total}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").as_bytes())
                    .await;
            }
            let payload = body[start as usize..=end as usize].to_vec();
            let shown_total = faults.wrong_total.unwrap_or(total);
            req.status = 206;
            (
                format!(
                    "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes {start}-{end}/{shown_total}\r\ncontent-length: {}\r\netag: \"{etag}\"\r\naccept-ranges: bytes\r\nconnection: close\r\n\r\n",
                    payload.len()
                ),
                payload,
            )
        }
        _ => {
            req.status = 200;
            (
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\netag: \"{etag}\"\r\nconnection: close\r\n\r\n",
                    body.len()
                ),
                body,
            )
        }
    };
    let cut_at = if range.is_some() { shared.faults.lock().unwrap().drop_after.pop_front() } else { None };
    shared.log.lock().unwrap().push(req);
    stream.write_all(head.as_bytes()).await?;
    let mut sent = 0usize;
    for chunk in payload.chunks(4096) {
        if let Some(limit) = cut_at
            && sent + chunk.len() > limit
        {
            stream.write_all(&chunk[..limit - sent]).await?;
            stream.flush().await?;
            return Ok(()); // connection closes with the body unfinished
        }
        stream.write_all(chunk).await?;
        sent += chunk.len();
        tokio::task::yield_now().await;
    }
    Ok(())
}

// ----- test data -----

const WORDS: &[&str] = &[
    "the",
    "little",
    "fox",
    "lantern",
    "garden",
    "river",
    "mountain",
    "friend",
    "smiled",
    "walked",
    "quietly",
    "café",
    "naïve",
    "résumé",
    "日本語",
    "小さな",
    "🎉",
    "🦊",
    "once",
    "upon",
    "a",
    "time",
    "there",
    "was",
    "boy",
    "named",
    "Tom",
    "and",
    "they",
    "played",
];

/// TinyStories-shaped text: stories of 200 to 1800 bytes, each followed by a line holding `<|endoftext|>`, with some
/// multi-byte characters mixed in. If `trailing_marker` is false the last story has no marker after it.
pub fn stories(seed: u64, approx_bytes: usize, trailing_marker: bool) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = String::new();
    while out.len() < approx_bytes {
        let target = rng.random_range(200..1800);
        let mut story = String::new();
        while story.len() < target {
            story.push_str(WORDS[rng.random_range(0..WORDS.len())]);
            story.push(if rng.random_range(0..12) == 0 { '\n' } else { ' ' });
        }
        out.push_str(story.trim_end());
        out.push_str("\n<|endoftext|>\n");
    }
    if !trailing_marker {
        out.truncate(out.len() - BOUNDARY.len());
    }
    out.into_bytes()
}

/// What the downloader must produce, computed the slow, obvious way: take windows of `shard_bytes`, cut each at its
/// last story boundary; the rest of the data is the final shard (for a prefix, only whole stories of it).
pub fn expected_shards(data: &[u8], shard_bytes: usize, prefix: Option<usize>) -> Vec<Vec<u8>> {
    let src = &data[..prefix.unwrap_or(data.len()).min(data.len())];
    let last_boundary_end =
        |w: &[u8]| w.windows(BOUNDARY.len()).rposition(|x| x == BOUNDARY).map(|i| i + BOUNDARY.len());
    let (mut shards, mut pos) = (Vec::new(), 0);
    loop {
        let remaining = src.len() - pos;
        if remaining >= shard_bytes {
            let cut =
                last_boundary_end(&src[pos..pos + shard_bytes]).expect("test data has a boundary in every window");
            shards.push(src[pos..pos + cut].to_vec());
            pos += cut;
        } else {
            let rest = &src[pos..];
            if prefix.is_some() {
                if let Some(cut) = last_boundary_end(rest) {
                    shards.push(rest[..cut].to_vec());
                }
            } else if !rest.is_empty() {
                shards.push(rest.to_vec());
            }
            return shards;
        }
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}
