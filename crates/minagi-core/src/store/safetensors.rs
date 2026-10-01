//! Export a checkpoint as a [safetensors](https://huggingface.co/docs/safetensors) file, for other tools.
//!
//! The format is eight bytes holding the length of a JSON header (little endian), the header (names, dtypes, shapes and
//! byte ranges, padded with spaces to a multiple of eight), then the raw tensor bytes. It is written without any
//! dependency and **streamed**: the header is computed from shapes alone, then tensors go to disk one at a time, so even
//! a model with hundreds of experts never has to fit in memory.
//!
//! Tensor names are the same as in the checkpoint (`tok_emb.weight`, `prelude.0.attn.qkv.weight`, `adapter.weight`,
//! `recur.0.mlp.router.weight`, `pool.gate`, ...). The tied output head is stored once, as `tok_emb.weight`. Every expert
//! is `pool.experts.<uid>.w1`, `.w3` (both `[d_ff, d_model]`) and `.w2` (`[d_model, d_ff]`); `uid` is the expert's stable
//! name, which pruning never renumbers. All values are `F32`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{Value, json};

use super::checkpoint::{Checkpoint, NamedTensors, Tensor};
use super::{Result, StoreError};

/// What was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsReport {
    pub tensors: usize,
    /// Size of `model.safetensors`.
    pub bytes: u64,
    pub experts: usize,
}

enum Source<'a> {
    Host(&'a Tensor),
    Expert { uid: u64, which: u8 },
}

struct Entry<'a> {
    name: String,
    shape: Vec<usize>,
    source: Source<'a>,
}

impl Entry<'_> {
    fn bytes(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64 * 4
    }
}

fn host_entries<'a>(set: &'a NamedTensors, skip: &[&str], out: &mut Vec<Entry<'a>>) {
    for (name, t) in set.iter() {
        if !skip.contains(&name) {
            out.push(Entry { name: name.to_string(), shape: t.shape.clone(), source: Source::Host(t) });
        }
    }
}

/// Write `<dest_dir>/model.safetensors` for the checkpoint at `checkpoint_dir`. `dest_dir` must exist. Experts are read
/// through a read-only tier, so the checkpoint is never touched.
pub fn export_safetensors(checkpoint_dir: &Path, dest_dir: &Path) -> Result<SafetensorsReport> {
    let ck = Checkpoint::read(checkpoint_dir)?;
    let man = &ck.data.manifest;
    let (d_model, d_ff) = match (man.d_model, man.d_ff) {
        (Some(d), Some(f)) => (d, f),
        _ => match man.parsed_cfg() {
            Ok(Some(p)) => (p.model.d_model as usize, p.model.pool_d_ff as usize),
            _ => return Err(StoreError::Invalid("the checkpoint does not say how large its experts are".into())),
        },
    };
    let uids = ck.uids();
    let mut entries: Vec<Entry> = Vec::new();
    // `head.weight` is the same tensor as `tok_emb.weight` (tied): store it once
    host_entries(&ck.data.core, &["head.weight"], &mut entries);
    host_entries(&ck.data.routers, &[], &mut entries);
    for &uid in &uids {
        for (which, (leaf, shape)) in
            [("w1", vec![d_ff, d_model]), ("w3", vec![d_ff, d_model]), ("w2", vec![d_model, d_ff])]
                .into_iter()
                .enumerate()
        {
            entries.push(Entry {
                name: format!("pool.experts.{uid}.{leaf}"),
                shape,
                source: Source::Expert { uid, which: which as u8 },
            });
        }
    }

    // header
    let mut offset = 0u64;
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        json!({
            "format": "pt",
            "writer": format!("minagi-core {}", env!("CARGO_PKG_VERSION")),
            "step": man.step.map(|s| s.to_string()).unwrap_or_default(),
            "experts": uids.len().to_string(),
        }),
    );
    for e in &entries {
        let end = offset + e.bytes();
        header.insert(e.name.clone(), json!({ "dtype": "F32", "shape": e.shape, "data_offsets": [offset, end] }));
        offset = end;
    }
    let mut header_bytes =
        serde_json::to_vec(&Value::Object(header)).map_err(|e| StoreError::Invalid(e.to_string()))?;
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }

    let path = dest_dir.join("model.safetensors");
    let tmp = dest_dir.join(".model.safetensors.tmp");
    let mut tiers = ck.open_tiers(2, true)?;
    let write = (|| -> Result<u64> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp).map_err(|e| StoreError::io_at(&tmp, e))?);
        let io = |e: std::io::Error| StoreError::io_at(&tmp, e);
        w.write_all(&(header_bytes.len() as u64).to_le_bytes()).map_err(io)?;
        w.write_all(&header_bytes).map_err(io)?;
        let mut written = 8 + header_bytes.len() as u64;
        let mut cache: Option<(u64, std::sync::Arc<super::tiers::ExpertEntry>)> = None;
        for e in &entries {
            let data: &[f32] = match &e.source {
                Source::Host(t) => &t.data,
                Source::Expert { uid, which } => {
                    if cache.as_ref().is_none_or(|(u, _)| u != uid) {
                        cache = Some((*uid, tiers.fetch(*uid)?));
                    }
                    let entry =
                        &cache.as_ref().map(|(_, e)| e).ok_or_else(|| StoreError::Invalid("no expert".into()))?;
                    match which {
                        0 => &entry.w1,
                        1 => &entry.w3,
                        _ => &entry.w2,
                    }
                }
            };
            let mut buf = Vec::with_capacity(data.len() * 4);
            for v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            w.write_all(&buf).map_err(io)?;
            written += buf.len() as u64;
        }
        w.flush().map_err(io)?;
        Ok(written)
    })();
    match write {
        Ok(bytes) => {
            std::fs::rename(&tmp, &path).map_err(|e| StoreError::io_at(&path, e))?;
            Ok(SafetensorsReport { tensors: entries.len(), bytes, experts: uids.len() })
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// One tensor of a safetensors file: `(name, dtype, shape, start, end)`, the byte range being relative to the data section.
pub type HeaderEntry = (String, String, Vec<usize>, u64, u64);

/// Read back the header of a safetensors file (for tests and tools).
pub fn read_header(path: &Path) -> Result<Vec<HeaderEntry>> {
    let bytes = std::fs::read(path).map_err(|e| StoreError::io_at(path, e))?;
    if bytes.len() < 8 {
        return Err(StoreError::Format("not a safetensors file".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8])) as usize;
    let header: Value =
        serde_json::from_slice(bytes.get(8..8 + n).ok_or_else(|| StoreError::Format("truncated header".into()))?)
            .map_err(|e| StoreError::Format(e.to_string()))?;
    let mut out = Vec::new();
    for (name, v) in header.as_object().into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let shape =
            v["shape"].as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        let off = &v["data_offsets"];
        out.push((
            name.clone(),
            v["dtype"].as_str().unwrap_or("").to_string(),
            shape,
            off[0].as_u64().unwrap_or(0),
            off[1].as_u64().unwrap_or(0),
        ));
    }
    Ok(out)
}
