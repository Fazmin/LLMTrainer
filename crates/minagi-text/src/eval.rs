//! Held-out evaluation text: one folder per domain under `val/`, read in order from the start so every evaluation
//! scores the same text and results are comparable over time.

use std::path::{Path, PathBuf};

use crate::error::{TextError, TextResult};
use crate::tokenizer::ByteTokenizer;

/// One sequential piece of a held-out file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalChunk {
    pub x: Vec<u16>,
    pub y: Vec<u16>,
    /// First chunk of a file: the engine resets its caches.
    pub new_text: bool,
}

struct Domain {
    name: String,
    files: Vec<PathBuf>,
}

pub struct EvalSet {
    domains: Vec<Domain>,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let ty = e.file_type()?;
        if ty.is_dir() {
            walk(&e.path(), out)?;
        } else if ty.is_file() {
            out.push(e.path());
        }
    }
    Ok(())
}

impl EvalSet {
    /// Open `root/val/<domain>/**`, limited to `enabled` domains when given.
    pub fn open(root: &Path, enabled: Option<&[String]>) -> TextResult<Self> {
        let val = root.join("val");
        let mut dirs: Vec<_> = std::fs::read_dir(&val)
            .map_err(|e| TextError::io(&val, e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
            .collect();
        dirs.sort_by_key(|e| e.file_name());
        let mut domains = Vec::new();
        for d in dirs {
            let name = d.file_name().to_string_lossy().to_string();
            if enabled.is_some_and(|en| !en.contains(&name)) {
                continue;
            }
            let mut files = Vec::new();
            walk(&d.path(), &mut files).map_err(|e| TextError::io(d.path(), e))?;
            if !files.is_empty() {
                domains.push(Domain { name, files });
            }
        }
        if domains.is_empty() {
            return Err(TextError::NoText(val.display().to_string()));
        }
        Ok(Self { domains })
    }

    pub fn domains(&self) -> Vec<String> {
        self.domains.iter().map(|d| d.name.clone()).collect()
    }

    /// Sequential chunks of `chunk` tokens from the start of the domain's files, until about `max_tokens` are scored.
    pub fn chunks(&self, domain: usize, chunk: usize, max_tokens: usize) -> TextResult<Vec<EvalChunk>> {
        let d = self.domains.get(domain).ok_or_else(|| TextError::Invalid(format!("no domain number {domain}")))?;
        let tok = ByteTokenizer::new();
        let mut out = Vec::new();
        let mut scored = 0usize;
        'files: for path in &d.files {
            let bytes = std::fs::read(path).map_err(|e| TextError::io(path, e))?;
            let t = tok.encode(&bytes);
            let mut i = 0;
            let mut first = true;
            while i + chunk < t.len() {
                out.push(EvalChunk {
                    x: t[i..i + chunk].to_vec(),
                    y: t[i + 1..i + chunk + 1].to_vec(),
                    new_text: first,
                });
                first = false;
                i += chunk;
                scored += chunk;
                if scored >= max_tokens {
                    break 'files;
                }
            }
        }
        if out.is_empty() {
            return Err(TextError::TooShort { chunk });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (lane, files) in [("stories", 3), ("arithmetic", 1)] {
            let d = dir.path().join("val").join(lane);
            std::fs::create_dir_all(&d).unwrap();
            for i in 0..files {
                std::fs::write(
                    d.join(format!("p{i}.txt")),
                    (0..1000).map(|k| ((k % 90) + 33) as u8).collect::<Vec<u8>>(),
                )
                .unwrap();
            }
        }
        dir
    }

    #[test]
    fn domains_are_listed_sorted_and_chunks_are_sequential() {
        let dir = tree();
        let set = EvalSet::open(dir.path(), None).unwrap();
        assert_eq!(set.domains(), vec!["arithmetic", "stories"]);
        let chunks = set.chunks(1, 100, 10_000).unwrap();
        // 3 files x 9 full chunks (the last 100 tokens need one more for their target).
        assert_eq!(chunks.len(), 27);
        assert!(chunks[0].new_text && !chunks[1].new_text && chunks[9].new_text);
        assert_eq!(&chunks[0].y[..99], &chunks[0].x[1..]);
        assert_eq!(chunks[0].y[99], chunks[1].x[0], "a chunk's last target is the next chunk's first input");
    }

    #[test]
    fn the_budget_limits_how_much_is_scored_and_results_are_repeatable() {
        let dir = tree();
        let set = EvalSet::open(dir.path(), None).unwrap();
        let a = set.chunks(1, 100, 350).unwrap();
        assert_eq!(a.len(), 4, "stops once at least 350 tokens are scored");
        assert_eq!(a, set.chunks(1, 100, 350).unwrap());
    }

    #[test]
    fn enabled_domains_filter_and_empty_is_an_error() {
        let dir = tree();
        let only = vec!["stories".to_string()];
        assert_eq!(EvalSet::open(dir.path(), Some(&only)).unwrap().domains(), vec!["stories"]);
        let none = vec!["nope".to_string()];
        assert!(matches!(EvalSet::open(dir.path(), Some(&none)), Err(TextError::NoText(_))));
        let set = EvalSet::open(dir.path(), None).unwrap();
        assert!(matches!(set.chunks(0, 5000, 100), Err(TextError::TooShort { .. })));
        assert!(set.chunks(9, 10, 10).is_err());
    }
}
