//! Small file-system helpers shared by the store: atomic replacement of a file, hard-link-or-copy,
//! and directory sizes.
//!
//! The rule for everything that is written is the same as in the Python reference: write to a
//! temporary name next to the destination and rename it into place, so an interrupted write can
//! never leave half a file where a whole one is expected.

use super::{Result, StoreError};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// How [`link_or_copy`] got a file into its new place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// A hard link: no bytes were copied, both names refer to the same immutable content.
    Linked,
    /// The bytes were copied (the file systems differ, or the file system has no hard links).
    Copied,
}

/// The temporary sibling name used while `dest` is being written: `<name>.tmp`.
///
/// It deliberately does not end in `.npz`, so the Python reference's cleanup of the experts folder
/// (which deletes every `*.npz` it does not know) and our own directory scans ignore it.
pub fn tmp_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    dest.with_file_name(name)
}

/// Rename `from` over `to`, replacing `to` if it exists.
///
/// On Windows a rename onto a file that another process (a virus scanner, an indexer, a reader
/// that did not ask for delete sharing) has open fails with a sharing violation or "access
/// denied"; those clear within milliseconds, so a few short retries make the replacement reliable
/// there. Elsewhere this is a plain `rename`.
pub fn rename_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        let mut attempt = 0;
        loop {
            match fs::rename(from, to) {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 10 && matches!(e.kind(), std::io::ErrorKind::PermissionDenied) => {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
                Err(e) => return Err(e),
            }
        }
    }
    #[cfg(not(windows))]
    {
        fs::rename(from, to)
    }
}

/// Atomically create or replace the file `dest`: `write` fills a buffered writer onto
/// `<dest>.tmp`, then the temporary is renamed over `dest`.
///
/// Returns the size of the finished file. On any error the temporary is removed and `dest` is left
/// exactly as it was. With `durable` the data is also forced to disk before the rename (slow on
/// some systems; the per-expert write-back does not ask for it, checkpoint files do).
pub fn write_atomic<F>(dest: &Path, durable: bool, write: F) -> Result<u64>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<()>,
{
    let tmp = tmp_path(dest);
    let result = (|| -> Result<u64> {
        let file = File::create(&tmp).map_err(|e| StoreError::io_at(&tmp, e))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        write(&mut w)?;
        w.flush().map_err(|e| StoreError::io_at(&tmp, e))?;
        let file = w.into_inner().map_err(|e| StoreError::io_at(&tmp, e.into_error()))?;
        let len = file.metadata().map_err(|e| StoreError::io_at(&tmp, e))?.len();
        if durable {
            file.sync_all().map_err(|e| StoreError::io_at(&tmp, e))?;
        }
        drop(file);
        rename_replace(&tmp, dest).map_err(|e| StoreError::io_at(dest, e))?;
        Ok(len)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Put `src` at `dst` without copying if the file system allows (a hard link), otherwise copy it.
///
/// Hard links are safe here because nothing in the store ever rewrites a file in place: every
/// writer creates a new file and renames it over the old name, which leaves the other link's
/// content untouched. That is what makes a checkpoint of a 1.6 GB expert pool cost a few hundred
/// directory entries instead of 1.6 GB of copying.
pub fn link_or_copy(src: &Path, dst: &Path) -> Result<LinkKind> {
    match fs::hard_link(src, dst) {
        Ok(()) => Ok(LinkKind::Linked),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(StoreError::io_at(src, e)),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(StoreError::io_at(dst, e)),
        Err(_) => {
            // Cross-device, or a file system without hard links: fall back to a real copy.
            fs::copy(src, dst).map_err(|e| StoreError::io_at(dst, e))?;
            Ok(LinkKind::Copied)
        }
    }
}

/// Total size in bytes of every file under `dir` (recursively). Missing directories count as 0.
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let Ok(rd) = fs::read_dir(dir) else { return 0 };
    for entry in rd.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += dir_size(&entry.path());
        } else {
            total += meta.len();
        }
    }
    total
}

/// Best-effort flush of a directory's entries to disk (no-op where unsupported).
pub fn sync_dir(dir: &Path) {
    #[cfg(unix)]
    if let Ok(f) = File::open(dir) {
        let _ = f.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// A fresh empty directory under the system temp dir for a unit test (the caller removes it).
#[cfg(test)]
pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "minagi-store-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_replaces_whole_or_not_at_all() {
        let d = scratch_dir("atomic");
        let f = d.join("x.npz");
        assert_eq!(write_atomic(&f, false, |w| Ok(w.write_all(b"first")?)).unwrap(), 5);
        assert_eq!(fs::read(&f).unwrap(), b"first");
        // a failing writer leaves the old file and no temporary behind
        let err = write_atomic(&f, false, |w| {
            w.write_all(b"partial")?;
            Err(StoreError::Format("boom".into()))
        });
        assert!(err.is_err());
        assert_eq!(fs::read(&f).unwrap(), b"first");
        assert!(!tmp_path(&f).exists());
        assert_eq!(write_atomic(&f, true, |w| Ok(w.write_all(b"second!")?)).unwrap(), 7);
        assert_eq!(fs::read(&f).unwrap(), b"second!");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tmp_name_is_a_sibling_that_does_not_look_like_an_expert_file() {
        let t = tmp_path(Path::new("/a/b/e00012.npz"));
        assert_eq!(t, Path::new("/a/b/e00012.npz.tmp"));
        assert!(!t.to_string_lossy().ends_with(".npz"));
    }

    #[test]
    fn link_or_copy_links_and_replacing_the_source_does_not_touch_the_link() {
        let d = scratch_dir("link");
        let src = d.join("a.bin");
        let dst = d.join("b.bin");
        fs::write(&src, b"old").unwrap();
        let kind = link_or_copy(&src, &dst).unwrap();
        assert_eq!(kind, LinkKind::Linked);
        // the store only ever replaces files by rename, so the link keeps the old content
        write_atomic(&src, false, |w| Ok(w.write_all(b"new")?)).unwrap();
        assert_eq!(fs::read(&src).unwrap(), b"new");
        assert_eq!(fs::read(&dst).unwrap(), b"old");
        assert!(matches!(link_or_copy(&d.join("missing"), &d.join("c")), Err(StoreError::IoAt { .. })));
        // an existing destination is an error, never silently overwritten
        assert!(matches!(link_or_copy(&src, &dst), Err(StoreError::IoAt { .. })));
        assert_eq!(fs::read(&dst).unwrap(), b"old");
        assert_eq!(dir_size(&d), 6);
        fs::remove_dir_all(&d).ok();
    }
}
