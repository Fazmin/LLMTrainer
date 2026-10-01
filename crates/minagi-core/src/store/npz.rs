//! NumPy `.npz`: a ZIP archive of `<key>.npy` entries.
//!
//! * [`NpzWriter`] writes *stored* (uncompressed) entries with deterministic timestamps, like
//!   `numpy.savez`.
//! * [`NpzReader`] reads stored or deflated entries (`numpy.savez` / `numpy.savez_compressed`).
//!
//! Keys are arbitrary strings; the entry name is `key + ".npy"` and the suffix is stripped again
//! on reading, exactly as `numpy.load` does.

use super::npy::{self, NpyArray};
use super::{Result, StoreError};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const SUFFIX: &str = ".npy";

/// Streaming writer: add arrays one at a time, then [`finish`](NpzWriter::finish).
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
}

impl<W: Write + Seek> NpzWriter<W> {
    pub fn new(w: W) -> Self {
        Self { zip: ZipWriter::new(w) }
    }

    /// Add one array under `key` (stored, uncompressed).
    pub fn add(&mut self, key: &str, arr: &NpyArray) -> Result<()> {
        let size = (arr.nbytes() + 128) as u64;
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .large_file(size >= u32::MAX as u64);
        self.zip.start_file(format!("{key}{SUFFIX}"), opts)?;
        npy::write_npy(&mut self.zip, arr)
    }

    /// Add one array under `key` straight from a slice (stored, uncompressed), without building an
    /// owning [`NpyArray`] first. Produces exactly the bytes [`add`](NpzWriter::add) would.
    pub fn add_slice<T: npy::NpyElement>(&mut self, key: &str, shape: &[usize], data: &[T]) -> Result<()> {
        let size = (std::mem::size_of_val(data) + 128) as u64;
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .large_file(size >= u32::MAX as u64);
        self.zip.start_file(format!("{key}{SUFFIX}"), opts)?;
        npy::write_npy_slice(&mut self.zip, shape, data)
    }

    /// Write the central directory and return the underlying writer.
    pub fn finish(self) -> Result<W> {
        Ok(self.zip.finish()?)
    }
}

/// Write `entries` to `path` as an `.npz` (entries keep their order).
pub fn write_npz_file<'a, I>(path: impl AsRef<Path>, entries: I) -> Result<()>
where
    I: IntoIterator<Item = (&'a str, &'a NpyArray)>,
{
    let mut w = NpzWriter::new(BufWriter::new(File::create(path)?));
    for (k, a) in entries {
        w.add(k, a)?;
    }
    w.finish()?.flush()?;
    Ok(())
}

/// Random-access reader over an `.npz`.
pub struct NpzReader<R: Read + Seek> {
    zip: ZipArchive<R>,
    keys: Vec<String>,
}

impl NpzReader<BufReader<File>> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(BufReader::new(File::open(path)?))
    }
}

impl<R: Read + Seek> NpzReader<R> {
    pub fn new(r: R) -> Result<Self> {
        let zip = ZipArchive::new(r)?;
        let keys = zip.file_names().filter_map(|n| n.strip_suffix(SUFFIX).map(str::to_string)).collect();
        Ok(Self { zip, keys })
    }

    /// Keys (entry names without `.npy`), in archive order.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn contains(&self, key: &str) -> bool {
        self.keys.iter().any(|k| k == key)
    }

    /// Read and decode one array.
    pub fn get(&mut self, key: &str) -> Result<NpyArray> {
        let name = format!("{key}{SUFFIX}");
        let mut entry = match self.zip.by_name(&name) {
            Ok(e) => e,
            Err(zip::result::ZipError::FileNotFound) => return Err(StoreError::MissingKey(key.to_string())),
            Err(e) => return Err(e.into()),
        };
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        npy::from_bytes(&buf)
    }

    /// Read only the `.npy` header of one entry (dtype and shape), without decoding the payload.
    pub fn header(&mut self, key: &str) -> Result<npy::NpyHeader> {
        let name = format!("{key}{SUFFIX}");
        let mut entry = match self.zip.by_name(&name) {
            Ok(e) => e,
            Err(zip::result::ZipError::FileNotFound) => return Err(StoreError::MissingKey(key.to_string())),
            Err(e) => return Err(e.into()),
        };
        npy::read_header(&mut entry)
    }

    /// Read every array, in archive order.
    pub fn read_all(&mut self) -> Result<Vec<(String, NpyArray)>> {
        let keys = self.keys.clone();
        keys.into_iter()
            .map(|k| {
                let a = self.get(&k)?;
                Ok((k, a))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample() -> Vec<(String, NpyArray)> {
        vec![
            ("embed".into(), NpyArray::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            ("recur.0.mlp.router.weight".into(), NpyArray::f32(vec![4], vec![0.1, -0.2, 0.3, f32::NAN]).unwrap()),
            ("adam|m|recur.0.attn.wq".into(), NpyArray::from_bf16(vec![3], &[1.0, 2.0, -3.5]).unwrap()),
            ("step".into(), NpyArray::scalar_f64(1234.0)),
            ("empty".into(), NpyArray::f32(vec![0, 5], vec![]).unwrap()),
            ("ids".into(), NpyArray::i64(vec![3], vec![1, -2, 3]).unwrap()),
        ]
    }

    #[test]
    fn roundtrip_in_memory_with_odd_keys() {
        let entries = sample();
        let mut w = NpzWriter::new(Cursor::new(Vec::new()));
        for (k, a) in &entries {
            w.add(k, a).unwrap();
        }
        let bytes = w.finish().unwrap().into_inner();
        let mut r = NpzReader::new(Cursor::new(bytes)).unwrap();
        let keys: Vec<&str> = r.keys().iter().map(String::as_str).collect();
        assert_eq!(keys, entries.iter().map(|e| e.0.as_str()).collect::<Vec<_>>());
        assert!(r.contains("adam|m|recur.0.attn.wq"));
        for (k, a) in &entries {
            assert!(r.get(k).unwrap().bit_eq(a), "{k}");
        }
        assert_eq!(r.read_all().unwrap().len(), entries.len());
    }

    #[test]
    fn entries_are_stored_not_compressed_and_deterministic() {
        let entries = sample();
        let build = || {
            let mut w = NpzWriter::new(Cursor::new(Vec::new()));
            for (k, a) in &entries {
                w.add(k, a).unwrap();
            }
            w.finish().unwrap().into_inner()
        };
        let (a, b) = (build(), build());
        assert_eq!(a, b, "output must be reproducible byte for byte");
        let mut z = zip::ZipArchive::new(Cursor::new(a)).unwrap();
        for i in 0..z.len() {
            let f = z.by_index(i).unwrap();
            assert_eq!(f.compression(), CompressionMethod::Stored, "{}", f.name());
            assert!(f.name().ends_with(".npy"));
        }
    }

    #[test]
    fn reads_deflated_archives() {
        let entries = sample();
        let mut zw = ZipWriter::new(Cursor::new(Vec::new()));
        for (k, a) in &entries {
            zw.start_file(
                format!("{k}.npy"),
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
            )
            .unwrap();
            npy::write_npy(&mut zw, a).unwrap();
        }
        let bytes = zw.finish().unwrap().into_inner();
        let mut r = NpzReader::new(Cursor::new(bytes)).unwrap();
        for (k, a) in &entries {
            assert!(r.get(k).unwrap().bit_eq(a), "{k}");
        }
    }

    #[test]
    fn add_slice_equals_add_and_header_describes_without_decoding() {
        let entries = sample();
        let mut a = NpzWriter::new(Cursor::new(Vec::new()));
        let mut b = NpzWriter::new(Cursor::new(Vec::new()));
        for (k, arr) in &entries {
            a.add(k, arr).unwrap();
        }
        let f = entries[0].1.as_f32().unwrap();
        b.add_slice("embed", &[2, 3], f).unwrap();
        let ba = a.finish().unwrap().into_inner();
        let bb = b.finish().unwrap().into_inner();
        let mut ra = NpzReader::new(Cursor::new(ba)).unwrap();
        let mut rb = NpzReader::new(Cursor::new(bb)).unwrap();
        assert!(ra.get("embed").unwrap().bit_eq(&rb.get("embed").unwrap()));
        let h = ra.header("adam|m|recur.0.attn.wq").unwrap();
        assert_eq!((h.descr.as_str(), h.shape.as_slice()), ("<i2", &[3usize][..]));
        assert!(matches!(ra.header("nope"), Err(StoreError::MissingKey(_))));
    }

    #[test]
    fn missing_key_and_bad_archive() {
        let mut w = NpzWriter::new(Cursor::new(Vec::new()));
        w.add("a", &NpyArray::scalar_f64(1.0)).unwrap();
        let mut r = NpzReader::new(Cursor::new(w.finish().unwrap().into_inner())).unwrap();
        assert!(matches!(r.get("b"), Err(StoreError::MissingKey(k)) if k == "b"));
        assert!(NpzReader::new(Cursor::new(b"not a zip".to_vec())).is_err());
    }

    #[test]
    fn file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("minagi-npz-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.npz");
        let entries = sample();
        write_npz_file(&path, entries.iter().map(|(k, a)| (k.as_str(), a))).unwrap();
        let mut r = NpzReader::open(&path).unwrap();
        for (k, a) in &entries {
            assert!(r.get(k).unwrap().bit_eq(a));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
