//! NumPy `.npy` format (v1.0 writer; v1.0/2.0/3.0 reader), little-endian, C order.
//!
//! The header is the Python-literal dict numpy itself writes,
//! `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`, padded with spaces and a final
//! newline so that the data starts at a multiple of 64 bytes (matching numpy >= 1.17 byte for
//! byte).

use super::bf16;
use super::{Result, StoreError};
use std::io::{Read, Write};

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const ALIGN: usize = 64;

/// Typed, owned array contents (flat, C order).
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    /// `<f4`
    F32(Vec<f32>),
    /// `<f8`
    F64(Vec<f64>),
    /// `<i2` (also the carrier for bf16 bit patterns, see [`NpyArray::from_bf16`])
    I16(Vec<i16>),
    /// `<i4`
    I32(Vec<i32>),
    /// `<i8`
    I64(Vec<i64>),
    /// `|u1`
    U8(Vec<u8>),
}

impl NpyData {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            NpyData::F32(v) => v.len(),
            NpyData::F64(v) => v.len(),
            NpyData::I16(v) => v.len(),
            NpyData::I32(v) => v.len(),
            NpyData::I64(v) => v.len(),
            NpyData::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// numpy `descr` string for this element type.
    pub fn descr(&self) -> &'static str {
        match self {
            NpyData::F32(_) => "<f4",
            NpyData::F64(_) => "<f8",
            NpyData::I16(_) => "<i2",
            NpyData::I32(_) => "<i4",
            NpyData::I64(_) => "<i8",
            NpyData::U8(_) => "|u1",
        }
    }

    fn elem_size(&self) -> usize {
        match self {
            NpyData::F32(_) | NpyData::I32(_) => 4,
            NpyData::F64(_) | NpyData::I64(_) => 8,
            NpyData::I16(_) => 2,
            NpyData::U8(_) => 1,
        }
    }
}

/// An n-dimensional array: shape plus flat C-order data. A 0-d array has an empty shape and
/// exactly one element.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    shape: Vec<usize>,
    data: NpyData,
}

impl NpyArray {
    /// Build an array, checking that `data.len()` equals the product of `shape`.
    pub fn new(shape: Vec<usize>, data: NpyData) -> Result<Self> {
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
        if n != Some(data.len()) {
            return Err(StoreError::Format(format!("shape {shape:?} does not match {} elements", data.len())));
        }
        Ok(Self { shape, data })
    }

    pub fn f32(shape: Vec<usize>, v: Vec<f32>) -> Result<Self> {
        Self::new(shape, NpyData::F32(v))
    }

    pub fn i16(shape: Vec<usize>, v: Vec<i16>) -> Result<Self> {
        Self::new(shape, NpyData::I16(v))
    }

    pub fn i64(shape: Vec<usize>, v: Vec<i64>) -> Result<Self> {
        Self::new(shape, NpyData::I64(v))
    }

    /// 0-d `<f8` scalar (e.g. a learning rate or step counter).
    pub fn scalar_f64(x: f64) -> Self {
        Self { shape: vec![], data: NpyData::F64(vec![x]) }
    }

    /// Pack `f32` values to bf16 (round-to-nearest-even) stored as `<i2`.
    pub fn from_bf16(shape: Vec<usize>, v: &[f32]) -> Result<Self> {
        Self::i16(shape, bf16::pack(v))
    }

    /// Interpret an `<i2` array as bf16 bit patterns and widen to `f32`.
    pub fn to_f32_from_bf16(&self) -> Option<Vec<f32>> {
        match &self.data {
            NpyData::I16(v) => Some(bf16::unpack(v)),
            _ => None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &NpyData {
        &self.data
    }

    pub fn into_data(self) -> NpyData {
        self.data
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            NpyData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match &self.data {
            NpyData::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i16(&self) -> Option<&[i16]> {
        match &self.data {
            NpyData::I16(v) => Some(v),
            _ => None,
        }
    }

    /// Total payload size in bytes (excluding the header).
    pub fn nbytes(&self) -> usize {
        self.data.len() * self.data.elem_size()
    }

    /// Exact equality including NaN payloads (bitwise for floats), unlike `PartialEq`.
    pub fn bit_eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        match (&self.data, &other.data) {
            (NpyData::F32(a), NpyData::F32(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (NpyData::F64(a), NpyData::F64(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (a, b) => a == b,
        }
    }
}

/// Spaces needed to bring `unpadded` (prefix + header + newline) to a multiple of 64.
/// numpy always pads at least one byte: an already aligned header gets a whole extra block.
fn padding(unpadded: usize) -> usize {
    ALIGN - (unpadded % ALIGN)
}

/// Header text (without magic/length prefix) for `descr`/`shape`, padded like numpy: spaces then
/// `\n`, total file prefix a multiple of 64 bytes.
fn header_text(descr: &str, shape: &[usize]) -> String {
    let shape_s = match shape {
        [] => "()".to_string(),
        [d] => format!("({d},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    // magic(6) + version(2) + len(2) + header + '\n'
    let pad = padding(10 + h.len() + 1);
    h.extend(std::iter::repeat_n(' ', pad));
    h.push('\n');
    h
}

/// The complete `.npy` prefix (magic, version 1.0, header length, header) for an array.
pub fn header_bytes(descr: &str, shape: &[usize]) -> Result<Vec<u8>> {
    let h = header_text(descr, shape);
    let len = u16::try_from(h.len())
        .map_err(|_| StoreError::Unsupported("npy v1.0 header longer than 65535 bytes".into()))?;
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    Ok(out)
}

/// An element type that can be written to a `.npy` stream straight from a slice.
pub trait NpyElement: Copy {
    /// numpy `descr` string (`<f4`, `<i2`, ...).
    const DESCR: &'static str;
    /// Append the little-endian bytes of `self`.
    fn push_le(self, out: &mut Vec<u8>);
}

macro_rules! npy_element {
    ($ty:ty, $descr:expr) => {
        impl NpyElement for $ty {
            const DESCR: &'static str = $descr;
            #[inline]
            fn push_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
npy_element!(f32, "<f4");
npy_element!(f64, "<f8");
npy_element!(i16, "<i2");
npy_element!(i32, "<i4");
npy_element!(i64, "<i8");
npy_element!(u8, "|u1");

/// Write `data` (flat, C order) as a complete `.npy` stream with the given shape, without first
/// building an owning [`NpyArray`]. This is what lets an expert's 4 MB weight matrices go to disk
/// without a copy.
pub fn write_npy_slice<W: Write, T: NpyElement>(w: &mut W, shape: &[usize], data: &[T]) -> Result<()> {
    let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if n != Some(data.len()) {
        return Err(StoreError::Format(format!("shape {shape:?} does not match {} elements", data.len())));
    }
    w.write_all(&header_bytes(T::DESCR, shape)?)?;
    let mut buf = Vec::with_capacity(1 << 16);
    for chunk in data.chunks(8192) {
        buf.clear();
        for &x in chunk {
            x.push_le(&mut buf);
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

/// Serialise `arr` as a complete `.npy` stream.
pub fn write_npy<W: Write>(w: &mut W, arr: &NpyArray) -> Result<()> {
    match &arr.data {
        NpyData::F32(v) => write_npy_slice(w, &arr.shape, v),
        NpyData::F64(v) => write_npy_slice(w, &arr.shape, v),
        NpyData::I16(v) => write_npy_slice(w, &arr.shape, v),
        NpyData::I32(v) => write_npy_slice(w, &arr.shape, v),
        NpyData::I64(v) => write_npy_slice(w, &arr.shape, v),
        NpyData::U8(v) => write_npy_slice(w, &arr.shape, v),
    }
}

/// Serialise to a fresh byte vector.
pub fn to_bytes(arr: &NpyArray) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(80 + arr.nbytes());
    write_npy(&mut out, arr)?;
    Ok(out)
}

/// Read a whole `.npy` stream.
pub fn read_npy<R: Read>(r: &mut R) -> Result<NpyArray> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    from_bytes(&bytes)
}

/// What the first bytes of a `.npy` say about the array that follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    /// numpy dtype string, for example `<f4`.
    pub descr: String,
    pub shape: Vec<usize>,
    pub fortran_order: bool,
}

impl NpyHeader {
    /// Number of elements (`None` if the shape overflows `usize`).
    pub fn numel(&self) -> Option<usize> {
        self.shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
    }
}

/// Read only the header of a `.npy` stream (a few hundred bytes), leaving the payload unread.
///
/// Used to describe the contents of large archives (for example 150 expert files) without
/// decoding any tensor.
pub fn read_header<R: Read>(r: &mut R) -> Result<NpyHeader> {
    let eof = |e: std::io::Error| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            StoreError::Format("truncated npy header".into())
        } else {
            StoreError::Io(e)
        }
    };
    let mut pre = [0u8; 10];
    r.read_exact(&mut pre).map_err(eof)?;
    if &pre[..6] != MAGIC {
        return Err(StoreError::Format("missing npy magic".into()));
    }
    let (major, minor) = (pre[6], pre[7]);
    let hdr_len = match major {
        1 => u16::from_le_bytes([pre[8], pre[9]]) as usize,
        2 | 3 => {
            let mut more = [0u8; 2];
            r.read_exact(&mut more).map_err(eof)?;
            u32::from_le_bytes([pre[8], pre[9], more[0], more[1]]) as usize
        }
        _ => return Err(StoreError::Unsupported(format!("npy format version {major}.{minor}"))),
    };
    if hdr_len > 1 << 20 {
        return Err(StoreError::Format(format!("npy header claims {hdr_len} bytes")));
    }
    let mut buf = vec![0u8; hdr_len];
    r.read_exact(&mut buf).map_err(eof)?;
    let text =
        std::str::from_utf8(&buf).map_err(|_| StoreError::Format("npy header is not valid utf-8/ascii".into()))?;
    let (descr, fortran_order, shape) = parse_header(text)?;
    Ok(NpyHeader { descr, shape, fortran_order })
}

/// Parse a `.npy` file image. The payload must be exactly the size the header promises.
pub fn from_bytes(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(StoreError::Format("missing npy magic".into()));
    }
    let (major, minor) = (bytes[6], bytes[7]);
    let (hdr_start, hdr_len): (usize, usize) = match major {
        1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(StoreError::Format("truncated npy header".into()));
            }
            (12, u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize)
        }
        _ => return Err(StoreError::Unsupported(format!("npy format version {major}.{minor}"))),
    };
    let hdr_end = hdr_start
        .checked_add(hdr_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| StoreError::Format("truncated npy header".into()))?;
    let header = std::str::from_utf8(&bytes[hdr_start..hdr_end])
        .map_err(|_| StoreError::Format("npy header is not valid utf-8/ascii".into()))?;
    let (descr, fortran, shape) = parse_header(header)?;
    if fortran {
        return Err(StoreError::Unsupported("fortran_order=True".into()));
    }
    let payload = &bytes[hdr_end..];
    let n = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| StoreError::Format("shape overflows usize".into()))?;
    macro_rules! decode {
        ($variant:ident, $ty:ty) => {{
            let w = std::mem::size_of::<$ty>();
            if payload.len() != n * w {
                return Err(StoreError::Format(format!(
                    "payload is {} bytes, header promises {} ({} x {w})",
                    payload.len(),
                    n * w,
                    n
                )));
            }
            NpyData::$variant(
                payload
                    .chunks_exact(w)
                    .map(|c| {
                        let arr: [u8; std::mem::size_of::<$ty>()] = c.try_into().unwrap();
                        <$ty>::from_le_bytes(arr)
                    })
                    .collect(),
            )
        }};
    }
    let data = match descr.as_str() {
        "<f4" => decode!(F32, f32),
        "<f8" => decode!(F64, f64),
        "<i2" => decode!(I16, i16),
        "<i4" => decode!(I32, i32),
        "<i8" => decode!(I64, i64),
        "|u1" | "<u1" => decode!(U8, u8),
        d if d.starts_with('>') => return Err(StoreError::Unsupported(format!("big-endian dtype {d}"))),
        d => return Err(StoreError::Unsupported(format!("dtype {d}"))),
    };
    NpyArray::new(shape, data)
}

/// Text following `'key':` (single or double quoted key), whitespace skipped.
fn value_after<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    for q in ['\'', '"'] {
        let pat = format!("{q}{key}{q}");
        if let Some(pos) = header.find(&pat) {
            let rest = header[pos + pat.len()..].trim_start();
            let rest = rest.strip_prefix(':')?;
            return Some(rest.trim_start());
        }
    }
    None
}

/// Parse the header dict into `(descr, fortran_order, shape)`.
fn parse_header(header: &str) -> Result<(String, bool, Vec<usize>)> {
    let bad = |m: &str| StoreError::Format(format!("bad npy header ({m}): {header:?}"));
    let h = header.trim();
    if !h.starts_with('{') || !h.ends_with('}') {
        return Err(bad("not a dict"));
    }
    let d = value_after(h, "descr").ok_or_else(|| bad("no descr"))?;
    let quote = d.chars().next().filter(|c| *c == '\'' || *c == '"');
    let Some(quote) = quote else {
        return Err(StoreError::Unsupported("structured / non-string descr".into()));
    };
    let d = &d[1..];
    let end = d.find(quote).ok_or_else(|| bad("unterminated descr"))?;
    let descr = d[..end].to_string();

    let f = value_after(h, "fortran_order").ok_or_else(|| bad("no fortran_order"))?;
    let fortran = if f.starts_with("False") {
        false
    } else if f.starts_with("True") {
        true
    } else {
        return Err(bad("fortran_order not a bool"));
    };

    let s = value_after(h, "shape").ok_or_else(|| bad("no shape"))?;
    let s = s.strip_prefix('(').ok_or_else(|| bad("shape not a tuple"))?;
    let close = s.find(')').ok_or_else(|| bad("unterminated shape"))?;
    let mut shape = Vec::new();
    for part in s[..close].split(',') {
        let part = part.trim().trim_end_matches('L');
        if part.is_empty() {
            continue;
        }
        shape.push(part.parse::<usize>().map_err(|_| bad("shape entry not an integer"))?);
    }
    Ok((descr, fortran, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.save` of `np.arange(6, dtype='<f4').reshape(2,3)` (generated with numpy 2.x).
    fn numpy_arange_2x3() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"\x93NUMPY\x01\x00v\x00");
        v.extend_from_slice(b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }");
        // padding to 128 bytes total (118-byte header incl. newline)
        let pad = 128 - v.len() - 1;
        v.extend(std::iter::repeat_n(b' ', pad));
        v.push(b'\n');
        for x in 0..6u32 {
            v.extend_from_slice(&(x as f32).to_le_bytes());
        }
        v
    }

    #[test]
    fn header_matches_numpy_layout() {
        let a = NpyArray::f32(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
        assert_eq!(to_bytes(&a).unwrap(), numpy_arange_2x3());
    }

    #[test]
    fn header_is_64_aligned_for_many_shapes() {
        for rank in 0..5 {
            for base in [0usize, 1, 7, 12, 123, 1234, 99999, 123456789] {
                let shape: Vec<usize> = (0..rank).map(|i| base + i).collect();
                for d in ["<f4", "<f8", "<i2", "|u1"] {
                    let h = header_bytes(d, &shape).unwrap();
                    assert_eq!(h.len() % 64, 0, "{d} {shape:?}");
                    assert_eq!(*h.last().unwrap(), b'\n');
                    let hl = u16::from_le_bytes([h[8], h[9]]) as usize;
                    assert_eq!(10 + hl, h.len());
                }
            }
        }
    }

    #[test]
    fn header_only_read_matches_full_read_and_slice_writer_matches_array_writer() {
        let a = NpyArray::f32(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
        let bytes = to_bytes(&a).unwrap();
        let h = read_header(&mut &bytes[..]).unwrap();
        assert_eq!(h, NpyHeader { descr: "<f4".into(), shape: vec![2, 3], fortran_order: false });
        assert_eq!(h.numel(), Some(6));
        let mut via_slice = Vec::new();
        write_npy_slice(&mut via_slice, &[2, 3], a.as_f32().unwrap()).unwrap();
        assert_eq!(via_slice, bytes);
        // 0-d and i16 too
        let s = NpyArray::scalar_f64(1.5);
        let sb = to_bytes(&s).unwrap();
        let sh = read_header(&mut &sb[..]).unwrap();
        assert_eq!((sh.descr.as_str(), sh.shape.len()), ("<f8", 0));
        // wrong element count is an error, truncation too
        assert!(write_npy_slice(&mut Vec::new(), &[2, 2], &[1.0f32]).is_err());
        assert!(matches!(read_header(&mut &bytes[..7]), Err(StoreError::Format(_))));
        assert!(matches!(read_header(&mut &b"nonsense nonsense"[..]), Err(StoreError::Format(_))));
    }

    #[test]
    fn padding_matches_numpy_rule() {
        // numpy: padlen = 64 - ((10 + len(header) + 1) % 64), never zero.
        assert_eq!(padding(64), 64);
        assert_eq!(padding(65), 63);
        assert_eq!(padding(127), 1);
        assert_eq!(padding(128), 64);
        assert_eq!(padding(70), 58);
    }

    #[test]
    fn roundtrip_all_dtypes_and_shapes() {
        let cases = vec![
            NpyArray::f32(vec![], vec![3.5]).unwrap(),
            NpyArray::f32(vec![0], vec![]).unwrap(),
            NpyArray::f32(vec![4, 0, 2], vec![]).unwrap(),
            NpyArray::f32(vec![5], vec![1.0, -0.0, f32::INFINITY, f32::NAN, 1e-45]).unwrap(),
            NpyArray::new(vec![2, 2], NpyData::F64(vec![1.0, -2.5, 1e300, f64::MIN_POSITIVE])).unwrap(),
            NpyArray::i16(vec![3, 1], vec![i16::MIN, -1, i16::MAX]).unwrap(),
            NpyArray::new(vec![2], NpyData::I32(vec![i32::MIN, i32::MAX])).unwrap(),
            NpyArray::i64(vec![2, 1, 2], vec![i64::MIN, -1, 0, i64::MAX]).unwrap(),
            NpyArray::new(vec![3], NpyData::U8(vec![0, 127, 255])).unwrap(),
            NpyArray::scalar_f64(0.1),
        ];
        for a in cases {
            let bytes = to_bytes(&a).unwrap();
            let b = from_bytes(&bytes).unwrap();
            assert!(a.bit_eq(&b), "{a:?} != {b:?}");
            assert_eq!(a.shape(), b.shape());
            let b2 = read_npy(&mut std::io::Cursor::new(&bytes)).unwrap();
            assert!(a.bit_eq(&b2));
        }
    }

    #[test]
    fn scalar_header_uses_empty_tuple() {
        let b = to_bytes(&NpyArray::scalar_f64(2.0)).unwrap();
        let text = std::str::from_utf8(&b[10..]).unwrap();
        assert!(text.contains("'descr': '<f8'"));
        assert!(text.contains("'shape': ()"));
    }

    #[test]
    fn reads_numpy_variants_of_the_header() {
        // v2.0 (u32 length), double-quoted keys, no trailing comma, 1-tuple.
        let hdr = "{\"descr\": \"<i2\", \"fortran_order\": False, \"shape\": (3,)}";
        let mut padded = hdr.to_string();
        while !(12 + padded.len() + 1).is_multiple_of(64) {
            padded.push(' ');
        }
        padded.push('\n');
        let mut v = Vec::new();
        v.extend_from_slice(b"\x93NUMPY\x02\x00");
        v.extend_from_slice(&(padded.len() as u32).to_le_bytes());
        v.extend_from_slice(padded.as_bytes());
        for x in [1i16, -2, 3] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        let a = from_bytes(&v).unwrap();
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.as_i16().unwrap(), &[1, -2, 3]);
    }

    #[test]
    fn rejects_bad_input() {
        let good = to_bytes(&NpyArray::f32(vec![2], vec![1.0, 2.0]).unwrap()).unwrap();
        // bad magic
        let mut b = good.clone();
        b[1] = b'X';
        assert!(matches!(from_bytes(&b), Err(StoreError::Format(_))));
        // truncated payload and trailing garbage
        assert!(matches!(from_bytes(&good[..good.len() - 1]), Err(StoreError::Format(_))));
        let mut b = good.clone();
        b.push(0);
        assert!(matches!(from_bytes(&b), Err(StoreError::Format(_))));
        // truncated header
        assert!(matches!(from_bytes(&good[..20]), Err(StoreError::Format(_))));
        assert!(matches!(from_bytes(&[]), Err(StoreError::Format(_))));
        // unsupported: fortran order, big endian, other dtype, version 4
        let mk = |h: &str| {
            let mut padded = h.to_string();
            while !(10 + padded.len() + 1).is_multiple_of(64) {
                padded.push(' ');
            }
            padded.push('\n');
            let mut v = b"\x93NUMPY\x01\x00".to_vec();
            v.extend_from_slice(&(padded.len() as u16).to_le_bytes());
            v.extend_from_slice(padded.as_bytes());
            v.extend_from_slice(&[0u8; 8]);
            v
        };
        assert!(matches!(
            from_bytes(&mk("{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }")),
            Err(StoreError::Unsupported(_))
        ));
        assert!(matches!(
            from_bytes(&mk("{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }")),
            Err(StoreError::Unsupported(_))
        ));
        assert!(matches!(
            from_bytes(&mk("{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }")),
            Err(StoreError::Unsupported(_))
        ));
        let mut v4 = good.clone();
        v4[6] = 4;
        assert!(matches!(from_bytes(&v4), Err(StoreError::Unsupported(_))));
    }

    #[test]
    fn shape_data_mismatch_is_an_error() {
        assert!(NpyArray::f32(vec![2, 2], vec![1.0; 3]).is_err());
        assert!(NpyArray::f32(vec![], vec![]).is_err());
    }

    #[test]
    fn bf16_helpers_roundtrip() {
        let a = NpyArray::from_bf16(vec![2, 2], &[1.0, -0.5, 1.2345, 1e-8]).unwrap();
        assert_eq!(a.data().descr(), "<i2");
        let back = a.to_f32_from_bf16().unwrap();
        assert_eq!(back[0], 1.0);
        assert_eq!(back[1], -0.5);
        assert!((back[2] - 1.2345).abs() < 0.02);
        let again = NpyArray::from_bf16(vec![2, 2], &back).unwrap();
        assert!(a.bit_eq(&again));
    }
}
