//! WeightLoaderCore — zero-copy weight views for Phase 1.
//!
//! [`MmapWeights`] is the in-memory f32 fixture behind the cycle smoke path.
//! [`SafeTensorsWeights`] parses a SafeTensors byte buffer with the official
//! `safetensors` crate (`std` feature off) and returns `&[u8]` views into that
//! buffer. [`bf16_view`] borrows a BF16 payload from that buffer and does not
//! decode it. GGUF stays out of this crate for now.
//!
//! The default build is `no_std` + `alloc`. Feature `host` (and this crate's
//! own tests) add `SafeTensorsMmap`, which maps a `.safetensors` file with
//! `memmap2` and lends those pages to [`SafeTensorsWeights`]. Tensor payloads
//! are not copied.
#![cfg_attr(all(not(test), not(feature = "host")), no_std)]

extern crate alloc;

mod safe_tensors;
pub use safe_tensors::SafeTensorsWeights;

#[cfg(any(test, feature = "host"))]
mod mmap;
#[cfg(any(test, feature = "host"))]
pub use mmap::{SafeTensorsMmap, WeightFileError};

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Element type stored in a tensor blob.
///
/// `F16` and `BF16` are SafeTensors storage tags. [`load_f32_tensor`] still
/// accepts only [`DType::F32`]. [`bf16_view`] borrows a BF16 payload and does
/// not convert it to f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32 => core::mem::size_of::<f32>(),
            DType::F16 | DType::BF16 => 2,
        }
    }
}

/// Shape and dtype for one named tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorInfo {
    pub fn numel(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }

    pub fn nbytes(&self) -> Option<usize> {
        self.numel()?.checked_mul(self.dtype.size_of())
    }
}

/// Failure from a weight lookup or blob decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightError {
    NotFound,
    Duplicate,
    EmptyName,
    DTypeMismatch,
    LengthMismatch,
    /// The byte buffer is not a SafeTensors file this loader can read.
    Parse,
    /// Storage dtype is outside the mapped set (`F32`, `F16`, `BF16`).
    UnsupportedDType,
}

/// Borrowed little-endian BF16 payload.
///
/// `bytes` points into the caller-owned buffer or mmap. `shape` points at the
/// loader index. Neither is a decoded `Vec<f32>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16View<'a> {
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// Borrow the BF16 tensor named `name`.
///
/// The dtype must be [`DType::BF16`] and the byte length must be
/// `2 * numel(shape)`. This does not decode elements and does not allocate
/// a payload copy. [`load_f32_tensor`] still rejects this dtype.
pub fn bf16_view<'a>(map: &'a impl WeightMap, name: &str) -> Result<Bf16View<'a>, WeightError> {
    let info = map.tensor_info(name)?;
    if info.dtype != DType::BF16 {
        return Err(WeightError::DTypeMismatch);
    }
    let nbytes = info.nbytes().ok_or(WeightError::LengthMismatch)?;
    let bytes = map.tensor_bytes(name)?;
    if bytes.len() != nbytes {
        return Err(WeightError::LengthMismatch);
    }
    Ok(Bf16View {
        shape: info.shape.as_slice(),
        bytes,
    })
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::NotFound => write!(f, "tensor not found"),
            WeightError::Duplicate => write!(f, "tensor name already registered"),
            WeightError::EmptyName => write!(f, "tensor name is empty"),
            WeightError::DTypeMismatch => write!(f, "tensor dtype does not match"),
            WeightError::LengthMismatch => write!(f, "tensor byte length does not match shape"),
            WeightError::Parse => write!(f, "weight buffer could not be parsed"),
            WeightError::UnsupportedDType => write!(f, "tensor dtype is not supported"),
        }
    }
}

/// Borrowed view of one tensor inside a [`WeightMap`].
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub info: TensorInfo,
    pub bytes: &'a [u8],
}

/// Named tensor table. P0 implementors may back this with memory rather than a file.
pub trait WeightMap {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightError>;
    fn tensor_info(&self, name: &str) -> Result<&TensorInfo, WeightError>;

    fn tensor(&self, name: &str) -> Result<TensorView<'_>, WeightError> {
        let info = self.tensor_info(name)?.clone();
        let bytes = self.tensor_bytes(name)?;
        Ok(TensorView { info, bytes })
    }
}

#[derive(Debug)]
struct Entry {
    info: TensorInfo,
    offset: usize,
    len: usize,
}

/// Owned byte blob plus a name index.
///
/// The buffer is the P0 stand-in for an mmap: callers borrow tensor bytes, they do not clone them.
/// `insert_f32` writes native-endian f32 bytes. `insert_bf16` stores the given
/// little-endian BF16 bytes and does not decode them. This is an in-memory
/// fixture format, not SafeTensors. A real file map is `SafeTensorsMmap` when
/// feature `host` is enabled.
#[derive(Debug, Default)]
pub struct MmapWeights {
    buf: Vec<u8>,
    index: BTreeMap<String, Entry>,
}

impl MmapWeights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_f32(
        &mut self,
        name: &str,
        shape: &[usize],
        data: &[f32],
    ) -> Result<(), WeightError> {
        if name.is_empty() {
            return Err(WeightError::EmptyName);
        }
        if self.index.contains_key(name) {
            return Err(WeightError::Duplicate);
        }
        let info = TensorInfo {
            shape: shape.to_vec(),
            dtype: DType::F32,
        };
        let numel = info.numel().ok_or(WeightError::LengthMismatch)?;
        if numel != data.len() {
            return Err(WeightError::LengthMismatch);
        }
        let nbytes = info.nbytes().ok_or(WeightError::LengthMismatch)?;
        let offset = self.buf.len();
        self.buf.reserve(nbytes);
        for value in data {
            self.buf.extend_from_slice(&value.to_ne_bytes());
        }
        self.index.insert(
            String::from(name),
            Entry {
                info,
                offset,
                len: nbytes,
            },
        );
        Ok(())
    }

    /// Store a little-endian BF16 blob.
    ///
    /// `data` is copied into the fixture buffer once, as raw bytes. Callers
    /// then borrow it. This does not expand BF16 to f32.
    pub fn insert_bf16(
        &mut self,
        name: &str,
        shape: &[usize],
        data: &[u8],
    ) -> Result<(), WeightError> {
        self.insert_bytes(name, DType::BF16, shape, data)
    }

    fn insert_bytes(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<(), WeightError> {
        if name.is_empty() {
            return Err(WeightError::EmptyName);
        }
        if self.index.contains_key(name) {
            return Err(WeightError::Duplicate);
        }
        let info = TensorInfo {
            shape: shape.to_vec(),
            dtype,
        };
        let nbytes = info.nbytes().ok_or(WeightError::LengthMismatch)?;
        if nbytes != bytes.len() {
            return Err(WeightError::LengthMismatch);
        }
        let offset = self.buf.len();
        self.buf.extend_from_slice(bytes);
        self.index.insert(
            String::from(name),
            Entry {
                info,
                offset,
                len: nbytes,
            },
        );
        Ok(())
    }
}

impl WeightMap for MmapWeights {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightError> {
        let entry = self.index.get(name).ok_or(WeightError::NotFound)?;
        self.buf
            .get(entry.offset..entry.offset + entry.len)
            .ok_or(WeightError::LengthMismatch)
    }

    fn tensor_info(&self, name: &str) -> Result<&TensorInfo, WeightError> {
        self.index
            .get(name)
            .map(|entry| &entry.info)
            .ok_or(WeightError::NotFound)
    }
}

/// Copy one f32 tensor out of a [`WeightMap`].
///
/// The byte view stays borrowed from the map; the returned `Vec<f32>` is a native-endian decode
/// so `tensor_core` can consume contiguous f32 without an alignment cast.
/// SafeTensors payloads are little-endian, which matches this decode on a little-endian host.
pub fn load_f32_tensor(
    map: &impl WeightMap,
    name: &str,
) -> Result<(TensorInfo, Vec<f32>), WeightError> {
    let view = map.tensor(name)?;
    if view.info.dtype != DType::F32 {
        return Err(WeightError::DTypeMismatch);
    }
    let expect = view.info.nbytes().ok_or(WeightError::LengthMismatch)?;
    if view.bytes.len() != expect || view.bytes.len() % DType::F32.size_of() != 0 {
        return Err(WeightError::LengthMismatch);
    }
    let mut data = Vec::with_capacity(view.bytes.len() / DType::F32.size_of());
    for chunk in view.bytes.chunks_exact(DType::F32.size_of()) {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(chunk);
        data.push(f32::from_ne_bytes(bytes));
    }
    Ok((view.info, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_weights_roundtrip_f32() {
        let mut weights = MmapWeights::new();
        weights.insert_f32("a", &[2], &[1.0, 2.0]).unwrap();
        weights.insert_f32("b", &[1, 2], &[3.0, 4.0]).unwrap();

        let (info_a, data_a) = load_f32_tensor(&weights, "a").unwrap();
        let (info_b, data_b) = load_f32_tensor(&weights, "b").unwrap();
        assert_eq!(info_a.shape, vec![2]);
        assert_eq!(info_a.dtype, DType::F32);
        assert_eq!(data_a, vec![1.0, 2.0]);
        assert_eq!(info_b.shape, vec![1, 2]);
        assert_eq!(data_b, vec![3.0, 4.0]);

        let bytes_a = weights.tensor_bytes("a").unwrap();
        let bytes_b = weights.tensor_bytes("b").unwrap();
        assert_eq!(bytes_a.len(), 8);
        assert_eq!(bytes_b.len(), 8);
        assert!(bytes_a.as_ptr() != bytes_b.as_ptr());
    }

    #[test]
    fn result_api_rejects_bad_lookup() {
        let mut weights = MmapWeights::new();
        assert_eq!(weights.tensor_bytes("missing"), Err(WeightError::NotFound));
        assert_eq!(
            weights.insert_f32("", &[1], &[1.0]),
            Err(WeightError::EmptyName)
        );
        weights.insert_f32("w", &[2], &[1.0, 2.0]).unwrap();
        assert_eq!(
            weights.insert_f32("w", &[2], &[1.0, 2.0]),
            Err(WeightError::Duplicate)
        );
        assert_eq!(
            weights.insert_f32("bad", &[2], &[1.0]),
            Err(WeightError::LengthMismatch)
        );
    }

    #[test]
    fn bf16_view_borrows_bytes_and_rejects_f32() {
        let mut weights = MmapWeights::new();
        // BF16 1.0 is 0x3F80, little-endian. Second element is not decoded here.
        let payload = [0x80, 0x3F, 0x00, 0xC0];
        weights.insert_bf16("row", &[1, 2], &payload).unwrap();
        let view = bf16_view(&weights, "row").unwrap();
        assert_eq!(view.shape, &[1, 2]);
        assert_eq!(view.bytes, &payload);
        assert_eq!(
            view.bytes.as_ptr(),
            weights.tensor_bytes("row").unwrap().as_ptr()
        );
        assert_eq!(view.bytes.len(), 4);
        assert_eq!(
            load_f32_tensor(&weights, "row"),
            Err(WeightError::DTypeMismatch)
        );

        weights.insert_f32("f32", &[1], &[1.0]).unwrap();
        assert_eq!(bf16_view(&weights, "f32"), Err(WeightError::DTypeMismatch));
        assert_eq!(bf16_view(&weights, "missing"), Err(WeightError::NotFound));
        assert_eq!(
            weights.insert_bf16("short", &[2], &[0x80, 0x3F]),
            Err(WeightError::LengthMismatch)
        );
    }
}
