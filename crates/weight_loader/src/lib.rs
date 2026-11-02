//! WeightLoaderCore — in-memory stand-in for SafeTensors / GGUF mapping (Phase 1).
//!
//! P0 keeps tensor bytes in an owned buffer and hands out zero-copy `&[u8]` views.
//! File mmap and real parsers stay on the P1 side of this API.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Element type stored in a tensor blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

impl DType {
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32 => core::mem::size_of::<f32>(),
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
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::NotFound => write!(f, "tensor not found"),
            WeightError::Duplicate => write!(f, "tensor name already registered"),
            WeightError::EmptyName => write!(f, "tensor name is empty"),
            WeightError::DTypeMismatch => write!(f, "tensor dtype is not f32"),
            WeightError::LengthMismatch => write!(f, "tensor byte length does not match shape"),
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
/// `insert_f32` writes native-endian f32 bytes. This is an in-memory fixture format, not SafeTensors.
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
}
