//! SafeTensors [`WeightMap`](crate::WeightMap).
//!
//! Built on the official `safetensors` crate with its `std` feature off.
//! [`SafeTensors::deserialize`](safetensors::SafeTensors::deserialize) borrows
//! the caller's buffer; tensor payloads are not copied.

use alloc::collections::BTreeMap;
use alloc::string::String;
use safetensors::tensor::TensorView as StView;
use safetensors::{Dtype as StDtype, SafeTensorError, SafeTensors};

use crate::{DType, TensorInfo, WeightError, WeightMap};

/// SafeTensors file borrowed from the caller's bytes.
///
/// `parse` accepts storage dtypes `F32`, `F16`, and `BF16`. Any other dtype
/// fails the whole buffer with [`WeightError::UnsupportedDType`]. Payload
/// bytes stay inside `bytes` (little-endian, row-major, as in the file).
#[derive(Debug)]
pub struct SafeTensorsWeights<'a> {
    inner: SafeTensors<'a>,
    index: BTreeMap<String, TensorInfo>,
}

impl<'a> SafeTensorsWeights<'a> {
    /// Parse a complete SafeTensors file.
    ///
    /// `bytes` is the whole file. The caller keeps that buffer alive (`Vec<u8>`
    /// today, mmap later); this type only borrows it.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, WeightError> {
        let inner = SafeTensors::deserialize(bytes).map_err(map_error)?;
        let mut index = BTreeMap::new();
        for (name, view) in inner.iter() {
            if name.is_empty() {
                return Err(WeightError::EmptyName);
            }
            let info = tensor_info(&view)?;
            if index.insert(String::from(name), info).is_some() {
                return Err(WeightError::Duplicate);
            }
        }
        Ok(Self { inner, index })
    }

    /// Number of tensors indexed from the header.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` when the file parsed and contained no tensors.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Tensor names in lexicographic order.
    ///
    /// The order matches the index built at parse time. Payload bytes are not touched.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}

fn tensor_info(view: &StView<'_>) -> Result<TensorInfo, WeightError> {
    let dtype = map_dtype(view.dtype())?;
    let info = TensorInfo {
        shape: view.shape().to_vec(),
        dtype,
    };
    let expect = info.nbytes().ok_or(WeightError::LengthMismatch)?;
    if view.data().len() != expect {
        return Err(WeightError::LengthMismatch);
    }
    Ok(info)
}

fn map_dtype(dtype: StDtype) -> Result<DType, WeightError> {
    match dtype {
        StDtype::F32 => Ok(DType::F32),
        StDtype::F16 => Ok(DType::F16),
        StDtype::BF16 => Ok(DType::BF16),
        _ => Err(WeightError::UnsupportedDType),
    }
}

fn map_error(err: SafeTensorError) -> WeightError {
    match err {
        SafeTensorError::TensorNotFound(_) => WeightError::NotFound,
        _ => WeightError::Parse,
    }
}

impl WeightMap for SafeTensorsWeights<'_> {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightError> {
        if !self.index.contains_key(name) {
            return Err(WeightError::NotFound);
        }
        let view = self.inner.tensor(name).map_err(map_error)?;
        Ok(view.data())
    }

    fn tensor_info(&self, name: &str) -> Result<&TensorInfo, WeightError> {
        self.index.get(name).ok_or(WeightError::NotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_f32_tensor, MmapWeights};

    fn le_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn write_fixture(
        tensors: &[(&str, StDtype, Vec<usize>, &[u8])],
    ) -> Result<Vec<u8>, SafeTensorError> {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                StView::new(*dtype, shape.clone(), bytes).map(|view| (*name, view))
            })
            .collect::<Result<Vec<_>, _>>()?;
        safetensors::serialize(views, None)
    }

    fn assert_borrowed_from(file: &[u8], view: &[u8]) {
        assert!(!view.is_empty());
        let file_start = file.as_ptr() as usize;
        let file_end = file_start + file.len();
        let view_start = view.as_ptr() as usize;
        let view_end = view_start + view.len();
        assert!(view_start >= file_start && view_end <= file_end);
    }

    #[test]
    fn parses_f32_f16_bf16_without_copying_payloads() {
        let f32_bytes = le_f32(&[1.0, -2.5]);
        let bias_bytes = le_f32(&[0.5]);
        // f16: 1.0 = 0x3C00, -2.0 = 0xC000. bf16: 1.0 = 0x3F80, -2.0 = 0xC000.
        let f16_bytes = [0x00, 0x3C, 0x00, 0xC0];
        let bf16_bytes = [0x80, 0x3F, 0x00, 0xC0];
        let file = write_fixture(&[
            ("w_f32", StDtype::F32, vec![2], &f32_bytes),
            ("w_f16", StDtype::F16, vec![2], &f16_bytes),
            ("w_bf16", StDtype::BF16, vec![1, 2], &bf16_bytes),
            ("bias", StDtype::F32, vec![], &bias_bytes),
        ])
        .unwrap();

        let weights = SafeTensorsWeights::parse(&file).unwrap();
        assert_eq!(weights.len(), 4);
        assert!(!weights.is_empty());
        assert_eq!(
            weights.names().collect::<Vec<_>>(),
            vec!["bias", "w_bf16", "w_f16", "w_f32"]
        );

        let f32_info = weights.tensor_info("w_f32").unwrap();
        assert_eq!(f32_info.dtype, DType::F32);
        assert_eq!(f32_info.shape, vec![2]);
        assert_eq!(f32_info.nbytes(), Some(8));
        let f32_view = weights.tensor_bytes("w_f32").unwrap();
        assert_eq!(f32_view, f32_bytes.as_slice());
        assert_borrowed_from(&file, f32_view);

        let f16_info = weights.tensor_info("w_f16").unwrap();
        assert_eq!(f16_info.dtype, DType::F16);
        assert_eq!(f16_info.shape, vec![2]);
        assert_eq!(f16_info.nbytes(), Some(4));
        let f16_view = weights.tensor_bytes("w_f16").unwrap();
        assert_eq!(f16_view, &f16_bytes);
        assert_borrowed_from(&file, f16_view);

        let bf16_info = weights.tensor_info("w_bf16").unwrap();
        assert_eq!(bf16_info.dtype, DType::BF16);
        assert_eq!(bf16_info.shape, vec![1, 2]);
        let bf16_view = weights.tensor_bytes("w_bf16").unwrap();
        assert_eq!(bf16_view, &bf16_bytes);
        assert_borrowed_from(&file, bf16_view);
        assert_ne!(f32_view.as_ptr(), f16_view.as_ptr());
        assert_ne!(f16_view.as_ptr(), bf16_view.as_ptr());

        let bias = weights.tensor("bias").unwrap();
        assert_eq!(bias.info.dtype, DType::F32);
        assert!(bias.info.shape.is_empty());
        assert_eq!(bias.bytes, bias_bytes.as_slice());
        assert_borrowed_from(&file, bias.bytes);

        assert_eq!(weights.tensor_bytes("missing"), Err(WeightError::NotFound));
        assert_eq!(
            load_f32_tensor(&weights, "w_f16"),
            Err(WeightError::DTypeMismatch)
        );

        if cfg!(target_endian = "little") {
            let (_, from_st) = load_f32_tensor(&weights, "w_f32").unwrap();
            let mut mmap = MmapWeights::new();
            mmap.insert_f32("w_f32", &[2], &[1.0, -2.5]).unwrap();
            let (_, from_mmap) = load_f32_tensor(&mmap, "w_f32").unwrap();
            assert_eq!(from_st, vec![1.0, -2.5]);
            assert_eq!(from_st, from_mmap);
            let (_, bias_vals) = load_f32_tensor(&weights, "bias").unwrap();
            assert_eq!(bias_vals, vec![0.5]);
        }
    }

    #[test]
    fn rejects_unmapped_dtype_for_the_whole_file() {
        let f32_bytes = le_f32(&[1.0]);
        let i64_bytes = 7i64.to_le_bytes();
        let file = write_fixture(&[
            ("keep", StDtype::F32, vec![1], &f32_bytes),
            ("drop", StDtype::I64, vec![1], &i64_bytes),
        ])
        .unwrap();
        assert_eq!(
            SafeTensorsWeights::parse(&file).unwrap_err(),
            WeightError::UnsupportedDType
        );
    }

    #[test]
    fn rejects_truncated_buffers() {
        let samples: &[&[u8]] = &[&[], b"short", &[0xff, 0, 0, 0, 0, 0, 0, 0], &[0xff; 8]];
        for bytes in samples {
            assert_eq!(
                SafeTensorsWeights::parse(bytes).unwrap_err(),
                WeightError::Parse
            );
        }
    }
}
