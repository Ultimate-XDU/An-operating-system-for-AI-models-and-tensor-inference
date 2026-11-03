//! BF16 byte views of Qwen2.5-1.5B-Instruct weights.
//!
//! Payloads stay little-endian BF16 slices borrowed from the caller's
//! [`WeightMap`] (a [`SafeTensorsWeights`] buffer or a host mmap behind it).
//! Nothing here decodes those bytes to `f32` or calls [`load_f32_tensor`].
//! Promotion to f32 belongs inside `tensor_core` ops
//! (`quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`).
//!
//! This is not a transformer forward, a KV cache, or an `lm_head` kernel.
//! The tied output matrix is [`embed_tokens`]. There is no `lm_head` key.
//!
//! [`WeightMap`]: weight_loader::WeightMap
//! [`SafeTensorsWeights`]: weight_loader::SafeTensorsWeights
//! [`load_f32_tensor`]: weight_loader::load_f32_tensor

use core::fmt;

use weight_loader::{bf16_view, Bf16View, WeightError, WeightMap};

use crate::qwen25_layout::{
    layer_tensor_keys, layer_tensor_shapes, EMBED_SHAPE, EMBED_TOKENS, FINAL_NORM,
    FINAL_NORM_SHAPE, NUM_HIDDEN_LAYERS,
};

/// Failure looking up one Qwen2.5 BF16 tensor.
///
/// `DType` means the stored dtype is not BF16. This path does not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenBf16Error {
    /// Layer index is outside `0..NUM_HIDDEN_LAYERS`.
    LayerOutOfRange,
    /// The SafeTensors name is not in the map.
    Missing,
    /// The tensor exists but is not BF16.
    DType,
    /// Rank or dimensions differ from the expected shape.
    Shape,
    /// The byte length does not match the tensor's own shape metadata.
    Length,
}

impl fmt::Display for QwenBf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QwenBf16Error::LayerOutOfRange => write!(f, "layer index is out of range"),
            QwenBf16Error::Missing => write!(f, "tensor not found"),
            QwenBf16Error::DType => write!(f, "tensor dtype is not bf16"),
            QwenBf16Error::Shape => write!(f, "tensor shape does not match the layout"),
            QwenBf16Error::Length => write!(f, "tensor byte length does not match shape"),
        }
    }
}

fn map_weight(err: WeightError) -> QwenBf16Error {
    match err {
        WeightError::NotFound => QwenBf16Error::Missing,
        WeightError::DTypeMismatch => QwenBf16Error::DType,
        WeightError::LengthMismatch => QwenBf16Error::Length,
        WeightError::Duplicate
        | WeightError::EmptyName
        | WeightError::Parse
        | WeightError::UnsupportedDType => QwenBf16Error::Length,
    }
}

/// Borrow one BF16 tensor whose shape equals `expected`.
///
/// `bytes` is the caller buffer (or mmap). The function does not allocate a
/// payload and does not convert BF16 to f32. Qwen entry points pass the
/// shapes from [`crate::qwen25_layout`]. A unit fixture may pass a smaller
/// `expected` so a fake table can be borrowed without the 1.5B allocation.
pub fn bf16_tensor<'a>(
    map: &'a impl WeightMap,
    name: &str,
    expected: &[usize],
) -> Result<Bf16View<'a>, QwenBf16Error> {
    let view = bf16_view(map, name).map_err(map_weight)?;
    if view.shape != expected {
        return Err(QwenBf16Error::Shape);
    }
    Ok(view)
}

/// `model.embed_tokens.weight`, shape [`EMBED_SHAPE`].
///
/// Also the tied output projection. The checkpoint has no `lm_head` tensor.
pub fn embed_tokens(map: &impl WeightMap) -> Result<Bf16View<'_>, QwenBf16Error> {
    bf16_tensor(map, EMBED_TOKENS, EMBED_SHAPE.as_slice())
}

/// `model.norm.weight`, shape [`FINAL_NORM_SHAPE`].
pub fn final_norm(map: &impl WeightMap) -> Result<Bf16View<'_>, QwenBf16Error> {
    bf16_tensor(map, FINAL_NORM, FINAL_NORM_SHAPE.as_slice())
}

/// Borrowed BF16 weights for one transformer block.
///
/// Fields are views. There is no `o_proj` bias and no MLP bias. This type
/// does not run attention or the MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen25LayerView<'a> {
    pub input_layernorm: Bf16View<'a>,
    pub q_weight: Bf16View<'a>,
    pub q_bias: Bf16View<'a>,
    pub k_weight: Bf16View<'a>,
    pub k_bias: Bf16View<'a>,
    pub v_weight: Bf16View<'a>,
    pub v_bias: Bf16View<'a>,
    pub o_weight: Bf16View<'a>,
    pub post_attention_layernorm: Bf16View<'a>,
    pub gate_weight: Bf16View<'a>,
    pub up_weight: Bf16View<'a>,
    pub down_weight: Bf16View<'a>,
}

/// Borrow every tensor in `model.layers.{index}`.
///
/// `index` must be in `0..NUM_HIDDEN_LAYERS` (0 and 27 are the ends).
/// Each tensor must be BF16 with the layout shape. Bytes stay in `map`.
pub fn layer(
    map: &impl WeightMap,
    index: usize,
) -> Result<Qwen25LayerView<'_>, QwenBf16Error> {
    if index >= NUM_HIDDEN_LAYERS {
        return Err(QwenBf16Error::LayerOutOfRange);
    }
    let keys = layer_tensor_keys(index);
    let shapes = layer_tensor_shapes();
    Ok(Qwen25LayerView {
        input_layernorm: bf16_tensor(map, &keys[0], shapes[0])?,
        q_weight: bf16_tensor(map, &keys[1], shapes[1])?,
        q_bias: bf16_tensor(map, &keys[2], shapes[2])?,
        k_weight: bf16_tensor(map, &keys[3], shapes[3])?,
        k_bias: bf16_tensor(map, &keys[4], shapes[4])?,
        v_weight: bf16_tensor(map, &keys[5], shapes[5])?,
        v_bias: bf16_tensor(map, &keys[6], shapes[6])?,
        o_weight: bf16_tensor(map, &keys[7], shapes[7])?,
        post_attention_layernorm: bf16_tensor(map, &keys[8], shapes[8])?,
        gate_weight: bf16_tensor(map, &keys[9], shapes[9])?,
        up_weight: bf16_tensor(map, &keys[10], shapes[10])?,
        down_weight: bf16_tensor(map, &keys[11], shapes[11])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen25_layout::{q_proj_bias_key, HIDDEN_SIZE, Q_PROJ_BIAS_SHAPE};
    use weight_loader::{DType, MmapWeights, SafeTensorsWeights};

    fn assert_borrowed_from(file: &[u8], view: &[u8]) {
        assert!(!view.is_empty());
        let file_start = file.as_ptr() as usize;
        let file_end = file_start + file.len();
        let view_start = view.as_ptr() as usize;
        let view_end = view_start + view.len();
        assert!(view_start >= file_start && view_end <= file_end);
    }

    /// Two token rows, four hidden columns. Not the 1.5B table.
    fn fake_embed_table() -> ([u8; 16], [usize; 2]) {
        let mut bytes = [0u8; 16];
        for (i, chunk) in bytes.chunks_exact_mut(2).enumerate() {
            let bits = 0x3F80u16.wrapping_add(i as u16);
            chunk.copy_from_slice(&bits.to_le_bytes());
        }
        (bytes, [2, 4])
    }

    #[test]
    fn missing_embed_is_err() {
        let weights = MmapWeights::new();
        assert_eq!(embed_tokens(&weights).unwrap_err(), QwenBf16Error::Missing);
        assert_eq!(
            bf16_tensor(&weights, EMBED_TOKENS, &[2, 4]).unwrap_err(),
            QwenBf16Error::Missing
        );
    }

    #[test]
    fn wrong_dtype_is_err() {
        let mut weights = MmapWeights::new();
        let (_, shape) = fake_embed_table();
        let f32s = [0.0f32; 8];
        weights.insert_f32(EMBED_TOKENS, &shape, &f32s).unwrap();
        assert_eq!(
            bf16_tensor(&weights, EMBED_TOKENS, &shape).unwrap_err(),
            QwenBf16Error::DType
        );
        assert_eq!(embed_tokens(&weights).unwrap_err(), QwenBf16Error::DType);
    }

    #[test]
    fn shape_mismatch_is_err() {
        let mut weights = MmapWeights::new();
        let (payload, shape) = fake_embed_table();
        weights.insert_bf16(EMBED_TOKENS, &shape, &payload).unwrap();
        assert_eq!(embed_tokens(&weights).unwrap_err(), QwenBf16Error::Shape);
        assert_eq!(
            bf16_tensor(&weights, EMBED_TOKENS, &[2, 3]).unwrap_err(),
            QwenBf16Error::Shape
        );
        let info = weights.tensor_info(EMBED_TOKENS).unwrap();
        assert_eq!(info.dtype, DType::BF16);
    }

    #[test]
    fn borrows_fake_bf16_embed_row_table() {
        let mut weights = MmapWeights::new();
        let (payload, shape) = fake_embed_table();
        weights.insert_bf16(EMBED_TOKENS, &shape, &payload).unwrap();
        let view = bf16_tensor(&weights, EMBED_TOKENS, &shape).unwrap();
        assert_eq!(view.shape, &shape);
        assert_eq!(view.bytes, &payload);
        assert_eq!(
            view.bytes.as_ptr(),
            weights.tensor_bytes(EMBED_TOKENS).unwrap().as_ptr()
        );
        let row_bytes = shape[1] * 2;
        assert_eq!(view.bytes.len(), shape[0] * row_bytes);
        assert_eq!(&view.bytes[row_bytes..row_bytes * 2], &payload[row_bytes..]);
        assert_ne!(view.bytes.len(), shape[0] * shape[1] * 4);
    }

    #[test]
    fn borrows_fake_embed_from_safetensors() {
        use safetensors::tensor::{Dtype as StDtype, TensorView as StView};

        let (payload, shape) = fake_embed_table();
        let view = StView::new(StDtype::BF16, shape.to_vec(), &payload).unwrap();
        let file = safetensors::serialize([(EMBED_TOKENS, view)], None).unwrap();
        let weights = SafeTensorsWeights::parse(&file).unwrap();
        let got = bf16_tensor(&weights, EMBED_TOKENS, &shape).unwrap();
        assert_eq!(got.shape, &shape);
        assert_eq!(got.bytes, &payload);
        assert_borrowed_from(&file, got.bytes);
        assert_eq!(embed_tokens(&weights).unwrap_err(), QwenBf16Error::Shape);
    }

    #[test]
    fn final_norm_borrows_layout_shape() {
        let mut weights = MmapWeights::new();
        let mut payload = vec![0u8; HIDDEN_SIZE * 2];
        payload[0] = 0x80;
        payload[1] = 0x3F;
        weights
            .insert_bf16(FINAL_NORM, &FINAL_NORM_SHAPE, &payload)
            .unwrap();
        let view = final_norm(&weights).unwrap();
        assert_eq!(view.shape, FINAL_NORM_SHAPE.as_slice());
        assert_eq!(
            view.bytes.as_ptr(),
            weights.tensor_bytes(FINAL_NORM).unwrap().as_ptr()
        );
        assert_eq!(view.bytes.len(), HIDDEN_SIZE * 2);
        assert_eq!(&view.bytes[..2], &[0x80, 0x3F]);
    }

    #[test]
    fn layer_ends_and_shape_and_dtype() {
        let empty = MmapWeights::new();
        assert_eq!(layer(&empty, 27).unwrap_err(), QwenBf16Error::Missing);
        assert_eq!(
            layer(&empty, 28).unwrap_err(),
            QwenBf16Error::LayerOutOfRange
        );

        let mut wrong_shape = MmapWeights::new();
        let key0 = crate::qwen25_layout::input_layernorm_key(0);
        wrong_shape
            .insert_bf16(&key0, &[2], &[0x80, 0x3F, 0x00, 0x00])
            .unwrap();
        assert_eq!(layer(&wrong_shape, 0).unwrap_err(), QwenBf16Error::Shape);

        let mut first_f32 = MmapWeights::new();
        let norm = vec![1.0f32; HIDDEN_SIZE];
        first_f32
            .insert_f32(&key0, &crate::qwen25_layout::INPUT_LAYERNORM_SHAPE, &norm)
            .unwrap();
        assert_eq!(layer(&first_f32, 0).unwrap_err(), QwenBf16Error::DType);

        let mut bias_map = MmapWeights::new();
        let mut bias_bytes = vec![0u8; HIDDEN_SIZE * 2];
        bias_bytes[2] = 0x80;
        bias_bytes[3] = 0x3F;
        let bias_key = q_proj_bias_key(27);
        bias_map
            .insert_bf16(&bias_key, &Q_PROJ_BIAS_SHAPE, &bias_bytes)
            .unwrap();
        let view = bf16_tensor(&bias_map, &bias_key, Q_PROJ_BIAS_SHAPE.as_slice()).unwrap();
        assert_eq!(view.shape, &[HIDDEN_SIZE]);
        assert_eq!(
            view.bytes.as_ptr(),
            bias_map.tensor_bytes(&bias_key).unwrap().as_ptr()
        );
        assert_eq!(view.bytes.len(), HIDDEN_SIZE * 2);
    }
}
