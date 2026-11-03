//! One Qwen2.5 attention layer: BF16 projections, bias, RoPE, KV write, GQA.
//!
//! This is the attention half of a transformer block. It does not RMSNorm,
//! it does not run SwiGLU, and it does not walk every layer. The caller
//! passes `proj_in` (the vector Q/K/V read) and a separate `residual` (the
//! stream `o_proj` is added onto). A full block will pass the input-norm
//! output as `proj_in` and the pre-norm stream as `residual`.
//!
//! Control flow for the cache follows ember (MIT, Copyright (c) 2026 Ali
//! Zulfiqar, <https://github.com/codewithfourtix/ember> `src/attention.rs`
//! and `Model::new_cache` in `src/model.rs`): [`attention_layer`] stores this
//! layer's K/V at the shared cursor and does not [`crate::KvCache::advance`].
//! The caller advances once after every layer of the token has stored.
//!
//! Numeric work stays in `tensor_core`: [`tensor_core::gemv_bf16`],
//! [`tensor_core::add_bf16_bias`], [`tensor_core::rope_f32_inplace`] (NeoX),
//! and [`tensor_core::gqa_attn_f32`] (`e / sum` softmax). Weights stay
//! little-endian BF16 views from [`crate::qwen25_bf16`]. Q/K/V have bias.
//! `o_proj` does not.

use alloc::vec::Vec;
use core::fmt;

use tensor_core::{
    add_bf16_bias, gemv_bf16, gqa_attn_f32, rope_f32_inplace, GqaMask, GqaShape, ROPE_THETA_QWEN25,
};

use crate::kv_cache::{KvCache, KvError};
use crate::qwen25_bf16::{bf16_tensor, QwenBf16Error};
use crate::qwen25_layout::{
    k_proj_bias_key, k_proj_weight_key, o_proj_weight_key, q_proj_bias_key, q_proj_weight_key,
    v_proj_bias_key, v_proj_weight_key, HEAD_DIM, HIDDEN_SIZE, NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
};
use weight_loader::WeightMap;

/// Widths for one attention layer.
///
/// Qwen2.5-1.5B passes [`AttnShape::qwen25_1_5b`]. A test shrinks the fields.
/// `head_dim` is even (NeoX). `num_q_heads` is a non-zero multiple of
/// `num_kv_heads`. `hidden` is the residual width and the K-projection input;
/// Qwen's `o_proj` happens to be square because `12 * 128 = 1536`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnShape {
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl AttnShape {
    /// 1536 hidden, 12 query heads, 2 KV heads, head dim 128, `rope_theta` 1e6.
    pub const fn qwen25_1_5b() -> Self {
        Self {
            hidden: HIDDEN_SIZE,
            num_q_heads: NUM_ATTENTION_HEADS,
            num_kv_heads: NUM_KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            rope_theta: ROPE_THETA_QWEN25,
        }
    }
}

/// Scratch for [`attention_layer`]. One allocation, reused across layers and tokens.
///
/// `max_seq` matches the [`KvCache`] this workspace is paired with.
#[derive(Debug, Clone)]
pub struct AttnWorkspace {
    shape: AttnShape,
    max_seq: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    probs: Vec<f32>,
    pack_k: Vec<f32>,
    pack_v: Vec<f32>,
}

impl AttnWorkspace {
    pub fn new(shape: AttnShape, max_seq: usize) -> Result<Self, AttnError> {
        let dims = dims_of(shape)?;
        if max_seq == 0 {
            return Err(AttnError::Shape);
        }
        let pack = dims
            .kv_heads
            .checked_mul(max_seq)
            .and_then(|n| n.checked_mul(dims.head_dim))
            .ok_or(AttnError::Shape)?;
        Ok(Self {
            shape,
            max_seq,
            q: zeros(dims.q_dim),
            k: zeros(dims.kv_dim),
            v: zeros(dims.kv_dim),
            att: zeros(dims.q_dim),
            proj: zeros(dims.hidden),
            probs: zeros(max_seq),
            pack_k: zeros(pack),
            pack_v: zeros(pack),
        })
    }

    pub fn shape(&self) -> AttnShape {
        self.shape
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
}

/// Failure from [`attention_layer`].
///
/// On `Err`, `residual` and the cache cursor and slots are unchanged.
/// The workspace is scratch and may hold a partial projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnError {
    /// A count is zero, `head_dim` is odd, the head counts do not divide,
    /// or a buffer length does not match [`AttnShape`].
    Shape,
    /// `pos` is not the cache cursor, or it does not fit in the RoPE `u32`.
    Position,
    /// A Q/K/V/O tensor is missing, not BF16, or not the shape this call asked for.
    Weight(QwenBf16Error),
    /// The cache rejected the layer, the cursor, or the packed prefix.
    Cache(KvError),
}

impl fmt::Display for AttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttnError::Shape => write!(f, "attention shape does not match the buffers"),
            AttnError::Position => write!(f, "attention position does not match the kv cursor"),
            AttnError::Weight(err) => write!(f, "attention weight: {err}"),
            AttnError::Cache(err) => write!(f, "kv cache: {err}"),
        }
    }
}

struct Dims {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
}

fn dims_of(shape: AttnShape) -> Result<Dims, AttnError> {
    if shape.hidden == 0
        || shape.num_q_heads == 0
        || shape.num_kv_heads == 0
        || shape.head_dim == 0
        || shape.head_dim % 2 != 0
        || shape.num_q_heads % shape.num_kv_heads != 0
    {
        return Err(AttnError::Shape);
    }
    let q_dim = shape
        .num_q_heads
        .checked_mul(shape.head_dim)
        .ok_or(AttnError::Shape)?;
    let kv_dim = shape
        .num_kv_heads
        .checked_mul(shape.head_dim)
        .ok_or(AttnError::Shape)?;
    Ok(Dims {
        hidden: shape.hidden,
        q_heads: shape.num_q_heads,
        kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        q_dim,
        kv_dim,
    })
}

fn zeros(n: usize) -> Vec<f32> {
    let mut out = Vec::new();
    out.resize(n, 0.0);
    out
}

/// Project, rotate, attend, and add `o_proj` onto `residual`.
///
/// `proj_in` is the Q/K/V input (`hidden` lanes). `residual` is updated with
/// `residual[i] += o_proj[i]` after the output GEMV. They are separate
/// borrows so a later block can pass a normalized vector and the pre-norm
/// stream. This function does not read `input_layernorm`.
///
/// Weights are borrowed by the Qwen2.5 key names at `layer`. Shapes come from
/// `shape`, not from the 1.5B constants, so a fixture can be small. Bytes stay
/// BF16. `q_proj`, `k_proj`, and `v_proj` biases are added after each GEMV.
/// There is no `o_proj` bias lookup.
///
/// Q and K are rotated in place with NeoX RoPE at `pos`. V is not rotated.
/// K and V are then packed with the committed prefix and scored by
/// [`gqa_attn_f32`] over that prefix (`GqaMask::Full`). The cache slot is
/// written only after that op and `o_proj` both accept the buffers. The
/// cursor stays put.
///
/// `pos` must equal `cache.len()`, and `work.max_seq()` must equal
/// `cache.max_seq()`.
pub fn attention_layer(
    residual: &mut [f32],
    proj_in: &[f32],
    map: &impl WeightMap,
    layer: usize,
    pos: usize,
    shape: AttnShape,
    cache: &mut KvCache,
    work: &mut AttnWorkspace,
) -> Result<(), AttnError> {
    let dims = dims_of(shape)?;
    if proj_in.len() != dims.hidden || residual.len() != dims.hidden {
        return Err(AttnError::Shape);
    }
    if work.shape != shape || work.max_seq != cache.max_seq() {
        return Err(AttnError::Shape);
    }
    if cache.shape().num_kv_heads != dims.kv_heads || cache.shape().head_dim != dims.head_dim {
        return Err(AttnError::Shape);
    }
    if layer >= cache.shape().num_layers {
        return Err(AttnError::Cache(KvError::Range));
    }
    if pos != cache.len() {
        return Err(AttnError::Position);
    }
    if pos >= cache.max_seq() {
        return Err(AttnError::Cache(KvError::Full));
    }
    let pos_u32 = u32::try_from(pos).map_err(|_| AttnError::Position)?;

    let q_w_shape = [dims.q_dim, dims.hidden];
    let q_b_shape = [dims.q_dim];
    let kv_w_shape = [dims.kv_dim, dims.hidden];
    let kv_b_shape = [dims.kv_dim];
    let o_w_shape = [dims.hidden, dims.q_dim];
    let q_w = bf16_tensor(map, &q_proj_weight_key(layer), &q_w_shape).map_err(AttnError::Weight)?;
    let q_b = bf16_tensor(map, &q_proj_bias_key(layer), &q_b_shape).map_err(AttnError::Weight)?;
    let k_w =
        bf16_tensor(map, &k_proj_weight_key(layer), &kv_w_shape).map_err(AttnError::Weight)?;
    let k_b = bf16_tensor(map, &k_proj_bias_key(layer), &kv_b_shape).map_err(AttnError::Weight)?;
    let v_w =
        bf16_tensor(map, &v_proj_weight_key(layer), &kv_w_shape).map_err(AttnError::Weight)?;
    let v_b = bf16_tensor(map, &v_proj_bias_key(layer), &kv_b_shape).map_err(AttnError::Weight)?;
    let o_w = bf16_tensor(map, &o_proj_weight_key(layer), &o_w_shape).map_err(AttnError::Weight)?;

    gemv_bf16(&mut work.q, q_w.bytes, proj_in, dims.q_dim, dims.hidden)
        .map_err(|_| AttnError::Shape)?;
    add_bf16_bias(&mut work.q, q_b.bytes).map_err(|_| AttnError::Shape)?;
    gemv_bf16(&mut work.k, k_w.bytes, proj_in, dims.kv_dim, dims.hidden)
        .map_err(|_| AttnError::Shape)?;
    add_bf16_bias(&mut work.k, k_b.bytes).map_err(|_| AttnError::Shape)?;
    gemv_bf16(&mut work.v, v_w.bytes, proj_in, dims.kv_dim, dims.hidden)
        .map_err(|_| AttnError::Shape)?;
    add_bf16_bias(&mut work.v, v_b.bytes).map_err(|_| AttnError::Shape)?;

    rope_f32_inplace(&mut work.q, pos_u32, dims.head_dim, shape.rope_theta)
        .map_err(|_| AttnError::Shape)?;
    rope_f32_inplace(&mut work.k, pos_u32, dims.head_dim, shape.rope_theta)
        .map_err(|_| AttnError::Shape)?;

    let seq = pos + 1;
    let packed = dims
        .kv_heads
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(dims.head_dim))
        .ok_or(AttnError::Shape)?;
    cache
        .pack_prefix_with_token(
            layer,
            pos,
            &work.k,
            &work.v,
            &mut work.pack_k,
            &mut work.pack_v,
        )
        .map_err(AttnError::Cache)?;
    gqa_attn_f32(
        &mut work.att,
        &work.q,
        &work.pack_k[..packed],
        &work.pack_v[..packed],
        &mut work.probs[..seq],
        GqaShape {
            num_q_heads: dims.q_heads,
            num_kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            seq_len: seq,
        },
        GqaMask::Full,
    )
    .map_err(|_| AttnError::Shape)?;
    gemv_bf16(
        &mut work.proj,
        o_w.bytes,
        &work.att,
        dims.hidden,
        dims.q_dim,
    )
    .map_err(|_| AttnError::Shape)?;

    cache
        .store(layer, &work.k, &work.v)
        .map_err(AttnError::Cache)?;
    add_assign(residual, &work.proj);
    Ok(())
}

/// `y[i] += x[i]`, left to right. Lengths match: both are `hidden`.
fn add_assign(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for i in 0..y.len() {
        y[i] += x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvShape;
    use crate::qwen25_bf16::bf16_tensor;
    use tensor_core::rope_f32_inplace;
    use weight_loader::{load_f32_tensor, MmapWeights, WeightError};

    fn pack(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bits = value.to_bits();
            assert_eq!(bits & 0xFFFF, 0, "test value {value} is not exact in bf16");
            let hi = u16::try_from(bits >> 16).unwrap();
            bytes.extend_from_slice(&hi.to_le_bytes());
        }
        bytes
    }

    fn zeros_f(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn eye(n: usize) -> Vec<f32> {
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    /// Row-major `[rows, cols]`. `columns[c][r]` is `W[r, c]`. Omitted columns stay 0.
    fn matrix(rows: usize, cols: usize, columns: &[&[f32]]) -> Vec<f32> {
        let mut w = vec![0.0; rows * cols];
        for (c, column) in columns.iter().enumerate() {
            assert_eq!(column.len(), rows);
            for (r, value) in column.iter().enumerate() {
                w[r * cols + c] = *value;
            }
        }
        w
    }

    fn insert(map: &mut MmapWeights, name: &str, shape: &[usize], values: &[f32]) {
        map.insert_bf16(name, shape, &pack(values)).unwrap();
    }

    struct Proj {
        q_w: Vec<f32>,
        q_b: Vec<f32>,
        k_w: Vec<f32>,
        k_b: Vec<f32>,
        v_w: Vec<f32>,
        v_b: Vec<f32>,
        o_w: Vec<f32>,
    }

    fn insert_proj(map: &mut MmapWeights, layer: usize, shape: AttnShape, proj: &Proj) {
        let q_dim = shape.num_q_heads * shape.head_dim;
        let kv_dim = shape.num_kv_heads * shape.head_dim;
        let hidden = shape.hidden;
        insert(map, &q_proj_weight_key(layer), &[q_dim, hidden], &proj.q_w);
        insert(map, &q_proj_bias_key(layer), &[q_dim], &proj.q_b);
        insert(map, &k_proj_weight_key(layer), &[kv_dim, hidden], &proj.k_w);
        insert(map, &k_proj_bias_key(layer), &[kv_dim], &proj.k_b);
        insert(map, &v_proj_weight_key(layer), &[kv_dim, hidden], &proj.v_w);
        insert(map, &v_proj_bias_key(layer), &[kv_dim], &proj.v_b);
        insert(map, &o_proj_weight_key(layer), &[hidden, q_dim], &proj.o_w);
    }

    fn cache_for(shape: AttnShape, layers: usize, max_seq: usize) -> KvCache {
        KvCache::new(KvShape {
            num_layers: layers,
            num_kv_heads: shape.num_kv_heads,
            head_dim: shape.head_dim,
            max_seq,
        })
        .unwrap()
    }

    fn gqa_2q1kv() -> AttnShape {
        AttnShape {
            hidden: 4,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_theta: ROPE_THETA_QWEN25,
        }
    }

    #[test]
    fn qwen25_attn_shape_matches_layout() {
        let shape = AttnShape::qwen25_1_5b();
        assert_eq!(shape.hidden, 1536);
        assert_eq!(shape.num_q_heads, 12);
        assert_eq!(shape.num_kv_heads, 2);
        assert_eq!(shape.head_dim, 128);
        assert_eq!(shape.num_q_heads * shape.head_dim, shape.hidden);
        assert_eq!(shape.head_dim, tensor_core::HEAD_DIM_QWEN25_1_5B);
        assert_eq!(shape.rope_theta, ROPE_THETA_QWEN25);
        assert_eq!(
            shape.num_q_heads / shape.num_kv_heads,
            tensor_core::GQA_Q_PER_KV_QWEN25_1_5B
        );
        assert!(AttnWorkspace::new(
            AttnShape {
                head_dim: 1,
                ..gqa_2q1kv()
            },
            2
        )
        .is_err());
        assert!(AttnWorkspace::new(
            AttnShape {
                num_q_heads: 3,
                num_kv_heads: 2,
                hidden: 6,
                head_dim: 2,
                rope_theta: 1.0,
            },
            2
        )
        .is_err());
        assert!(AttnWorkspace::new(gqa_2q1kv(), 0).is_err());
    }

    #[test]
    fn grouped_step_matches_hand_fixture() {
        // 4 query heads, 2 KV heads, head_dim 2, one token at position 0.
        // RoPE at position 0 is the identity. seq = 1, so each query head
        // copies its KV head's V. Contiguous groups: heads 0,1 read KV 0
        // and heads 2,3 read KV 1. x is e0, so each projection is column 0.
        let shape = AttnShape {
            hidden: 8,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            rope_theta: ROPE_THETA_QWEN25,
        };
        let mut weights = MmapWeights::new();
        insert_proj(
            &mut weights,
            0,
            shape,
            &Proj {
                q_w: zeros_f(8 * 8),
                q_b: zeros_f(8),
                k_w: matrix(4, 8, &[&[5.0, 6.0, 7.0, 8.0]]),
                k_b: zeros_f(4),
                v_w: matrix(4, 8, &[&[1.0, 2.0, 3.0, 4.0]]),
                v_b: zeros_f(4),
                o_w: eye(8),
            },
        );
        // An o_proj bias in the map is not read. 100 would move y if it were.
        insert(
            &mut weights,
            "model.layers.0.self_attn.o_proj.bias",
            &[8],
            &[100.0; 8],
        );
        let q_key = q_proj_weight_key(0);
        let view = bf16_tensor(&weights, &q_key, &[8, 8]).unwrap();
        assert_eq!(
            view.bytes.as_ptr(),
            weights.tensor_bytes(&q_key).unwrap().as_ptr()
        );
        assert_eq!(
            load_f32_tensor(&weights, &q_key).unwrap_err(),
            WeightError::DTypeMismatch
        );

        let x = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut residual = x;
        let mut cache = cache_for(shape, 1, 2);
        let mut work = AttnWorkspace::new(shape, 2).unwrap();
        attention_layer(
            &mut residual,
            &x,
            &weights,
            0,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        // att = [1, 2, 1, 2, 3, 4, 3, 4], y = x + att.
        assert_eq!(residual, [2.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[5.0, 6.0]);
        assert_eq!(cache.key_head(0, 1, 0).unwrap(), &[7.0, 8.0]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 2.0]);
        assert_eq!(cache.value_head(0, 1, 0).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn bias_is_added_after_the_gemv() {
        // Position 0, one KV head, two query heads. Column 0 of K is [2, 0]
        // plus bias [1, 0.5]. Column 0 of V is [4, 0] plus bias [0.5, 1].
        // Q bias is nonzero and is not stored; seq = 1 copies V either way.
        // An o_proj bias of 100 is present and must not move y.
        let shape = gqa_2q1kv();
        let mut weights = MmapWeights::new();
        insert_proj(
            &mut weights,
            0,
            shape,
            &Proj {
                q_w: zeros_f(4 * 4),
                q_b: vec![0.25, 0.0, 0.0, 0.5],
                k_w: matrix(2, 4, &[&[2.0, 0.0]]),
                k_b: vec![1.0, 0.5],
                v_w: matrix(2, 4, &[&[4.0, 0.0]]),
                v_b: vec![0.5, 1.0],
                o_w: eye(4),
            },
        );
        insert(
            &mut weights,
            "model.layers.0.self_attn.o_proj.bias",
            &[4],
            &[100.0; 4],
        );
        let x = [1.0f32, 0.0, 0.0, 0.0];
        let mut residual = x;
        let mut cache = cache_for(shape, 1, 2);
        let mut work = AttnWorkspace::new(shape, 2).unwrap();
        attention_layer(
            &mut residual,
            &x,
            &weights,
            0,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        // V = [4.5, 1] copied to both query heads. y = x + att.
        assert_eq!(residual, [5.5, 1.0, 4.5, 1.0]);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[3.0, 0.5]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[4.5, 1.0]);
    }

    #[test]
    fn second_token_mixes_both_kv_heads_from_the_cache() {
        // Two query heads and two KV heads (group size 1), head_dim 2.
        // Q and K stay 0, so every score is 0 and the softmax is uniform 1/2
        // on the second token. RoPE of a zero vector stays zero, including
        // at position 1. V is column 0 at the first token and column 1 at
        // the second.
        //
        // Token 0 V: head0 [4, 0], head1 [0, 6].
        // Token 1 V: head0 [0, 8], head1 [2, 0].
        // Token 1 mix: head0 [2, 4], head1 [1, 3].
        // A time-major pack read as gqa's head-major layout would mix to
        // [2, 3] and [1, 4] instead.
        let shape = AttnShape {
            hidden: 4,
            num_q_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            rope_theta: ROPE_THETA_QWEN25,
        };
        let mut weights = MmapWeights::new();
        insert_proj(
            &mut weights,
            0,
            shape,
            &Proj {
                q_w: zeros_f(4 * 4),
                q_b: zeros_f(4),
                k_w: zeros_f(4 * 4),
                k_b: zeros_f(4),
                v_w: matrix(4, 4, &[&[4.0, 0.0, 0.0, 6.0], &[0.0, 8.0, 2.0, 0.0]]),
                v_b: zeros_f(4),
                o_w: eye(4),
            },
        );
        let mut cache = cache_for(shape, 1, 4);
        let mut work = AttnWorkspace::new(shape, 4).unwrap();

        let x0 = [1.0f32, 0.0, 0.0, 0.0];
        let mut residual = x0;
        attention_layer(
            &mut residual,
            &x0,
            &weights,
            0,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        assert_eq!(residual, [5.0, 0.0, 0.0, 6.0]);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[4.0, 0.0]);
        assert_eq!(cache.value_head(0, 1, 0).unwrap(), &[0.0, 6.0]);
        cache.advance().unwrap();

        let x1 = [0.0f32, 1.0, 0.0, 0.0];
        residual = x1;
        attention_layer(
            &mut residual,
            &x1,
            &weights,
            0,
            1,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        assert_eq!(residual, [2.0, 5.0, 1.0, 3.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[4.0, 0.0]);
        assert_eq!(cache.value_head(0, 1, 0).unwrap(), &[0.0, 6.0]);
        assert_eq!(cache.value_head(0, 0, 1).unwrap(), &[0.0, 8.0]);
        assert_eq!(cache.value_head(0, 1, 1).unwrap(), &[2.0, 0.0]);
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn rope_rotates_k_and_leaves_v() {
        let shape = gqa_2q1kv();
        let mut weights = MmapWeights::new();
        insert_proj(
            &mut weights,
            0,
            shape,
            &Proj {
                q_w: zeros_f(4 * 4),
                q_b: zeros_f(4),
                k_w: zeros_f(2 * 4),
                k_b: vec![1.0, 0.0],
                v_w: zeros_f(2 * 4),
                v_b: vec![4.0, 8.0],
                o_w: eye(4),
            },
        );
        let x = [0.0f32; 4];
        let mut residual = x;
        let mut cache = cache_for(shape, 1, 2);
        let mut work = AttnWorkspace::new(shape, 2).unwrap();
        attention_layer(
            &mut residual,
            &x,
            &weights,
            0,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[4.0, 8.0]);
        cache.advance().unwrap();

        residual = x;
        attention_layer(
            &mut residual,
            &x,
            &weights,
            0,
            1,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        let mut rotated = [1.0f32, 0.0];
        rope_f32_inplace(&mut rotated, 1, 2, ROPE_THETA_QWEN25).unwrap();
        assert_ne!(rotated, [1.0, 0.0]);
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &rotated);
        assert_eq!(cache.value_head(0, 0, 1).unwrap(), &[4.0, 8.0]);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn two_layers_store_then_one_advance() {
        let shape = gqa_2q1kv();
        let mut weights = MmapWeights::new();
        for (layer, v_bias) in [vec![1.0f32, 0.0], vec![0.0f32, 1.0]]
            .into_iter()
            .enumerate()
        {
            insert_proj(
                &mut weights,
                layer,
                shape,
                &Proj {
                    q_w: zeros_f(4 * 4),
                    q_b: zeros_f(4),
                    k_w: zeros_f(2 * 4),
                    k_b: zeros_f(2),
                    v_w: zeros_f(2 * 4),
                    v_b: v_bias,
                    o_w: eye(4),
                },
            );
        }
        let mut cache = cache_for(shape, 2, 2);
        let mut work = AttnWorkspace::new(shape, 2).unwrap();
        let mut residual = [0.0f32; 4];
        let proj = residual;
        attention_layer(
            &mut residual,
            &proj,
            &weights,
            0,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        assert_eq!(residual, [1.0, 0.0, 1.0, 0.0]);
        assert_eq!(cache.len(), 0);
        let proj = residual;
        attention_layer(
            &mut residual,
            &proj,
            &weights,
            1,
            0,
            shape,
            &mut cache,
            &mut work,
        )
        .unwrap();
        assert_eq!(residual, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
        assert_eq!(cache.value_head(1, 0, 0).unwrap(), &[0.0, 1.0]);
        cache.advance().unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn errors_do_not_write_residual_or_cache() {
        let shape = gqa_2q1kv();
        let mut residual = [7.0f32; 4];
        let before = residual;
        let x = [1.0f32, 0.0, 0.0, 0.0];
        let empty = MmapWeights::new();
        let mut cache = cache_for(shape, 1, 2);
        let mut work = AttnWorkspace::new(shape, 2).unwrap();

        assert_eq!(
            attention_layer(
                &mut residual,
                &x,
                &empty,
                0,
                1,
                shape,
                &mut cache,
                &mut work
            ),
            Err(AttnError::Position)
        );
        assert_eq!(residual, before);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[0.0, 0.0]);

        assert_eq!(
            attention_layer(
                &mut residual,
                &x,
                &empty,
                0,
                0,
                shape,
                &mut cache,
                &mut work
            )
            .unwrap_err(),
            AttnError::Weight(QwenBf16Error::Missing)
        );
        assert_eq!(residual, before);
        assert_eq!(cache.len(), 0);

        let mut f32_w = MmapWeights::new();
        f32_w
            .insert_f32(&q_proj_weight_key(0), &[4, 4], &[0.0; 16])
            .unwrap();
        assert_eq!(
            attention_layer(
                &mut residual,
                &x,
                &f32_w,
                0,
                0,
                shape,
                &mut cache,
                &mut work
            )
            .unwrap_err(),
            AttnError::Weight(QwenBf16Error::DType)
        );
        assert_eq!(residual, before);

        let mut short = [7.0f32; 3];
        let short_before = short;
        assert_eq!(
            attention_layer(&mut short, &x, &empty, 0, 0, shape, &mut cache, &mut work),
            Err(AttnError::Shape)
        );
        assert_eq!(short, short_before);

        let mut work_small = AttnWorkspace::new(shape, 1).unwrap();
        let mut cache_wide = cache_for(shape, 1, 2);
        assert_eq!(
            attention_layer(
                &mut residual,
                &x,
                &empty,
                0,
                0,
                shape,
                &mut cache_wide,
                &mut work_small
            ),
            Err(AttnError::Shape)
        );
        assert_eq!(residual, before);
        assert_eq!(cache_wide.len(), 0);

        // Cursor already at max_seq. pos == len, so this is Full, and the
        // earlier slot stays. Weights are not consulted.
        let mut cache_full = cache_for(shape, 1, 1);
        let mut work_full = AttnWorkspace::new(shape, 1).unwrap();
        cache_full.store(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache_full.advance().unwrap();
        assert_eq!(cache_full.len(), 1);
        assert_eq!(
            attention_layer(
                &mut residual,
                &x,
                &empty,
                0,
                1,
                shape,
                &mut cache_full,
                &mut work_full
            )
            .unwrap_err(),
            AttnError::Cache(KvError::Full)
        );
        assert_eq!(residual, before);
        assert_eq!(cache_full.len(), 1);
        assert_eq!(cache_full.key_head(0, 0, 0).unwrap(), &[1.0, 2.0]);
        assert_eq!(cache_full.value_head(0, 0, 0).unwrap(), &[3.0, 4.0]);
    }
}
