//! ModelCore — unified MinimalistModel trait (Phase 1).
//!
//! `QwenStub` is a CPU toy: one embedding row plus one `lm_head` GEMV. It is not a Qwen block.
//! Real 1.5B Instruct names and shapes live in [`qwen25_layout`].
//! [`qwen25_bf16`] borrows those tensors as BF16 bytes. The stub does not call either.
#![cfg_attr(not(test), no_std)]

pub mod qwen25_bf16;
pub mod qwen25_layout;

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use tensor_core::gemv_f32;
use weight_loader::{load_f32_tensor, DType, MmapWeights, TensorInfo, WeightError, WeightMap};

/// Phase-1 model surface: one token forward.
pub trait MinimalistModel {
    type KvCache;
    fn forward_token(&mut self, token: u32, pos: usize, kv: &mut Self::KvCache) -> &[f32];
}

/// Counts tokens this stub has forwarded. There is no real KV tensor in P0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCache {
    pub len: usize,
    pub last_pos: usize,
}

/// Weight or shape problem while building [`QwenStub`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelError {
    Weight(WeightError),
    BadRank,
    BadShape,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Weight(err) => write!(f, "weight error: {err}"),
            ModelError::BadRank => write!(f, "expected a rank-2 f32 tensor"),
            ModelError::BadShape => write!(f, "tok_embeddings and lm_head shapes disagree"),
        }
    }
}

impl From<WeightError> for ModelError {
    fn from(err: WeightError) -> Self {
        ModelError::Weight(err)
    }
}

/// Identity embeddings and a shift `lm_head`.
///
/// `tok_embeddings` is `I`. Row `next` of `lm_head` is the unit vector at column `t` when
/// `next == (t + 1) % vocab`. Greedy decode therefore walks `t → (t + 1) mod vocab`.
/// `dim` equals `vocab` so that shift is a square permutation.
pub fn cycle_fixture(vocab: usize) -> Result<MmapWeights, ModelError> {
    if vocab == 0 {
        return Err(ModelError::BadShape);
    }
    let n = vocab.checked_mul(vocab).ok_or(ModelError::BadShape)?;
    let mut embed = Vec::new();
    embed.resize(n, 0.0);
    let mut head = Vec::new();
    head.resize(n, 0.0);
    for token in 0..vocab {
        embed[token * vocab + token] = 1.0;
        let next = (token + 1) % vocab;
        head[next * vocab + token] = 1.0;
    }
    let mut weights = MmapWeights::new();
    weights.insert_f32("tok_embeddings", &[vocab, vocab], &embed)?;
    weights.insert_f32("lm_head", &[vocab, vocab], &head)?;
    Ok(weights)
}

/// Single-GEMV model over two named f32 matrices.
#[derive(Debug)]
pub struct QwenStub {
    vocab: usize,
    dim: usize,
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    logits: Vec<f32>,
}

impl QwenStub {
    pub fn from_weights(weights: &impl WeightMap) -> Result<Self, ModelError> {
        let (embed_info, embed) = load_f32_tensor(weights, "tok_embeddings")?;
        let (head_info, lm_head) = load_f32_tensor(weights, "lm_head")?;
        let (vocab, dim) = rank2(&embed_info)?;
        let (head_vocab, head_dim) = rank2(&head_info)?;
        if vocab == 0 || dim == 0 || vocab != head_vocab || dim != head_dim {
            return Err(ModelError::BadShape);
        }
        if embed.len() != vocab * dim || lm_head.len() != vocab * dim {
            return Err(ModelError::BadShape);
        }
        let mut logits = Vec::new();
        logits.resize(vocab, 0.0);
        Ok(Self {
            vocab,
            dim,
            embed,
            lm_head,
            logits,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl MinimalistModel for QwenStub {
    type KvCache = KvCache;

    fn forward_token(&mut self, token: u32, pos: usize, kv: &mut Self::KvCache) -> &[f32] {
        let token = token as usize;
        assert!(token < self.vocab, "token out of range");
        let start = token * self.dim;
        let x = &self.embed[start..start + self.dim];
        gemv_f32(&mut self.logits, &self.lm_head, x, self.vocab, self.dim);
        kv.len = kv.len.saturating_add(1);
        kv.last_pos = pos;
        &self.logits
    }
}

/// First maximum. Ties keep the lower index.
pub fn greedy_token(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &value) in logits.iter().enumerate() {
        if value > best {
            best = value;
            best_i = i;
        }
    }
    best_i as u32
}

fn rank2(info: &TensorInfo) -> Result<(usize, usize), ModelError> {
    if info.dtype != DType::F32 {
        return Err(ModelError::Weight(WeightError::DTypeMismatch));
    }
    match info.shape.as_slice() {
        [vocab, dim] => Ok((*vocab, *dim)),
        _ => Err(ModelError::BadRank),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_token_smoke() {
        let weights = cycle_fixture(4).unwrap();
        let mut model = QwenStub::from_weights(&weights).unwrap();
        assert_eq!(model.vocab(), 4);
        assert_eq!(model.dim(), 4);

        let mut kv = KvCache::default();
        let logits = model.forward_token(0, 0, &mut kv);
        assert_eq!(logits, &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(greedy_token(logits), 1);
        assert_eq!(
            kv,
            KvCache {
                len: 1,
                last_pos: 0
            }
        );

        let logits = model.forward_token(1, 1, &mut kv);
        assert_eq!(logits, &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(greedy_token(logits), 2);
        assert_eq!(kv.len, 2);
        assert_eq!(kv.last_pos, 1);
    }

    #[test]
    fn from_weights_rejects_missing_tensor() {
        let mut weights = MmapWeights::new();
        weights
            .insert_f32("tok_embeddings", &[1, 1], &[1.0])
            .unwrap();
        let err = QwenStub::from_weights(&weights).unwrap_err();
        assert_eq!(err, ModelError::Weight(WeightError::NotFound));
    }
}
