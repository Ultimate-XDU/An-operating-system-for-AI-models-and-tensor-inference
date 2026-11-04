//! Per-layer K/V cache for one GQA model.
//!
//! Control flow follows ember `KvCache` (MIT, Copyright (c) 2026 Ali Zulfiqar,
//! <https://github.com/codewithfourtix/ember> `src/attention.rs`, allocated from
//! `Model::new_cache` in `src/model.rs`): every layer writes this token at the
//! same cursor, then the caller [`KvCache::advance`]s once. [`KvCache::store`]
//! does not move the cursor. Ember caps context at 4096; the caller passes
//! `max_seq` here.
//!
//! The bytes are not ember's time-major `[pos, kv_dim]` buffer. Each layer is
//! `[num_kv_heads, max_seq, head_dim]`. Tokens `0..seq` of one KV head are
//! contiguous, so attention reads that slice in place (`kv_stride = max_seq`)
//! instead of copying the prefix down to length `seq`. Scores stay in
//! [`tensor_core::gqa_attn_f32`].
//!
//! Qwen2.5-1.5B is [`KvShape::qwen25_1_5b`]: 28 layers, 2 KV heads, head dim
//! 128. A test passes a smaller [`KvShape`]. This type does not run attention.

use alloc::vec::Vec;
use core::fmt;

use crate::qwen25_layout::{HEAD_DIM, NUM_HIDDEN_LAYERS, NUM_KEY_VALUE_HEADS};

/// Geometry of one [`KvCache`].
///
/// Qwen2.5-1.5B passes [`KvShape::qwen25_1_5b`]. Fields are public so a test
/// can shrink them. [`KvCache::new`] rejects a zero dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl KvShape {
    /// 28 layers, 2 KV heads, head dim 128. `max_seq` is the caller's cap.
    pub const fn qwen25_1_5b(max_seq: usize) -> Self {
        Self {
            num_layers: NUM_HIDDEN_LAYERS,
            num_kv_heads: NUM_KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            max_seq,
        }
    }
}

/// Failure from a cache write or read.
///
/// On `Err`, the cursor and the stored tokens are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvError {
    /// A dimension is zero, a product overflows, or a K/V vector has the wrong length.
    Shape,
    /// Layer, head, or position is outside this allocation.
    Range,
    /// [`KvCache::store`] or [`KvCache::advance`] when the cursor is `max_seq`.
    Full,
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvError::Shape => write!(f, "kv shape does not match the buffers"),
            KvError::Range => write!(f, "kv index is out of range"),
            KvError::Full => write!(f, "kv cache is full"),
        }
    }
}

/// K and V for every layer, plus one shared write cursor.
///
/// `len()` is the number of tokens committed by [`KvCache::advance`]. A store
/// writes at that index in every layer the caller names. After each layer of
/// this token has stored, one `advance` publishes the token.
///
/// ```text
/// len = 0
/// store(layer 0); store(layer 1);   // both write position 0
/// advance();                         // len = 1
/// store(layer 0); store(layer 1);   // both write position 1
/// advance();
/// ```
///
/// Slot `(layer, head, pos, d)` is
/// `((layer * num_kv_heads + head) * max_seq + pos) * head_dim + d`.
/// [`KvCache::clear`] sets the cursor back to 0 and leaves the slots as they
/// were. The next store overwrites position 0.
#[derive(Debug, Clone)]
pub struct KvCache {
    shape: KvShape,
    /// Tokens committed by [`KvCache::advance`].
    len: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    /// Allocate zeroed K and V. The cursor starts at 0.
    pub fn new(shape: KvShape) -> Result<Self, KvError> {
        if shape.num_layers == 0
            || shape.num_kv_heads == 0
            || shape.head_dim == 0
            || shape.max_seq == 0
        {
            return Err(KvError::Shape);
        }
        let per_layer = shape
            .num_kv_heads
            .checked_mul(shape.max_seq)
            .and_then(|n| n.checked_mul(shape.head_dim))
            .ok_or(KvError::Shape)?;
        let total = shape
            .num_layers
            .checked_mul(per_layer)
            .ok_or(KvError::Shape)?;
        let mut k = Vec::new();
        k.resize(total, 0.0);
        let v = k.clone();
        Ok(Self {
            shape,
            len: 0,
            k,
            v,
        })
    }

    pub fn shape(&self) -> KvShape {
        self.shape
    }

    /// Tokens committed by [`KvCache::advance`].
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_seq(&self) -> usize {
        self.shape.max_seq
    }

    /// Write `key` and `value` for `layer` at the current cursor.
    ///
    /// Both vectors are `[num_kv_heads, head_dim]`, one token, head-major.
    /// A second store at the same cursor overwrites that layer's slot.
    /// The cursor does not move.
    pub fn store(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<(), KvError> {
        self.store_at(layer, self.len, key, value)
    }

    /// Write `key` and `value` for `layer` at `pos`. The cursor does not move.
    ///
    /// Same layout as [`KvCache::store`]. Prefill uses this to drop a whole
    /// prompt chunk into `len..len+n` before one [`KvCache::advance`] per
    /// token. `pos >= max_seq` is [`KvError::Full`].
    pub fn store_at(
        &mut self,
        layer: usize,
        pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvError> {
        let kv_dim = self.kv_dim()?;
        if key.len() != kv_dim || value.len() != kv_dim {
            return Err(KvError::Shape);
        }
        if layer >= self.shape.num_layers {
            return Err(KvError::Range);
        }
        if pos >= self.shape.max_seq {
            return Err(KvError::Full);
        }
        let head_dim = self.shape.head_dim;
        for head in 0..self.shape.num_kv_heads {
            let dst = self.offset(layer, head, pos)?;
            let src = head * head_dim;
            self.k[dst..dst + head_dim].copy_from_slice(&key[src..src + head_dim]);
            self.v[dst..dst + head_dim].copy_from_slice(&value[src..src + head_dim]);
        }
        Ok(())
    }

    /// Pack K and V for `gqa_attn_f32` from slots `0..seq`.
    ///
    /// Destination layout is `[num_kv_heads, seq, head_dim]`. `dst_k` and
    /// `dst_v` must be at least that long. `seq` may pass the cursor when
    /// [`KvCache::store_at`] has already filled those slots. This does not
    /// write the cache and does not move the cursor.
    pub fn pack_prefix(
        &self,
        layer: usize,
        seq: usize,
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> Result<(), KvError> {
        if layer >= self.shape.num_layers {
            return Err(KvError::Range);
        }
        if seq > self.shape.max_seq {
            return Err(KvError::Full);
        }
        let head_dim = self.shape.head_dim;
        let need = self
            .shape
            .num_kv_heads
            .checked_mul(seq)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or(KvError::Shape)?;
        if dst_k.len() < need || dst_v.len() < need {
            return Err(KvError::Shape);
        }
        for head in 0..self.shape.num_kv_heads {
            for t in 0..seq {
                let src = self.offset(layer, head, t)?;
                let dst = (head * seq + t) * head_dim;
                dst_k[dst..dst + head_dim].copy_from_slice(&self.k[src..src + head_dim]);
                dst_v[dst..dst + head_dim].copy_from_slice(&self.v[src..src + head_dim]);
            }
        }
        Ok(())
    }

    /// Publish the token every layer just stored. One call per token.
    pub fn advance(&mut self) -> Result<(), KvError> {
        if self.len >= self.shape.max_seq {
            return Err(KvError::Full);
        }
        self.len += 1;
        Ok(())
    }

    /// Set the cursor to 0. Slots are left as stored until a later write.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Key lanes for one KV head at `pos`. `pos` is an allocated slot, not
    /// only a committed token: the in-progress store sits at `len()` until
    /// [`KvCache::advance`].
    pub fn key_head(&self, layer: usize, head: usize, pos: usize) -> Result<&[f32], KvError> {
        let off = self.offset(layer, head, pos)?;
        Ok(&self.k[off..off + self.shape.head_dim])
    }

    /// Value lanes for one KV head at `pos`. Same index rules as [`KvCache::key_head`].
    pub fn value_head(&self, layer: usize, head: usize, pos: usize) -> Result<&[f32], KvError> {
        let off = self.offset(layer, head, pos)?;
        Ok(&self.v[off..off + self.shape.head_dim])
    }

    /// One layer of K, layout `[num_kv_heads, max_seq, head_dim]`.
    ///
    /// Attention uses this with `kv_stride = max_seq` so a prefix of length
    /// `seq <= max_seq` is the leading `seq` tokens of each head. No copy.
    pub(crate) fn layer_keys(&self, layer: usize) -> Result<&[f32], KvError> {
        let (start, n) = self.layer_span(layer)?;
        Ok(&self.k[start..start + n])
    }

    /// One layer of V. Same layout as [`KvCache::layer_keys`].
    pub(crate) fn layer_values(&self, layer: usize) -> Result<&[f32], KvError> {
        let (start, n) = self.layer_span(layer)?;
        Ok(&self.v[start..start + n])
    }

    fn layer_span(&self, layer: usize) -> Result<(usize, usize), KvError> {
        if layer >= self.shape.num_layers {
            return Err(KvError::Range);
        }
        let n = self
            .shape
            .num_kv_heads
            .checked_mul(self.shape.max_seq)
            .and_then(|h| h.checked_mul(self.shape.head_dim))
            .ok_or(KvError::Shape)?;
        let start = layer.checked_mul(n).ok_or(KvError::Shape)?;
        Ok((start, n))
    }

    fn kv_dim(&self) -> Result<usize, KvError> {
        self.shape
            .num_kv_heads
            .checked_mul(self.shape.head_dim)
            .ok_or(KvError::Shape)
    }

    fn offset(&self, layer: usize, head: usize, pos: usize) -> Result<usize, KvError> {
        let shape = self.shape;
        if layer >= shape.num_layers || head >= shape.num_kv_heads || pos >= shape.max_seq {
            return Err(KvError::Range);
        }
        let head_stride = shape
            .max_seq
            .checked_mul(shape.head_dim)
            .ok_or(KvError::Shape)?;
        let layer_stride = shape
            .num_kv_heads
            .checked_mul(head_stride)
            .ok_or(KvError::Shape)?;
        layer
            .checked_mul(layer_stride)
            .and_then(|base| {
                head.checked_mul(head_stride)
                    .map(|head_off| base + head_off)
            })
            .and_then(|base| {
                pos.checked_mul(shape.head_dim)
                    .map(|pos_off| base + pos_off)
            })
            .ok_or(KvError::Shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(layers: usize, kv_heads: usize, head_dim: usize, max_seq: usize) -> KvShape {
        KvShape {
            num_layers: layers,
            num_kv_heads: kv_heads,
            head_dim,
            max_seq,
        }
    }

    #[test]
    fn qwen25_shape_is_28_layers_2kv_128() {
        let shape = KvShape::qwen25_1_5b(4);
        assert_eq!(shape.num_layers, 28);
        assert_eq!(shape.num_kv_heads, 2);
        assert_eq!(shape.head_dim, 128);
        assert_eq!(shape.max_seq, 4);
        assert_eq!(shape.head_dim, crate::qwen25_layout::HEAD_DIM);
        let cache = KvCache::new(shape).unwrap();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_seq(), 4);
        let slot = cache.key_head(27, 1, 3).unwrap();
        assert_eq!(slot.len(), 128);
        assert!(slot.iter().all(|lane| *lane == 0.0));
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[0.0; 128][..]);

        assert_eq!(
            KvCache::new(KvShape {
                max_seq: 0,
                ..shape
            })
            .unwrap_err(),
            KvError::Shape
        );
        assert_eq!(
            KvCache::new(KvShape {
                num_layers: 0,
                num_kv_heads: 2,
                head_dim: 2,
                max_seq: 1,
            })
            .unwrap_err(),
            KvError::Shape
        );
    }

    #[test]
    fn store_all_layers_then_advance_once() {
        let mut cache = KvCache::new(shape(2, 2, 2, 3)).unwrap();
        // Two KV heads, head_dim 2. Head 0 then head 1.
        let k0 = [1.0f32, 2.0, 3.0, 4.0];
        let v0 = [5.0f32, 6.0, 7.0, 8.0];
        let k1 = [9.0f32, 10.0, 11.0, 12.0];
        let v1 = [13.0f32, 14.0, 15.0, 16.0];
        cache.store(0, &k0, &v0).unwrap();
        cache.store(1, &k1, &v1).unwrap();
        // Both layers wrote position 0. The cursor waits for one advance.
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[1.0, 2.0]);
        assert_eq!(cache.key_head(0, 1, 0).unwrap(), &[3.0, 4.0]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[5.0, 6.0]);
        assert_eq!(cache.value_head(1, 1, 0).unwrap(), &[15.0, 16.0]);
        assert_eq!(cache.key_head(1, 0, 0).unwrap(), &[9.0, 10.0]);

        // A second store at the same cursor overwrites that layer only.
        let k_repl = [30.0f32, 31.0, 32.0, 33.0];
        let v_repl = [34.0f32, 35.0, 36.0, 37.0];
        cache.store(0, &k_repl, &v_repl).unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[30.0, 31.0]);
        assert_eq!(cache.value_head(0, 1, 0).unwrap(), &[36.0, 37.0]);
        assert_eq!(cache.key_head(1, 0, 0).unwrap(), &[9.0, 10.0]);

        cache.advance().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let k0b = [17.0f32, 18.0, 19.0, 20.0];
        let v0b = [21.0f32, 22.0, 23.0, 24.0];
        cache.store(0, &k0b, &v0b).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &[17.0, 18.0]);
        assert_eq!(cache.value_head(0, 1, 1).unwrap(), &[23.0, 24.0]);
        // Position 0 of layer 0 is the replacement, not the second token.
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[30.0, 31.0]);
        // Layer 1 has not stored this token.
        assert_eq!(cache.key_head(1, 0, 1).unwrap(), &[0.0, 0.0]);
        assert_eq!(cache.value_head(1, 1, 0).unwrap(), &[15.0, 16.0]);

        cache.advance().unwrap();
        cache.advance().unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.advance().unwrap_err(), KvError::Full);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.store(0, &k0, &v0).unwrap_err(), KvError::Full);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[30.0, 31.0]);

        assert_eq!(cache.store(2, &k0, &v0).unwrap_err(), KvError::Range);
        assert_eq!(cache.store(0, &[1.0], &v0).unwrap_err(), KvError::Shape);
        assert_eq!(cache.key_head(0, 2, 0).unwrap_err(), KvError::Range);
        assert_eq!(cache.key_head(0, 0, 3).unwrap_err(), KvError::Range);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[30.0, 31.0]);
    }

    #[test]
    fn clear_resets_the_cursor() {
        let mut cache = KvCache::new(shape(1, 1, 2, 2)).unwrap();
        cache.store(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.advance().unwrap();
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        // The old slot stays until the next store overwrites position 0.
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[1.0, 2.0]);
        cache.store(0, &[8.0, 9.0], &[6.0, 7.0]).unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[8.0, 9.0]);
        assert_eq!(cache.value_head(0, 0, 0).unwrap(), &[6.0, 7.0]);
    }

    #[test]
    fn store_at_writes_ahead_of_the_cursor_and_packs() {
        let mut cache = KvCache::new(shape(1, 2, 2, 4)).unwrap();
        cache
            .store_at(0, 1, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &[1.0, 2.0]);
        assert_eq!(cache.value_head(0, 1, 1).unwrap(), &[7.0, 8.0]);
        assert_eq!(cache.key_head(0, 0, 0).unwrap(), &[0.0, 0.0]);
        let mut dst_k = vec![0.0f32; 2 * 2 * 2];
        let mut dst_v = vec![0.0f32; 2 * 2 * 2];
        cache.pack_prefix(0, 2, &mut dst_k, &mut dst_v).unwrap();
        // Head 0, positions 0 then 1, then head 1.
        assert_eq!(dst_k, vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        assert_eq!(dst_v, vec![0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 0);
        assert_eq!(
            cache
                .store_at(0, 4, &[1.0, 0.0, 0.0, 0.0], &[0.0; 4])
                .unwrap_err(),
            KvError::Full
        );
        assert_eq!(cache.key_head(0, 0, 1).unwrap(), &[1.0, 2.0]);
    }
}
