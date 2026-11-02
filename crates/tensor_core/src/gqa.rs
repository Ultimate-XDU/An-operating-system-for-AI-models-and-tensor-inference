//! Grouped-query attention scores and the weighted value mix.
//!
//! One query position against a cached key/value sequence. Activations are
//! f32 for the whole op: Q, K, V, scores, probabilities, and the output.
//! There is no weight tensor here. Qwen2.5 Q/K/V projections stay on
//! [`crate::gemv_bf16`] later; this function starts after those projections
//! (and after RoPE on Q and K).
//!
//! Qwen2.5-1.5B passes [`NUM_ATTENTION_HEADS_QWEN25_1_5B`] (12),
//! [`NUM_KV_HEADS_QWEN25_1_5B`] (2), and [`crate::HEAD_DIM_QWEN25_1_5B`]
//! (128). That is [`GQA_Q_PER_KV_QWEN25_1_5B`] (6) query heads per KV head.
//! The counts live on [`GqaShape`] so a test can shrink them. The op does
//! not hard-code the 1.5B width.
//!
//! # Layout
//!
//! Row-major, one batch, one query token. This is the logical HF Qwen2
//! eager layout after the head view, without a batch axis: Q is
//! `[num_q_heads, head_dim]`, and K/V are `[num_kv_heads, seq_len, head_dim]`.
//!
//! ```text
//! q[(q_head * head_dim) + d]
//! k[((kv_head * seq_len) + t) * head_dim + d]
//! v[((kv_head * seq_len) + t) * head_dim + d]
//! y[(q_head * head_dim) + d]
//! ```
//!
//! `probs` is scratch of length `seq_len`, not an output the model reads.
//! On success it holds the last query head's probabilities, with `0.0` in
//! every masked slot. Earlier heads are overwritten.
//!
//! # Groups
//!
//! `q_per_kv = num_q_heads / num_kv_heads` (exact; otherwise
//! [`GqaError::Shape`]). Query head `h` reads KV head `h / q_per_kv`.
//! Heads `[g * q_per_kv, (g + 1) * q_per_kv)` share KV head `g`. That is
//! HF `repeat_kv` / `repeat_interleave` on the head axis, not a round-robin
//! `h % num_kv_heads` map. Group size 1 (MHA) and one KV head (MQA) are the
//! same rule.
//!
//! # Scale
//!
//! `scale = 1 / sqrt(head_dim)` in f32, once per call:
//! `1.0 / sqrtf(head_dim as f32)`. [`attn_scale_f32`] is that value.
//! `head_dim as f32` is exact for every head width up to `2^24` (Qwen uses
//! 128). Each score is `dot * scale` after the dot is finished. The scale
//! is not folded into the lane products, and the dot is not divided by
//! `sqrt(head_dim)` (multiply by the reciprocal is the fingerprint).
//!
//! # Score, softmax, mix
//!
//! One thread. Products are formed first, then added. The op does not call
//! `mul_add`. For each query head, in head order:
//!
//! 1. Dots, left to right over `d = 0 .. head_dim`:
//!    `acc += q[d] * k[t, d]`. Masked keys are not read.
//! 2. `score[t] = acc * scale`.
//! 3. Softmax over the visible key indexes, left to right. Both masks keep
//!    key `0`, so the max starts at that score (not at `-inf`):
//!    - `max` is the first score, then `if score > max { max = score }`
//!      (`>` , not `>=`).
//!    - `e = expf(score - max)`, `sum += e`.
//!    - `prob = e / sum` (division, not `e * (1/sum)`; those differ by an ulp).
//! 4. Value mix, one output lane at a time. For each `d`, left to right
//!    over visible keys: `acc += prob[t] * v[t, d]`. Masked keys are not
//!    read. A visible key whose `expf` underflowed to zero is still a term.
//!
//! Masked keys are omitted. They are not stored as `-inf` and then run
//! through `expf`. For finite visible scores that matches an additive `-inf`
//! mask (`exp(-inf) = 0` does not move `max` or `sum`) and it does not read
//! garbage V in slots the query is not allowed to see. A finite `max` makes
//! `expf(0) = 1`, so `sum` is at least 1 before the other visible terms.
//!
//! # Causal mask
//!
//! [`GqaMask::Full`] scores every cached key. Decode whose cache is already
//! the prefix `0..=query_pos` (length `query_pos + 1`) uses this.
//!
//! [`GqaMask::Causal`] keeps key index `t` when `t <= query_pos`. Slots
//! past the query are dropped before the dot. `query_pos >= seq_len - 1`
//! keeps every key, the same set as [`GqaMask::Full`]. Key `0` is always
//! in that set (`query_pos` is a `usize`).
//!
//! Prefill over many query positions is not this function. KV-cache
//! allocation is not this function.
//!
//! On [`GqaError::Shape`], `y` and `probs` are left unchanged.
//!
//! Fingerprint intent (the run hash is later): weights stay
//! `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`. This op adds no second
//! dtype. Its exact-sensitive choices are the serial dot, the f32 scale,
//! and the serial max / `expf` / sum / divide softmax above.

use core::fmt;

/// Qwen2.5-1.5B `num_attention_heads`.
pub const NUM_ATTENTION_HEADS_QWEN25_1_5B: usize = 12;

/// Qwen2.5-1.5B `num_key_value_heads`.
pub const NUM_KV_HEADS_QWEN25_1_5B: usize = 2;

/// Query heads that share one KV head: 12 / 2.
///
/// Head `h` reads KV head `h / GQA_Q_PER_KV_QWEN25_1_5B`.
pub const GQA_Q_PER_KV_QWEN25_1_5B: usize = 6;

/// Counts for [`gqa_attn_f32`].
///
/// `num_q_heads` must be a non-zero multiple of `num_kv_heads`.
/// `head_dim` and `seq_len` must be non-zero. Named fields so a caller
/// cannot swap the sequence length with the head width by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaShape {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}

/// Which cached keys the single query position may score.
///
/// See the module docs. Key index `0` is visible for both variants, so the
/// softmax max starts at that key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaMask {
    /// Score every key in `0..seq_len`.
    Full,
    /// Score key `t` only when `t <= query_pos`.
    Causal {
        /// Position of this query. Key indexes are positions in the cache.
        query_pos: usize,
    },
}

/// Shape failure from a GQA op.
///
/// On `Err`, `y` and the probability scratch are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaError {
    /// A count is zero, `num_q_heads` is not a multiple of `num_kv_heads`,
    /// or a buffer length does not match the layout in the module docs.
    Shape,
}

impl fmt::Display for GqaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GqaError::Shape => write!(f, "gqa shape does not match the buffers"),
        }
    }
}

/// `1 / sqrt(head_dim)` in f32.
///
/// IEEE-754 `sqrtf`, then one division. `head_dim == 0` is
/// [`GqaError::Shape`]. [`gqa_attn_f32`] multiplies each finished dot by
/// this value.
pub fn attn_scale_f32(head_dim: usize) -> Result<f32, GqaError> {
    if head_dim == 0 {
        return Err(GqaError::Shape);
    }
    Ok(1.0 / sqrt_f32(head_dim as f32))
}

/// Scores, softmax, and the value mix for one query position.
///
/// `y` and `q` are `[num_q_heads, head_dim]`. `k` and `v` are
/// `[num_kv_heads, seq_len, head_dim]`. `probs` has length `seq_len`.
/// Those counts are [`GqaShape`]. Layout, groups, scale, softmax order,
/// and [`GqaMask`]: see the module docs. Qwen2.5-1.5B passes 12, 2, and
/// [`crate::HEAD_DIM_QWEN25_1_5B`].
///
/// On [`GqaError::Shape`], `y` and `probs` are unchanged.
pub fn gqa_attn_f32(
    y: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &mut [f32],
    shape: GqaShape,
    mask: GqaMask,
) -> Result<(), GqaError> {
    let q_per_kv = checked_shape(y.len(), q.len(), k.len(), v.len(), probs.len(), shape)?;
    let GqaShape {
        num_q_heads,
        head_dim,
        seq_len,
        ..
    } = shape;
    // `head_dim > 0` was checked above. Same scale for every head.
    let scale = attn_scale_f32(head_dim)?;
    for q_head in 0..num_q_heads {
        let kv_head = q_head / q_per_kv;
        let q_row = &q[q_head * head_dim..(q_head + 1) * head_dim];
        for (t, slot) in probs.iter_mut().enumerate() {
            if !key_kept(t, mask) {
                continue;
            }
            let k_at = row_offset(kv_head, t, seq_len, head_dim);
            *slot = dot_ltr(q_row, &k[k_at..k_at + head_dim]) * scale;
        }
        // Key 0 is visible for every mask, so `probs[0]` is a real score.
        softmax_kept(probs, mask);
        let y_row = &mut y[q_head * head_dim..(q_head + 1) * head_dim];
        for (d, out) in y_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (t, prob) in probs.iter().enumerate() {
                if !key_kept(t, mask) {
                    continue;
                }
                let v_at = row_offset(kv_head, t, seq_len, head_dim) + d;
                acc += *prob * v[v_at];
            }
            *out = acc;
        }
    }
    Ok(())
}

/// `true` when this mask scores key index `t`.
///
/// `t == 0` is always kept. [`GqaMask::Causal`] with `query_pos >= seq_len - 1`
/// keeps every index in `0..seq_len`, the same set as [`GqaMask::Full`].
fn key_kept(t: usize, mask: GqaMask) -> bool {
    match mask {
        GqaMask::Full => true,
        GqaMask::Causal { query_pos } => t <= query_pos,
    }
}

fn checked_shape(
    y_len: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
    probs_len: usize,
    shape: GqaShape,
) -> Result<usize, GqaError> {
    let GqaShape {
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    } = shape;
    if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 || seq_len == 0 {
        return Err(GqaError::Shape);
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(GqaError::Shape);
    }
    let q_per_kv = num_q_heads / num_kv_heads;
    let q_elems = num_q_heads.checked_mul(head_dim).ok_or(GqaError::Shape)?;
    let kv_elems = num_kv_heads
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or(GqaError::Shape)?;
    if y_len != q_elems || q_len != q_elems || k_len != kv_elems || v_len != kv_elems {
        return Err(GqaError::Shape);
    }
    if probs_len != seq_len {
        return Err(GqaError::Shape);
    }
    Ok(q_per_kv)
}

fn row_offset(kv_head: usize, t: usize, seq_len: usize, head_dim: usize) -> usize {
    (kv_head * seq_len + t) * head_dim
}

/// Left to right. Multiply, then add. Not `mul_add`.
fn dot_ltr(q: &[f32], k: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for d in 0..q.len() {
        acc += q[d] * k[d];
    }
    acc
}

/// In-place softmax over visible keys. `scores[0]` is a visible score.
///
/// Max, then `expf(score - max)` summed left to right, then `e / sum`.
/// Masked slots are set to `0.0` and are not part of `max` or `sum`.
fn softmax_kept(scores: &mut [f32], mask: GqaMask) {
    debug_assert!(key_kept(0, mask));
    let mut max_score = scores[0];
    for (t, score) in scores.iter().enumerate().skip(1) {
        if key_kept(t, mask) && *score > max_score {
            max_score = *score;
        }
    }
    let mut sum = 0.0f32;
    for (t, score) in scores.iter_mut().enumerate() {
        if !key_kept(t, mask) {
            continue;
        }
        let e = exp_f32(*score - max_score);
        *score = e;
        sum += e;
    }
    for (t, score) in scores.iter_mut().enumerate() {
        if key_kept(t, mask) {
            *score /= sum;
        } else {
            *score = 0.0;
        }
    }
}

// `f32::{exp, sqrt}` live in `std`. This crate is `no_std` outside tests,
// so call libm: IEEE `sqrtf` and `expf`, not a polynomial stand-in.
#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn expf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
}

fn exp_f32(x: f32) -> f32 {
    // SAFETY: expf is defined for every f32 bit pattern.
    unsafe { expf(x) }
}

fn sqrt_f32(x: f32) -> f32 {
    // SAFETY: sqrtf is defined for every f32 bit pattern, including NaN,
    // infinities, and negatives (those return NaN).
    unsafe { sqrtf(x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HEAD_DIM_QWEN25_1_5B;

    fn shape(num_q_heads: usize, num_kv_heads: usize, head_dim: usize, seq_len: usize) -> GqaShape {
        GqaShape {
            num_q_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        }
    }

    #[test]
    fn qwen25_head_counts_and_scale_bits() {
        assert_eq!(NUM_ATTENTION_HEADS_QWEN25_1_5B, 12);
        assert_eq!(NUM_KV_HEADS_QWEN25_1_5B, 2);
        assert_eq!(GQA_Q_PER_KV_QWEN25_1_5B, 6);
        assert_eq!(
            NUM_ATTENTION_HEADS_QWEN25_1_5B / NUM_KV_HEADS_QWEN25_1_5B,
            GQA_Q_PER_KV_QWEN25_1_5B
        );
        // hidden 1536, kv proj width 256.
        assert_eq!(NUM_ATTENTION_HEADS_QWEN25_1_5B * HEAD_DIM_QWEN25_1_5B, 1536);
        assert_eq!(NUM_KV_HEADS_QWEN25_1_5B * HEAD_DIM_QWEN25_1_5B, 256);

        assert_eq!(attn_scale_f32(0), Err(GqaError::Shape));
        // 1/sqrt(1), 1/sqrt(4), 1/sqrt(128). 128 is the 1.5B head dim.
        assert_eq!(attn_scale_f32(1).unwrap().to_bits(), 0x3f80_0000);
        assert_eq!(attn_scale_f32(4).unwrap().to_bits(), 0x3f00_0000);
        assert_eq!(attn_scale_f32(4).unwrap(), 0.5);
        assert_eq!(
            attn_scale_f32(HEAD_DIM_QWEN25_1_5B).unwrap().to_bits(),
            0x3db5_04f3
        );
    }

    #[test]
    fn gqa_groups_are_contiguous_blocks() {
        // head_dim = 1, scale = 1. Four query heads, two KV heads.
        // Heads 0 and 1 share KV 0. Heads 2 and 3 share KV 1.
        // Q is equal inside each group, so a correct map makes y equal
        // inside the group. Round-robin `h % 2` would split them.
        //
        // KV0 keys [1, 0], values [10, 0]. Scores for q = 1 are [1, 0].
        // Softmax bits are libm expf(0) and expf(-1), then e/sum.
        // Forgetting the scale is impossible at head_dim 1; the softmax
        // pin is in `softmax_is_max_sub_expf_sum_div`.
        // KV1 keys [0, 2], values [0, 7]. Scores for q = 4 are [0, 8].
        let q = [1.0f32, 1.0, 4.0, 4.0];
        let k = [1.0f32, 0.0, 0.0, 2.0];
        let v = [10.0f32, 0.0, 0.0, 7.0];
        let mut y = [0.0; 4];
        let mut probs = [0.0; 2];
        gqa_attn_f32(
            &mut y,
            &q,
            &k,
            &v,
            &mut probs,
            shape(4, 2, 1, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(y[0].to_bits(), 0x40e9_f052);
        assert_eq!(y[1].to_bits(), y[0].to_bits());
        assert_eq!(y[2].to_bits(), 0x40df_ecc5);
        assert_eq!(y[3].to_bits(), y[2].to_bits());
        assert_ne!(y[0].to_bits(), y[2].to_bits());
        // Scratch is the last head (head 3), same scores as head 2: [0, 8].
        assert_eq!(probs[0].to_bits(), 0x39af_d1ef);
        assert_eq!(probs[1].to_bits(), 0x3f7f_ea06);
    }

    #[test]
    fn causal_mask_drops_future_keys_and_matches_prefix() {
        // Same cache as the grouping test. query_pos = 0 keeps only t = 0,
        // so each head copies that KV head's first value. The future value
        // (0 or 7) must not move y. A single visible score still softmaxes
        // to 1, including when the score itself is large (here 1 and 0).
        let q = [1.0f32, 1.0, 4.0, 4.0];
        let k = [1.0f32, 0.0, 0.0, 2.0];
        let v = [10.0f32, 0.0, 0.0, 7.0];
        let mut y = [0.0; 4];
        let mut probs = [9.0; 2];
        gqa_attn_f32(
            &mut y,
            &q,
            &k,
            &v,
            &mut probs,
            shape(4, 2, 1, 2),
            GqaMask::Causal { query_pos: 0 },
        )
        .unwrap();
        assert_eq!(y, [10.0, 10.0, 0.0, 0.0]);
        assert_eq!(probs, [1.0, 0.0]);

        // Full attention on the truncated cache is the same mix.
        let mut y_prefix = [0.0; 4];
        let mut probs_prefix = [0.0; 1];
        gqa_attn_f32(
            &mut y_prefix,
            &q,
            &[1.0, 0.0],
            &[10.0, 0.0],
            &mut probs_prefix,
            shape(4, 2, 1, 1),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(y_prefix, y);

        // query_pos at the last key, and past the cache, keep every key.
        let mut y_full = [0.0; 4];
        let mut probs_full = [0.0; 2];
        gqa_attn_f32(
            &mut y_full,
            &q,
            &k,
            &v,
            &mut probs_full,
            shape(4, 2, 1, 2),
            GqaMask::Full,
        )
        .unwrap();
        let mut y_last = [0.0; 4];
        let mut probs_last = [0.0; 2];
        gqa_attn_f32(
            &mut y_last,
            &q,
            &k,
            &v,
            &mut probs_last,
            shape(4, 2, 1, 2),
            GqaMask::Causal { query_pos: 1 },
        )
        .unwrap();
        assert_eq!(y_last, y_full);
        let mut y_past = [0.0; 4];
        let mut probs_past = [0.0; 2];
        gqa_attn_f32(
            &mut y_past,
            &q,
            &k,
            &v,
            &mut probs_past,
            shape(4, 2, 1, 2),
            GqaMask::Causal {
                query_pos: usize::MAX,
            },
        )
        .unwrap();
        assert_eq!(y_past, y_full);

        // One key, score = 50*50 = 2500. exp(2500) overflows. Subtracting
        // the max leaves exp(0) = 1, so y is V and not inf.
        let mut y_one = [0.0];
        let mut probs_one = [0.0];
        gqa_attn_f32(
            &mut y_one,
            &[50.0],
            &[50.0],
            &[3.5],
            &mut probs_one,
            shape(1, 1, 1, 1),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(y_one, [3.5]);
        assert_eq!(probs_one, [1.0]);
    }

    #[test]
    fn softmax_is_max_sub_expf_sum_div() {
        // head_dim = 4, scale = 1/2 exactly. q · k0 = 2, q · k1 = 0,
        // scores [1, 0]. Bits are expf(0), expf(-1), sum left to right,
        // then e/sum (not e * (1/sum)). V mix is prob0*4 + prob1*8.
        // Scores [2, 0] (the unscaled dots) would be a different softmax.
        let q = [2.0f32, 0.0, 0.0, 0.0];
        let mut k = [0.0f32; 8];
        k[0] = 1.0;
        let mut v = [0.0f32; 8];
        v[0] = 4.0;
        v[4] = 8.0;
        let mut y = [0.0; 4];
        let mut probs = [0.0; 2];
        gqa_attn_f32(
            &mut y,
            &q,
            &k,
            &v,
            &mut probs,
            shape(1, 1, 4, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(probs[0].to_bits(), 0x3f3b_26a8);
        assert_eq!(probs[1].to_bits(), 0x3e89_b2b1);
        assert_eq!(y[0].to_bits(), 0x40a2_6cac);
        assert_eq!(&y[1..], &[0.0, 0.0, 0.0]);

        // Equal logits of 100. expf(100) is inf. Subtracting the max first
        // keeps a uniform 1/2, so y = 0.5*2 + 0.5*6 = 4.
        let mut y_big = [0.0];
        let mut probs_big = [0.0; 2];
        gqa_attn_f32(
            &mut y_big,
            &[1.0],
            &[100.0, 100.0],
            &[2.0, 6.0],
            &mut probs_big,
            shape(1, 1, 1, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(probs_big, [0.5, 0.5]);
        assert_eq!(y_big, [4.0]);

        // head_dim = 128, dot = 1 against a zero key. score = 1/sqrt(128).
        // Pins scale, expf(-scale), the left-to-right sum, and e/sum together.
        let mut q128 = [0.0f32; HEAD_DIM_QWEN25_1_5B];
        q128[0] = 1.0;
        let mut k128 = [0.0f32; HEAD_DIM_QWEN25_1_5B * 2];
        k128[0] = 1.0;
        let mut v128 = [0.0f32; HEAD_DIM_QWEN25_1_5B * 2];
        v128[0] = 2.0;
        v128[HEAD_DIM_QWEN25_1_5B] = 6.0;
        let mut y128 = [0.0; HEAD_DIM_QWEN25_1_5B];
        let mut probs128 = [0.0; 2];
        gqa_attn_f32(
            &mut y128,
            &q128,
            &k128,
            &v128,
            &mut probs128,
            shape(1, 1, HEAD_DIM_QWEN25_1_5B, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(probs128[0].to_bits(), 0x3f05_a737);
        assert_eq!(probs128[1].to_bits(), 0x3ef4_b194);
        assert_eq!(y128[0].to_bits(), 0x407a_58ca);
        assert!(y128[1..].iter().all(|lane| *lane == 0.0));
    }

    #[test]
    fn dot_reduction_is_left_to_right() {
        // q = [1, 1, 1, 1], scale = 1/2.
        // k0 = [2^24, 1, 1, 1]. Left to right the 1s fall off 2^24, so the
        // dot stays 16777216 and the score stays 8388608.
        // k1 = [1, 1, 1, 2^24]. The 1s add first (3), then 3 + 2^24 rounds
        // to 16777220 and the score is 8388610.
        // Right to left on k0 would match k1's dot and swap the mix.
        // y[0] = p0*2 + p1*8 with those two scores. The reversed mix is
        // 0x402dc61f, not the pin below.
        let q = [1.0f32, 1.0, 1.0, 1.0];
        let k = [
            16777216.0, 1.0, 1.0, 1.0, // t0
            1.0, 1.0, 1.0, 16777216.0, // t1
        ];
        let v = [
            2.0, 0.0, 0.0, 0.0, // t0
            8.0, 0.0, 0.0, 0.0, // t1
        ];
        let mut y = [0.0; 4];
        let mut probs = [0.0; 2];
        gqa_attn_f32(
            &mut y,
            &q,
            &k,
            &v,
            &mut probs,
            shape(1, 1, 4, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(y[0].to_bits(), 0x40e9_1cef);
        assert_eq!(&y[1..], &[0.0, 0.0, 0.0]);
        assert_ne!(y[0].to_bits(), 0x402d_c61f);
    }

    #[test]
    fn qwen_width_six_queries_share_one_kv() {
        // 12 x 128 against 2 KV heads, seq = 2, Q = 0 so every score is 0
        // and the softmax is uniform 1/2. The scale cancels. KV0 values
        // average to 4 in lane 0; KV1 averages to 20. Heads 0..5 must match
        // KV0 and heads 6..11 must match KV1.
        const H: usize = HEAD_DIM_QWEN25_1_5B;
        const Q_HEADS: usize = NUM_ATTENTION_HEADS_QWEN25_1_5B;
        const KV_HEADS: usize = NUM_KV_HEADS_QWEN25_1_5B;
        let q = [0.0f32; Q_HEADS * H];
        let k = [0.0f32; KV_HEADS * 2 * H];
        let mut v = [0.0f32; KV_HEADS * 2 * H];
        v[0] = 2.0;
        v[H] = 6.0;
        v[2 * H] = 10.0;
        v[3 * H] = 30.0;
        let mut y = [7.0f32; Q_HEADS * H];
        let mut probs = [0.0; 2];
        gqa_attn_f32(
            &mut y,
            &q,
            &k,
            &v,
            &mut probs,
            shape(Q_HEADS, KV_HEADS, H, 2),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(probs, [0.5, 0.5]);
        for head in 0..Q_HEADS {
            let lane0 = y[head * H];
            if head < GQA_Q_PER_KV_QWEN25_1_5B {
                assert_eq!(lane0, 4.0);
            } else {
                assert_eq!(lane0, 20.0);
            }
            assert!(y[head * H + 1..(head + 1) * H]
                .iter()
                .all(|lane| *lane == 0.0));
        }

        // Causal at position 0 keeps t = 0 only: 2 for the first group, 10
        // for the second. The future lanes (6 and 30) are not read.
        let mut y_c = [7.0f32; Q_HEADS * H];
        let mut probs_c = [9.0; 2];
        gqa_attn_f32(
            &mut y_c,
            &q,
            &k,
            &v,
            &mut probs_c,
            shape(Q_HEADS, KV_HEADS, H, 2),
            GqaMask::Causal { query_pos: 0 },
        )
        .unwrap();
        assert_eq!(probs_c, [1.0, 0.0]);
        for head in 0..Q_HEADS {
            let expect = if head < GQA_Q_PER_KV_QWEN25_1_5B {
                2.0
            } else {
                10.0
            };
            assert_eq!(y_c[head * H], expect);
        }
    }

    #[test]
    fn shape_errors_do_not_write() {
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [4.0f32, 5.0, 6.0, 7.0];
        let mut y = [7.0f32, 8.0, 9.0, 10.0];
        let y_before = y;
        let mut probs = [3.0f32, 4.0];
        let probs_before = probs;

        // 3 query heads is not a multiple of 2 KV heads.
        assert_eq!(
            gqa_attn_f32(
                &mut y,
                &q[..3],
                &k,
                &v,
                &mut probs,
                shape(3, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y, y_before);
        assert_eq!(probs, probs_before);

        assert_eq!(
            gqa_attn_f32(
                &mut y[..3],
                &q[..3],
                &k,
                &v,
                &mut probs,
                shape(3, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(probs, probs_before);

        // Length mismatches, including a scratch sized for every head
        // instead of one sequence.
        let mut probs_wide = [3.0f32; 8];
        assert_eq!(
            gqa_attn_f32(
                &mut y,
                &q,
                &k,
                &v,
                &mut probs_wide,
                shape(4, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(probs_wide, [3.0; 8]);
        assert_eq!(y, y_before);

        assert_eq!(
            gqa_attn_f32(
                &mut y[..2],
                &q,
                &k,
                &v,
                &mut probs,
                shape(4, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y, y_before);
        assert_eq!(probs, probs_before);

        let mut y_ok = y;
        assert_eq!(
            gqa_attn_f32(
                &mut y_ok,
                &q,
                &k[..3],
                &v,
                &mut probs,
                shape(4, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y_ok, y_before);
        assert_eq!(
            gqa_attn_f32(
                &mut y_ok,
                &q,
                &k,
                &v[..3],
                &mut probs,
                shape(4, 2, 1, 2),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(probs, probs_before);

        let mut empty_y = [];
        let mut empty_p = [];
        assert_eq!(
            gqa_attn_f32(
                &mut empty_y,
                &[],
                &[],
                &[],
                &mut empty_p,
                shape(0, 2, 1, 1),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(
            gqa_attn_f32(
                &mut y[..1],
                &q[..1],
                &k[..1],
                &v[..1],
                &mut probs[..1],
                shape(1, 0, 1, 1),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(
            gqa_attn_f32(
                &mut y[..1],
                &q[..1],
                &k[..1],
                &v[..1],
                &mut probs[..1],
                shape(1, 1, 0, 1),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(
            gqa_attn_f32(
                &mut y[..1],
                &q[..1],
                &[],
                &[],
                &mut empty_p,
                shape(1, 1, 1, 0),
                GqaMask::Full
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y, y_before);
        assert_eq!(probs, probs_before);

        // MHA (group size 1) and MQA (one KV head) are the same rule, not
        // shape errors. seq = 1 copies V.
        let mut y_mha = [0.0; 2];
        let mut p_mha = [0.0];
        gqa_attn_f32(
            &mut y_mha,
            &[9.0, 8.0],
            &[1.0, 2.0],
            &[3.0, 4.0],
            &mut p_mha,
            shape(2, 2, 1, 1),
            GqaMask::Full,
        )
        .unwrap();
        assert_eq!(y_mha, [3.0, 4.0]);
        let mut y_mqa = [0.0; 2];
        gqa_attn_f32(
            &mut y_mqa,
            &[9.0, 8.0],
            &[1.0],
            &[5.0],
            &mut p_mha,
            shape(2, 1, 1, 1),
            GqaMask::Causal { query_pos: 0 },
        )
        .unwrap();
        assert_eq!(y_mqa, [5.0, 5.0]);
    }
}
