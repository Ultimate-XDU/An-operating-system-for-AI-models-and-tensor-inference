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
//! [`gqa_attn_f32_strided`] is the same chains when K and V are stored as
//! `[num_kv_heads, kv_stride, head_dim]` with `kv_stride >= seq_len` (the
//! cache slot stride). Tokens `0..seq_len` of one head are still contiguous.
//! An optional last-token overlay supplies position `seq_len - 1` from a
//! side buffer so decode does not copy the prefix. `kv_stride == seq_len`
//! and no overlay is [`gqa_attn_f32`].
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
//! 4. Value mix. Each output lane is its own left-to-right chain over
//!    visible keys: `acc += prob[t] * v[t, d]` (product, then add; not
//!    `mul_add`). Lanes do not depend on each other, so several lanes of
//!    one head — and the query heads that share a KV head — may advance
//!    together. That does not reorder a lane's adds. Masked keys are not
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
    gqa_attn_f32_strided(y, q, k, v, None, None, probs, shape, mask, shape.seq_len)
}

/// [`gqa_attn_f32`] against a strided KV cache.
///
/// `k` and `v` are `[num_kv_heads, kv_stride, head_dim]`. `kv_stride` is the
/// allocated token stride (`max_seq` on the cache). Only tokens
/// `0..seq_len` participate. When `last_k` and `last_v` are both `Some`,
/// they are `[num_kv_heads, head_dim]` and replace token `seq_len - 1`
/// (the slot in `k`/`v` at that index is not read). Both must be `Some` or
/// both `None`. `kv_stride == seq_len` and no overlay matches
/// [`gqa_attn_f32`] bit for bit.
///
/// On [`GqaError::Shape`], `y` and `probs` are unchanged.
pub fn gqa_attn_f32_strided(
    y: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    last_k: Option<&[f32]>,
    last_v: Option<&[f32]>,
    probs: &mut [f32],
    shape: GqaShape,
    mask: GqaMask,
    kv_stride: usize,
) -> Result<(), GqaError> {
    let q_per_kv = checked_strided(
        y.len(),
        q.len(),
        k.len(),
        v.len(),
        probs.len(),
        last_k.map(|s| s.len()),
        last_v.map(|s| s.len()),
        shape,
        kv_stride,
    )?;
    // `head_dim > 0` was checked above. Same scale for every head.
    let scale = 1.0 / sqrt_f32(shape.head_dim as f32);
    let visible = visible_len(shape.seq_len, mask);
    let grouped = q_per_kv <= MAX_Q
        && q_per_kv
            .checked_mul(shape.seq_len)
            .is_some_and(|n| n <= MAX_SCORE_SLOTS);
    if grouped {
        for kv_head in 0..shape.num_kv_heads {
            let q_base = kv_head * q_per_kv;
            let q_elems = q_per_kv * shape.head_dim;
            let y_group = &mut y[q_base * shape.head_dim..q_base * shape.head_dim + q_elems];
            let q_group = &q[q_base * shape.head_dim..q_base * shape.head_dim + q_elems];
            let out_probs = if kv_head + 1 == shape.num_kv_heads {
                Some(&mut *probs)
            } else {
                None
            };
            attend_group(
                y_group,
                q_group,
                k,
                v,
                last_k,
                last_v,
                out_probs,
                q_per_kv,
                kv_head,
                shape.seq_len,
                kv_stride,
                shape.head_dim,
                visible,
                scale,
            );
        }
    } else {
        for q_head in 0..shape.num_q_heads {
            let kv_head = q_head / q_per_kv;
            attend_one(
                y,
                q,
                k,
                v,
                last_k,
                last_v,
                probs,
                q_head,
                kv_head,
                shape.seq_len,
                kv_stride,
                shape.head_dim,
                visible,
                scale,
            );
        }
    }
    Ok(())
}

/// Query heads kept on the stack together. Qwen2.5-1.5B uses 6.
const MAX_Q: usize = 8;
/// `q_per_kv * seq_len` score slots. 6 × 512 covers S1 (`seq_len` ≤ 256).
const MAX_SCORE_SLOTS: usize = 4096;

/// Both masks keep a prefix of the cache. Key 0 is always in it.
///
/// [`GqaMask::Full`] keeps `0..seq_len`. [`GqaMask::Causal`] keeps
/// `0..=query_pos`, which is every key once `query_pos >= seq_len - 1`.
fn visible_len(seq_len: usize, mask: GqaMask) -> usize {
    match mask {
        GqaMask::Full => seq_len,
        GqaMask::Causal { query_pos } => {
            if query_pos >= seq_len - 1 {
                seq_len
            } else {
                query_pos + 1
            }
        }
    }
}

fn checked_strided(
    y_len: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
    probs_len: usize,
    last_k_len: Option<usize>,
    last_v_len: Option<usize>,
    shape: GqaShape,
    kv_stride: usize,
) -> Result<usize, GqaError> {
    let GqaShape {
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    } = shape;
    if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 || seq_len == 0 || kv_stride == 0 {
        return Err(GqaError::Shape);
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(GqaError::Shape);
    }
    let overlay = match (last_k_len, last_v_len) {
        (None, None) => false,
        (Some(_), Some(_)) => true,
        _ => return Err(GqaError::Shape),
    };
    let min_stride = if overlay {
        if seq_len == 1 {
            1
        } else {
            seq_len - 1
        }
    } else {
        seq_len
    };
    if kv_stride < min_stride {
        return Err(GqaError::Shape);
    }
    let q_per_kv = num_q_heads / num_kv_heads;
    let q_elems = num_q_heads.checked_mul(head_dim).ok_or(GqaError::Shape)?;
    let kv_elems = num_kv_heads
        .checked_mul(kv_stride)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or(GqaError::Shape)?;
    if y_len != q_elems || q_len != q_elems || k_len != kv_elems || v_len != kv_elems {
        return Err(GqaError::Shape);
    }
    if probs_len != seq_len {
        return Err(GqaError::Shape);
    }
    if let (Some(lk), Some(lv)) = (last_k_len, last_v_len) {
        let kv_dim = num_kv_heads.checked_mul(head_dim).ok_or(GqaError::Shape)?;
        if lk != kv_dim || lv != kv_dim {
            return Err(GqaError::Shape);
        }
    }
    Ok(q_per_kv)
}

/// In-place softmax over `0..visible`. `scores[0]` is a visible score.
///
/// Max, then `expf(score - max)` summed left to right, then `e / sum`.
/// Slots `visible..` are set to `0.0` and are not part of `max` or `sum`.
/// That is the same set as the old per-key mask: both [`GqaMask`] variants
/// keep a prefix that starts at key 0.
fn softmax_prefix(scores: &mut [f32], visible: usize) {
    debug_assert!(visible >= 1 && visible <= scores.len());
    let mut max_score = scores[0];
    for score in scores.iter().take(visible).skip(1) {
        if *score > max_score {
            max_score = *score;
        }
    }
    let mut sum = 0.0f32;
    for score in &mut scores[..visible] {
        let e = exp_f32(*score - max_score);
        *score = e;
        sum += e;
    }
    for score in &mut scores[..visible] {
        *score /= sum;
    }
    for score in &mut scores[visible..] {
        *score = 0.0;
    }
}

fn row_slice<'a>(
    base: &'a [f32],
    last: Option<&'a [f32]>,
    kv_head: usize,
    t: usize,
    seq_len: usize,
    kv_stride: usize,
    head_dim: usize,
) -> &'a [f32] {
    if let Some(last) = last {
        if t + 1 == seq_len {
            let s = kv_head * head_dim;
            return &last[s..s + head_dim];
        }
    }
    let off = (kv_head * kv_stride + t) * head_dim;
    &base[off..off + head_dim]
}

/// Left to right over `d`. One accumulator per query. Not `mul_add`.
fn dot_queries(acc: &mut [f32], qs: &[&[f32]], k: &[f32]) {
    for a in acc.iter_mut() {
        *a = 0.0;
    }
    for d in 0..k.len() {
        let kd = k[d];
        for (a, q) in acc.iter_mut().zip(qs.iter()) {
            *a += q[d] * kd;
        }
    }
}

fn attend_one(
    y: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    last_k: Option<&[f32]>,
    last_v: Option<&[f32]>,
    probs: &mut [f32],
    q_head: usize,
    kv_head: usize,
    seq_len: usize,
    kv_stride: usize,
    head_dim: usize,
    visible: usize,
    scale: f32,
) {
    let q_row = &q[q_head * head_dim..(q_head + 1) * head_dim];
    let mut acc = [0.0f32; 1];
    for t in 0..visible {
        let k_row = row_slice(k, last_k, kv_head, t, seq_len, kv_stride, head_dim);
        dot_queries(&mut acc, &[q_row], k_row);
        probs[t] = acc[0] * scale;
    }
    softmax_prefix(probs, visible);
    let y_row = &mut y[q_head * head_dim..(q_head + 1) * head_dim];
    mix_queries(
        y_row, probs, seq_len, 1, v, last_v, kv_head, seq_len, kv_stride, head_dim, visible,
    );
}

fn attend_group(
    y: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    last_k: Option<&[f32]>,
    last_v: Option<&[f32]>,
    out_probs: Option<&mut [f32]>,
    q_per_kv: usize,
    kv_head: usize,
    seq_len: usize,
    kv_stride: usize,
    head_dim: usize,
    visible: usize,
    scale: f32,
) {
    let mut scores = [0.0f32; MAX_SCORE_SLOTS];
    let mut qrows: [&[f32]; MAX_Q] = [&[]; MAX_Q];
    for qi in 0..q_per_kv {
        let s = qi * head_dim;
        qrows[qi] = &q[s..s + head_dim];
    }
    if q_per_kv == 6 && head_dim == 128 {
        let qptrs = [
            qrows[0].as_ptr(),
            qrows[1].as_ptr(),
            qrows[2].as_ptr(),
            qrows[3].as_ptr(),
            qrows[4].as_ptr(),
            qrows[5].as_ptr(),
        ];
        score_6x128(
            &mut scores,
            &qptrs,
            k,
            last_k,
            kv_head,
            seq_len,
            kv_stride,
            visible,
            scale,
        );
    } else {
        let mut acc = [0.0f32; MAX_Q];
        for t in 0..visible {
            let k_row = row_slice(k, last_k, kv_head, t, seq_len, kv_stride, head_dim);
            dot_queries(&mut acc[..q_per_kv], &qrows[..q_per_kv], k_row);
            for qi in 0..q_per_kv {
                scores[qi * seq_len + t] = acc[qi] * scale;
            }
        }
    }
    for qi in 0..q_per_kv {
        let row = qi * seq_len;
        softmax_prefix(&mut scores[row..row + seq_len], visible);
    }
    mix_queries(
        y, &scores, seq_len, q_per_kv, v, last_v, kv_head, seq_len, kv_stride, head_dim, visible,
    );
    if let Some(out) = out_probs {
        let last = (q_per_kv - 1) * seq_len;
        out.copy_from_slice(&scores[last..last + seq_len]);
    }
}

/// Six query dots against one or two K rows of length 128.
///
/// Each query keeps one left-to-right accumulator. A second token is a
/// second accumulator, not a reorder of the first.
fn score_6x128(
    scores: &mut [f32],
    qs: &[*const f32; 6],
    k: &[f32],
    last_k: Option<&[f32]>,
    kv_head: usize,
    seq_len: usize,
    kv_stride: usize,
    visible: usize,
    scale: f32,
) {
    let mut t = 0usize;
    while t + 1 < visible {
        let k0 = row_slice(k, last_k, kv_head, t, seq_len, kv_stride, 128);
        let k1 = row_slice(k, last_k, kv_head, t + 1, seq_len, kv_stride, 128);
        // SAFETY: each q pointer and both K rows have 128 lanes. `qrows`
        // in the caller keeps those allocations borrowed for this call.
        let (a, b) = unsafe { dot_6x128_pair(qs, k0.as_ptr(), k1.as_ptr()) };
        for qi in 0..6 {
            scores[qi * seq_len + t] = a[qi] * scale;
            scores[qi * seq_len + t + 1] = b[qi] * scale;
        }
        t += 2;
    }
    if t < visible {
        let k0 = row_slice(k, last_k, kv_head, t, seq_len, kv_stride, 128);
        // SAFETY: same 128-lane pointers as the pair path.
        let a = unsafe { dot_6x128(qs, k0.as_ptr()) };
        for qi in 0..6 {
            scores[qi * seq_len + t] = a[qi] * scale;
        }
    }
}

/// SAFETY: `qs[i]` and `k` each point at 128 f32 lanes.
unsafe fn dot_6x128(qs: &[*const f32; 6], k: *const f32) -> [f32; 6] {
    let q0 = qs[0];
    let q1 = qs[1];
    let q2 = qs[2];
    let q3 = qs[3];
    let q4 = qs[4];
    let q5 = qs[5];
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut a4 = 0.0f32;
    let mut a5 = 0.0f32;
    let mut d = 0usize;
    while d < 128 {
        let u = *k.add(d);
        a0 += *q0.add(d) * u;
        a1 += *q1.add(d) * u;
        a2 += *q2.add(d) * u;
        a3 += *q3.add(d) * u;
        a4 += *q4.add(d) * u;
        a5 += *q5.add(d) * u;
        d += 1;
    }
    [a0, a1, a2, a3, a4, a5]
}

/// SAFETY: `qs[i]`, `k0`, and `k1` each point at 128 f32 lanes.
unsafe fn dot_6x128_pair(
    qs: &[*const f32; 6],
    k0: *const f32,
    k1: *const f32,
) -> ([f32; 6], [f32; 6]) {
    let q0 = qs[0];
    let q1 = qs[1];
    let q2 = qs[2];
    let q3 = qs[3];
    let q4 = qs[4];
    let q5 = qs[5];
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut a4 = 0.0f32;
    let mut a5 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut b3 = 0.0f32;
    let mut b4 = 0.0f32;
    let mut b5 = 0.0f32;
    let mut d = 0usize;
    while d < 128 {
        let x0 = *q0.add(d);
        let x1 = *q1.add(d);
        let x2 = *q2.add(d);
        let x3 = *q3.add(d);
        let x4 = *q4.add(d);
        let x5 = *q5.add(d);
        let u = *k0.add(d);
        let w = *k1.add(d);
        a0 += x0 * u;
        b0 += x0 * w;
        a1 += x1 * u;
        b1 += x1 * w;
        a2 += x2 * u;
        b2 += x2 * w;
        a3 += x3 * u;
        b3 += x3 * w;
        a4 += x4 * u;
        b4 += x4 * w;
        a5 += x5 * u;
        b5 += x5 * w;
        d += 1;
    }
    ([a0, a1, a2, a3, a4, a5], [b0, b1, b2, b3, b4, b5])
}

fn mix_queries(
    y: &mut [f32],
    probs: &[f32],
    seq_len: usize,
    q_per_kv: usize,
    v: &[f32],
    last_v: Option<&[f32]>,
    kv_head: usize,
    kv_seq: usize,
    kv_stride: usize,
    head_dim: usize,
    visible: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::gemv_bf16_kernel::avx2_available() {
        // SAFETY: `avx2_available` checked OSXSAVE, AVX, XCR0, and AVX2.
        // Shape checks put every V row and every y lane in range.
        unsafe {
            mix_queries_avx2(
                y, probs, seq_len, q_per_kv, v, last_v, kv_head, kv_seq, kv_stride, head_dim,
                visible,
            );
        }
        return;
    }
    mix_queries_scalar(
        y, probs, seq_len, q_per_kv, v, last_v, kv_head, kv_seq, kv_stride, head_dim, visible, 0,
    );
}

/// Portable value mix. Each lane adds `prob[t] * v[t, d]` from `t = 0`.
fn mix_queries_scalar(
    y: &mut [f32],
    probs: &[f32],
    seq_len: usize,
    q_per_kv: usize,
    v: &[f32],
    last_v: Option<&[f32]>,
    kv_head: usize,
    kv_seq: usize,
    kv_stride: usize,
    head_dim: usize,
    visible: usize,
    d_start: usize,
) {
    let mut d0 = d_start;
    while d0 < head_dim {
        let n = (head_dim - d0).min(8);
        let mut acc = [[0.0f32; 8]; MAX_Q];
        for t in 0..visible {
            let row = row_slice(v, last_v, kv_head, t, kv_seq, kv_stride, head_dim);
            for qi in 0..q_per_kv {
                let p = probs[qi * seq_len + t];
                for i in 0..n {
                    acc[qi][i] += p * row[d0 + i];
                }
            }
        }
        for qi in 0..q_per_kv {
            let dst = qi * head_dim + d0;
            y[dst..dst + n].copy_from_slice(&acc[qi][..n]);
        }
        d0 += n;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mix_queries_avx2(
    y: &mut [f32],
    probs: &[f32],
    seq_len: usize,
    q_per_kv: usize,
    v: &[f32],
    last_v: Option<&[f32]>,
    kv_head: usize,
    kv_seq: usize,
    kv_stride: usize,
    head_dim: usize,
    visible: usize,
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let y_ptr = y.as_mut_ptr();
    let prob_ptr = probs.as_ptr();
    let mut d0 = 0usize;
    while d0 + 8 <= head_dim {
        let mut acc = [_mm256_setzero_ps(); MAX_Q];
        for t in 0..visible {
            let row = row_slice(v, last_v, kv_head, t, kv_seq, kv_stride, head_dim);
            let vv = _mm256_loadu_ps(row.as_ptr().add(d0));
            if q_per_kv == 6 {
                let p0 = _mm256_set1_ps(*prob_ptr.add(t));
                let p1 = _mm256_set1_ps(*prob_ptr.add(seq_len + t));
                let p2 = _mm256_set1_ps(*prob_ptr.add(2 * seq_len + t));
                let p3 = _mm256_set1_ps(*prob_ptr.add(3 * seq_len + t));
                let p4 = _mm256_set1_ps(*prob_ptr.add(4 * seq_len + t));
                let p5 = _mm256_set1_ps(*prob_ptr.add(5 * seq_len + t));
                acc[0] = _mm256_add_ps(acc[0], _mm256_mul_ps(p0, vv));
                acc[1] = _mm256_add_ps(acc[1], _mm256_mul_ps(p1, vv));
                acc[2] = _mm256_add_ps(acc[2], _mm256_mul_ps(p2, vv));
                acc[3] = _mm256_add_ps(acc[3], _mm256_mul_ps(p3, vv));
                acc[4] = _mm256_add_ps(acc[4], _mm256_mul_ps(p4, vv));
                acc[5] = _mm256_add_ps(acc[5], _mm256_mul_ps(p5, vv));
            } else {
                for qi in 0..q_per_kv {
                    let p = _mm256_set1_ps(*prob_ptr.add(qi * seq_len + t));
                    acc[qi] = _mm256_add_ps(acc[qi], _mm256_mul_ps(p, vv));
                }
            }
        }
        for qi in 0..q_per_kv {
            _mm256_storeu_ps(y_ptr.add(qi * head_dim + d0), acc[qi]);
        }
        d0 += 8;
    }
    if d0 < head_dim {
        mix_queries_scalar(
            y, probs, seq_len, q_per_kv, v, last_v, kv_head, kv_seq, kv_stride, head_dim, visible,
            d0,
        );
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

    #[test]
    fn strided_overlay_matches_packed_and_keeps_lane_order() {
        // Two KV heads, head_dim 8 (one AVX lane group), seq 5 stored at
        // stride 8. The gap between heads is poison. The last token is not
        // in the cache buffer. Lane 0 of V is the left-to-right cancellation
        // pattern; a pairwise token sum would not yield 3 after prob 1.
        const HEADS: usize = 2;
        const DIM: usize = 8;
        const SEQ: usize = 8;
        const STRIDE: usize = 12;
        let big = 16_777_216.0f32;
        let pattern = [big, 1.0, 1.0, 1.0, -big, 1.0, 1.0, 1.0];
        // Q = 0 → every score is 0 → softmax is uniform 1/seq. Lane 0 of V
        // is the cancellation pattern, so the mix order is visible.
        let q = vec![0.0f32; 4 * DIM];
        let mut k_packed = vec![0.0f32; HEADS * SEQ * DIM];
        let mut v_packed = vec![0.0f32; HEADS * SEQ * DIM];
        let mut k_cache = vec![99.0f32; HEADS * STRIDE * DIM];
        let mut v_cache = vec![99.0f32; HEADS * STRIDE * DIM];
        let mut last_k = vec![0.0f32; HEADS * DIM];
        let mut last_v = vec![0.0f32; HEADS * DIM];
        for head in 0..HEADS {
            for t in 0..SEQ {
                for d in 0..DIM {
                    let packed = (head * SEQ + t) * DIM + d;
                    let value = if d == 0 {
                        pattern[t]
                    } else if t + 1 == SEQ {
                        (head + d) as f32
                    } else {
                        0.0
                    };
                    let key = if d == 0 { 1.0 } else { 0.0 };
                    k_packed[packed] = key;
                    v_packed[packed] = value;
                    if t + 1 == SEQ {
                        last_k[head * DIM + d] = key;
                        last_v[head * DIM + d] = value;
                    } else {
                        let slot = (head * STRIDE + t) * DIM + d;
                        k_cache[slot] = key;
                        v_cache[slot] = value;
                    }
                }
            }
        }
        let gqa_shape = shape(4, HEADS, DIM, SEQ);
        let mut y_packed = vec![0.0f32; 4 * DIM];
        let mut p_packed = vec![0.0f32; SEQ];
        gqa_attn_f32(
            &mut y_packed,
            &q,
            &k_packed,
            &v_packed,
            &mut p_packed,
            gqa_shape,
            GqaMask::Full,
        )
        .unwrap();
        let mut y_stride = vec![7.0f32; 4 * DIM];
        let mut p_stride = vec![7.0f32; SEQ];
        gqa_attn_f32_strided(
            &mut y_stride,
            &q,
            &k_cache,
            &v_cache,
            Some(&last_k),
            Some(&last_v),
            &mut p_stride,
            gqa_shape,
            GqaMask::Full,
            STRIDE,
        )
        .unwrap();
        for (i, (got, expect)) in y_stride.iter().zip(y_packed.iter()).enumerate() {
            assert_eq!(got.to_bits(), expect.to_bits(), "y[{i}]");
        }
        for (i, (got, expect)) in p_stride.iter().zip(p_packed.iter()).enumerate() {
            assert_eq!(got.to_bits(), expect.to_bits(), "p[{i}]");
        }
        let p = 1.0f32 / SEQ as f32;
        let mut ltr = 0.0f32;
        for value in pattern {
            ltr += p * value;
        }
        let mut pairwise = 0.0f32;
        for chunk in pattern.chunks(2) {
            let mut pair = 0.0f32;
            for value in chunk {
                pair += p * *value;
            }
            pairwise += pair;
        }
        assert_eq!(y_packed[0].to_bits(), ltr.to_bits());
        assert_ne!(ltr.to_bits(), pairwise.to_bits());

        // Causal keeps the prefix and still matches the packed call.
        let mut y_c = vec![0.0f32; 4 * DIM];
        let mut p_c = vec![0.0f32; SEQ];
        gqa_attn_f32_strided(
            &mut y_c,
            &q,
            &k_cache,
            &v_cache,
            Some(&last_k),
            Some(&last_v),
            &mut p_c,
            gqa_shape,
            GqaMask::Causal { query_pos: 2 },
            STRIDE,
        )
        .unwrap();
        let mut y_pref = vec![0.0f32; 4 * DIM];
        let mut p_pref = vec![0.0f32; SEQ];
        gqa_attn_f32(
            &mut y_pref,
            &q,
            &k_packed,
            &v_packed,
            &mut p_pref,
            gqa_shape,
            GqaMask::Causal { query_pos: 2 },
        )
        .unwrap();
        for (i, (got, expect)) in y_c.iter().zip(y_pref.iter()).enumerate() {
            assert_eq!(got.to_bits(), expect.to_bits(), "causal y[{i}]");
        }
        assert_eq!(p_c, p_pref);
        assert_eq!(p_c[3], 0.0);

        // A short stride, or only one of the overlay buffers, writes nothing.
        let y_before = y_stride.clone();
        let p_before = p_stride.clone();
        assert_eq!(
            gqa_attn_f32_strided(
                &mut y_stride,
                &q,
                &k_cache,
                &v_cache,
                Some(&last_k),
                Some(&last_v),
                &mut p_stride,
                gqa_shape,
                GqaMask::Full,
                SEQ - 2,
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y_stride, y_before);
        assert_eq!(p_stride, p_before);
        assert_eq!(
            gqa_attn_f32_strided(
                &mut y_stride,
                &q,
                &k_packed,
                &v_packed,
                Some(&last_k),
                None,
                &mut p_stride,
                gqa_shape,
                GqaMask::Full,
                SEQ,
            ),
            Err(GqaError::Shape)
        );
        assert_eq!(y_stride, y_before);
    }

    #[test]
    fn qwen_width_strided_matches_packed_bits() {
        const H: usize = HEAD_DIM_QWEN25_1_5B;
        const Q_HEADS: usize = NUM_ATTENTION_HEADS_QWEN25_1_5B;
        const KV_HEADS: usize = NUM_KV_HEADS_QWEN25_1_5B;
        const SEQ: usize = 7;
        const STRIDE: usize = 11;
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if state == 0 {
                state = 0xA5A5_1234;
            }
            let mut bits = state;
            if (bits >> 23) & 0xFF == 0xFF {
                bits = (bits & 0x807F_FFFF) | (120 << 23);
            }
            f32::from_bits(bits)
        };
        let mut q = vec![0.0f32; Q_HEADS * H];
        for lane in &mut q {
            *lane = next();
        }
        let mut k_packed = vec![0.0f32; KV_HEADS * SEQ * H];
        let mut v_packed = vec![0.0f32; KV_HEADS * SEQ * H];
        for lane in k_packed.iter_mut().chain(v_packed.iter_mut()) {
            *lane = next();
        }
        let mut k_cache = vec![0.0f32; KV_HEADS * STRIDE * H];
        let mut v_cache = vec![0.0f32; KV_HEADS * STRIDE * H];
        let mut last_k = vec![0.0f32; KV_HEADS * H];
        let mut last_v = vec![0.0f32; KV_HEADS * H];
        for head in 0..KV_HEADS {
            for t in 0..SEQ - 1 {
                let src = (head * SEQ + t) * H;
                let dst = (head * STRIDE + t) * H;
                k_cache[dst..dst + H].copy_from_slice(&k_packed[src..src + H]);
                v_cache[dst..dst + H].copy_from_slice(&v_packed[src..src + H]);
            }
            let src = (head * SEQ + SEQ - 1) * H;
            last_k[head * H..(head + 1) * H].copy_from_slice(&k_packed[src..src + H]);
            last_v[head * H..(head + 1) * H].copy_from_slice(&v_packed[src..src + H]);
        }
        // Poison the unread slot at the write cursor and the gap past seq.
        for head in 0..KV_HEADS {
            for t in (SEQ - 1)..STRIDE {
                let dst = (head * STRIDE + t) * H;
                for lane in &mut k_cache[dst..dst + H] {
                    *lane = 1.0e20;
                }
                for lane in &mut v_cache[dst..dst + H] {
                    *lane = -1.0e20;
                }
            }
        }
        let gqa_shape = shape(Q_HEADS, KV_HEADS, H, SEQ);
        let mut y_packed = vec![0.0f32; Q_HEADS * H];
        let mut p_packed = vec![0.0f32; SEQ];
        gqa_attn_f32(
            &mut y_packed,
            &q,
            &k_packed,
            &v_packed,
            &mut p_packed,
            gqa_shape,
            GqaMask::Full,
        )
        .unwrap();
        let mut y_stride = vec![0.0f32; Q_HEADS * H];
        let mut p_stride = vec![0.0f32; SEQ];
        gqa_attn_f32_strided(
            &mut y_stride,
            &q,
            &k_cache,
            &v_cache,
            Some(&last_k),
            Some(&last_v),
            &mut p_stride,
            gqa_shape,
            GqaMask::Full,
            STRIDE,
        )
        .unwrap();
        for (i, (got, expect)) in y_stride.iter().zip(&y_packed).enumerate() {
            assert_eq!(got.to_bits(), expect.to_bits(), "y[{i}]");
        }
        for (i, (got, expect)) in p_stride.iter().zip(&p_packed).enumerate() {
            assert_eq!(got.to_bits(), expect.to_bits(), "p[{i}]");
        }
    }
}
