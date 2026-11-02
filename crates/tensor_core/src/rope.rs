//! Rotary position embedding for Qwen2.5-style heads.
//!
//! Activations are f32. There is no weight tensor: Q and K are rotated
//! after the projection. Qwen2.5 passes [`ROPE_THETA_QWEN25`] (`1e6`).
//! Qwen2.5-1.5B uses [`HEAD_DIM_QWEN25_1_5B`] (`128` = 1536 / 12 query heads).
//! K for that model is 2 heads of the same width. One call covers one head
//! or a batch of heads that share a single position (one token's Q, or one
//! token's K).
//!
//! # Pairing
//!
//! Interleaved even/odd. Dimension `2i` rotates with dimension `2i+1`:
//!
//! ```text
//! y[2i]   = x[2i] * cos - x[2i+1] * sin
//! y[2i+1] = x[2i] * sin + x[2i+1] * cos
//! ```
//!
//! Products are formed first, then added or subtracted. The op does not
//! call `mul_add`. This is the adjacent-lane layout from the RoPE paper
//! and from llama.cpp's non-NeoX rope. It stays the contract for this
//! module: a later switch to half-split lanes would move every Q and K bit.
//!
//! Hugging Face Qwen2 `rotate_half`, and llama.cpp Qwen2
//! (`LLAMA_ROPE_TYPE_NEOX`), pair lane `i` with lane `i + head_dim/2` and
//! repeat the frequency table across both halves. Same angles, different
//! index permutation. These functions do not apply that permutation.
//!
//! # Frequencies
//!
//! Each pair has its own table entry, independent of the other pairs
//! (not the iterative `freq *= theta**(-2/head_dim)` recurrence, which
//! rounds differently):
//!
//! ```text
//! inv_freq[i] = 1 / powf(theta, (2*i) / head_dim)
//! angle       = f32(position) * inv_freq[i]
//! cos, sin    = cosf(angle), sinf(angle)
//! ```
//!
//! `powf`, `cosf`, and `sinf` are libm. There is no small-angle shortcut
//! and no extra polynomial. [`rope_inv_freqs`] writes that table.
//! [`rope_f32`] and [`rope_f32_inplace`] evaluate each entry once per call
//! and reuse the resulting `cos`/`sin` for every head in the batch.
//!
//! `position as f32` is exact for every integer up to `2^24`. Qwen2.5
//! `max_position_embeddings` is 32768, so every position that config
//! allows is an exact f32. Position `0` has angle `0`, `cosf(0) == 1`,
//! `sinf(0) == 0`, and the rotation is the identity on finite inputs.
//!
//! On [`RopeError::Shape`], output slices are left unchanged.

use core::fmt;

/// Qwen2.5 `rope_theta` (`config.json`, `1000000.0`).
///
/// [`rope_f32`] and [`rope_f32_inplace`] take `theta` as an argument so a
/// test can move it. The Qwen2.5 track passes this value.
/// See `docs/qwen25-1.5b-instruct-layout.md`.
pub const ROPE_THETA_QWEN25: f32 = 1_000_000.0;

/// Qwen2.5-1.5B head dimension: `hidden_size / num_attention_heads` = 1536 / 12.
///
/// The rotate functions take `head_dim` as an argument. This constant is the
/// width those calls pass for the 1.5B checkpoint (Q = 12 heads, K = 2 heads).
pub const HEAD_DIM_QWEN25_1_5B: usize = 128;

/// Shape failure from a RoPE op.
///
/// On `Err`, output slices passed by the caller are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeError {
    /// `head_dim` is zero or odd, a buffer is empty, or a length is not a
    /// positive multiple of `head_dim` (or of `head_dim / 2` for the
    /// frequency table).
    Shape,
}

impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RopeError::Shape => write!(f, "rope shape does not match the buffers"),
        }
    }
}

/// Write `head_dim / 2` inverse frequencies for one head.
///
/// `inv_freq[i] = 1 / powf(theta, (2*i) / head_dim)`. `head_dim` must be
/// even and non-zero, and `inv_freq.len()` must be `head_dim / 2`.
/// The entries do not depend on position. [`rope_f32`] uses this table.
pub fn rope_inv_freqs(inv_freq: &mut [f32], head_dim: usize, theta: f32) -> Result<(), RopeError> {
    let n_pairs = even_head_dim(head_dim)?;
    if inv_freq.len() != n_pairs {
        return Err(RopeError::Shape);
    }
    for (i, slot) in inv_freq.iter_mut().enumerate() {
        *slot = inv_freq_at(i, head_dim, theta);
    }
    Ok(())
}

/// Out-of-place RoPE. `y` and `x` are one head or several heads, same length.
///
/// Each contiguous `head_dim` block is one head. Every head uses `position`.
/// `head_dim` must be even and non-zero. `x.len()` must be a positive
/// multiple of `head_dim` and equal to `y.len()`.
///
/// Pairing and frequencies: see the module docs. Qwen2.5-1.5B passes
/// [`HEAD_DIM_QWEN25_1_5B`] and [`ROPE_THETA_QWEN25`].
pub fn rope_f32(
    y: &mut [f32],
    x: &[f32],
    position: u32,
    head_dim: usize,
    theta: f32,
) -> Result<(), RopeError> {
    rotate(y, Some(x), position, head_dim, theta)
}

/// In-place RoPE. Same pairing, frequencies, and shape rules as [`rope_f32`].
pub fn rope_f32_inplace(
    x: &mut [f32],
    position: u32,
    head_dim: usize,
    theta: f32,
) -> Result<(), RopeError> {
    rotate(x, None, position, head_dim, theta)
}

/// `head_dim / 2` when `head_dim` is a usable even width.
fn even_head_dim(head_dim: usize) -> Result<usize, RopeError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(RopeError::Shape);
    }
    Ok(head_dim / 2)
}

/// Number of heads in a buffer of `len` lanes.
fn head_count(len: usize, head_dim: usize) -> Result<usize, RopeError> {
    let _n_pairs = even_head_dim(head_dim)?;
    if len == 0 || len % head_dim != 0 {
        return Err(RopeError::Shape);
    }
    Ok(len / head_dim)
}

fn rotate(
    y: &mut [f32],
    x: Option<&[f32]>,
    position: u32,
    head_dim: usize,
    theta: f32,
) -> Result<(), RopeError> {
    if let Some(src) = x {
        if src.len() != y.len() {
            return Err(RopeError::Shape);
        }
    }
    let n_heads = head_count(y.len(), head_dim)?;
    let n_pairs = head_dim / 2;
    let position = position as f32;
    for pair in 0..n_pairs {
        // One (cos, sin) for this pair, then every head. That is the
        // position's frequency table: pair `i` does not depend on pair `i-1`.
        let (cos, sin) = pair_cos_sin(pair, head_dim, theta, position);
        for head in 0..n_heads {
            let off = head * head_dim + pair * 2;
            let (even, odd) = if let Some(src) = x {
                (src[off], src[off + 1])
            } else {
                (y[off], y[off + 1])
            };
            // Multiply, then add or subtract. Not `mul_add`.
            let even_cos = even * cos;
            let odd_sin = odd * sin;
            let even_sin = even * sin;
            let odd_cos = odd * cos;
            y[off] = even_cos - odd_sin;
            y[off + 1] = even_sin + odd_cos;
        }
    }
    Ok(())
}

fn inv_freq_at(pair: usize, head_dim: usize, theta: f32) -> f32 {
    let exponent = (2 * pair) as f32 / (head_dim as f32);
    1.0 / pow_f32(theta, exponent)
}

fn pair_cos_sin(pair: usize, head_dim: usize, theta: f32, position: f32) -> (f32, f32) {
    let angle = position * inv_freq_at(pair, head_dim, theta);
    (cos_f32(angle), sin_f32(angle))
}

// `f32::{sin, cos, powf}` live in `std`. This crate is `no_std` outside
// tests, so call libm: IEEE `powf`, and `sinf`/`cosf`, not a polynomial
// stand-in. Same pattern as `sqrtf` in the BF16 module.
#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
}

fn sin_f32(x: f32) -> f32 {
    // SAFETY: sinf is defined for every f32 bit pattern.
    unsafe { sinf(x) }
}

fn cos_f32(x: f32) -> f32 {
    // SAFETY: cosf is defined for every f32 bit pattern.
    unsafe { cosf(x) }
}

fn pow_f32(x: f32, y: f32) -> f32 {
    // SAFETY: powf is defined for every f32 pair, including the Qwen
    // domain (finite theta > 0, exponent in [0, 1)).
    unsafe { powf(x, y) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen25_theta_and_head_dim_are_exact() {
        assert_eq!(ROPE_THETA_QWEN25, 1e6);
        // 1e6 = 5^6 * 2^6, and 5^6 fits in the f32 mantissa.
        assert_eq!(ROPE_THETA_QWEN25.to_bits(), 0x4974_2400);
        assert_eq!(HEAD_DIM_QWEN25_1_5B, 128);
        assert_eq!(1536 / 12, HEAD_DIM_QWEN25_1_5B);
    }

    #[test]
    fn position_zero_is_identity() {
        // Finite lanes, theta = 1e6. angle = 0, cos = 1, sin = +0.
        let heads = [1.0f32, -2.0, 3.5, -4.25];
        let mut y = [0.0; 4];
        rope_f32(&mut y, &heads, 0, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(y, heads);

        let mut wide = [0.0f32; HEAD_DIM_QWEN25_1_5B];
        for (i, lane) in wide.iter_mut().enumerate() {
            *lane = if i % 2 == 0 {
                (i as f32) + 0.5
            } else {
                -((i as f32) + 0.25)
            };
        }
        let mut out = [0.0; HEAD_DIM_QWEN25_1_5B];
        rope_f32(&mut out, &wide, 0, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(out, wide);

        let mut inplace = wide;
        rope_f32_inplace(&mut inplace, 0, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(inplace, wide);
    }

    #[test]
    fn position_one_head_dim_two_is_rotation_by_one_radian() {
        // inv_freq[0] = 1 / theta^0 = 1. (1, 0) -> (cos(1), sin(1)).
        // Bits are libm sinf(1) / cosf(1), not a polynomial approximation.
        let mut y = [0.0; 2];
        rope_f32(&mut y, &[1.0, 0.0], 1, 2, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(y[0].to_bits(), 0x3f0a_5140);
        assert_eq!(y[1].to_bits(), 0x3f57_6aa4);

        // (3, 4) -> (3 cos - 4 sin, 3 sin + 4 cos), same angle.
        let mut y24 = [0.0; 2];
        rope_f32(&mut y24, &[3.0, 4.0], 1, 2, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(y24[0].to_bits(), 0xbfdf_5b68);
        assert_eq!(y24[1].to_bits(), 0x4095_f09e);

        let mut inplace = [3.0, 4.0];
        rope_f32_inplace(&mut inplace, 1, 2, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(inplace, y24);
    }

    #[test]
    fn second_pair_uses_its_own_frequency() {
        // head_dim = 4, position = 1, x = [0, 0, 1, 0].
        // Pair 0 stays zero. Pair 1 has exponent 2/4 = 1/2,
        // inv_freq = 1/1000, angle = 0.001.
        // sin(0.001) is one ulp under the angle, so this is not the
        // sin(x) ≈ x shortcut. cos is not 1.
        let mut table = [0.0; 2];
        rope_inv_freqs(&mut table, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(table[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(table[1].to_bits(), (1.0f32 / 1000.0).to_bits());
        assert_eq!(table[1].to_bits(), 0x3a83_126f);

        let mut y = [9.0; 4];
        rope_f32(&mut y, &[0.0, 0.0, 1.0, 0.0], 1, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(y[0].to_bits(), 0);
        assert_eq!(y[1].to_bits(), 0);
        assert_eq!(y[2].to_bits(), 0x3f7f_fff8);
        assert_eq!(y[3].to_bits(), 0x3a83_126e);
        assert_ne!(y[2].to_bits(), 0x3f0a_5140); // not cos(1)
        assert_ne!(y[3].to_bits(), table[1].to_bits()); // not the raw angle
    }

    #[test]
    fn position_three_head_dim_four_matches_libm_bits() {
        // x = [1, 2, 3, 4], theta = 1e6, position = 3.
        // Pair 0 angle = 3. Pair 1 angle = 3/1000.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0; 4];
        rope_f32(&mut y, &x, 3, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(
            y.map(f32::to_bits),
            [0xbfa2_d884, 0xbfeb_5fee, 0x403f_3b2c, 0x4080_4995]
        );

        let mut inplace = x;
        rope_f32_inplace(&mut inplace, 3, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(inplace, y);

        // Two heads, one position: the same table is applied to both.
        let batch = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mut both = [0.0; 8];
        rope_f32(&mut both, &batch, 3, 4, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(&both[..4], &y);
        assert_eq!(&both[4..], &y);
    }

    #[test]
    fn qwen_head_pairs_adjacent_lanes_not_halves() {
        // head_dim = 128, position = 1, x = e_0.
        // Even/odd puts (cos 1, sin 1) in lanes 0 and 1.
        // Half-split (NeoX / HF rotate_half) would put sin in lane 64.
        let mut x = [0.0f32; HEAD_DIM_QWEN25_1_5B];
        x[0] = 1.0;
        let mut y = [0.0; HEAD_DIM_QWEN25_1_5B];
        rope_f32(&mut y, &x, 1, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(y[0].to_bits(), 0x3f0a_5140);
        assert_eq!(y[1].to_bits(), 0x3f57_6aa4);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[64], 0.0);
        assert_eq!(y[127], 0.0);

        // Last pair at position 8: x[126] = 1, x[127] = -2.
        // inv_freq[63] = 1 / theta^(126/128).
        let mut table = [0.0; HEAD_DIM_QWEN25_1_5B / 2];
        rope_inv_freqs(&mut table, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(table[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(table[32].to_bits(), (1.0f32 / 1000.0).to_bits());
        assert_eq!(table[63].to_bits(), 0x35a6_8e4c);

        let mut last = [0.0f32; HEAD_DIM_QWEN25_1_5B];
        last[126] = 1.0;
        last[127] = -2.0;
        rope_f32_inplace(&mut last, 8, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25).unwrap();
        assert_eq!(last[126].to_bits(), 0x3f80_00a7);
        assert_eq!(last[127].to_bits(), 0xbfff_ffad);
        assert_eq!(last[64], 0.0);
    }

    #[test]
    fn odd_length_is_rejected_and_buffers_stay() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut y = x;
        // Length 5 is odd and not a multiple of head_dim 4.
        assert_eq!(
            rope_f32(&mut y, &x, 1, 4, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(y, x);

        let mut inplace = [9.0f32, 8.0, 7.0];
        // head_dim itself is odd.
        assert_eq!(
            rope_f32_inplace(&mut inplace, 1, 3, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(inplace, [9.0, 8.0, 7.0]);

        let mut six = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let six_before = six;
        assert_eq!(
            rope_f32_inplace(&mut six, 2, 3, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(six, six_before);

        let mut empty = [];
        assert_eq!(
            rope_f32_inplace(&mut empty, 0, 2, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(
            rope_f32_inplace(&mut [1.0, 2.0], 0, 0, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );

        let mut short = [7.0, 7.0];
        assert_eq!(
            rope_f32(&mut short, &x[..4], 0, 4, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(short, [7.0, 7.0]);

        // Not a multiple of an even head_dim, even when the length is even.
        let six_src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut six_copy = six_src;
        assert_eq!(
            rope_f32(&mut six_copy, &six_src, 1, 4, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(six_copy, six_src);

        let mut table = [3.0, 4.0, 5.0];
        assert_eq!(
            rope_inv_freqs(&mut table, 3, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(table, [3.0, 4.0, 5.0]);
        assert_eq!(
            rope_inv_freqs(&mut table, 4, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
        assert_eq!(table, [3.0, 4.0, 5.0]);
        assert_eq!(
            rope_inv_freqs(&mut [], 0, ROPE_THETA_QWEN25),
            Err(RopeError::Shape)
        );
    }
}
