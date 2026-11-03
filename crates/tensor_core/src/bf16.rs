//! BF16 weight bytes, f32 accumulate.
//!
//! Phase1 main-track weights stay little-endian BF16 views. These ops decode
//! each element to f32 inside the loop and accumulate in f32. They do not
//! upcast a whole matrix first, and they do not call `load_f32_tensor`.
//! Activations between ops stay f32, including RMSNorm input and output.
//!
//! Fingerprint intent (the run hash comes later):
//! - `quant_or_dtype` = [`QUANT_OR_DTYPE`] (`bf16`)
//! - `acc_path` = [`ACC_PATH`] (`bf16_w_fp32_acc`)
//!
//! RMSNorm does not open a second accumulate path. `Σx²` is a single-thread
//! left-to-right f32 sum (multiply, then add). Gamma is BF16 on the main
//! track ([`rmsnorm_bf16_weight`]) or f32 on the reference ([`rmsnorm_f32`]).

use core::fmt;

/// Fingerprint field `quant_or_dtype` for the Phase1 main track.
pub const QUANT_OR_DTYPE: &str = "bf16";

/// Fingerprint field `acc_path`.
///
/// Storage is BF16. GEMV decodes each weight to f32 and adds in f32.
/// RMSNorm keeps that contract: BF16 gamma, f32 math, f32 activations.
/// Its exact-sensitive step is the serial `Σx²` reduction, not a second dtype.
pub const ACC_PATH: &str = "bf16_w_fp32_acc";

/// Qwen2.5 `rms_norm_eps` (`config.json`).
///
/// [`rmsnorm_f32`] and [`rmsnorm_bf16_weight`] take `eps` as an argument so a
/// test can move it. The Qwen2.5 track passes this value (`1e-6`).
/// See `docs/qwen25-1.5b-instruct-layout.md`.
pub const RMS_NORM_EPS_QWEN25: f32 = 1e-6;

/// Shape or index failure from a tensor op.
///
/// On `Err`, output slices passed by the caller are left unchanged.
/// [`rmsnorm_f32`] uses [`Bf16OpError::Shape`] for a length mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16OpError {
    /// Rank, width, byte length, or RMSNorm vector length does not match.
    Shape,
    /// `token_id` is outside `[0, vocab)`.
    TokenOutOfBounds,
    /// A weight row would be read past the byte buffer.
    RowOutOfBounds,
}

impl fmt::Display for Bf16OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bf16OpError::Shape => write!(f, "shape does not match the buffers"),
            Bf16OpError::TokenOutOfBounds => write!(f, "token id is out of bounds"),
            Bf16OpError::RowOutOfBounds => write!(f, "bf16 row is out of bounds"),
        }
    }
}

/// IEEE BF16 as the top 16 bits of f32. `le` is little-endian.
#[inline]
pub fn bf16_le_to_f32(le: [u8; 2]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes(le)) << 16)
}

/// `y[i] = sum_j bf16_to_f32(W[i, j]) * x[j]`.
///
/// `w` is row-major little-endian BF16, `m` rows by `k` columns, `2 * m * k` bytes.
/// `x` and `y` are f32. The sum is f32, left to right across the row.
pub fn gemv_bf16(
    y: &mut [f32],
    w: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
) -> Result<(), Bf16OpError> {
    if y.len() != m || x.len() != k {
        return Err(Bf16OpError::Shape);
    }
    if k == 0 {
        if !w.is_empty() {
            return Err(Bf16OpError::Shape);
        }
        y.fill(0.0);
        return Ok(());
    }
    let row_bytes = k.checked_mul(2).ok_or(Bf16OpError::Shape)?;
    if w.len() / row_bytes < m {
        return Err(Bf16OpError::RowOutOfBounds);
    }
    let need = row_bytes.checked_mul(m).ok_or(Bf16OpError::Shape)?;
    if w.len() != need {
        return Err(Bf16OpError::Shape);
    }
    for i in 0..m {
        let row = &w[i * row_bytes..(i + 1) * row_bytes];
        let mut acc = 0.0f32;
        for j in 0..k {
            let mut le = [0u8; 2];
            le.copy_from_slice(&row[j * 2..j * 2 + 2]);
            acc += bf16_le_to_f32(le) * x[j];
        }
        y[i] = acc;
    }
    Ok(())
}

/// `y[i] += bf16_to_f32(bias[i])`.
///
/// `bias` is little-endian BF16, `2 * y.len()` bytes. Each element is promoted
/// inside this loop and then added in f32. The vector is not copied into a
/// `Vec<f32>` first, and the add is not fused into a GEMV dot: Qwen2.5 adds
/// the Q/K/V bias after `gemv_bf16` has finished the row. `o_proj` and the MLP
/// have no bias; callers simply do not call this for those projections.
///
/// A matching empty pair (`y` and `bias` both empty) is a no-op. Any other
/// length, including an odd byte length, is [`Bf16OpError::Shape`] and leaves
/// `y` unchanged.
pub fn add_bf16_bias(y: &mut [f32], bias: &[u8]) -> Result<(), Bf16OpError> {
    let nbytes = y.len().checked_mul(2).ok_or(Bf16OpError::Shape)?;
    if bias.len() != nbytes {
        return Err(Bf16OpError::Shape);
    }
    for i in 0..y.len() {
        let mut le = [0u8; 2];
        le.copy_from_slice(&bias[i * 2..i * 2 + 2]);
        y[i] += bf16_le_to_f32(le);
    }
    Ok(())
}

/// Copy embedding row `token_id` into `out` as f32.
///
/// `table` is a row-major little-endian BF16 matrix. `shape` is `[vocab, hidden]`.
/// `out.len()` must be `hidden`. Decode happens per element; the table is not
/// copied to an f32 allocation.
pub fn embed_gather_bf16(
    out: &mut [f32],
    table: &[u8],
    shape: &[usize],
    token_id: u32,
) -> Result<(), Bf16OpError> {
    let (vocab, hidden) = match *shape {
        [vocab, hidden] => (vocab, hidden),
        _ => return Err(Bf16OpError::Shape),
    };
    if out.len() != hidden {
        return Err(Bf16OpError::Shape);
    }
    let row = match usize::try_from(token_id) {
        Ok(row) if row < vocab => row,
        _ => return Err(Bf16OpError::TokenOutOfBounds),
    };
    let row_bytes = hidden.checked_mul(2).ok_or(Bf16OpError::Shape)?;
    let start = row.checked_mul(row_bytes).ok_or(Bf16OpError::Shape)?;
    let end = start.checked_add(row_bytes).ok_or(Bf16OpError::Shape)?;
    let Some(raw) = table.get(start..end) else {
        return Err(Bf16OpError::RowOutOfBounds);
    };
    let need = row_bytes.checked_mul(vocab).ok_or(Bf16OpError::Shape)?;
    if table.len() != need {
        return Err(Bf16OpError::Shape);
    }
    for j in 0..hidden {
        let mut le = [0u8; 2];
        le.copy_from_slice(&raw[j * 2..j * 2 + 2]);
        out[j] = bf16_le_to_f32(le);
    }
    Ok(())
}

/// `y[i] = weight[i] * (x[i] * inv_rms)` with an f32 gamma.
///
/// `inv_rms = 1 / sqrt(Σx² / n + eps)`, with IEEE-754 `sqrtf` (not an
/// approximate rsqrt). `Σx²` is a single-thread, left-to-right f32 sum of
/// `x[j] * x[j]` (multiply, then add; not `f32::mul_add`).
/// `n == x.len()`. Empty `x`, or any slice whose length is not `n`, returns
/// [`Bf16OpError::Shape`] and leaves `y` unchanged.
///
/// Qwen2.5 passes [`RMS_NORM_EPS_QWEN25`]. `eps = 0` and an all-zero `x` is
/// `0 * inf` and yields NaN; the Qwen default keeps that vector at zero.
pub fn rmsnorm_f32(y: &mut [f32], weight: &[f32], x: &[f32], eps: f32) -> Result<(), Bf16OpError> {
    let n = rms_len(y, x)?;
    if weight.len() != n {
        return Err(Bf16OpError::Shape);
    }
    let inv_rms = inv_rms_serial(x, eps);
    for i in 0..n {
        y[i] = weight[i] * (x[i] * inv_rms);
    }
    Ok(())
}

/// `y[i] = bf16_to_f32(weight[i]) * (x[i] * inv_rms)`.
///
/// `x` and `y` are f32. `weight` is little-endian BF16 gamma, `2 * n` bytes,
/// decoded per element inside the loop. The vector is not copied to an f32
/// allocation. The sum of squares matches [`rmsnorm_f32`]: one thread, left
/// to right, f32. Same fingerprint as GEMV ([`ACC_PATH`]).
///
/// On `Err`, `y` is unchanged. A length mismatch (including an odd byte
/// length, a short gamma, or `n == 0`) is [`Bf16OpError::Shape`].
pub fn rmsnorm_bf16_weight(
    y: &mut [f32],
    weight: &[u8],
    x: &[f32],
    eps: f32,
) -> Result<(), Bf16OpError> {
    let n = rms_len(y, x)?;
    let nbytes = n.checked_mul(2).ok_or(Bf16OpError::Shape)?;
    if weight.len() != nbytes {
        return Err(Bf16OpError::Shape);
    }
    let inv_rms = inv_rms_serial(x, eps);
    for i in 0..n {
        let mut le = [0u8; 2];
        le.copy_from_slice(&weight[i * 2..i * 2 + 2]);
        y[i] = bf16_le_to_f32(le) * (x[i] * inv_rms);
    }
    Ok(())
}

/// Reject an empty vector. `y` must be the same length as `x`.
fn rms_len(y: &[f32], x: &[f32]) -> Result<usize, Bf16OpError> {
    if x.is_empty() || y.len() != x.len() {
        return Err(Bf16OpError::Shape);
    }
    Ok(x.len())
}

/// `1 / sqrt(mean(x²) + eps)`, mean taken after a serial sum.
///
/// `sqrt` is IEEE-754 `sqrtf`, not an approximate rsqrt.
/// `x.len() as f32` is exact for lengths up to `2^24` (Qwen2.5 hidden size is 1536).
fn inv_rms_serial(x: &[f32], eps: f32) -> f32 {
    let mut sum_sq = 0.0f32;
    for &value in x {
        sum_sq += value * value;
    }
    let mean = sum_sq / (x.len() as f32);
    1.0 / sqrt_f32(mean + eps)
}

// `f32::sqrt` lives in `std`. This crate is `no_std` outside tests, so call
// libm `sqrtf`: correctly rounded IEEE-754 sqrt, not an approximate rsqrt.
#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn sqrtf(x: f32) -> f32;
}

fn sqrt_f32(x: f32) -> f32 {
    // SAFETY: sqrtf is defined for every f32 bit pattern, including NaN,
    // infinities, and negatives (those return NaN).
    unsafe { sqrtf(x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_f32;

    /// Exact BF16 value: low 16 bits of the f32 are already zero.
    fn le_bf16(value: f32) -> [u8; 2] {
        let bits = value.to_bits();
        assert_eq!(bits & 0xFFFF, 0, "test value {value} is not exact in bf16");
        u16::try_from(bits >> 16).unwrap().to_le_bytes()
    }

    fn pack(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&le_bf16(*value));
        }
        bytes
    }

    #[test]
    fn fingerprint_intent_names_bf16_f32_acc() {
        assert_eq!(QUANT_OR_DTYPE, "bf16");
        assert_eq!(ACC_PATH, "bf16_w_fp32_acc");
        // RMSNorm stays on that path. Qwen2.5's eps is the documented default.
        assert_eq!(RMS_NORM_EPS_QWEN25, 1e-6f32);
        assert_eq!(RMS_NORM_EPS_QWEN25.to_bits(), 0x3586_37BD);
    }

    #[test]
    fn bf16_le_matches_top_half_of_f32() {
        assert_eq!(bf16_le_to_f32(le_bf16(1.0)), 1.0);
        assert_eq!(bf16_le_to_f32([0x80, 0x3F]), 1.0);
        assert_eq!(bf16_le_to_f32([0x00, 0xC0]), -2.0);
        assert_eq!(bf16_le_to_f32([0x3F, 0x80]), f32::from_bits(0x803F_0000));
        assert_ne!(bf16_le_to_f32([0x3F, 0x80]), 1.0);
    }

    #[test]
    fn add_bf16_bias_adds_after_promotion() {
        // 10 and 20 are already f32. 0.5 and 1.0 are exact BF16. The add is
        // one f32 `+=` per lane, after the whole bias element is promoted.
        let mut y = [10.0f32, 20.0];
        add_bf16_bias(&mut y, &pack(&[0.5, 1.0])).unwrap();
        assert_eq!(y, [10.5, 21.0]);

        let mut kept = [7.0f32, 8.0, 9.0];
        let before = kept;
        assert_eq!(
            add_bf16_bias(&mut kept, &pack(&[0.5, 1.0])),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(kept, before);
        // Odd byte length: not a BF16 vector. `y` stays put.
        assert_eq!(
            add_bf16_bias(&mut kept[..1], &[0x00]),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(kept, before);

        let mut empty = [];
        assert_eq!(add_bf16_bias(&mut empty, &[]), Ok(()));
    }

    #[test]
    fn gemv_bf16_matches_hand_values_and_f32_path() {
        // W is 2x3, row-major. Every entry is exact in BF16.
        let w_f32 = [1.0f32, 2.0, 0.5, -1.0, 4.0, 1.5];
        let w = pack(&w_f32);
        let x = [1.0f32, -2.0, 0.5];
        let mut y_bf16 = [0.0; 2];
        let mut y_f32 = [0.0; 2];
        gemv_bf16(&mut y_bf16, &w, &x, 2, 3).unwrap();
        gemv_f32(&mut y_f32, &w_f32, &x, 2, 3);
        // row0 = 1 - 4 + 0.25 = -2.75; row1 = -1 - 8 + 0.75 = -8.25
        assert_eq!(y_bf16, [-2.75, -8.25]);
        assert_eq!(y_bf16, y_f32);
    }

    #[test]
    fn gemv_bf16_reads_little_endian_weights() {
        let mut y = [0.0];
        gemv_bf16(&mut y, &[0x80, 0x3F], &[2.0], 1, 1).unwrap();
        assert_eq!(y, [2.0]);

        gemv_bf16(&mut y, &[0x3F, 0x80], &[2.0], 1, 1).unwrap();
        assert_eq!(y[0], f32::from_bits(0x803F_0000) * 2.0);
        assert_ne!(y[0], 2.0);
    }

    #[test]
    fn gemv_bf16_rejects_short_rows_and_bad_shape() {
        // One row of k=2. Asking for 2 rows reads past the buffer.
        let w = [0x80u8, 0x3F, 0x00, 0x40];
        let x = [1.0f32, 1.0];
        let mut y = [7.0, 8.0];
        assert_eq!(
            gemv_bf16(&mut y, &w, &x, 2, 2),
            Err(Bf16OpError::RowOutOfBounds)
        );
        assert_eq!(y, [7.0, 8.0]);

        let mut y_short = [7.0];
        assert_eq!(
            gemv_bf16(&mut y_short, &w, &x, 2, 2),
            Err(Bf16OpError::Shape)
        );
        let mut y_ok = [0.0, 0.0];
        let x_short = [1.0f32];
        assert_eq!(
            gemv_bf16(&mut y_ok, &w, &x_short, 2, 2),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(y_ok, [0.0, 0.0]);

        // Trailing byte: both declared rows exist, length still disagrees.
        let mut y_one = [0.0, 0.0];
        let x_one = [1.0f32, 1.0];
        let wide = [0x80u8, 0x3F, 0x00, 0x40, 0x00];
        assert_eq!(
            gemv_bf16(&mut y_one, &wide, &x_one, 2, 1),
            Err(Bf16OpError::Shape)
        );
    }

    #[test]
    fn embed_gather_bf16_reads_token_row() {
        // [vocab=3, hidden=2]
        let table = pack(&[1.0, 2.0, 3.0, -0.5, 0.5, 4.0]);
        let shape = [3usize, 2];
        let mut out = [0.0; 2];
        embed_gather_bf16(&mut out, &table, &shape, 1).unwrap();
        assert_eq!(out, [3.0, -0.5]);
        embed_gather_bf16(&mut out, &table, &shape, 2).unwrap();
        assert_eq!(out, [0.5, 4.0]);
        embed_gather_bf16(&mut out, &table, &shape, 0).unwrap();
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn embed_gather_bf16_reads_little_endian() {
        let mut out = [0.0];
        embed_gather_bf16(&mut out, &[0x80, 0x3F], &[1, 1], 0).unwrap();
        assert_eq!(out, [1.0]);
        embed_gather_bf16(&mut out, &[0x3F, 0x80], &[1, 1], 0).unwrap();
        assert_eq!(out[0], f32::from_bits(0x803F_0000));
        assert_ne!(out[0], 1.0);
    }

    #[test]
    fn embed_gather_bf16_rejects_oob_token_and_row() {
        let table = pack(&[1.0, 2.0]);
        let shape = [2usize, 1];
        let mut out = [9.0];
        assert_eq!(
            embed_gather_bf16(&mut out, &table, &shape, 2),
            Err(Bf16OpError::TokenOutOfBounds)
        );
        assert_eq!(
            embed_gather_bf16(&mut out, &table, &shape, u32::MAX),
            Err(Bf16OpError::TokenOutOfBounds)
        );
        assert_eq!(out, [9.0]);

        // Declared vocab covers token 1, but the bytes stop after row 0.
        let short = pack(&[1.0]);
        assert_eq!(
            embed_gather_bf16(&mut out, &short, &shape, 1),
            Err(Bf16OpError::RowOutOfBounds)
        );
        assert_eq!(out, [9.0]);
        // Token 0's row is present, but the table length is still wrong.
        assert_eq!(
            embed_gather_bf16(&mut out, &short, &shape, 0),
            Err(Bf16OpError::Shape)
        );

        assert_eq!(
            embed_gather_bf16(&mut out, &table, &[2], 0),
            Err(Bf16OpError::Shape)
        );
        let mut wide = [0.0, 0.0];
        assert_eq!(
            embed_gather_bf16(&mut wide, &table, &shape, 0),
            Err(Bf16OpError::Shape)
        );
    }

    #[test]
    fn rmsnorm_matches_hand_values() {
        // n = 4, x = [8, 0, 0, 0]. Σx² = 64, mean = 16, inv_rms = 1/4.
        // y = w * (x * 1/4) = [3, 0, 0, 0] for w = [1.5, 4, -2, 0.5].
        let x = [8.0f32, 0.0, 0.0, 0.0];
        let w = [1.5f32, 4.0, -2.0, 0.5];
        let mut y = [0.0; 4];
        rmsnorm_f32(&mut y, &w, &x, 0.0).unwrap();
        assert_eq!(y, [3.0, 0.0, 0.0, 0.0]);

        let mut y_bf16 = [0.0; 4];
        rmsnorm_bf16_weight(&mut y_bf16, &pack(&w), &x, 0.0).unwrap();
        assert_eq!(y_bf16, y);

        // Constant magnitude: |x| = 2, n = 2, mean = 4, inv_rms = 1/2.
        // Sign of x is kept. y = [3 * 1, 4 * -1].
        let x_sign = [2.0f32, -2.0];
        let w_sign = [3.0f32, 4.0];
        let mut y_sign = [0.0; 2];
        rmsnorm_f32(&mut y_sign, &w_sign, &x_sign, 0.0).unwrap();
        assert_eq!(y_sign, [3.0, -4.0]);
        let mut y_sign_bf16 = [0.0; 2];
        rmsnorm_bf16_weight(&mut y_sign_bf16, &pack(&w_sign), &x_sign, 0.0).unwrap();
        assert_eq!(y_sign_bf16, y_sign);
    }

    #[test]
    fn rmsnorm_serial_sum_squares_is_left_to_right() {
        // squares = [2^24, 1, 1, 1]. Left to right the 1s fall below the ulp
        // of 2^24, so Σx² stays 16777216. Ones-first would be 16777220 and
        // would not yield y[0] == 2. n = 4, mean = 2^22, inv_rms = 2^-11.
        let big = 4096.0f32; // 2^12; square is 2^24
        let x = [big, 1.0, 1.0, 1.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let mut y = [0.0; 4];
        rmsnorm_f32(&mut y, &w, &x, 0.0).unwrap();
        let small = f32::from_bits(0x3a00_0000); // 2^-11
        assert_eq!(y, [2.0, small, small, small]);

        let mut y_bf16 = [0.0; 4];
        rmsnorm_bf16_weight(&mut y_bf16, &pack(&w), &x, 0.0).unwrap();
        assert_eq!(y_bf16, y);
    }

    #[test]
    fn rmsnorm_eps_qwen25_scales_and_zero_stays_zero() {
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let packed = pack(&w);
        let mut y_zero_eps = [0.0; 4];
        let mut y_eps = [0.0; 4];
        rmsnorm_f32(&mut y_zero_eps, &w, &x, 0.0).unwrap();
        rmsnorm_bf16_weight(&mut y_eps, &packed, &x, RMS_NORM_EPS_QWEN25).unwrap();
        assert_eq!(y_zero_eps, [1.0, 1.0, 1.0, 1.0]);
        // 1/sqrt(1 + 1e-6) in f32. Locked so a mean/eps reorder fails.
        let scaled = f32::from_bits(0x3f7f_fff8);
        assert_eq!(y_eps, [scaled, scaled, scaled, scaled]);
        assert!(scaled < 1.0);

        let zeros = [0.0f32, 0.0];
        let w0 = [3.0f32, -1.0];
        let mut y_default = [7.0; 2];
        rmsnorm_bf16_weight(&mut y_default, &pack(&w0), &zeros, RMS_NORM_EPS_QWEN25).unwrap();
        assert_eq!(y_default, [0.0, 0.0]);

        let mut y_nan = [7.0; 2];
        rmsnorm_f32(&mut y_nan, &w0, &zeros, 0.0).unwrap();
        assert!(y_nan[0].is_nan() && y_nan[1].is_nan());
    }

    #[test]
    fn rmsnorm_bf16_weight_reads_little_endian() {
        // n = 1, x = 1, eps = 0 → y = gamma. LE [00, 40] is f32 2.0.
        let mut y = [0.0];
        rmsnorm_bf16_weight(&mut y, &[0x00, 0x40], &[1.0], 0.0).unwrap();
        assert_eq!(y, [2.0]);

        rmsnorm_bf16_weight(&mut y, &[0x40, 0x00], &[1.0], 0.0).unwrap();
        assert_eq!(y[0], f32::from_bits(0x0040_0000));
        assert_ne!(y[0], 2.0);
    }

    #[test]
    fn rmsnorm_rejects_length_mismatch() {
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 1.0];
        let packed = pack(&w);
        let mut y = [7.0, 8.0];

        assert_eq!(
            rmsnorm_f32(&mut y, &w, &x[..1], 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(y, [7.0, 8.0]);

        let mut y_short = [7.0];
        assert_eq!(
            rmsnorm_f32(&mut y_short, &w, &x, 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(y_short, [7.0]);
        assert_eq!(
            rmsnorm_f32(&mut y, &w[..1], &x, 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(y, [7.0, 8.0]);

        assert_eq!(
            rmsnorm_bf16_weight(&mut y, &packed[..2], &x, 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(
            rmsnorm_bf16_weight(&mut y, &[0x00, 0x40, 0x00], &x, 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(
            rmsnorm_bf16_weight(&mut y_short, &packed, &x, 0.0),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(y, [7.0, 8.0]);
        assert_eq!(y_short, [7.0]);

        let mut empty = [];
        assert_eq!(
            rmsnorm_f32(&mut empty, &[], &[], RMS_NORM_EPS_QWEN25),
            Err(Bf16OpError::Shape)
        );
        assert_eq!(
            rmsnorm_bf16_weight(&mut empty, &[], &[], RMS_NORM_EPS_QWEN25),
            Err(Bf16OpError::Shape)
        );
    }
}
