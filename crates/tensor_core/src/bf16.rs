//! BF16 weight bytes, f32 accumulate.
//!
//! Phase1 main-track weights stay little-endian BF16 views. These ops decode
//! each element to f32 inside the loop and accumulate in f32. They do not
//! upcast a whole matrix first, and they do not call `load_f32_tensor`.
//!
//! Fingerprint intent (the run hash comes later):
//! - `quant_or_dtype` = [`QUANT_OR_DTYPE`] (`bf16`)
//! - `acc_path` = [`ACC_PATH`] (`bf16_w_fp32_acc`)

use core::fmt;

/// Fingerprint field `quant_or_dtype` for the Phase1 main track.
pub const QUANT_OR_DTYPE: &str = "bf16";

/// Fingerprint field `acc_path`.
///
/// Storage is BF16. The dot product decodes each weight to f32 and adds in f32.
pub const ACC_PATH: &str = "bf16_w_fp32_acc";

/// Shape or index failure from a BF16 op.
///
/// On `Err`, output slices passed by the caller are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16OpError {
    /// Rank, width, or byte length does not match the declared shape.
    Shape,
    /// `token_id` is outside `[0, vocab)`.
    TokenOutOfBounds,
    /// A weight row would be read past the byte buffer.
    RowOutOfBounds,
}

impl fmt::Display for Bf16OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bf16OpError::Shape => write!(f, "bf16 op shape does not match the buffers"),
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
}
