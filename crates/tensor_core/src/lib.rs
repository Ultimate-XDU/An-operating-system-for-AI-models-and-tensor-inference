//! TensorCore — Phase 1 CPU ops.
//!
//! [`gemv_f32`] is the f32 fixture path used by the cycle smoke.
//! [`gemv_bf16`] and [`embed_gather_bf16`] are the main-track numeric path:
//! little-endian BF16 weight bytes, decoded to f32 inside the op, accumulated
//! in f32. The whole matrix is not loaded as f32.
//!
//! Fingerprint intent (the run hash is later): `quant_or_dtype=bf16`
//! ([`QUANT_OR_DTYPE`]), `acc_path=bf16_w_fp32_acc` ([`ACC_PATH`]).
#![cfg_attr(not(test), no_std)]

mod bf16;
pub use bf16::{
    bf16_le_to_f32, embed_gather_bf16, gemv_bf16, Bf16OpError, ACC_PATH, QUANT_OR_DTYPE,
};

/// y[i] = sum_j W[i,j] * x[j]
pub fn gemv_f32(y: &mut [f32], w: &[f32], x: &[f32], m: usize, k: usize) {
    assert_eq!(y.len(), m);
    assert_eq!(x.len(), k);
    assert_eq!(w.len(), m * k);
    for i in 0..m {
        let mut acc = 0.0f32;
        let row = &w[i * k..(i + 1) * k];
        for j in 0..k {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::gemv_f32;

    #[test]
    fn gemv_f32_row_major_dots() {
        // W is 2x3, row-major. y = W x.
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0; 2];
        gemv_f32(&mut y, &w, &x, 2, 3);
        assert_eq!(y, [6.0, 15.0]);
    }
}
