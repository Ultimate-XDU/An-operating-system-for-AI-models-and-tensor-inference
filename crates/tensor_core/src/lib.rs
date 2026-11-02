//! TensorCore — Phase 1 CPU ops.
//!
//! [`gemv_f32`] is the f32 fixture path used by the cycle smoke.
//! [`gemv_bf16`], [`embed_gather_bf16`], and [`rmsnorm_bf16_weight`] are the
//! main-track numeric path: little-endian BF16 weight bytes, decoded to f32
//! inside the op. Activations stay f32. The whole matrix is not loaded as f32.
//! [`rmsnorm_f32`] is the f32-gamma reference for the same reduction.
//! [`rope_f32`] and [`rope_f32_inplace`] rotate Q or K heads in f32. There is
//! no weight byte view: the frequency table is `powf`, then libm `cosf`/`sinf`.
//! Qwen2.5 passes [`ROPE_THETA_QWEN25`] (`1e6`) and, for the 1.5B model,
//! [`HEAD_DIM_QWEN25_1_5B`] (`128`). Pairing is interleaved even/odd
//! (lane `2i` with lane `2i+1`); see the `rope` module.
//!
//! Fingerprint intent (the run hash is later): `quant_or_dtype=bf16`
//! ([`QUANT_OR_DTYPE`]), `acc_path=bf16_w_fp32_acc` ([`ACC_PATH`]).
//! RMSNorm's `Σx²` is the exact-sensitive reduction on that path: one thread,
//! left to right, f32. Qwen2.5 passes [`RMS_NORM_EPS_QWEN25`] (`1e-6`).
#![cfg_attr(not(test), no_std)]

mod bf16;
mod rope;
pub use bf16::{
    bf16_le_to_f32, embed_gather_bf16, gemv_bf16, rmsnorm_bf16_weight, rmsnorm_f32, Bf16OpError,
    ACC_PATH, QUANT_OR_DTYPE, RMS_NORM_EPS_QWEN25,
};
pub use rope::{
    rope_f32, rope_f32_inplace, rope_inv_freqs, RopeError, HEAD_DIM_QWEN25_1_5B, ROPE_THETA_QWEN25,
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
