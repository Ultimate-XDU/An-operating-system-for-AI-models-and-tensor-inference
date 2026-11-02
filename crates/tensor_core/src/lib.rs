//! TensorCore — Phase 1 CPU ops (GEMV, RMSNorm, Softmax stubs).
#![cfg_attr(not(test), no_std)]

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
