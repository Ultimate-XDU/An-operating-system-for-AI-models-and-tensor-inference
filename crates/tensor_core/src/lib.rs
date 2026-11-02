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
