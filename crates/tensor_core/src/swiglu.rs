//! SwiGLU MLP for one token, HF Qwen2 / llama.cpp form.
//!
//! Activations stay f32. Gate, up, and down weights are little-endian BF16
//! views on the main track ([`swiglu_bf16`]) or f32 on the fixture path
//! ([`swiglu_f32`]). Each projection is [`crate::gemv_bf16`] or
//! [`crate::gemv_f32`]: the row is decoded inside that GEMV, the dot is f32,
//! left to right, multiply then add. This module does not reimplement the
//! dot and does not upcast a whole matrix first.
//!
//! Qwen2.5-1.5B passes [`HIDDEN_SIZE_QWEN25_1_5B`] (1536) and
//! [`INTERMEDIATE_SIZE_QWEN25_1_5B`] (8960). The counts live on
//! [`SwigluShape`] so a test can shrink them. The op does not hard-code the
//! 1.5B width. `model_core::qwen25_layout` already pins hidden size 1536;
//! this crate does not depend on `model_core` (that crate depends on this
//! one). The MLP width is not in that layout module, so both counts live
//! here, next to the op, the same way head dim lives in `rope`.
//!
//! # Formula
//!
//! No bias on gate, up, or down. Qwen2.5 stores none on `mlp.*` (QKV bias
//! is a different projection). For a hidden vector `x`:
//!
//! ```text
//! gate = gemv(gate_proj, x)          // [intermediate, hidden]
//! up   = gemv(up_proj, x)            // [intermediate, hidden]
//! h[i] = silu(gate[i]) * up[i]       // left to right, one multiply
//! y    = gemv(down_proj, h)          // [hidden, intermediate]
//! ```
//!
//! [`silu_f32`] is `z / (1 + expf(-z))`: libm `expf`, then one add, then
//! one division. That is llama.cpp `ggml_silu_f32` and the real-arithmetic
//! identity `z * sigmoid(z)`. It is not `z * (1 / (1 + expf(-z)))`. Those
//! differ by an ulp (same split as GQA's `e / sum` versus `e * (1/sum)`).
//! The product is formed after SiLU returns. The op does not call `mul_add`.
//!
//! # Weights
//!
//! Row-major `[out, in]`, the SafeTensors / PyTorch linear layout:
//!
//! ```text
//! mlp.gate_proj.weight    BF16    [intermediate, hidden]
//! mlp.up_proj.weight      BF16    [intermediate, hidden]
//! mlp.down_proj.weight    BF16    [hidden, intermediate]
//! ```
//!
//! BF16 storage is 2 bytes per element, little-endian. For 1.5B that is
//! `[8960, 1536]`, `[8960, 1536]`, and `[1536, 8960]`.
//!
//! # Scratch
//!
//! `gate`, `up`, and `h` each have length `intermediate`. On success they
//! hold the three vectors in the formula (`h` is not written back over
//! `gate` or `up`). `y` has length `hidden`. On [`SwigluError::Shape`],
//! `y`, `gate`, `up`, and `h` are left unchanged. Shape is checked before
//! any store. [`crate::gemv_bf16`] also returns before it writes a row.
//!
//! One call is one token's MLP. The transformer block is not this function.
//!
//! Fingerprint intent (the run hash is later): `quant_or_dtype=bf16`,
//! `acc_path=bf16_w_fp32_acc`. Exact-sensitive steps are the three serial
//! GEMVs, `expf` inside SiLU, the division form of SiLU, and the
//! left-to-right elementwise multiply.

use core::fmt;

/// Qwen2.5-1.5B `hidden_size` (`config.json`).
///
/// Same number as `model_core::qwen25_layout::HIDDEN_SIZE`. Gate and up
/// take a vector of this length; down writes one. See
/// `docs/qwen25-1.5b-instruct-layout.md`.
pub const HIDDEN_SIZE_QWEN25_1_5B: usize = 1536;

/// Qwen2.5-1.5B `intermediate_size` (`config.json`).
///
/// `mlp.gate_proj` and `mlp.up_proj` each write a vector of this length.
/// `mlp.down_proj` reads it. See `docs/qwen25-1.5b-instruct-layout.md`.
pub const INTERMEDIATE_SIZE_QWEN25_1_5B: usize = 8960;

/// Widths for [`swiglu_bf16`] and [`swiglu_f32`].
///
/// Both fields must be non-zero. Named so a caller cannot swap
/// `hidden` with `intermediate` by position. Qwen2.5-1.5B passes
/// [`HIDDEN_SIZE_QWEN25_1_5B`] and [`INTERMEDIATE_SIZE_QWEN25_1_5B`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwigluShape {
    pub hidden: usize,
    pub intermediate: usize,
}

/// Shape failure from a SwiGLU op.
///
/// On `Err`, `y` and the gate / up / hidden scratch are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluError {
    /// A dimension is zero, a product overflows `usize`, or a buffer
    /// length does not match the layout in the module docs.
    Shape,
}

impl fmt::Display for SwigluError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwigluError::Shape => write!(f, "swiglu shape does not match the buffers"),
        }
    }
}

/// SiLU(z) = z / (1 + exp(-z)).
///
/// libm `expf`, then one addition, then one division. Not
/// `z * (1 / (1 + expf(-z)))`.
#[inline]
pub fn silu_f32(z: f32) -> f32 {
    z / (1.0 + exp_f32(-z))
}

/// `h[i] = silu(gate[i]) * up[i]`, left to right.
///
/// The product uses the finished SiLU value. This is not `mul_add`.
/// All three slices must have the same non-zero length. On
/// [`SwigluError::Shape`], `h` is unchanged.
///
/// [`swiglu_bf16`] runs this between the up projection and the down
/// projection. Qwen2 applies SiLU to the gate projection only.
pub fn swiglu_mul_f32(h: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), SwigluError> {
    if gate.is_empty() || up.len() != gate.len() || h.len() != gate.len() {
        return Err(SwigluError::Shape);
    }
    mul_ltr(h, gate, up);
    Ok(())
}

/// SwiGLU on f32 weights. Same graph as [`swiglu_bf16`].
///
/// `gate_w` and `up_w` are row-major `[intermediate, hidden]`. `down_w` is
/// row-major `[hidden, intermediate]`. No bias is added after any GEMV.
/// `gate`, `up`, and `h` are scratch of length `intermediate`. On success
/// they hold the formula's three vectors. On [`SwigluError::Shape`], `y`
/// and the scratch are unchanged.
///
/// The argument list is the three projections plus the three scratch
/// vectors, in formula order. Gate and up share a shape, so the names stay
/// positional, same as [`crate::gemv_f32`].
#[allow(clippy::too_many_arguments)]
pub fn swiglu_f32(
    y: &mut [f32],
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    gate: &mut [f32],
    up: &mut [f32],
    h: &mut [f32],
    shape: SwigluShape,
) -> Result<(), SwigluError> {
    acts_match(y, x, gate, up, h, shape)?;
    let SwigluShape {
        hidden,
        intermediate,
    } = shape;
    let gate_len = f32_matrix_len(intermediate, hidden)?;
    let down_len = f32_matrix_len(hidden, intermediate)?;
    if gate_w.len() != gate_len || up_w.len() != gate_len || down_w.len() != down_len {
        return Err(SwigluError::Shape);
    }
    crate::gemv_f32(gate, gate_w, x, intermediate, hidden);
    crate::gemv_f32(up, up_w, x, intermediate, hidden);
    mul_ltr(h, gate, up);
    crate::gemv_f32(y, down_w, h, hidden, intermediate);
    Ok(())
}

/// SwiGLU on little-endian BF16 weight views.
///
/// `gate_w` is `mlp.gate_proj.weight`, `up_w` is `mlp.up_proj.weight`,
/// `down_w` is `mlp.down_proj.weight`. Shapes, the missing bias, SiLU, and
/// scratch: see the module docs. Qwen2.5-1.5B passes
/// [`HIDDEN_SIZE_QWEN25_1_5B`] and [`INTERMEDIATE_SIZE_QWEN25_1_5B`].
///
/// Projections call [`crate::gemv_bf16`]. Activations stay f32. On
/// [`SwigluError::Shape`], `y`, `gate`, `up`, and `h` are unchanged.
/// Argument order matches [`swiglu_f32`]: weights, then scratch, then shape.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_bf16(
    y: &mut [f32],
    x: &[f32],
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    gate: &mut [f32],
    up: &mut [f32],
    h: &mut [f32],
    shape: SwigluShape,
) -> Result<(), SwigluError> {
    acts_match(y, x, gate, up, h, shape)?;
    let SwigluShape {
        hidden,
        intermediate,
    } = shape;
    let gate_bytes = bf16_matrix_bytes(intermediate, hidden)?;
    let down_bytes = bf16_matrix_bytes(hidden, intermediate)?;
    if gate_w.len() != gate_bytes || up_w.len() != gate_bytes || down_w.len() != down_bytes {
        return Err(SwigluError::Shape);
    }
    // Lengths match what `gemv_bf16` requires, and `hidden` / `intermediate`
    // are non-zero, so each call returns `Ok` without taking the `k == 0`
    // path. Errors inside that op are returned before it stores a row.
    crate::gemv_bf16(gate, gate_w, x, intermediate, hidden).map_err(|_| SwigluError::Shape)?;
    crate::gemv_bf16(up, up_w, x, intermediate, hidden).map_err(|_| SwigluError::Shape)?;
    mul_ltr(h, gate, up);
    crate::gemv_bf16(y, down_w, h, hidden, intermediate).map_err(|_| SwigluError::Shape)?;
    Ok(())
}

fn acts_match(
    y: &[f32],
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    h: &[f32],
    shape: SwigluShape,
) -> Result<(), SwigluError> {
    let SwigluShape {
        hidden,
        intermediate,
    } = shape;
    if hidden == 0 || intermediate == 0 {
        return Err(SwigluError::Shape);
    }
    if y.len() != hidden || x.len() != hidden {
        return Err(SwigluError::Shape);
    }
    if gate.len() != intermediate || up.len() != intermediate || h.len() != intermediate {
        return Err(SwigluError::Shape);
    }
    Ok(())
}

/// Element count of a row-major `[rows, cols]` f32 matrix.
fn f32_matrix_len(rows: usize, cols: usize) -> Result<usize, SwigluError> {
    rows.checked_mul(cols).ok_or(SwigluError::Shape)
}

/// Byte length of a row-major `[rows, cols]` little-endian BF16 matrix.
///
/// Same split [`crate::gemv_bf16`] uses: `row_bytes = cols * 2`, then
/// `row_bytes * rows`.
fn bf16_matrix_bytes(rows: usize, cols: usize) -> Result<usize, SwigluError> {
    let row_bytes = cols.checked_mul(2).ok_or(SwigluError::Shape)?;
    row_bytes.checked_mul(rows).ok_or(SwigluError::Shape)
}

/// Left to right. SiLU finishes, then one multiply. Not `mul_add`.
fn mul_ltr(h: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(h.len(), gate.len());
    debug_assert_eq!(h.len(), up.len());
    for i in 0..gate.len() {
        let activated = silu_f32(gate[i]);
        h[i] = activated * up[i];
    }
}

// `f32::exp` lives in `std`. This crate is `no_std` outside tests, so call
// libm `expf`. Same pattern as `expf` in GQA and `sqrtf` in the BF16 module.
#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn expf(x: f32) -> f32;
}

fn exp_f32(x: f32) -> f32 {
    // SAFETY: expf is defined for every f32 bit pattern.
    unsafe { expf(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(hidden: usize, intermediate: usize) -> SwigluShape {
        SwigluShape {
            hidden,
            intermediate,
        }
    }

    /// Exact BF16 value: low 16 bits of the f32 are already zero.
    fn le_bf16(value: f32) -> [u8; 2] {
        let bits = value.to_bits();
        assert_eq!(bits & 0xffff, 0, "test value {value} is not exact in bf16");
        u16::try_from(bits >> 16).unwrap().to_le_bytes()
    }

    fn pack(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&le_bf16(*value));
        }
        bytes
    }

    /// Independent reference. Dots are left to right. SiLU is the division
    /// form, not a call into [`silu_f32`], so a reciprocal-multiply rewrite
    /// of that function fails this check.
    fn reference(
        x: &[f32],
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
        hidden: usize,
        intermediate: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut gate = vec![0.0; intermediate];
        let mut up = vec![0.0; intermediate];
        for i in 0..intermediate {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            let grow = &gate_w[i * hidden..(i + 1) * hidden];
            let urow = &up_w[i * hidden..(i + 1) * hidden];
            for j in 0..hidden {
                g += grow[j] * x[j];
                u += urow[j] * x[j];
            }
            gate[i] = g;
            up[i] = u;
        }
        let mut h = vec![0.0; intermediate];
        for i in 0..intermediate {
            let activated = gate[i] / (1.0 + exp_f32(-gate[i]));
            h[i] = activated * up[i];
        }
        let mut y = vec![0.0; hidden];
        for i in 0..hidden {
            let mut acc = 0.0f32;
            let row = &down_w[i * intermediate..(i + 1) * intermediate];
            for j in 0..intermediate {
                acc += row[j] * h[j];
            }
            y[i] = acc;
        }
        (gate, up, h, y)
    }

    #[test]
    fn qwen25_mlp_dims_match_layout_doc() {
        assert_eq!(HIDDEN_SIZE_QWEN25_1_5B, 1536);
        assert_eq!(INTERMEDIATE_SIZE_QWEN25_1_5B, 8960);
        // hidden = num_attention_heads * head_dim, pinned by GQA and RoPE.
        assert_eq!(
            crate::NUM_ATTENTION_HEADS_QWEN25_1_5B * crate::HEAD_DIM_QWEN25_1_5B,
            HIDDEN_SIZE_QWEN25_1_5B
        );
        // gate_proj / up_proj byte length, and down_proj (same element count).
        assert_eq!(
            INTERMEDIATE_SIZE_QWEN25_1_5B * HIDDEN_SIZE_QWEN25_1_5B * 2,
            27_525_120
        );
    }

    #[test]
    fn silu_and_elementwise_match_hand_values() {
        // 0 / (1 + expf(0)) = 0. -0 / 2 keeps the sign.
        assert_eq!(silu_f32(0.0).to_bits(), 0);
        assert_eq!(silu_f32(-0.0).to_bits(), 0x8000_0000);

        // 1 / (1 + expf(-1)). Same bits as sigmoid(1).
        assert_eq!(silu_f32(1.0).to_bits(), 0x3f3b_26a8);
        // -1 / (1 + expf(1)).
        assert_eq!(silu_f32(-1.0).to_bits(), 0xbe89_b2b1);
        // 2 / (1 + expf(-2)).
        assert_eq!(silu_f32(2.0).to_bits(), 0x3fe1_7bea);

        // 3 / (1 + expf(-3)). Reciprocal-multiply form is the other ulp.
        assert_eq!(silu_f32(3.0).to_bits(), 0x4036_e4ed);
        assert_ne!(silu_f32(3.0).to_bits(), 0x4036_e4ec);
        // -1.5 / (1 + expf(1.5)). Reciprocal-multiply form is 0xbe8c_1a52.
        assert_eq!(silu_f32(-1.5).to_bits(), 0xbe8c_1a51);
        assert_ne!(silu_f32(-1.5).to_bits(), 0xbe8c_1a52);

        // h = [silu(0)*4, silu(1)*2, silu(-1)*(-3), silu(3)*0.5]
        //    = [0, silu(1)*2, -3*silu(-1), silu(3)/2]
        let gate = [0.0f32, 1.0, -1.0, 3.0];
        let up = [4.0f32, 2.0, -3.0, 0.5];
        let mut h = [7.0; 4];
        swiglu_mul_f32(&mut h, &gate, &up).unwrap();
        assert_eq!(
            h.map(f32::to_bits),
            [0x0000_0000, 0x3fbb_26a8, 0x3f4e_8c0a, 0x3fb6_e4ed]
        );

        let mut short = [9.0, 8.0];
        let short_before = short;
        assert_eq!(
            swiglu_mul_f32(&mut short, &gate, &up),
            Err(SwigluError::Shape)
        );
        assert_eq!(short, short_before);
        assert_eq!(
            swiglu_mul_f32(&mut h, &gate[..3], &up),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            h.map(f32::to_bits),
            [0x0000_0000, 0x3fbb_26a8, 0x3f4e_8c0a, 0x3fb6_e4ed]
        );
        let mut empty = [];
        assert_eq!(
            swiglu_mul_f32(&mut empty, &[], &[]),
            Err(SwigluError::Shape)
        );
    }

    #[test]
    fn swiglu_matches_hand_values_and_reference_loop() {
        // hidden = 2, intermediate = 2, x = [1, 0].
        // gate = [1, 0], up = [2, 0], h = [silu(1)*2, 0].
        // The 3 in up row 0 is multiplied by x[1] = 0. Using x[0] for every
        // column would make up[0] = 5, not 2.
        // down row0 = [0.5, 3] → 0.5*(silu(1)*2) + 3*0 = silu(1).
        // down row1 = [1, 5]   → silu(1)*2.
        // A bias on gate would move the first lane off 1 and change both
        // output bits. SiLU on `up` instead of `gate` would store silu(2)
        // (0x3fe1_7bea) in h[0], not silu(1)*2.
        let x = [1.0f32, 0.0];
        let gate_w = [1.0f32, 0.0, 0.0, 1.0];
        let up_w = [2.0f32, 3.0, 0.0, 5.0];
        let down_w = [0.5f32, 3.0, 1.0, 5.0];
        let mut y = [7.0; 2];
        let mut gate = [7.0; 2];
        let mut up = [7.0; 2];
        let mut h = [7.0; 2];
        swiglu_f32(
            &mut y,
            &x,
            &gate_w,
            &up_w,
            &down_w,
            &mut gate,
            &mut up,
            &mut h,
            shape(2, 2),
        )
        .unwrap();
        assert_eq!(gate, [1.0, 0.0]);
        assert_eq!(up, [2.0, 0.0]);
        assert_eq!(h[0].to_bits(), 0x3fbb_26a8);
        assert_eq!(h[1].to_bits(), 0);
        assert_eq!(y[0].to_bits(), 0x3f3b_26a8);
        assert_eq!(y[1].to_bits(), 0x3fbb_26a8);

        let (g_ref, u_ref, h_ref, y_ref) = reference(&x, &gate_w, &up_w, &down_w, 2, 2);
        assert_eq!(gate, g_ref.as_slice());
        assert_eq!(up, u_ref.as_slice());
        assert_eq!(h, h_ref.as_slice());
        assert_eq!(y, y_ref.as_slice());

        let gate_b = pack(&gate_w);
        let up_b = pack(&up_w);
        let down_b = pack(&down_w);
        let mut y_b = [8.0; 2];
        let mut gate_s = [8.0; 2];
        let mut up_s = [8.0; 2];
        let mut h_s = [8.0; 2];
        swiglu_bf16(
            &mut y_b,
            &x,
            &gate_b,
            &up_b,
            &down_b,
            &mut gate_s,
            &mut up_s,
            &mut h_s,
            shape(2, 2),
        )
        .unwrap();
        assert_eq!(y_b, y);
        assert_eq!(gate_s, gate);
        assert_eq!(up_s, up);
        assert_eq!(h_s, h);

        // Wider fixture: every lane is a real left-to-right dot, then SiLU.
        // Weights are exact in BF16, so the byte view matches the f32 path
        // bit for bit. Swapped gate/up weights would not.
        let x3 = [1.0f32, -2.0, 0.5];
        let gate3 = [1.0f32, 0.5, -1.0, 2.0, 1.0, 0.5];
        let up3 = [0.5f32, 2.0, 1.0, -1.0, 0.5, 4.0];
        let down3 = [1.0f32, 0.5, -2.0, 1.0, 0.5, -0.5];
        let (g3, u3, h3, y3) = reference(&x3, &gate3, &up3, &down3, 3, 2);
        let mut y_f = [0.0; 3];
        let mut g_f = [0.0; 2];
        let mut u_f = [0.0; 2];
        let mut h_f = [0.0; 2];
        swiglu_f32(
            &mut y_f,
            &x3,
            &gate3,
            &up3,
            &down3,
            &mut g_f,
            &mut u_f,
            &mut h_f,
            shape(3, 2),
        )
        .unwrap();
        assert_eq!(g_f, g3.as_slice());
        assert_eq!(u_f, u3.as_slice());
        assert_eq!(h_f, h3.as_slice());
        assert_eq!(y_f, y3.as_slice());

        let mut y_bf = [0.0; 3];
        let mut g_bf = [0.0; 2];
        let mut u_bf = [0.0; 2];
        let mut h_bf = [0.0; 2];
        swiglu_bf16(
            &mut y_bf,
            &x3,
            &pack(&gate3),
            &pack(&up3),
            &pack(&down3),
            &mut g_bf,
            &mut u_bf,
            &mut h_bf,
            shape(3, 2),
        )
        .unwrap();
        assert_eq!(y_bf, y_f);
        assert_eq!(g_bf, g_f);
        assert_eq!(u_bf, u_f);
        assert_eq!(h_bf, h_f);

        // Little-endian gate weight. [0x80, 0x3F] is 1.0. The swapped pair
        // is not 1, so y moves. up = 2, down = 2, x = 1 → y = 2 * silu(1) * 2.
        let mut y_le = [0.0];
        let mut g_le = [0.0];
        let mut u_le = [0.0];
        let mut h_le = [0.0];
        swiglu_bf16(
            &mut y_le,
            &[1.0],
            &[0x80, 0x3F],
            &[0x00, 0x40],
            &[0x00, 0x40],
            &mut g_le,
            &mut u_le,
            &mut h_le,
            shape(1, 1),
        )
        .unwrap();
        assert_eq!(g_le, [1.0]);
        assert_eq!(u_le, [2.0]);
        assert_eq!(h_le[0].to_bits(), 0x3fbb_26a8);
        // 2 * (silu(1) * 2). Exponent bumps are exact on this finite value.
        assert_eq!(y_le[0].to_bits(), 0x403b_26a8);

        let mut y_swap = [0.0];
        swiglu_bf16(
            &mut y_swap,
            &[1.0],
            &[0x3F, 0x80],
            &[0x00, 0x40],
            &[0x00, 0x40],
            &mut g_le,
            &mut u_le,
            &mut h_le,
            shape(1, 1),
        )
        .unwrap();
        assert_ne!(y_swap, y_le);
        assert_ne!(g_le[0], 1.0);
    }

    #[test]
    fn shape_errors_do_not_write() {
        let x = [1.0f32, 0.0];
        let gate_w = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5];
        let up_w = [2.0f32, 0.0, 4.0, 8.0, 1.0, 1.0];
        let down_w = [0.5f32, 3.0, 1.0, 1.0, 5.0, 0.5];
        let gate_b = pack(&gate_w);
        let up_b = pack(&up_w);
        let down_b = pack(&down_w);
        // hidden = 2, intermediate = 3. Sentinel in every output.
        let mut y = [7.0f32, 8.0];
        let y_before = y;
        let mut gate = [3.0f32, 4.0, 5.0];
        let gate_before = gate;
        let mut up = [6.0f32, 7.0, 8.0];
        let up_before = up;
        let mut h = [9.0f32, 1.0, 2.0];
        let h_before = h;

        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w,
                &up_w,
                &down_w,
                &mut gate,
                &mut up,
                &mut h,
                shape(0, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b,
                &up_b,
                &down_b,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 0),
            ),
            Err(SwigluError::Shape)
        );

        assert_eq!(
            swiglu_f32(
                &mut y[..1],
                &x,
                &gate_w,
                &up_w,
                &down_w,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x[..1],
                &gate_b,
                &up_b,
                &down_b,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w,
                &up_w,
                &down_w,
                &mut gate[..2],
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b,
                &up_b,
                &down_b,
                &mut gate,
                &mut up[..2],
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w,
                &up_w,
                &down_w,
                &mut gate,
                &mut up,
                &mut h[..2],
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );

        // Weight lengths: one short row, one trailing element, one odd byte.
        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w[..5],
                &up_w,
                &down_w,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w,
                &up_w[..5],
                &down_w,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        let down_long = [0.5f32, 3.0, 1.0, 1.0, 5.0, 0.5, 9.0];
        assert_eq!(
            swiglu_f32(
                &mut y,
                &x,
                &gate_w,
                &up_w,
                &down_long,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b[..gate_b.len() - 1],
                &up_b,
                &down_b,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b,
                &up_b[..up_b.len() - 2],
                &down_b,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );
        let mut down_extra = down_b.clone();
        down_extra.push(0);
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b,
                &up_b,
                &down_extra,
                &mut gate,
                &mut up,
                &mut h,
                shape(2, 3),
            ),
            Err(SwigluError::Shape)
        );

        // Swapping hidden and intermediate disagrees with x and the scratch.
        assert_eq!(
            swiglu_bf16(
                &mut y,
                &x,
                &gate_b,
                &up_b,
                &down_b,
                &mut gate,
                &mut up,
                &mut h,
                shape(3, 2),
            ),
            Err(SwigluError::Shape)
        );

        assert_eq!(y, y_before);
        assert_eq!(gate, gate_before);
        assert_eq!(up, up_before);
        assert_eq!(h, h_before);

        // Huge shapes do not panic and do not write. The matrix-length
        // helpers themselves reject products that overflow `usize`.
        assert_eq!(bf16_matrix_bytes(usize::MAX, 2), Err(SwigluError::Shape));
        assert_eq!(
            bf16_matrix_bytes(2, usize::MAX / 2 + 1),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            f32_matrix_len(usize::MAX, usize::MAX),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_bf16(
                &mut y[..1],
                &x[..1],
                &[],
                &[],
                &[],
                &mut gate[..1],
                &mut up[..1],
                &mut h[..1],
                shape(usize::MAX, usize::MAX),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(
            swiglu_f32(
                &mut y[..1],
                &x[..1],
                &[],
                &[],
                &[],
                &mut gate[..1],
                &mut up[..1],
                &mut h[..1],
                shape(usize::MAX / 2 + 1, 2),
            ),
            Err(SwigluError::Shape)
        );
        assert_eq!(y, y_before);
        assert_eq!(gate, gate_before);
        assert_eq!(up, up_before);
        assert_eq!(h, h_before);

        let mut empty_y = [];
        let mut empty_gate = [];
        let mut empty_up = [];
        let mut empty_h = [];
        assert_eq!(
            swiglu_bf16(
                &mut empty_y,
                &[],
                &[],
                &[],
                &[],
                &mut empty_gate,
                &mut empty_up,
                &mut empty_h,
                shape(0, 0),
            ),
            Err(SwigluError::Shape)
        );
    }
}
