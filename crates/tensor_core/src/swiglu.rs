//! SwiGLU middle: elementwise SiLU(gate) ⊙ up.
//!
//! Activations are f32 for the whole op. There is no weight tensor.
//! Qwen2 MLP is `down(SiLU(gate) ⊙ up)` after `gate_proj` and `up_proj`.
//! Those three GEMVs stay on [`crate::gemv_bf16`] later (no MLP bias).
//! [`swiglu_mul_f32`] is only the elementwise product in between.
//!
//! Qwen2.5-1.5B `intermediate_size` is [`INTERMEDIATE_SIZE_QWEN25_1_5B`]
//! (8960). `gate`, `up`, and `out` are that width on the 1.5B track. The
//! op does not hard-code it: any positive equal length is legal. `hidden`
//! for that model is 1536, which is the later down-proj output, not this
//! buffer.
//!
//! # SiLU
//!
//! One thread, lane `0` then lane `1`, through the end. [`silu_f32`] is
//!
//! ```text
//! silu(x) = x / (1 + expf(-x))
//! out[i]  = silu(gate[i]) * up[i]
//! ```
//!
//! `expf` is libm. The denominator is `1 + expf(-x)`, then one division.
//! `x * (1 / (1 + expf(-x)))` is a different rounding (they split near
//! zero) and is not this function. The multiply by `up` runs after that
//! quotient is finished. `(gate * up) / (1 + expf(-gate))` is a different
//! rounding and is not this function. The op does not call `mul_add` and
//! does not use a polynomial sigmoid.
//!
//! HF `hidden_act = silu` is this function (`nn.SiLU`: `x * sigmoid(x)`).
//! Specials are whatever that libm expression returns. There is no clamp.
//!
//! An empty buffer, or any length that is not the same positive `n` on
//! `out`, `gate`, and `up`, is [`SwigluError::Shape`]. On that error,
//! `out` is left unchanged.
//!
//! Fingerprint intent (the run hash is later): weights stay
//! `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`. This op adds no
//! second dtype. Its exact-sensitive step is the serial `expf`, the add
//! of one, the division, then the multiply by `up`.

use core::fmt;

/// Qwen2.5-1.5B `intermediate_size` (`config.json`).
///
/// `mlp.gate_proj` and `mlp.up_proj` are `[8960, 1536]`. `mlp.down_proj`
/// is `[1536, 8960]`. [`swiglu_mul_f32`] sees the 8960-wide activations.
/// See `docs/qwen25-1.5b-instruct-layout.md`.
pub const INTERMEDIATE_SIZE_QWEN25_1_5B: usize = 8960;

/// Shape failure from a SwiGLU op.
///
/// On `Err`, `out` is left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluError {
    /// A buffer is empty, or `out`, `gate`, and `up` are not the same length.
    Shape,
}

impl fmt::Display for SwigluError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwigluError::Shape => write!(f, "swiglu shape does not match the buffers"),
        }
    }
}

/// `x / (1 + expf(-x))` in f32.
///
/// Libm `expf`, then add `1`, then divide. [`swiglu_mul_f32`] multiplies
/// the result by `up` only after this returns.
pub fn silu_f32(x: f32) -> f32 {
    let denom = 1.0 + exp_f32(-x);
    x / denom
}

/// `out[i] = silu(gate[i]) * up[i]`.
///
/// `out`, `gate`, and `up` must be the same non-empty length. Qwen2.5-1.5B
/// passes [`INTERMEDIATE_SIZE_QWEN25_1_5B`]. Formula, order, and the
/// rejected rewrites: see the module docs.
///
/// On [`SwigluError::Shape`], `out` is unchanged.
pub fn swiglu_mul_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), SwigluError> {
    if gate.is_empty() || gate.len() != up.len() || out.len() != gate.len() {
        return Err(SwigluError::Shape);
    }
    for i in 0..gate.len() {
        let activated = silu_f32(gate[i]);
        out[i] = activated * up[i];
    }
    Ok(())
}

// `f32::exp` lives in `std`. This crate is `no_std` outside tests, so call
// libm `expf`: the IEEE `exp` for f32, not a polynomial stand-in.
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

    #[test]
    fn qwen25_intermediate_size_is_8960() {
        // config.json intermediate_size. hidden 1536 is the down-proj
        // output, not the width of this op.
        assert_eq!(INTERMEDIATE_SIZE_QWEN25_1_5B, 8960);
    }

    #[test]
    fn silu_is_div_by_one_plus_expf() {
        // Exact identities of `x / (1 + expf(-x))`.
        assert_eq!(silu_f32(0.0).to_bits(), 0x0000_0000);
        assert_eq!(silu_f32(-0.0).to_bits(), 0x8000_0000);
        // expf(-20) underflows to 0, so SiLU(20) is 20.
        assert_eq!(silu_f32(20.0).to_bits(), 20.0f32.to_bits());
        assert_eq!(silu_f32(20.0), 20.0);
        // 16 is one ulp below the identity: expf(-16) is still visible.
        assert_eq!(silu_f32(16.0).to_bits(), 0x417f_fffe);
        assert_ne!(silu_f32(16.0).to_bits(), 16.0f32.to_bits());
        // expf(100) overflows, so SiLU(-100) is -0.
        assert_eq!(silu_f32(-100.0).to_bits(), 0x8000_0000);

        // silu(1) = 1 / (1 + expf(-1)); silu(-1) = -1 / (1 + expf(1)).
        assert_eq!(silu_f32(1.0).to_bits(), 0x3f3b_26a8);
        assert_eq!(silu_f32(-1.0).to_bits(), 0xbe89_b2b1);
        assert_eq!(silu_f32(2.0).to_bits(), 0x3fe1_7bea);
        assert_eq!(silu_f32(-2.0).to_bits(), 0xbe74_20a9);

        // Near zero: 2^-25 is small enough that expf(-x) rounds to 1,
        // so SiLU is exactly x/2.
        let tiny = f32::from_bits(0x3300_0000);
        assert_eq!(silu_f32(tiny).to_bits(), 0x3280_0000);
        assert_eq!(silu_f32(tiny), tiny * 0.5);

        // 1e-7 is where the reciprocal form rounds the other way.
        // x / (1 + expf(-x)) is 0x3356_bf96.
        // x * (1 / (1 + expf(-x))) is 0x3356_bf97.
        let near = f32::from_bits(0x33d6_bf95);
        assert_eq!(silu_f32(near).to_bits(), 0x3356_bf96);
        assert_ne!(silu_f32(near).to_bits(), 0x3356_bf97);

        // -88 is the same split, one ulp, on a large negative tail.
        assert_eq!(silu_f32(-88.0).to_bits(), 0x8335_4ddc);
        assert_ne!(silu_f32(-88.0).to_bits(), 0x8335_4ddb);
    }

    #[test]
    fn swiglu_mul_hand_vector() {
        // Lane notes, left to right:
        // 0: SiLU(0) = 0, times -1 → -0.
        // 20 * 0.5 = 10 exactly (SiLU identity).
        // SiLU(-100) = -0, times -3 → +0.
        // SiLU(1) * 3, bits below.
        // SiLU(2^-25) = 2^-26, times 4 → 2^-24.
        let tiny = f32::from_bits(0x3300_0000);
        let gate = [0.0f32, 20.0, -100.0, 1.0, tiny];
        let up = [-1.0f32, 0.5, -3.0, 3.0, 4.0];
        let mut out = [0.0; 5];
        swiglu_mul_f32(&mut out, &gate, &up).unwrap();

        assert_eq!(out[0].to_bits(), 0x8000_0000);
        assert_eq!(out[1], 10.0);
        assert_eq!(out[1].to_bits(), 0x4120_0000);
        assert_eq!(out[2].to_bits(), 0x0000_0000);
        assert_eq!(out[3].to_bits(), 0x400c_5cfe);
        assert_eq!(out[4].to_bits(), 0x3380_0000);
        assert_eq!(out[4], f32::from_bits(0x3380_0000));

        // Each lane is the scalar SiLU, then the multiply. up = 1 keeps
        // the SiLU bits, including the 1e-7 division pin.
        let gate_more = [-2.0f32, 0.5, -1.0, f32::from_bits(0x33d6_bf95)];
        let up_more = [1.0f32, -4.0, -2.0, 1.0];
        let mut out_more = [1.0; 4];
        swiglu_mul_f32(&mut out_more, &gate_more, &up_more).unwrap();
        assert_eq!(out_more[0].to_bits(), 0xbe74_20a9);
        assert_eq!(out_more[1].to_bits(), 0xbf9f_597f);
        assert_eq!(out_more[2].to_bits(), 0x3f09_b2b1);
        assert_eq!(out_more[3].to_bits(), 0x3356_bf96);
    }

    #[test]
    fn swiglu_mul_is_silu_then_up() {
        // gate = -19.5 (exact), up = 0.1f32.
        // SiLU finishes, then * up: 0xb1e3_b063.
        // (gate * up) / (1 + expf(-gate)) is the next ulp, 0xb1e3_b064.
        let gate = [-19.5f32];
        let up = [0.1f32];
        assert_eq!(gate[0].to_bits(), 0xc19c_0000);
        assert_eq!(up[0].to_bits(), 0x3dcc_cccd);
        let mut out = [0.0];
        swiglu_mul_f32(&mut out, &gate, &up).unwrap();
        assert_eq!(out[0].to_bits(), 0xb1e3_b063);
        assert_ne!(out[0].to_bits(), 0xb1e3_b064);
        assert_eq!(out[0].to_bits(), (silu_f32(gate[0]) * up[0]).to_bits());
    }

    #[test]
    fn swiglu_mul_rejects_shape_and_leaves_out() {
        let gate = [1.0f32, -1.0, 2.0];
        let up = [4.0f32, 5.0, 6.0];
        let mut out = [7.0f32, 8.0, 9.0];
        swiglu_mul_f32(&mut out, &gate, &up).unwrap();
        let kept = out;

        let up_short = [4.0f32, 5.0];
        assert_eq!(
            swiglu_mul_f32(&mut out, &gate, &up_short),
            Err(SwigluError::Shape)
        );
        assert_eq!(out, kept);

        let mut short = [7.0f32, 8.0];
        assert_eq!(
            swiglu_mul_f32(&mut short, &gate, &up),
            Err(SwigluError::Shape)
        );
        assert_eq!(short, [7.0, 8.0]);

        let mut long = [7.0f32, 8.0, 9.0, 10.0];
        assert_eq!(
            swiglu_mul_f32(&mut long, &gate, &up),
            Err(SwigluError::Shape)
        );
        assert_eq!(long, [7.0, 8.0, 9.0, 10.0]);

        let mut empty = [];
        assert_eq!(
            swiglu_mul_f32(&mut empty, &[], &[]),
            Err(SwigluError::Shape)
        );

        let mut sentinel = [7.0f32];
        assert_eq!(
            swiglu_mul_f32(&mut sentinel, &[], &[]),
            Err(SwigluError::Shape)
        );
        assert_eq!(sentinel, [7.0]);
        assert_eq!(
            swiglu_mul_f32(&mut sentinel, &gate, &[4.0, 5.0]),
            Err(SwigluError::Shape)
        );
        assert_eq!(sentinel, [7.0]);
    }
}
