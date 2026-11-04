//! Prompt-chunk GEMM: many tokens, one weight read, same bits as `gemv_bf16`.
//!
//! `Y[b, i] = sum_j bf16_to_f32(W[i, j]) * X[b, j]`. Each pair `(b, i)` is its
//! own f32 chain, `j` from `0` to `k - 1`, multiply then add. Not FMA, and not
//! a tree of partial dots. A lane never mixes two tokens or two rows, so the
//! bits match one [`crate::gemv_bf16`] per token.
//!
//! `W` stays the little-endian BF16 view. This does not pack it and does not
//! build an f32 copy of the matrix. Activations are staged into a caller
//! scratch so a column `j` across the chunk is contiguous; that scratch is not
//! a second weight matrix.
//!
//! Rows split across [`crate::gemv_bf16_threads`] when this crate is built with
//! `std` (or under `cargo test` of this crate). Workers do not reduce into
//! each other.

use crate::bf16_le_to_f32;
use crate::gemv_bf16_kernel::GEMV_THREADS_MAX;
use crate::Bf16OpError;

#[cfg(any(feature = "std", test))]
use crate::gemv_bf16_kernel::gemv_row_spans;

/// Rows whose accumulators share one pass over a K tile.
///
/// Scratch for one worker is `GEMM_ROW_BLOCK * batch` f32s.
pub const GEMM_ROW_BLOCK: usize = 16;

/// K tile. 64 columns of a 128-token chunk are 32 KiB, one L1-sized X slice.
const GEMM_J_BLOCK: usize = 64;

/// f32s [`gemm_bf16`] needs in `scratch` for this `(batch, k, threads)`.
///
/// Layout is `[k, batch]` activation staging, then one
/// `[GEMM_ROW_BLOCK, batch]` accumulator per worker. `threads` is clamped to
/// `1..=GEMV_THREADS_MAX`.
pub fn gemm_bf16_scratch_elems(batch: usize, k: usize, threads: usize) -> Option<usize> {
    let threads = threads.clamp(1, GEMV_THREADS_MAX);
    let stage = k.checked_mul(batch)?;
    let acc = threads.checked_mul(GEMM_ROW_BLOCK)?.checked_mul(batch)?;
    stage.checked_add(acc)
}

/// `Y[b, i] = W[i, :] · X[b, :]` for `b` in `0..batch`.
///
/// `x` and `y` are token-major: token `b` is contiguous, length `k` in `x`
/// and length `m` in `y`. `w` is row-major little-endian BF16, `m` by `k`.
/// `scratch` must hold [`gemm_bf16_scratch_elems`] for `batch`, `k`, and the
/// current [`crate::gemv_bf16_threads`]. On [`Bf16OpError::Shape`], `y` is
/// left unchanged.
pub fn gemm_bf16(
    y: &mut [f32],
    w: &[u8],
    x: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    scratch: &mut [f32],
) -> Result<(), Bf16OpError> {
    gemm_bf16_at_threads(
        y,
        w,
        x,
        batch,
        m,
        k,
        scratch,
        crate::gemv_bf16_threads(),
        Isa::Auto,
    )
}

pub(crate) fn gemm_bf16_at_threads(
    y: &mut [f32],
    w: &[u8],
    x: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    scratch: &mut [f32],
    threads: usize,
    isa: Isa,
) -> Result<(), Bf16OpError> {
    let y_need = batch.checked_mul(m).ok_or(Bf16OpError::Shape)?;
    let x_need = batch.checked_mul(k).ok_or(Bf16OpError::Shape)?;
    let w_need = m
        .checked_mul(k)
        .and_then(|n| n.checked_mul(2))
        .ok_or(Bf16OpError::Shape)?;
    if y.len() != y_need || x.len() != x_need || w.len() != w_need {
        return Err(Bf16OpError::Shape);
    }
    if m == 0 || batch == 0 {
        return Ok(());
    }
    if k == 0 {
        y.fill(0.0);
        return Ok(());
    }
    let threads = threads.clamp(1, GEMV_THREADS_MAX);
    let need = gemm_bf16_scratch_elems(batch, k, threads).ok_or(Bf16OpError::Shape)?;
    if scratch.len() < need {
        return Err(Bf16OpError::Shape);
    }
    let (stage, acc) = scratch.split_at_mut(k * batch);
    transpose_batch(stage, x, batch, k);
    let isa = resolve_isa(isa);
    dispatch(y, w, stage, acc, batch, m, k, threads, isa);
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) enum Isa {
    Auto,
    Scalar,
    Avx2,
}

fn resolve_isa(isa: Isa) -> Isa {
    match isa {
        Isa::Auto => {
            #[cfg(target_arch = "x86_64")]
            {
                if avx2_available() {
                    return Isa::Avx2;
                }
            }
            Isa::Scalar
        }
        Isa::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                if avx2_available() {
                    return Isa::Avx2;
                }
            }
            Isa::Scalar
        }
        Isa::Scalar => Isa::Scalar,
    }
}

fn transpose_batch(stage: &mut [f32], x: &[f32], batch: usize, k: usize) {
    // `stage` is `[k, batch]`: column `j` is `batch` contiguous lanes.
    for b in 0..batch {
        let src = &x[b * k..(b + 1) * k];
        for j in 0..k {
            stage[j * batch + b] = src[j];
        }
    }
}

fn dispatch(
    y: &mut [f32],
    w: &[u8],
    stage: &[f32],
    acc: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    threads: usize,
    isa: Isa,
) {
    #[cfg(any(feature = "std", test))]
    {
        dispatch_parallel(y, w, stage, acc, batch, m, k, threads, isa);
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        let _ = threads;
        let acc = &mut acc[..GEMM_ROW_BLOCK * batch];
        gemm_rows(y, w, stage, acc, batch, m, k, 0, m, isa);
    }
}

#[cfg(any(feature = "std", test))]
fn dispatch_parallel(
    y: &mut [f32],
    w: &[u8],
    stage: &[f32],
    acc: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    threads: usize,
    isa: Isa,
) {
    let spans = gemv_row_spans(m, threads);
    let slice = spans.as_slice();
    if slice.len() <= 1 {
        let acc = &mut acc[..GEMM_ROW_BLOCK * batch];
        gemm_rows(y, w, stage, acc, batch, m, k, 0, m, isa);
        return;
    }
    let stride = GEMM_ROW_BLOCK * batch;
    debug_assert!(acc.len() >= slice.len() * stride);
    let y_ptr = y.as_mut_ptr();
    let w_ptr = w.as_ptr();
    let x_ptr = stage.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    std::thread::scope(|scope| {
        for (slot, &(start, end)) in slice.iter().enumerate().skip(1) {
            // SAFETY: `slot` selects a private accumulator. Row spans are
            // disjoint, so each worker writes a distinct `y[b * m + row]`.
            // `w` and `stage` are shared reads and outlive the scope.
            let job = SpanJob {
                y: y_ptr,
                w: w_ptr,
                x: x_ptr,
                acc: unsafe { acc_ptr.add(slot * stride) },
                batch,
                m,
                k,
                start,
                end,
                isa,
            };
            scope.spawn(move || unsafe { job.run() });
        }
        let (start, end) = slice[0];
        unsafe {
            SpanJob {
                y: y_ptr,
                w: w_ptr,
                x: x_ptr,
                acc: acc_ptr,
                batch,
                m,
                k,
                start,
                end,
                isa,
            }
            .run();
        }
    });
}

#[cfg(any(feature = "std", test))]
struct SpanJob {
    y: *mut f32,
    w: *const u8,
    x: *const f32,
    acc: *mut f32,
    batch: usize,
    m: usize,
    k: usize,
    start: usize,
    end: usize,
    isa: Isa,
}

// The caller keeps the buffers alive until the scope joins, and each job
// writes a disjoint row span plus its own accumulator.
#[cfg(any(feature = "std", test))]
unsafe impl Send for SpanJob {}

#[cfg(any(feature = "std", test))]
impl SpanJob {
    /// # Safety
    ///
    /// See [`gemm_rows_raw`].
    unsafe fn run(self) {
        gemm_rows_raw(
            self.y, self.w, self.x, self.acc, self.batch, self.m, self.k, self.start, self.end,
            self.isa,
        );
    }
}

/// # Safety
///
/// `y_ptr` is `batch * m` f32s, `w_ptr` is `m * k * 2` bytes, `x_ptr` is
/// `k * batch` f32s, and `acc_ptr` is `GEMM_ROW_BLOCK * batch` f32s. Those
/// ranges stay valid for the call. This invocation is the only writer of
/// `y[b * m + row]` for `row` in `start..end` and of `acc_ptr`.
#[cfg(any(feature = "std", test))]
unsafe fn gemm_rows_raw(
    y_ptr: *mut f32,
    w_ptr: *const u8,
    x_ptr: *const f32,
    acc_ptr: *mut f32,
    batch: usize,
    m: usize,
    k: usize,
    start: usize,
    end: usize,
    isa: Isa,
) {
    let y = core::slice::from_raw_parts_mut(y_ptr, batch * m);
    let w = core::slice::from_raw_parts(w_ptr, m * k * 2);
    let stage = core::slice::from_raw_parts(x_ptr, k * batch);
    let acc = core::slice::from_raw_parts_mut(acc_ptr, GEMM_ROW_BLOCK * batch);
    gemm_rows(y, w, stage, acc, batch, m, k, start, end, isa);
}

fn gemm_rows(
    y: &mut [f32],
    w: &[u8],
    stage: &[f32],
    acc: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    row0: usize,
    row1: usize,
    isa: Isa,
) {
    debug_assert!(row1 <= m && row0 <= row1);
    debug_assert_eq!(w.len(), m * k * 2);
    debug_assert_eq!(stage.len(), k * batch);
    let row_bytes = k * 2;
    let mut row = row0;
    while row < row1 {
        let rb = (row1 - row).min(GEMM_ROW_BLOCK);
        acc[..rb * batch].fill(0.0);
        let mut j0 = 0usize;
        while j0 < k {
            let jb = (k - j0).min(GEMM_J_BLOCK);
            for r in 0..rb {
                let wrow = unsafe { w.as_ptr().add((row + r) * row_bytes + j0 * 2) };
                let a = &mut acc[r * batch..(r + 1) * batch];
                accumulate_row(a, wrow, stage, batch, j0, jb, isa);
            }
            j0 += jb;
        }
        for r in 0..rb {
            let src = &acc[r * batch..(r + 1) * batch];
            for b in 0..batch {
                y[b * m + row + r] = src[b];
            }
        }
        row += rb;
    }
}

fn accumulate_row(
    acc: &mut [f32],
    wrow: *const u8,
    stage: &[f32],
    batch: usize,
    j0: usize,
    jb: usize,
    isa: Isa,
) {
    let mut wf = [0.0f32; GEMM_J_BLOCK];
    debug_assert!(jb <= GEMM_J_BLOCK);
    for j in 0..jb {
        // SAFETY: `wrow` points at column `j0` of this row, and `j < jb`
        // stays inside the row (`j0 + jb <= k`).
        wf[j] = unsafe { load_bf16(wrow.add(j * 2)) };
    }
    let mut t = 0usize;
    #[cfg(target_arch = "x86_64")]
    if matches!(isa, Isa::Avx2) {
        // SAFETY: `Isa::Avx2` is only selected after `avx2_available`.
        t = unsafe { accumulate_avx2(acc, &wf[..jb], stage, batch, j0, t) };
    }
    if t < batch {
        accumulate_scalar(&mut acc[t..], &wf[..jb], stage, batch, j0, t);
    }
}

fn accumulate_scalar(
    acc: &mut [f32],
    wf: &[f32],
    stage: &[f32],
    batch: usize,
    j0: usize,
    t0: usize,
) {
    for (j, &wj) in wf.iter().enumerate() {
        let xcol = &stage[(j0 + j) * batch + t0..(j0 + j) * batch + batch];
        for (lane, slot) in acc.iter_mut().enumerate() {
            *slot += wj * xcol[lane];
        }
    }
}

#[inline]
unsafe fn load_bf16(p: *const u8) -> f32 {
    bf16_le_to_f32([*p, *p.add(1)])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    acc: &mut [f32],
    wf: &[f32],
    stage: &[f32],
    batch: usize,
    j0: usize,
    mut t: usize,
) -> usize {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };
    let acc_ptr = acc.as_mut_ptr();
    let x_ptr = stage.as_ptr();
    while t + 32 <= batch {
        let mut a0 = _mm256_loadu_ps(acc_ptr.add(t));
        let mut a1 = _mm256_loadu_ps(acc_ptr.add(t + 8));
        let mut a2 = _mm256_loadu_ps(acc_ptr.add(t + 16));
        let mut a3 = _mm256_loadu_ps(acc_ptr.add(t + 24));
        for (j, wj) in wf.iter().enumerate() {
            let w = _mm256_broadcast_ss(wj);
            let xp = x_ptr.add((j0 + j) * batch + t);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(w, _mm256_loadu_ps(xp)));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(w, _mm256_loadu_ps(xp.add(8))));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(w, _mm256_loadu_ps(xp.add(16))));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(w, _mm256_loadu_ps(xp.add(24))));
        }
        _mm256_storeu_ps(acc_ptr.add(t), a0);
        _mm256_storeu_ps(acc_ptr.add(t + 8), a1);
        _mm256_storeu_ps(acc_ptr.add(t + 16), a2);
        _mm256_storeu_ps(acc_ptr.add(t + 24), a3);
        t += 32;
    }
    while t + 8 <= batch {
        let mut a0 = _mm256_loadu_ps(acc_ptr.add(t));
        for (j, wj) in wf.iter().enumerate() {
            let w = _mm256_broadcast_ss(wj);
            let xp = x_ptr.add((j0 + j) * batch + t);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(w, _mm256_loadu_ps(xp)));
        }
        _mm256_storeu_ps(acc_ptr.add(t), a0);
        t += 8;
    }
    t
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            // SAFETY: CPUID and XGETBV are specified on x86_64. XGETBV runs
            // only after OSXSAVE is set.
            let yes = unsafe { avx2_cpuid() };
            CACHE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_cpuid() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
    let leaf1 = __cpuid(1);
    let osxsave = leaf1.ecx & (1 << 27) != 0;
    let avx = leaf1.ecx & (1 << 28) != 0;
    if !osxsave || !avx || _xgetbv(0) & 0x6 != 0x6 {
        return false;
    }
    let leaf7 = __cpuid_count(7, 0);
    leaf7.ebx & (1 << 5) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_bf16;

    fn pack(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn pattern_w(n: usize) -> Vec<u8> {
        // Finite BF16 only. Exponent 126 stays near 0.5; the low 7 bits are
        // the fraction. NaN payloads are not part of the exactness contract.
        let words: Vec<u16> = (0..n)
            .map(|i| {
                let sign = if i % 5 == 0 { 0x8000 } else { 0 };
                let frac = ((i as u16).wrapping_mul(17)) & 0x007F;
                sign | 0x3F00 | frac
            })
            .collect();
        pack(&words)
    }

    fn pattern_x(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 17) as f32) * 0.015625 - 1.5).collect()
    }

    fn scratch_for(batch: usize, k: usize, threads: usize) -> Vec<f32> {
        vec![0.0; gemm_bf16_scratch_elems(batch, k, threads).unwrap()]
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn scalar_oracle(w: &[u8], x: &[f32], batch: usize, m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; batch * m];
        for i in 0..m {
            for b in 0..batch {
                let mut acc = 0.0f32;
                let row = i * k * 2;
                let xb = &x[b * k..(b + 1) * k];
                for j in 0..k {
                    let le = [w[row + j * 2], w[row + j * 2 + 1]];
                    acc += bf16_le_to_f32(le) * xb[j];
                }
                y[b * m + i] = acc;
            }
        }
        y
    }

    fn assert_matches(batch: usize, m: usize, k: usize, threads: usize, isa: Isa) {
        let w = if k == 0 || m == 0 {
            Vec::new()
        } else {
            pattern_w(m * k)
        };
        let x = pattern_x(batch * k);
        let expect = scalar_oracle(&w, &x, batch, m, k);
        let mut y = vec![0.0f32; batch * m];
        let mut scratch = scratch_for(batch, k.max(1), threads);
        if k == 0 {
            scratch.clear();
        }
        gemm_bf16_at_threads(&mut y, &w, &x, batch, m, k, &mut scratch, threads, isa).unwrap();
        assert!(
            bits_eq(&y, &expect),
            "batch={batch} m={m} k={k} threads={threads}"
        );
        if k > 0 && m > 0 {
            for b in 0..batch {
                let mut one = vec![0.0f32; m];
                gemv_bf16(&mut one, &w, &x[b * k..(b + 1) * k], m, k).unwrap();
                assert!(
                    bits_eq(&y[b * m..(b + 1) * m], &one),
                    "token {b} diverged from gemv"
                );
            }
        }
    }

    #[test]
    fn gemm_matches_scalar_and_per_token_gemv() {
        let shapes = [
            (1, 1, 1),
            (1, 8, 8),
            (3, 5, 7),
            (8, 16, 64),
            (13, 17, 65),
            (32, 32, 64),
            (33, 31, 63),
            (64, 16, 40),
            (7, 4, 1),
            (4, 1, 9),
            (128, 17, 70),
            (128, 32, 128),
        ];
        for &(batch, m, k) in &shapes {
            assert_matches(batch, m, k, 1, Isa::Scalar);
            assert_matches(batch, m, k, 1, Isa::Auto);
            assert_matches(batch, m, k, 8, Isa::Auto);
            #[cfg(target_arch = "x86_64")]
            {
                assert_matches(batch, m, k, 3, Isa::Avx2);
            }
        }
    }

    #[test]
    fn gemm_zero_k_fills_zero_and_rejects_a_short_scratch() {
        let mut y = vec![1.0f32; 6];
        gemm_bf16(&mut y, &[], &[0.0; 0], 2, 3, 0, &mut []).unwrap();
        assert_eq!(y, vec![0.0; 6]);

        let mut y = vec![4.0f32; 4];
        let x = vec![1.0f32; 4];
        let w = pattern_w(4);
        let err = gemm_bf16(&mut y, &w, &x, 2, 2, 2, &mut []).unwrap_err();
        assert_eq!(err, Bf16OpError::Shape);
        assert_eq!(y, vec![4.0; 4]);

        let err = gemm_bf16(&mut y, &w, &x, 2, 2, 3, &mut vec![0.0; 64]).unwrap_err();
        assert_eq!(err, Bf16OpError::Shape);
        assert_eq!(y, vec![4.0; 4]);
    }
}
