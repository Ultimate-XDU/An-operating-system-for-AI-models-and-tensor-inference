//! Row-parallel `gemv_bf16` with the scalar left-to-right f32 chain.
//!
//! Several rows share one pass over `x`. Each row still does
//! `acc += bf16_to_f32(w) * x[j]` for `j` from `0` to `k - 1` (multiply,
//! then add; not FMA). On x86_64 with AVX2, groups of four rows sit in XMM
//! lanes. A lane is only added to itself, in column order, so the bits
//! match the one-row scalar loop. The tile is XMM on purpose: a VEX xmm
//! op clears the upper half of that ymm, and a wider transpose spilled
//! the accumulators.
//!
//! Callers have already checked `w.len() == y.len() * x.len() * 2`.

use crate::bf16_le_to_f32;

/// Dispatch. `w`, `x`, and `y` match a validated [`crate::gemv_bf16`] call
/// with `k > 0`. `m == 0` writes nothing.
pub(crate) fn gemv_bf16_rows(y: &mut [f32], w: &[u8], x: &[f32]) {
    let m = y.len();
    let k = x.len();
    debug_assert!(k > 0);
    debug_assert_eq!(w.len(), m.saturating_mul(k).saturating_mul(2));
    if m == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: `avx2_available` checked OSXSAVE, AVX, XCR0, and AVX2.
            unsafe { gemv_bf16_avx2(y, w, x) };
            return;
        }
    }
    gemv_bf16_scalar(y, w, x);
}

/// Eight independent left-to-right chains, then a scalar tail of at most
/// seven rows. Same bits as one row at a time. This is the non-AVX2 path.
pub(crate) fn gemv_bf16_scalar(y: &mut [f32], w: &[u8], x: &[f32]) {
    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let mut row = 0;
    while row + 8 <= m {
        let mut acc = [0.0f32; 8];
        for j in 0..k {
            let xj = x[j];
            for r in 0..8 {
                let off = (row + r) * row_bytes + j * 2;
                // SAFETY: `off + 1 < w.len()`. `row + r < m` and `j < k`,
                // and the caller checked `w.len() == m * k * 2`.
                let le = unsafe { [*w.as_ptr().add(off), *w.as_ptr().add(off + 1)] };
                acc[r] += bf16_le_to_f32(le) * xj;
            }
        }
        y[row..row + 8].copy_from_slice(&acc);
        row += 8;
    }
    while row < m {
        let start = row * row_bytes;
        y[row] = dot_row(&w[start..start + row_bytes], x);
        row += 1;
    }
}

/// One row, left to right. The reference the fast paths must match.
fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (j, &xj) in x.iter().enumerate() {
        let mut le = [0u8; 2];
        le.copy_from_slice(&row[j * 2..j * 2 + 2]);
        acc += bf16_le_to_f32(le) * xj;
    }
    acc
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

/// Sixteen rows, then eight, then a scalar tail.
///
/// Each XMM holds four row accumulators. Columns are applied from `j = 0`
/// upward, multiply then add, so each row matches the scalar chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2(y: &mut [f32], w: &[u8], x: &[f32]) {
    use core::arch::x86_64::_mm_storeu_ps;

    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();
    let mut row = 0usize;
    while row + 16 <= m {
        let p0 = panel_ptrs(w_ptr, row, row_bytes);
        let p1 = panel_ptrs(w_ptr, row + 8, row_bytes);
        let (a0, a1, b0, b1) = dot_rows16(p0, p1, x_ptr, k);
        _mm_storeu_ps(y_ptr.add(row), a0);
        _mm_storeu_ps(y_ptr.add(row + 4), a1);
        _mm_storeu_ps(y_ptr.add(row + 8), b0);
        _mm_storeu_ps(y_ptr.add(row + 12), b1);
        row += 16;
    }
    if row + 8 <= m {
        let p0 = panel_ptrs(w_ptr, row, row_bytes);
        let (a0, a1) = dot_rows8(p0, x_ptr, k);
        _mm_storeu_ps(y_ptr.add(row), a0);
        _mm_storeu_ps(y_ptr.add(row + 4), a1);
        row += 8;
    }
    while row < m {
        let mut acc = 0.0f32;
        let base = w_ptr.add(row * row_bytes);
        for j in 0..k {
            acc += load_bf16(base.add(j * 2)) * *x_ptr.add(j);
        }
        *y_ptr.add(row) = acc;
        row += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn panel_ptrs(w: *const u8, row: usize, row_bytes: usize) -> [*const u8; 8] {
    let base = w.add(row * row_bytes);
    let mut ptrs = [base; 8];
    for r in 0..8 {
        ptrs[r] = base.add(r * row_bytes);
    }
    ptrs
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_bf16(p: *const u8) -> f32 {
    bf16_le_to_f32([*p, *p.add(1)])
}

/// Four columns × eight rows. Low four lanes of each column are rows 0..4,
/// high four are rows 4..8. `j` is a multiple of 4 and `j + 4 <= k`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cols_8x4(rows: [*const u8; 8], j: usize) -> [core::arch::x86_64::__m128i; 4] {
    use core::arch::x86_64::{
        _mm_loadl_epi64, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
        _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    };

    let off = j * 2;
    let r0 = _mm_loadl_epi64(rows[0].add(off) as *const _);
    let r1 = _mm_loadl_epi64(rows[1].add(off) as *const _);
    let r2 = _mm_loadl_epi64(rows[2].add(off) as *const _);
    let r3 = _mm_loadl_epi64(rows[3].add(off) as *const _);
    let r4 = _mm_loadl_epi64(rows[4].add(off) as *const _);
    let r5 = _mm_loadl_epi64(rows[5].add(off) as *const _);
    let r6 = _mm_loadl_epi64(rows[6].add(off) as *const _);
    let r7 = _mm_loadl_epi64(rows[7].add(off) as *const _);

    let t0 = _mm_unpacklo_epi16(r0, r1);
    let t1 = _mm_unpacklo_epi16(r2, r3);
    let t2 = _mm_unpacklo_epi16(r4, r5);
    let t3 = _mm_unpacklo_epi16(r6, r7);
    let b0 = _mm_unpacklo_epi32(t0, t1);
    let b1 = _mm_unpackhi_epi32(t0, t1);
    let b2 = _mm_unpacklo_epi32(t2, t3);
    let b3 = _mm_unpackhi_epi32(t2, t3);
    [
        _mm_unpacklo_epi64(b0, b2),
        _mm_unpackhi_epi64(b0, b2),
        _mm_unpacklo_epi64(b1, b3),
        _mm_unpackhi_epi64(b1, b3),
    ]
}

/// `acc += promoted column`, low four rows then high four. Multiply, then add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn madd4(
    acc_lo: core::arch::x86_64::__m128,
    acc_hi: core::arch::x86_64::__m128,
    col: core::arch::x86_64::__m128i,
    x: *const f32,
    zero: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m128, core::arch::x86_64::__m128) {
    use core::arch::x86_64::{
        _mm_add_ps, _mm_broadcast_ss, _mm_castsi128_ps, _mm_mul_ps, _mm_unpackhi_epi16,
        _mm_unpacklo_epi16,
    };
    // `unpack(0, bf16)` puts each u16 in the high half of a u32: the same
    // bits as `u16 as u32 << 16`.
    let lo = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, col));
    let hi = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, col));
    let xv = _mm_broadcast_ss(&*x);
    (
        _mm_add_ps(acc_lo, _mm_mul_ps(lo, xv)),
        _mm_add_ps(acc_hi, _mm_mul_ps(hi, xv)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_rows8(
    rows: [*const u8; 8],
    x: *const f32,
    k: usize,
) -> (core::arch::x86_64::__m128, core::arch::x86_64::__m128) {
    use core::arch::x86_64::{_mm_loadu_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_storeu_ps};

    let zero = _mm_setzero_si128();
    let mut acc_lo = _mm_setzero_ps();
    let mut acc_hi = _mm_setzero_ps();
    let mut j = 0usize;
    let k4 = k & !3;
    while j + 4 <= k4 {
        prefetch_ahead(rows, j, k);
        let cols = cols_8x4(rows, j);
        (acc_lo, acc_hi) = madd4(acc_lo, acc_hi, cols[0], x.add(j), zero);
        (acc_lo, acc_hi) = madd4(acc_lo, acc_hi, cols[1], x.add(j + 1), zero);
        (acc_lo, acc_hi) = madd4(acc_lo, acc_hi, cols[2], x.add(j + 2), zero);
        (acc_lo, acc_hi) = madd4(acc_lo, acc_hi, cols[3], x.add(j + 3), zero);
        j += 4;
    }
    if j == k {
        return (acc_lo, acc_hi);
    }
    let mut tmp = [0.0f32; 8];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc_lo);
    _mm_storeu_ps(tmp.as_mut_ptr().add(4), acc_hi);
    while j < k {
        let xj = *x.add(j);
        for r in 0..8 {
            tmp[r] += load_bf16(rows[r].add(j * 2)) * xj;
        }
        j += 1;
    }
    (
        _mm_loadu_ps(tmp.as_ptr()),
        _mm_loadu_ps(tmp.as_ptr().add(4)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_rows16(
    rows0: [*const u8; 8],
    rows1: [*const u8; 8],
    x: *const f32,
    k: usize,
) -> (
    core::arch::x86_64::__m128,
    core::arch::x86_64::__m128,
    core::arch::x86_64::__m128,
    core::arch::x86_64::__m128,
) {
    use core::arch::x86_64::{_mm_loadu_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_storeu_ps};

    let zero = _mm_setzero_si128();
    let mut a_lo = _mm_setzero_ps();
    let mut a_hi = _mm_setzero_ps();
    let mut b_lo = _mm_setzero_ps();
    let mut b_hi = _mm_setzero_ps();
    let mut j = 0usize;
    let k4 = k & !3;
    while j + 4 <= k4 {
        // Rows are `k` apart, so a column step jumps a whole row and the
        // hardware prefetcher drops the stream. Pull the line 64 BF16 ahead.
        prefetch_ahead(rows0, j, k);
        prefetch_ahead(rows1, j, k);
        let c0 = cols_8x4(rows0, j);
        let c1 = cols_8x4(rows1, j);
        (a_lo, a_hi) = madd4(a_lo, a_hi, c0[0], x.add(j), zero);
        (b_lo, b_hi) = madd4(b_lo, b_hi, c1[0], x.add(j), zero);
        (a_lo, a_hi) = madd4(a_lo, a_hi, c0[1], x.add(j + 1), zero);
        (b_lo, b_hi) = madd4(b_lo, b_hi, c1[1], x.add(j + 1), zero);
        (a_lo, a_hi) = madd4(a_lo, a_hi, c0[2], x.add(j + 2), zero);
        (b_lo, b_hi) = madd4(b_lo, b_hi, c1[2], x.add(j + 2), zero);
        (a_lo, a_hi) = madd4(a_lo, a_hi, c0[3], x.add(j + 3), zero);
        (b_lo, b_hi) = madd4(b_lo, b_hi, c1[3], x.add(j + 3), zero);
        j += 4;
    }
    if j == k {
        return (a_lo, a_hi, b_lo, b_hi);
    }
    let mut t0 = [0.0f32; 8];
    let mut t1 = [0.0f32; 8];
    _mm_storeu_ps(t0.as_mut_ptr(), a_lo);
    _mm_storeu_ps(t0.as_mut_ptr().add(4), a_hi);
    _mm_storeu_ps(t1.as_mut_ptr(), b_lo);
    _mm_storeu_ps(t1.as_mut_ptr().add(4), b_hi);
    while j < k {
        let xj = *x.add(j);
        for r in 0..8 {
            t0[r] += load_bf16(rows0[r].add(j * 2)) * xj;
            t1[r] += load_bf16(rows1[r].add(j * 2)) * xj;
        }
        j += 1;
    }
    (
        _mm_loadu_ps(t0.as_ptr()),
        _mm_loadu_ps(t0.as_ptr().add(4)),
        _mm_loadu_ps(t1.as_ptr()),
        _mm_loadu_ps(t1.as_ptr().add(4)),
    )
}

/// Pull each row's line 64 BF16 ahead of `j`. Rows sit `k * 2` bytes apart.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prefetch_ahead(rows: [*const u8; 8], j: usize, k: usize) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    if j + 64 < k {
        for r in 0..8 {
            _mm_prefetch(rows[r].add((j + 64) * 2) as *const i8, _MM_HINT_T0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gemv_bf16, HIDDEN_SIZE_QWEN25_1_5B, INTERMEDIATE_SIZE_QWEN25_1_5B};

    /// Tied `model.embed_tokens` rows. `model_core` pins the same count;
    /// this crate does not depend on it.
    const VOCAB_TIED_QWEN25_1_5B: usize = 151936;

    fn naive(y: &mut [f32], w: &[u8], x: &[f32]) {
        let m = y.len();
        let k = x.len();
        let row_bytes = k * 2;
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
    }

    fn assert_same_bits(got: &[f32], expect: &[f32]) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "index {i}: got {g:?} ({:#010x}) expect {e:?} ({:#010x})",
                g.to_bits(),
                e.to_bits()
            );
        }
    }

    fn xorshift(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        if x == 0 {
            x = 0xA5A5_1234;
        }
        *state = x;
        x
    }

    /// Finite BF16 payload (exponent not all-ones) plus a full-range f32 `x`.
    fn patterned(m: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut state = seed;
        let mut w = vec![0u8; m * k * 2];
        for chunk in w.chunks_mut(2) {
            let mut bits = xorshift(&mut state) as u16;
            if (bits >> 7) & 0xFF == 0xFF {
                bits = (bits & 0x807F) | (0x70 << 7);
            }
            chunk.copy_from_slice(&bits.to_le_bytes());
        }
        let mut x = vec![0.0f32; k];
        for xj in &mut x {
            let mut bits = xorshift(&mut state);
            if (bits >> 23) & 0xFF == 0xFF {
                bits = (bits & 0x807F_FFFF) | (120 << 23);
            }
            *xj = f32::from_bits(bits);
        }
        (w, x)
    }

    fn assert_paths_match(m: usize, k: usize, w: &[u8], x: &[f32]) {
        assert_eq!(w.len(), m * k * 2);
        assert_eq!(x.len(), k);
        let mut y_naive = vec![7.0f32; m];
        let mut y_scalar = vec![8.0f32; m];
        let mut y_pub = vec![9.0f32; m];
        naive(&mut y_naive, w, x);
        gemv_bf16_scalar(&mut y_scalar, w, x);
        gemv_bf16(&mut y_pub, w, x, m, k).unwrap();
        assert_same_bits(&y_scalar, &y_naive);
        assert_same_bits(&y_pub, &y_naive);
        #[cfg(target_arch = "x86_64")]
        if super::avx2_available() {
            let mut y_avx = vec![3.0f32; m];
            unsafe { gemv_bf16_avx2(&mut y_avx, w, x) };
            assert_same_bits(&y_avx, &y_naive);
        }
    }

    fn le_bf16(value: f32) -> [u8; 2] {
        let bits = value.to_bits();
        assert_eq!(bits & 0xFFFF, 0, "{value} is not an exact bf16");
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
    fn left_to_right_chain_is_not_a_tree_or_blocked_sum() {
        // 2^24 is exact in BF16. ULP there is 2, so +1 does not move it.
        // Left to right: four +1s are lost on the way up, the cancellation
        // happens, then the last three +1s survive → 3.
        // A 4-wide partial sum cancels first and yields 0. An adjacent-pair
        // tree yields 4. `x` is all ones.
        let big = 16_777_216.0f32;
        let pattern = [big, 1.0, 1.0, 1.0, -big, 1.0, 1.0, 1.0];
        let w = pack(&pattern);
        let x = [1.0f32; 8];
        let mut y = [0.0f32; 1];
        gemv_bf16(&mut y, &w, &x, 1, 8).unwrap();
        assert_eq!(y[0].to_bits(), 3.0f32.to_bits());

        // Every row of a 16-row panel uses the same chain, including rows
        // that land in the second XMM group.
        let mut wide = Vec::with_capacity(16 * w.len());
        for _ in 0..16 {
            wide.extend_from_slice(&w);
        }
        let mut y16 = vec![0.0f32; 16];
        gemv_bf16(&mut y16, &wide, &x, 16, 8).unwrap();
        assert!(y16.iter().all(|v| v.to_bits() == 3.0f32.to_bits()));
    }

    #[test]
    fn lanes_stay_on_their_own_row() {
        // Identity: row i is e_i, so a swapped lane shows up as the wrong x.
        let k = 16;
        let m = 16;
        let mut w = vec![0u8; m * k * 2];
        let one = le_bf16(1.0);
        for i in 0..m {
            let off = (i * k + i) * 2;
            w[off..off + 2].copy_from_slice(&one);
        }
        let x: Vec<f32> = (0..k).map(|j| (j as f32) * 0.5 - 3.0).collect();
        assert_paths_match(m, k, &w, &x);
    }

    #[test]
    fn scalar_and_avx2_match_naive_on_tails_and_qwen_dims() {
        let shapes = [
            (1, 1),
            (1, 7),
            (1, 8),
            (1, 9),
            (7, 1),
            (7, 7),
            (7, 8),
            (7, 9),
            (8, 1),
            (8, 7),
            (8, 8),
            (8, 9),
            (9, 9),
            (15, 17),
            (16, 16),
            (17, 15),
            (24, 31),
            (31, 24),
            (3, HIDDEN_SIZE_QWEN25_1_5B),
            (8, HIDDEN_SIZE_QWEN25_1_5B),
            (9, HIDDEN_SIZE_QWEN25_1_5B),
            (16, HIDDEN_SIZE_QWEN25_1_5B),
            (17, HIDDEN_SIZE_QWEN25_1_5B),
            (24, HIDDEN_SIZE_QWEN25_1_5B),
            (8, INTERMEDIATE_SIZE_QWEN25_1_5B),
            (16, INTERMEDIATE_SIZE_QWEN25_1_5B),
            (17, INTERMEDIATE_SIZE_QWEN25_1_5B),
            (15, INTERMEDIATE_SIZE_QWEN25_1_5B - 1),
            // Tied-head inner size. Full vocab is 151936 rows; a panel uses
            // the same `k` and the same kernel (rows do not cross).
            (32, HIDDEN_SIZE_QWEN25_1_5B),
            (48, HIDDEN_SIZE_QWEN25_1_5B),
        ];
        assert_eq!(VOCAB_TIED_QWEN25_1_5B, 151936);
        assert_eq!(HIDDEN_SIZE_QWEN25_1_5B, 1536);
        assert_eq!(INTERMEDIATE_SIZE_QWEN25_1_5B, 8960);
        for (n, (m, k)) in shapes.iter().copied().enumerate() {
            let (w, x) = patterned(m, k, 0xC0FF_EE00 ^ (n as u32).wrapping_mul(0x9E37_79B9));
            assert_paths_match(m, k, &w, &x);
        }
    }

    #[test]
    fn qwen_projection_panels_match_naive_bit_for_bit() {
        // Q / O: [1536, 1536]. Gate / up panel: many rows of k = 1536.
        // Down: k = 8960. These are the decode GEMV shapes; the tied head
        // is the same `k` as Q with `VOCAB_TIED_QWEN25_1_5B` rows.
        let panels = [
            (HIDDEN_SIZE_QWEN25_1_5B, HIDDEN_SIZE_QWEN25_1_5B, 0x11),
            (256, HIDDEN_SIZE_QWEN25_1_5B, 0x22),
            (128, INTERMEDIATE_SIZE_QWEN25_1_5B, 0x33),
            (HIDDEN_SIZE_QWEN25_1_5B, 64, 0x44),
        ];
        assert_eq!(VOCAB_TIED_QWEN25_1_5B, 151936);
        for (m, k, seed) in panels {
            let (w, x) = patterned(m, k, seed);
            assert_paths_match(m, k, &w, &x);
        }
    }

    #[test]
    fn every_bf16_pattern_matches_through_simd_and_tail() {
        // One live weight per row so a NaN is not added to another NaN
        // (ADDPS and ADDSS may keep different payloads in that case). Zeros
        // around it still run the full left-to-right chain. Column 0 is inside
        // an AVX2 tile; column 16 of a length-17 row is the scalar tail.
        for (k, live) in [(16usize, 0usize), (17, 16)] {
            let m = 65536usize;
            let mut w = vec![0u8; m * k * 2];
            for row in 0..m {
                let off = (row * k + live) * 2;
                w[off..off + 2].copy_from_slice(&(row as u16).to_le_bytes());
            }
            let mut x = vec![0.0f32; k];
            x[live] = 1.0;
            assert_paths_match(m, k, &w, &x);
        }
    }

    /// BF16 exponents away from 0 and 255, and f32 `x` in the same range,
    /// so products and the running sum stay normal.
    fn normal_panel(m: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut state = seed | 1;
        let mut w = vec![0u8; m * k * 2];
        for chunk in w.chunks_mut(2) {
            let bits = xorshift(&mut state);
            let bf = (bits & 0x807F) | (0x70 << 7) | ((bits & 0x7) << 7);
            chunk.copy_from_slice(&(bf as u16).to_le_bytes());
        }
        let mut x = vec![0.0f32; k];
        for xj in &mut x {
            let bits = xorshift(&mut state);
            *xj = f32::from_bits((bits & 0x807F_FFFF) | (0x70 << 23));
        }
        (w, x)
    }

    fn elapsed_secs(iters: usize, mut body: impl FnMut()) -> f64 {
        body();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            body();
        }
        start.elapsed().as_secs_f64() / iters as f64
    }

    /// Release-only timing against the old one-row loop. Debug `cargo test`
    /// returns without a factor so the default suite stays short.
    #[test]
    fn gemv_bf16_large_shapes_beat_naive_loop() {
        if cfg!(debug_assertions) {
            return;
        }
        // Gate / up is the bulk of a token. Down flips the rectangle.
        // The tied head is the same `k` as gate, with more rows.
        let shapes = [
            (
                INTERMEDIATE_SIZE_QWEN25_1_5B,
                HIDDEN_SIZE_QWEN25_1_5B,
                "gate_up",
            ),
            (
                HIDDEN_SIZE_QWEN25_1_5B,
                INTERMEDIATE_SIZE_QWEN25_1_5B,
                "down",
            ),
            (4096, HIDDEN_SIZE_QWEN25_1_5B, "lm_head_panel"),
            (HIDDEN_SIZE_QWEN25_1_5B, HIDDEN_SIZE_QWEN25_1_5B, "q_or_o"),
        ];
        for (m, k, name) in shapes {
            // Normal magnitudes. Random bit patterns include BF16 subnormals,
            // and subnormal fp32 math is microcoded, so it does not measure
            // the decode path. Qwen weights are normal BF16.
            let (w, x) = normal_panel(m, k, 0x51);
            let mut y_fast = vec![0.0f32; m];
            let mut y_naive = vec![0.0f32; m];
            let fast = elapsed_secs(3, || {
                gemv_bf16(&mut y_fast, &w, &x, m, k).unwrap();
                std::hint::black_box(&y_fast);
            });
            let slow = elapsed_secs(3, || {
                naive(&mut y_naive, &w, &x);
                std::hint::black_box(&y_naive);
            });
            assert_same_bits(&y_fast, &y_naive);
            let speedup = slow / fast;
            let macs = (m as f64) * (k as f64);
            eprintln!(
                "gemv_bf16 {name} [{m}, {k}]: naive {slow:.6}s fast {fast:.6}s speedup {speedup:.2}x ({:.2} Gmac/s)",
                macs / fast / 1e9
            );
            // Floor, not the measured factor. This host's one-row loop is
            // already close to f32-add throughput, so the ratio sits around
            // 3.5–5×. The printed line is the number to quote. A regression
            // back to that one-row loop still fails.
            assert!(
                speedup >= 2.5,
                "{name} speedup {speedup:.2}x, want >= 2.5x versus the naive loop"
            );
        }
    }
}
