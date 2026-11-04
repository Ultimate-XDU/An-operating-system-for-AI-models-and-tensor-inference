//! Row-parallel `gemv_bf16` with the scalar left-to-right f32 chain.
//!
//! Several rows share one pass over `x`. Each row still does
//! `acc += bf16_to_f32(w) * x[j]` for `j` from `0` to `k - 1` (multiply,
//! then add; not FMA). On x86_64 with AVX2, knife 10 cuts the row panel
//! by the rectangle. Tall short-K GEMVs (gate, up, tied head) keep the
//! 32-row block. Long-K and square projections (down, Q, K, V, O) use an
//! 8-row register block: one YMM accumulator, eight columns, each column
//! added before the next temporary is built. A lane is only added to
//! itself, in column order, so the bits match the one-row scalar loop.
//! The add is `vaddps` after `vmulps`, not `vfmadd`.
//!
//! Callers have already checked `w.len() == y.len() * x.len() * 2`.
//!
//! Optional panel pack ([`set_gemv_bf16_panel_pack`]) copies each 32-row
//! K-tile from row-major mmap order into a column-major BF16 scratch, then
//! dots that scratch. The scratch is one tile, not a second copy of the
//! matrix. Default is off: on the measured host the extra pass was slower
//! than this strided kernel, and it did not reach 2×. Knife 9 also leaves
//! software prefetch off ([`set_gemv_bf16_prefetch`]). With several workers
//! each walking its own rows, the extra `_mm_prefetch` competed with the
//! hardware prefetcher and cut multi-thread throughput. `MYOS_GEMV_PREFETCH=1`
//! turns that prefetch back on.
//!
//! Row-parallel workers live outside this file. They call [`gemv_bf16_rows`]
//! on a contiguous span from [`gemv_row_spans`]. A span does not share an
//! accumulator with another span.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::bf16_le_to_f32;

/// Showdown `n_threads`. Matches llama.cpp S1 BF16 `-t 8`.
pub const SHOWDOWN_THREADS: usize = 8;

/// Upper bound for [`set_gemv_bf16_threads`]. The row-span table is this long.
pub const GEMV_THREADS_MAX: usize = 64;

/// AVX2 panel width. Spans use whole panels when every worker can take one.
const ROW_PANEL: usize = 32;

/// Knife 10. At or above this many rows, and with `k` at or below
/// [`SHORT_K`], the 32-row block stays on. Gate, up, and the tied head
/// land here. Fewer rows, or a longer K, use the 8-row register block.
const TALL_ROWS: usize = 4096;
/// See [`TALL_ROWS`]. Down's K is 8960 and does not take this path.
const SHORT_K: usize = 2048;

/// Off by default. See the module docs.
static PANEL_PACK: AtomicBool = AtomicBool::new(false);
/// Off by default (knife 9). The strided kernel can still prefetch when
/// [`set_gemv_bf16_prefetch`] is set.
static PREFETCH: AtomicBool = AtomicBool::new(false);
/// Workers for the next [`crate::gemv_bf16`] call. The host sets this from
/// `--threads`. Default is the showdown width, not 1.
static GEMV_THREADS: AtomicUsize = AtomicUsize::new(SHOWDOWN_THREADS);

/// `true`: pack each 32-row tile into column-major scratch before the dot.
///
/// The hot path reads this once per GEMV. Safe to call before any GEMV; not
/// meant to flip mid-call.
pub fn set_gemv_bf16_panel_pack(enabled: bool) {
    PANEL_PACK.store(enabled, Ordering::Relaxed);
}

/// `true` enables `_mm_prefetch` in the BF16 GEMV. Default is `false`.
pub fn set_gemv_bf16_prefetch(enabled: bool) {
    PREFETCH.store(enabled, Ordering::Relaxed);
}

fn panel_pack_enabled() -> bool {
    PANEL_PACK.load(Ordering::Relaxed)
}

/// `true` when the knife-6 panel pack is on. Default is `false`.
pub fn gemv_bf16_panel_pack() -> bool {
    panel_pack_enabled()
}

fn prefetch_enabled() -> bool {
    PREFETCH.load(Ordering::Relaxed)
}

/// Worker count for the next host GEMV. Clamped to `1..=GEMV_THREADS_MAX`.
///
/// `0` becomes 1. Values above [`GEMV_THREADS_MAX`] become that max. The CLI
/// rejects both instead of clamping, so a fingerprint cannot advertise a
/// width the pool will not run.
pub fn set_gemv_bf16_threads(n: usize) {
    GEMV_THREADS.store(n.clamp(1, GEMV_THREADS_MAX), Ordering::Relaxed);
}

/// Worker count stored by [`set_gemv_bf16_threads`]. Default [`SHOWDOWN_THREADS`].
pub fn gemv_bf16_threads() -> usize {
    GEMV_THREADS
        .load(Ordering::Relaxed)
        .clamp(1, GEMV_THREADS_MAX)
}

/// Contiguous `[start, end)` row spans, deterministic in `(m, threads)`.
///
/// When at least one 32-row panel fits on every worker, spans are whole
/// panels and the last span also takes `m % 32` leftover rows. Otherwise
/// rows are split as evenly as possible; the first `m % threads` spans get
/// one extra row. Empty spans are not emitted. `threads` is clamped to
/// `1..=min(m, GEMV_THREADS_MAX)`.
#[derive(Clone, Copy)]
pub struct GemvRowSpans {
    spans: [(usize, usize); GEMV_THREADS_MAX],
    len: u8,
}

impl GemvRowSpans {
    pub fn as_slice(&self) -> &[(usize, usize)] {
        &self.spans[..self.len as usize]
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }
}

impl core::fmt::Debug for GemvRowSpans {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

pub fn gemv_row_spans(m: usize, threads: usize) -> GemvRowSpans {
    let mut out = GemvRowSpans {
        spans: [(0, 0); GEMV_THREADS_MAX],
        len: 0,
    };
    if m == 0 {
        return out;
    }
    let threads = threads.clamp(1, GEMV_THREADS_MAX).min(m);
    let panels = m / ROW_PANEL;
    if panels >= threads {
        let base = panels / threads;
        let extra = panels % threads;
        let tail = m % ROW_PANEL;
        let mut row = 0usize;
        for t in 0..threads {
            let n_panels = base + usize::from(t < extra);
            let mut n_rows = n_panels * ROW_PANEL;
            if t + 1 == threads {
                n_rows += tail;
            }
            let end = row + n_rows;
            out.spans[out.len as usize] = (row, end);
            out.len += 1;
            row = end;
        }
        debug_assert_eq!(row, m);
        return out;
    }
    let base = m / threads;
    let rem = m % threads;
    let mut row = 0usize;
    for t in 0..threads {
        let n = base + usize::from(t < rem);
        if n == 0 {
            break;
        }
        out.spans[out.len as usize] = (row, row + n);
        out.len += 1;
        row += n;
    }
    debug_assert_eq!(row, m);
    out
}

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

/// Knife 10 row cut.
///
/// Tall short-K rectangles (gate, up, tied head: many rows, `k` around
/// 1536) stay on the 32-row block. Everything else — long K (down) and
/// the square Q/O projections — uses the 8-row register block. Same-run
/// release timings on this machine: gate/up 32-row is faster; down and
/// Q/O 8-row is faster. Not a pack and not a prefetch switch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2(y: &mut [f32], w: &[u8], x: &[f32]) {
    if panel_pack_enabled() {
        gemv_bf16_avx2_packed(y, w, x);
        return;
    }
    if y.len() >= TALL_ROWS && x.len() <= SHORT_K {
        gemv_bf16_avx2_rows32(y, w, x);
    } else {
        gemv_bf16_avx2_rows8(y, w, x);
    }
}

/// 32-row YMM panel. Used for tall short-K GEMVs and the microbench.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2_rows32(y: &mut [f32], w: &[u8], x: &[f32]) {
    use core::arch::x86_64::_mm256_storeu_ps;

    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();
    let mut row = 0usize;
    while row + 32 <= m {
        let mut rows = [w_ptr; 32];
        let base = w_ptr.add(row * row_bytes);
        for r in 0..32 {
            rows[r] = base.add(r * row_bytes);
        }
        let acc = dot_rows32_intrin(&rows, x_ptr, k);
        _mm256_storeu_ps(y_ptr.add(row), acc[0]);
        _mm256_storeu_ps(y_ptr.add(row + 8), acc[1]);
        _mm256_storeu_ps(y_ptr.add(row + 16), acc[2]);
        _mm256_storeu_ps(y_ptr.add(row + 24), acc[3]);
        row += 32;
    }
    gemv_bf16_avx2_from16(y, w, x, row);
}

/// Eight rows per panel. See [`gemv_bf16_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2_rows8(y: &mut [f32], w: &[u8], x: &[f32]) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_broadcast_ss, _mm256_castsi256_ps, _mm256_cvtepu16_epi32,
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_slli_epi32, _mm256_storeu_ps,
        _mm_loadu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    };

    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();
    let mut row = 0usize;
    let k8 = k & !7;
    while row + 8 <= m {
        let base = w_ptr.add(row * row_bytes);
        let mut rows = [base; 8];
        for i in 0..8 {
            rows[i] = base.add(i * row_bytes);
        }
        let mut acc = _mm256_setzero_ps();
        let mut j = 0usize;
        while j < k8 {
            let off = j * 2;
            let r0 = _mm_loadu_si128(rows[0].add(off) as *const _);
            let r1 = _mm_loadu_si128(rows[1].add(off) as *const _);
            let r2 = _mm_loadu_si128(rows[2].add(off) as *const _);
            let r3 = _mm_loadu_si128(rows[3].add(off) as *const _);
            let r4 = _mm_loadu_si128(rows[4].add(off) as *const _);
            let r5 = _mm_loadu_si128(rows[5].add(off) as *const _);
            let r6 = _mm_loadu_si128(rows[6].add(off) as *const _);
            let r7 = _mm_loadu_si128(rows[7].add(off) as *const _);
            let a0 = _mm_unpacklo_epi16(r0, r1);
            let a1 = _mm_unpackhi_epi16(r0, r1);
            let a2 = _mm_unpacklo_epi16(r2, r3);
            let a3 = _mm_unpackhi_epi16(r2, r3);
            let a4 = _mm_unpacklo_epi16(r4, r5);
            let a5 = _mm_unpackhi_epi16(r4, r5);
            let a6 = _mm_unpacklo_epi16(r6, r7);
            let a7 = _mm_unpackhi_epi16(r6, r7);
            let b0 = _mm_unpacklo_epi32(a0, a2);
            let b1 = _mm_unpackhi_epi32(a0, a2);
            let b2 = _mm_unpacklo_epi32(a1, a3);
            let b3 = _mm_unpackhi_epi32(a1, a3);
            let b4 = _mm_unpacklo_epi32(a4, a6);
            let b5 = _mm_unpackhi_epi32(a4, a6);
            let b6 = _mm_unpacklo_epi32(a5, a7);
            let b7 = _mm_unpackhi_epi32(a5, a7);
            // Each column is promoted and added before the next one is built.
            macro_rules! madd_col {
                ($col:expr, $t:expr) => {{
                    let w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32($col), 16));
                    let xv = _mm256_broadcast_ss(&*x_ptr.add(j + $t));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(w, xv));
                }};
            }
            madd_col!(_mm_unpacklo_epi64(b0, b4), 0);
            madd_col!(_mm_unpackhi_epi64(b0, b4), 1);
            madd_col!(_mm_unpacklo_epi64(b1, b5), 2);
            madd_col!(_mm_unpackhi_epi64(b1, b5), 3);
            madd_col!(_mm_unpacklo_epi64(b2, b6), 4);
            madd_col!(_mm_unpackhi_epi64(b2, b6), 5);
            madd_col!(_mm_unpacklo_epi64(b3, b7), 6);
            madd_col!(_mm_unpackhi_epi64(b3, b7), 7);
            j += 8;
        }
        if j == k {
            _mm256_storeu_ps(y_ptr.add(row), acc);
        } else {
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            while j < k {
                let xj = *x_ptr.add(j);
                for r in 0..8 {
                    tmp[r] += load_bf16(rows[r].add(j * 2)) * xj;
                }
                j += 1;
            }
            _mm256_storeu_ps(y_ptr.add(row), _mm256_loadu_ps(tmp.as_ptr()));
        }
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

/// Knife-5a body: sixteen rows, then eight, then a scalar tail, from `row`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2_from16(y: &mut [f32], w: &[u8], x: &[f32], mut row: usize) {
    use core::arch::x86_64::_mm_storeu_ps;

    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();
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

/// Eight columns × eight rows. Column `t` lane `r` is row `r`, column `j+t`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cols_8x8(rows: [*const u8; 8], j: usize) -> [core::arch::x86_64::__m128i; 8] {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    };
    let off = j * 2;
    let r0 = _mm_loadu_si128(rows[0].add(off) as *const _);
    let r1 = _mm_loadu_si128(rows[1].add(off) as *const _);
    let r2 = _mm_loadu_si128(rows[2].add(off) as *const _);
    let r3 = _mm_loadu_si128(rows[3].add(off) as *const _);
    let r4 = _mm_loadu_si128(rows[4].add(off) as *const _);
    let r5 = _mm_loadu_si128(rows[5].add(off) as *const _);
    let r6 = _mm_loadu_si128(rows[6].add(off) as *const _);
    let r7 = _mm_loadu_si128(rows[7].add(off) as *const _);
    let a0 = _mm_unpacklo_epi16(r0, r1);
    let a1 = _mm_unpackhi_epi16(r0, r1);
    let a2 = _mm_unpacklo_epi16(r2, r3);
    let a3 = _mm_unpackhi_epi16(r2, r3);
    let a4 = _mm_unpacklo_epi16(r4, r5);
    let a5 = _mm_unpackhi_epi16(r4, r5);
    let a6 = _mm_unpacklo_epi16(r6, r7);
    let a7 = _mm_unpackhi_epi16(r6, r7);
    let b0 = _mm_unpacklo_epi32(a0, a2);
    let b1 = _mm_unpackhi_epi32(a0, a2);
    let b2 = _mm_unpacklo_epi32(a1, a3);
    let b3 = _mm_unpackhi_epi32(a1, a3);
    let b4 = _mm_unpacklo_epi32(a4, a6);
    let b5 = _mm_unpackhi_epi32(a4, a6);
    let b6 = _mm_unpacklo_epi32(a5, a7);
    let b7 = _mm_unpackhi_epi32(a5, a7);
    [
        _mm_unpacklo_epi64(b0, b4),
        _mm_unpackhi_epi64(b0, b4),
        _mm_unpacklo_epi64(b1, b5),
        _mm_unpackhi_epi64(b1, b5),
        _mm_unpacklo_epi64(b2, b6),
        _mm_unpackhi_epi64(b2, b6),
        _mm_unpacklo_epi64(b3, b7),
        _mm_unpackhi_epi64(b3, b7),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn madd8(
    acc: core::arch::x86_64::__m256,
    col: core::arch::x86_64::__m128i,
    x: *const f32,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_broadcast_ss, _mm256_castsi256_ps, _mm256_cvtepu16_epi32,
        _mm256_mul_ps, _mm256_slli_epi32,
    };
    // Shift, not FMA: same bits as `u16 as u32 << 16`, then mul, then add.
    let w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(col), 16));
    let xv = _mm256_broadcast_ss(&*x);
    _mm256_add_ps(acc, _mm256_mul_ps(w, xv))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_rows32_intrin(
    rows: &[*const u8; 32],
    x: *const f32,
    k: usize,
) -> [core::arch::x86_64::__m256; 4] {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let p = |g: usize| {
        let mut panel = [rows[0]; 8];
        for i in 0..8 {
            panel[i] = rows[g * 8 + i];
        }
        panel
    };
    let p0 = p(0);
    let p1 = p(1);
    let p2 = p(2);
    let p3 = p(3);
    let mut j = 0usize;
    let k8 = k & !7;
    while j < k8 {
        if j % 32 == 0 {
            prefetch_ahead(p0, j, k);
            prefetch_ahead(p1, j, k);
            prefetch_ahead(p2, j, k);
            prefetch_ahead(p3, j, k);
        }
        let c0 = cols_8x8(p0, j);
        let c1 = cols_8x8(p1, j);
        for t in 0..8 {
            let xp = x.add(j + t);
            a0 = madd8(a0, c0[t], xp);
            a1 = madd8(a1, c1[t], xp);
        }
        let c2 = cols_8x8(p2, j);
        let c3 = cols_8x8(p3, j);
        for t in 0..8 {
            let xp = x.add(j + t);
            a2 = madd8(a2, c2[t], xp);
            a3 = madd8(a3, c3[t], xp);
        }
        j += 8;
    }
    if j == k {
        return [a0, a1, a2, a3];
    }
    let mut tmp = [0.0f32; 32];
    _mm256_storeu_ps(tmp.as_mut_ptr(), a0);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), a1);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(16), a2);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(24), a3);
    while j < k {
        let xj = *x.add(j);
        for r in 0..32 {
            tmp[r] += load_bf16(rows[r].add(j * 2)) * xj;
        }
        j += 1;
    }
    [
        _mm256_loadu_ps(tmp.as_ptr()),
        _mm256_loadu_ps(tmp.as_ptr().add(8)),
        _mm256_loadu_ps(tmp.as_ptr().add(16)),
        _mm256_loadu_ps(tmp.as_ptr().add(24)),
    ]
}

/// Pull each row's line 64 BF16 ahead of `j`. Rows sit `k * 2` bytes apart.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prefetch_ahead(rows: [*const u8; 8], j: usize, k: usize) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    if prefetch_enabled() && j + 64 < k {
        for r in 0..8 {
            _mm_prefetch(rows[r].add((j + 64) * 2) as *const i8, _MM_HINT_T0);
        }
    }
}

/// 32 rows × this many BF16 columns. 256 × 32 × 2 = 16 KiB, one L1 tile.
#[cfg(target_arch = "x86_64")]
const PACK_COLS: usize = 256;

/// Column `j` is 32 little-endian BF16 (64 bytes). Group `g` of 8 rows is
/// at byte `g * 16` inside the column, in [`cols_8x8`] order.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
struct ColPanel {
    bytes: [u8; 32 * PACK_COLS * 2],
}

/// Pack path. Same row groups and the same left-to-right mul-then-add as
/// [`gemv_bf16_avx2`]. The tile scratch dies with the call.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2_packed(y: &mut [f32], w: &[u8], x: &[f32]) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    let m = y.len();
    let k = x.len();
    let row_bytes = k * 2;
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();
    let mut row = 0usize;
    while row + 32 <= m {
        let base = w_ptr.add(row * row_bytes);
        let mut rows = [base; 32];
        for r in 0..32 {
            rows[r] = base.add(r * row_bytes);
        }
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut j = 0usize;
        let k8 = k & !7;
        while j < k8 {
            let n = (k8 - j).min(PACK_COLS) & !7;
            debug_assert!(n >= 8);
            let mut panel = core::mem::MaybeUninit::<ColPanel>::uninit();
            pack_cols32(&rows, j, n, k, panel.as_mut_ptr() as *mut u8);
            acc = dot_packed_tile(panel.as_ptr() as *const u8, x_ptr.add(j), n, acc);
            j += n;
        }
        if j < k {
            let mut tmp = [0.0f32; 32];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc[0]);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc[1]);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(16), acc[2]);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(24), acc[3]);
            while j < k {
                let xj = *x_ptr.add(j);
                for r in 0..32 {
                    tmp[r] += load_bf16(rows[r].add(j * 2)) * xj;
                }
                j += 1;
            }
            _mm256_storeu_ps(y_ptr.add(row), _mm256_loadu_ps(tmp.as_ptr()));
            _mm256_storeu_ps(y_ptr.add(row + 8), _mm256_loadu_ps(tmp.as_ptr().add(8)));
            _mm256_storeu_ps(y_ptr.add(row + 16), _mm256_loadu_ps(tmp.as_ptr().add(16)));
            _mm256_storeu_ps(y_ptr.add(row + 24), _mm256_loadu_ps(tmp.as_ptr().add(24)));
        } else {
            _mm256_storeu_ps(y_ptr.add(row), acc[0]);
            _mm256_storeu_ps(y_ptr.add(row + 8), acc[1]);
            _mm256_storeu_ps(y_ptr.add(row + 16), acc[2]);
            _mm256_storeu_ps(y_ptr.add(row + 24), acc[3]);
        }
        row += 32;
    }
    gemv_bf16_avx2_from16(y, w, x, row);
}

/// Transpose `n` columns (a multiple of 8) of 32 rows into `dst`.
///
/// `rows[r]` points at column 0. Output column `t` is 64 bytes at
/// `dst + t * 64`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_cols32(rows: &[*const u8; 32], j0: usize, n: usize, k: usize, dst: *mut u8) {
    use core::arch::x86_64::_mm_store_si128;

    let mut panels = [[rows[0]; 8]; 4];
    for g in 0..4 {
        for i in 0..8 {
            panels[g][i] = rows[g * 8 + i];
        }
    }
    let mut t = 0usize;
    while t < n {
        if t % 32 == 0 {
            prefetch_ahead(panels[0], j0 + t, k);
            prefetch_ahead(panels[1], j0 + t, k);
            prefetch_ahead(panels[2], j0 + t, k);
            prefetch_ahead(panels[3], j0 + t, k);
        }
        for g in 0..4 {
            let cols = cols_8x8(panels[g], j0 + t);
            for u in 0..8 {
                let p = dst.add((t + u) * 64 + g * 16);
                _mm_store_si128(p as *mut _, cols[u]);
            }
        }
        t += 8;
    }
}

/// Continue the four YMM chains over a packed panel. Column order is `0..n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_packed_tile(
    col: *const u8,
    x: *const f32,
    n: usize,
    mut acc: [core::arch::x86_64::__m256; 4],
) -> [core::arch::x86_64::__m256; 4] {
    use core::arch::x86_64::_mm_load_si128;

    let mut j = 0usize;
    while j + 4 <= n {
        for t in 0..4 {
            let p = col.add((j + t) * 64);
            let xp = x.add(j + t);
            acc[0] = madd8(acc[0], _mm_load_si128(p as *const _), xp);
            acc[1] = madd8(acc[1], _mm_load_si128(p.add(16) as *const _), xp);
            acc[2] = madd8(acc[2], _mm_load_si128(p.add(32) as *const _), xp);
            acc[3] = madd8(acc[3], _mm_load_si128(p.add(48) as *const _), xp);
        }
        j += 4;
    }
    while j < n {
        let p = col.add(j * 64);
        let xp = x.add(j);
        acc[0] = madd8(acc[0], _mm_load_si128(p as *const _), xp);
        acc[1] = madd8(acc[1], _mm_load_si128(p.add(16) as *const _), xp);
        acc[2] = madd8(acc[2], _mm_load_si128(p.add(32) as *const _), xp);
        acc[3] = madd8(acc[3], _mm_load_si128(p.add(48) as *const _), xp);
        j += 1;
    }
    acc
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
            // Column-major tile scratch. Same bits, including across K tiles.
            let mut y_pack = vec![4.0f32; m];
            unsafe { gemv_bf16_avx2_packed(&mut y_pack, w, x) };
            assert_same_bits(&y_pack, &y_naive);
        }
    }

    #[test]
    fn panel_pack_flag_matches_scalar_and_restores() {
        let (w, x) = patterned(40, 100, 0x66);
        let mut y_naive = vec![0.0f32; 40];
        naive(&mut y_naive, &w, &x);
        set_gemv_bf16_panel_pack(true);
        let mut y = vec![1.0f32; 40];
        gemv_bf16(&mut y, &w, &x, 40, 100).unwrap();
        set_gemv_bf16_panel_pack(false);
        set_gemv_bf16_prefetch(false);
        assert_same_bits(&y, &y_naive);
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

        // 32 rows: one full YMM panel (four accumulators, eight lanes each).
        let mut wide = Vec::with_capacity(32 * w.len());
        for _ in 0..32 {
            wide.extend_from_slice(&w);
        }
        let mut y32 = vec![0.0f32; 32];
        gemv_bf16(&mut y32, &wide, &x, 32, 8).unwrap();
        assert!(y32.iter().all(|v| v.to_bits() == 3.0f32.to_bits()));
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
        #[cfg(target_arch = "x86_64")]
        {
            // Hot panel: 32×256 BF16 is 16 KiB, so this is the compute ceiling
            // rather than the big-shape bandwidth ceiling.
            let (m, k) = (32usize, 256usize);
            let (w, x) = normal_panel(m, k, 7);
            let mut y_fast = vec![0.0f32; m];
            let mut y_prev = vec![0.0f32; m];
            let mut y_naive = vec![0.0f32; m];
            let reps = 4000usize;
            let fast = elapsed_secs(3, || {
                for _ in 0..reps {
                    // threads=1 kernel. The public `gemv_bf16` path is the
                    // row pool; this bench stays the single-thread SIMD floor.
                    super::gemv_bf16_rows(&mut y_fast, &w, &x);
                }
                std::hint::black_box(&y_fast);
            });
            let prev = elapsed_secs(3, || unsafe {
                for _ in 0..reps {
                    super::gemv_bf16_avx2_from16(&mut y_prev, &w, &x, 0);
                }
                std::hint::black_box(&y_prev);
            });
            let slow = elapsed_secs(3, || {
                for _ in 0..reps {
                    naive(&mut y_naive, &w, &x);
                }
                std::hint::black_box(&y_naive);
            });
            let macs = (reps * m * k) as f64;
            eprintln!(
                "gemv_bf16 hot_32x256: naive {:.2} Gmac/s knife5a {:.2} fast {:.2} vs_naive {:.2}x vs_knife5a {:.2}x",
                macs / slow / 1e9,
                macs / prev / 1e9,
                macs / fast / 1e9,
                slow / fast,
                prev / fast
            );
        }
        for (m, k, name) in shapes {
            // Normal magnitudes. Random bit patterns include BF16 subnormals,
            // and subnormal fp32 math is microcoded, so it does not measure
            // the decode path. Qwen weights are normal BF16.
            let (w, x) = normal_panel(m, k, 0x51);
            let mut y_fast = vec![0.0f32; m];
            #[cfg(target_arch = "x86_64")]
            let mut y_prev = vec![0.0f32; m];
            let mut y_naive = vec![0.0f32; m];
            let fast = elapsed_secs(3, || {
                super::gemv_bf16_rows(&mut y_fast, &w, &x);
                std::hint::black_box(&y_fast);
            });
            // Knife 5a: the 16-row XMM kernel, still in this file as the tail.
            #[cfg(target_arch = "x86_64")]
            let prev = elapsed_secs(3, || unsafe {
                super::gemv_bf16_avx2_from16(&mut y_prev, &w, &x, 0);
                std::hint::black_box(&y_prev);
            });
            #[cfg(not(target_arch = "x86_64"))]
            let prev = slow;
            let slow = elapsed_secs(3, || {
                naive(&mut y_naive, &w, &x);
                std::hint::black_box(&y_naive);
            });
            assert_same_bits(&y_fast, &y_naive);
            #[cfg(target_arch = "x86_64")]
            assert_same_bits(&y_prev, &y_naive);
            #[cfg(target_arch = "x86_64")]
            let pack = if super::avx2_available() {
                let mut y_pack = vec![0.0f32; m];
                let pack = elapsed_secs(3, || unsafe {
                    super::gemv_bf16_avx2_packed(&mut y_pack, &w, &x);
                    std::hint::black_box(&y_pack);
                });
                assert_same_bits(&y_pack, &y_naive);
                pack
            } else {
                fast
            };
            #[cfg(not(target_arch = "x86_64"))]
            let pack = fast;
            // Alternate the two kernels so a cold first sample does not
            // decide the ratio. `fast` above is still the public path.
            #[cfg(target_arch = "x86_64")]
            let (legacy, paired) = if super::avx2_available() {
                let mut y_old = vec![0.0f32; m];
                let mut y_new = vec![0.0f32; m];
                unsafe { super::gemv_bf16_avx2_rows32(&mut y_old, &w, &x) };
                super::gemv_bf16_rows(&mut y_new, &w, &x);
                let reps = 8usize;
                let mut t_old = 0.0f64;
                let mut t_new = 0.0f64;
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    unsafe { super::gemv_bf16_avx2_rows32(&mut y_old, &w, &x) };
                    t_old += t0.elapsed().as_secs_f64();
                    let t1 = std::time::Instant::now();
                    super::gemv_bf16_rows(&mut y_new, &w, &x);
                    t_new += t1.elapsed().as_secs_f64();
                    std::hint::black_box(&y_old);
                    std::hint::black_box(&y_new);
                }
                assert_same_bits(&y_old, &y_naive);
                assert_same_bits(&y_new, &y_naive);
                (t_old / reps as f64, t_new / reps as f64)
            } else {
                (fast, fast)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (legacy, paired) = (fast, fast);
            let speedup = slow / fast;
            let vs_prev = prev / fast;
            let vs_k9 = legacy / paired;
            let macs = (m as f64) * (k as f64);
            eprintln!(
                "gemv_bf16 {name} [{m}, {k}]: naive {slow:.6}s knife5a {prev:.6}s knife9 {legacy:.6}s knife10 {paired:.6}s vs_naive {speedup:.2}x vs_knife5a {vs_prev:.2}x vs_knife9 {vs_k9:.2}x ({:.2} Gmac/s)",
                macs / paired / 1e9
            );
            eprintln!(
                "gemv_bf16 {name} pack [{m}, {k}]: {pack:.6}s vs_strided {:.2}x ({:.2} Gmac/s) — not the default",
                fast / pack,
                macs / pack / 1e9
            );
            // Floor, not the measured factor. This host's one-row loop is
            // already close to f32-add throughput. The printed line is the
            // number to quote. A regression back to that loop still fails.
            assert!(
                speedup >= 2.5,
                "{name} speedup {speedup:.2}x, want >= 2.5x versus the naive loop"
            );
        }
    }

    #[test]
    fn showdown_default_is_eight_threads_and_pack_starts_off() {
        assert_eq!(super::SHOWDOWN_THREADS, 8);
        assert_eq!(super::GEMV_THREADS_MAX, 64);
        // The atomic is initialized to the showdown width. Tests in this
        // file do not call `set_gemv_bf16_threads`.
        assert_eq!(super::gemv_bf16_threads(), 8);
    }

    #[test]
    fn row_spans_cover_qwen_shapes_without_gaps() {
        fn assert_cover(m: usize, threads: usize, expect: &[(usize, usize)]) {
            let spans = super::gemv_row_spans(m, threads);
            assert_eq!(spans.as_slice(), expect, "m={m} threads={threads}");
            if m == 0 {
                return;
            }
            assert_eq!(spans.as_slice().first().unwrap().0, 0);
            assert_eq!(spans.as_slice().last().unwrap().1, m);
            for pair in spans.as_slice().windows(2) {
                assert_eq!(pair[0].1, pair[1].0);
                assert!(pair[0].0 < pair[0].1);
            }
        }
        assert_cover(0, 8, &[]);
        assert_cover(1, 8, &[(0, 1)]);
        assert_cover(4, 8, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        // 32 rows is one panel, which is fewer panels than 8 workers, so the
        // split is by rows (4 each), not by panel.
        assert_cover(
            32,
            8,
            &[
                (0, 4),
                (4, 8),
                (8, 12),
                (12, 16),
                (16, 20),
                (20, 24),
                (24, 28),
                (28, 32),
            ],
        );
        let mut kv = [(0, 0); 8];
        for t in 0..8 {
            kv[t] = (t * 32, (t + 1) * 32);
        }
        assert_cover(256, 8, &kv);
        let mut q = [(0, 0); 8];
        for t in 0..8 {
            q[t] = (t * 192, (t + 1) * 192);
        }
        assert_cover(1536, 8, &q);
        let mut gate = [(0, 0); 8];
        for t in 0..8 {
            gate[t] = (t * 1120, (t + 1) * 1120);
        }
        assert_cover(8960, 8, &gate);
        // 151936 / 32 = 4748 panels. 4748 = 8*593 + 4, so the first four
        // workers take one extra panel. No tail rows.
        let wide = 594 * 32;
        let narrow = 593 * 32;
        let mut head = [(0, 0); 8];
        let mut row = 0;
        for t in 0..8 {
            let n = if t < 4 { wide } else { narrow };
            head[t] = (row, row + n);
            row += n;
        }
        assert_eq!(row, 151936);
        assert_cover(151936, 8, &head);
        assert_cover(65, 2, &[(0, 32), (32, 65)]);
        assert_cover(7, 3, &[(0, 3), (3, 5), (5, 7)]);
        assert_cover(40, 1, &[(0, 40)]);
        assert_cover(40, 0, &[(0, 40)]);
    }

    #[test]
    fn row_parallel_matches_scalar_cancellation_and_naive() {
        let big = 16_777_216.0f32;
        let pattern = [big, 1.0, 1.0, 1.0, -big, 1.0, 1.0, 1.0];
        let row = pack(&pattern);
        let x = [1.0f32; 8];
        for m in [1usize, 7, 8, 9, 17, 31, 32, 33, 48, 64, 256] {
            let mut w = Vec::with_capacity(m * row.len());
            for _ in 0..m {
                w.extend_from_slice(&row);
            }
            let mut y1 = vec![0.0f32; m];
            let mut y8 = vec![9.0f32; m];
            crate::bf16::gemv_bf16_at_threads(&mut y1, &w, &x, m, 8, 1).unwrap();
            crate::bf16::gemv_bf16_at_threads(&mut y8, &w, &x, m, 8, 8).unwrap();
            assert!(
                y1.iter().all(|v| v.to_bits() == 3.0f32.to_bits()),
                "m={m} serial lost the left-to-right chain"
            );
            assert_same_bits(&y8, &y1);
        }

        let shapes = [
            (15, 17),
            (32, 64),
            (33, 40),
            (48, 31),
            (64, 128),
            (96, 63),
            (128, 64),
            (256, 48),
        ];
        for threads in [1usize, 2, 3, 7, 8] {
            for (i, (m, k)) in shapes.iter().copied().enumerate() {
                let (w, x) = patterned(m, k, 0x1700 ^ (i as u32) ^ ((threads as u32) << 8));
                let mut expect = vec![0.0f32; m];
                naive(&mut expect, &w, &x);
                let mut got = vec![1.0f32; m];
                crate::bf16::gemv_bf16_at_threads(&mut got, &w, &x, m, k, threads).unwrap();
                assert_same_bits(&got, &expect);
            }
        }
    }

    #[test]
    fn parallel_callers_do_not_mix_rows() {
        let (w1, x1) = patterned(48, 32, 1);
        let (w2, x2) = patterned(48, 32, 2);
        let mut y1 = vec![0.0f32; 48];
        let mut y2 = vec![0.0f32; 48];
        let mut n1 = vec![0.0f32; 48];
        let mut n2 = vec![0.0f32; 48];
        naive(&mut n1, &w1, &x1);
        naive(&mut n2, &w2, &x2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                crate::bf16::gemv_bf16_at_threads(&mut y1, &w1, &x1, 48, 32, 8).unwrap();
            });
            scope.spawn(|| {
                crate::bf16::gemv_bf16_at_threads(&mut y2, &w2, &x2, 48, 32, 8).unwrap();
            });
        });
        assert_same_bits(&y1, &n1);
        assert_same_bits(&y2, &n2);
    }

    /// Knife 6 pack stays off. Turning it on under the row pool still matches
    /// the scalar chain. The flag is process-global; both pack and the
    /// strided kernel match `naive`, so a racing restore cannot change bits.
    #[test]
    fn packed_rows_under_eight_threads_match_naive() {
        let (w, x) = normal_panel(96, 80, 9);
        let mut expect = vec![0.0f32; 96];
        naive(&mut expect, &w, &x);
        super::set_gemv_bf16_panel_pack(true);
        let mut got = vec![0.0f32; 96];
        let result = crate::bf16::gemv_bf16_at_threads(&mut got, &w, &x, 96, 80, 8);
        super::set_gemv_bf16_panel_pack(false);
        result.unwrap();
        assert_same_bits(&got, &expect);
    }

    /// Soft record: threads=8 versus threads=1 on the gate/up shape.
    /// Not a ≥2× gate. Debug builds skip the timing.
    #[test]
    fn release_gate_threads8_versus_threads1() {
        if cfg!(debug_assertions) {
            return;
        }
        let (m, k) = (INTERMEDIATE_SIZE_QWEN25_1_5B, HIDDEN_SIZE_QWEN25_1_5B);
        let (w, x) = normal_panel(m, k, 0x77);
        let mut y1 = vec![0.0f32; m];
        let mut y8 = vec![0.0f32; m];
        let reps = 4usize;
        let t1 = elapsed_secs(3, || {
            for _ in 0..reps {
                crate::bf16::gemv_bf16_at_threads(&mut y1, &w, &x, m, k, 1).unwrap();
            }
            std::hint::black_box(&y1);
        });
        let t8 = elapsed_secs(3, || {
            for _ in 0..reps {
                crate::bf16::gemv_bf16_at_threads(&mut y8, &w, &x, m, k, 8).unwrap();
            }
            std::hint::black_box(&y8);
        });
        assert_same_bits(&y8, &y1);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        eprintln!(
            "knife7_gemv_threads gate_up [{m}, {k}]: threads1_x{reps} {t1:.6}s threads8_x{reps} {t8:.6}s speedup {:.3}x per_call_1 {:.6}s per_call_8 {:.6}s cores={cores}",
            t1 / t8,
            t1 / reps as f64,
            t8 / reps as f64
        );
    }
}
