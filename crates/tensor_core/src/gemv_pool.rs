//! Row partition for `gemv_bf16`.
//!
//! The numeric kernel ([`crate::gemv_bf16_kernel::gemv_bf16_rows`]) stays
//! free of `std` and is what each worker runs. This module only assigns
//! contiguous output rows and joins them. There is no cross-row reduction.
//!
//! The pool is fixed-size and grows up to [`crate::GEMV_THREADS_MAX`] minus
//! the caller. Worker `i` always receives span `i + 1` of the current call
//! (span 0 runs on the caller). Scheduling cannot reorder a row's
//! `j = 0..k-1` chain, because that chain never leaves its worker.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::gemv_bf16_kernel::gemv_bf16_rows;
use crate::gemv_row_spans;

/// One contiguous row span of one matrix. `rows == 0` is a hole (this
/// worker has no rows of that segment) and is skipped.
#[derive(Clone, Copy)]
struct Piece {
    y: *mut f32,
    w: *const u8,
    rows: usize,
}

/// Upper bound on projections that share one `x` in a single worker wave.
/// Q/K/V is 3 and gate/up is 2. A longer list falls back to one wave per
/// matrix inside [`crate::gemv_bf16_same_x`].
pub(crate) const GEMV_FUSED_MAX: usize = 8;

struct Task {
    pieces: [Piece; GEMV_FUSED_MAX],
    n: u8,
    x: *const f32,
    k: usize,
    done: Arc<Done>,
}

// The caller keeps every `y`/`w` and `x` alive until `Done` reaches zero.
// Pieces in one task are disjoint writes. Pieces across tasks of one call
// are disjoint row spans. `x` and every `w` are only read.
unsafe impl Send for Task {}

/// Borrowed segment for [`gemv_bf16_same_x_parallel`]. `y` is `rows` f32s
/// and `w` is `rows * x.len() * 2` bytes. Segments of one call must not
/// alias `y`.
#[derive(Clone, Copy)]
pub(crate) struct SegPtr {
    pub y: *mut f32,
    pub w: *const u8,
    pub rows: usize,
}

struct Done {
    left: AtomicUsize,
    failed: AtomicBool,
    mu: Mutex<()>,
    cv: Condvar,
}

impl Done {
    fn new(n: usize) -> Self {
        Self {
            left: AtomicUsize::new(n),
            failed: AtomicBool::new(false),
            mu: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    fn finish_one(&self) {
        let prev = self.left.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            let _guard = self.mu.lock().unwrap_or_else(|err| err.into_inner());
            self.cv.notify_one();
        }
    }

    fn wait(&self) {
        let mut guard = self.mu.lock().unwrap_or_else(|err| err.into_inner());
        while self.left.load(Ordering::Acquire) != 0 {
            guard = self.cv.wait(guard).unwrap_or_else(|err| err.into_inner());
        }
    }
}

struct Pool {
    txs: Vec<Sender<Task>>,
    joins: Vec<JoinHandle<()>>,
}

impl Pool {
    fn ensure(&mut self, workers: usize) {
        while self.txs.len() < workers {
            let (tx, rx) = mpsc::channel();
            let id = self.txs.len();
            let handle = thread::Builder::new()
                .name(format!("myos-gemv-{id}"))
                .spawn(move || worker_loop(rx))
                .unwrap_or_else(|err| panic!("spawn myos-gemv-{id}: {err}"));
            self.txs.push(tx);
            self.joins.push(handle);
        }
    }
}

fn worker_loop(rx: Receiver<Task>) {
    while let Ok(task) = rx.recv() {
        let failed = catch_unwind(AssertUnwindSafe(|| unsafe { run_task(&task) })).is_err();
        if failed {
            task.done.failed.store(true, Ordering::Release);
        }
        task.done.finish_one();
    }
}

/// # Safety
///
/// For each piece with `rows > 0`, `y` is `rows` exclusive f32s and `w` is
/// `rows * k * 2` readable bytes. `task.x` is `k` readable f32s. Those
/// ranges stay valid until this function returns. No other thread writes
/// the same `y` lanes. `k > 0`.
unsafe fn run_task(task: &Task) {
    let x = std::slice::from_raw_parts(task.x, task.k);
    let n = task.n as usize;
    for piece in task.pieces.iter().take(n) {
        if piece.rows == 0 {
            continue;
        }
        let y = std::slice::from_raw_parts_mut(piece.y, piece.rows);
        let nbytes = piece.rows.saturating_mul(task.k).saturating_mul(2);
        let w = std::slice::from_raw_parts(piece.w, nbytes);
        gemv_bf16_rows(y, w, x);
    }
}

fn blank_pieces() -> [Piece; GEMV_FUSED_MAX] {
    [Piece {
        y: core::ptr::null_mut(),
        w: core::ptr::null(),
        rows: 0,
    }; GEMV_FUSED_MAX]
}

fn with_senders<R>(workers: usize, body: impl FnOnce(&[Sender<Task>]) -> R) -> R {
    static POOL: Mutex<Pool> = Mutex::new(Pool {
        txs: Vec::new(),
        joins: Vec::new(),
    });
    let guard = POOL.lock().unwrap_or_else(|err| err.into_inner());
    // Clone under the lock, then drop it before the GEMV so another call can
    // enqueue. Workers do not take this mutex.
    let mut pool = guard;
    pool.ensure(workers);
    let txs: Vec<Sender<Task>> = pool.txs[..workers].to_vec();
    drop(pool);
    body(&txs)
}

/// Partition `y` into `threads` spans and run [`gemv_bf16_rows`] on each.
///
/// `w.len() == y.len() * x.len() * 2` and `x` is non-empty. Those are the
/// checks in [`crate::gemv_bf16`]. `threads <= 1`, or a single span, stays
/// on this thread and does not touch the pool.
pub(crate) fn gemv_bf16_rows_parallel(y: &mut [f32], w: &[u8], x: &[f32], threads: usize) {
    if y.is_empty() {
        return;
    }
    let spans = gemv_row_spans(y.len(), threads);
    let slice = spans.as_slice();
    if slice.len() <= 1 {
        gemv_bf16_rows(y, w, x);
        return;
    }
    let k = x.len();
    let row_bytes = match k.checked_mul(2) {
        Some(n) => n,
        None => {
            gemv_bf16_rows(y, w, x);
            return;
        }
    };
    debug_assert_eq!(slice[0].0, 0);
    debug_assert_eq!(slice.last().map(|s| s.1), Some(y.len()));

    let workers = slice.len() - 1;
    let done = Arc::new(Done::new(workers));
    let y_ptr = y.as_mut_ptr();
    let w_ptr = w.as_ptr();
    let x_ptr = x.as_ptr();

    with_senders(workers, |txs| {
        for (worker, &(start, end)) in slice.iter().skip(1).enumerate() {
            let rows = end - start;
            let mut pieces = blank_pieces();
            pieces[0] = Piece {
                y: unsafe { y_ptr.add(start) },
                w: unsafe { w_ptr.add(start * row_bytes) },
                rows,
            };
            let task = Task {
                pieces,
                n: 1,
                x: x_ptr,
                k,
                done: Arc::clone(&done),
            };
            txs[worker]
                .send(task)
                .unwrap_or_else(|_| panic!("myos-gemv-{worker} stopped"));
        }
        let (start, end) = slice[0];
        let rows = end - start;
        let mut pieces = blank_pieces();
        pieces[0] = Piece {
            y: unsafe { y_ptr.add(start) },
            w: unsafe { w_ptr.add(start * row_bytes) },
            rows,
        };
        unsafe {
            run_task(&Task {
                pieces,
                n: 1,
                x: x_ptr,
                k,
                done: Arc::clone(&done),
            });
        }
        done.wait();
    });
    if done.failed.load(Ordering::Acquire) {
        panic!("gemv worker panicked");
    }
}

/// One worker wave over several GEMVs that share `x`.
///
/// Worker `t` runs its [`gemv_row_spans`] slice of segment 0, then segment
/// 1, and so on, and only then joins. Each piece is the knife-9 row kernel:
/// `j` from `0` to `k - 1`, multiply then add, no cross-row reduction, no
/// pack, no software prefetch. `x` stays the vector those kernels already
/// reread from L1; this does not retile rows or the register block.
///
/// `segs` is empty, or each `w` is `rows * x.len() * 2` and `x` is non-empty.
/// `y` ranges in one call are disjoint. `threads <= 1` or a single span
/// stays on this thread.
pub(crate) fn gemv_bf16_same_x_parallel(segs: &[SegPtr], x: &[f32], threads: usize) {
    if segs.is_empty() || x.is_empty() {
        return;
    }
    let k = x.len();
    let row_bytes = match k.checked_mul(2) {
        Some(n) => n,
        None => return,
    };
    let mut nspan = 0usize;
    for seg in segs {
        nspan = nspan.max(gemv_row_spans(seg.rows, threads).len());
    }
    if nspan <= 1 {
        for seg in segs {
            if seg.rows == 0 {
                continue;
            }
            unsafe {
                run_task(&Task {
                    pieces: {
                        let mut pieces = blank_pieces();
                        pieces[0] = Piece {
                            y: seg.y,
                            w: seg.w,
                            rows: seg.rows,
                        };
                        pieces
                    },
                    n: 1,
                    x: x.as_ptr(),
                    k,
                    done: Arc::new(Done::new(0)),
                });
            }
        }
        return;
    }
    let workers = nspan - 1;
    let done = Arc::new(Done::new(workers));
    let x_ptr = x.as_ptr();
    let build = |worker: usize, done: Arc<Done>| -> Task {
        let mut pieces = blank_pieces();
        for (i, seg) in segs.iter().enumerate() {
            let spans = gemv_row_spans(seg.rows, threads);
            let slice = spans.as_slice();
            if worker >= slice.len() {
                continue;
            }
            let (start, end) = slice[worker];
            pieces[i] = Piece {
                y: unsafe { seg.y.add(start) },
                w: unsafe { seg.w.add(start * row_bytes) },
                rows: end - start,
            };
        }
        Task {
            pieces,
            n: segs.len() as u8,
            x: x_ptr,
            k,
            done,
        }
    };
    with_senders(workers, |txs| {
        for worker in 0..workers {
            let task = build(worker + 1, Arc::clone(&done));
            txs[worker]
                .send(task)
                .unwrap_or_else(|_| panic!("myos-gemv-{worker} stopped"));
        }
        unsafe {
            run_task(&build(0, Arc::clone(&done)));
        }
        done.wait();
    });
    if done.failed.load(Ordering::Acquire) {
        panic!("gemv worker panicked");
    }
}
