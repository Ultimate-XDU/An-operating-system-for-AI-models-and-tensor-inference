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

struct Task {
    y: *mut f32,
    rows: usize,
    w: *const u8,
    x: *const f32,
    k: usize,
    done: Arc<Done>,
}

// The caller keeps `y`/`w`/`x` alive until `Done` reaches zero, and each
// task writes a disjoint row span. `x` is only read.
unsafe impl Send for Task {}

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
/// `task.y` is `rows` exclusive f32s, `task.w` is `rows * k * 2` readable
/// bytes, and `task.x` is `k` readable f32s. Those ranges stay valid until
/// this function returns. No other thread writes them.
unsafe fn run_task(task: &Task) {
    let y = std::slice::from_raw_parts_mut(task.y, task.rows);
    let nbytes = task.rows.saturating_mul(task.k).saturating_mul(2);
    let w = std::slice::from_raw_parts(task.w, nbytes);
    let x = std::slice::from_raw_parts(task.x, task.k);
    gemv_bf16_rows(y, w, x);
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
            let task = Task {
                y: unsafe { y_ptr.add(start) },
                rows,
                w: unsafe { w_ptr.add(start * row_bytes) },
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
        unsafe {
            run_task(&Task {
                y: y_ptr.add(start),
                rows,
                w: w_ptr.add(start * row_bytes),
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
