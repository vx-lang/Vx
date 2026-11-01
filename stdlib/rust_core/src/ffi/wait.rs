//===- wait.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A thread waits until an atomic word changes, and another thread wakes it. The channel and
// the `Mutex` of `std` use these two calls to block an OS thread while they wait
// (docs/discussions/waiting_tasks_channels_and_task_errors.md, W2 and M7).
//
// The words live in Vx memory. The runtime keeps a fixed table of buckets, each a lock and a
// condition variable, and a word uses the bucket of its address. A waiter takes the lock, reads
// the word, and sleeps only if the word still holds the value that it saw. A waker changes the
// word before it calls `vx_wake_all`, which takes the same lock. So a wake can never come
// between the read and the sleep, and no wake is lost.
//
// A task of the shared runtime (sched.rs) does not sleep: it parks. Under the lock it goes on
// the list of fibers of the bucket, and a wake puts each fiber of the list back on the queue.
//
//===----------------------------------------------------------------------===//

use crate::ffi::sched::{self, FiberRef};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex};

const BUCKETS: usize = 64;

struct Bucket {
    /// The fibers that park on a word of this bucket.
    lock: Mutex<Vec<FiberRef>>,
    cond: Condvar,
}

static TABLE: [Bucket; BUCKETS] = [const {
    Bucket {
        lock: Mutex::new(Vec::new()),
        cond: Condvar::new(),
    }
}; BUCKETS];

fn bucket(word: *const i64) -> &'static Bucket {
    // A word is 8-byte aligned, so the three low bits of its address are always 0.
    &TABLE[(word as usize >> 3) % BUCKETS]
}

/// Sleep while the atomic word at `word` holds `value`. A task of the shared runtime parks in
/// place of the sleep. The call can return early, when another word of the same bucket changes.
/// So the caller reads its word again and loops.
///
/// # Safety
///
/// `word` points to a live, 8-byte aligned `i64` that other threads change only atomically.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while(word: *const i64, value: i64) {
    wait(word, value, false);
}

/// Sleeps or parks while the word holds `value`, and with `or_cancel`, while the block of this
/// task or thread was not cancelled.
unsafe fn wait(word: *const i64, value: i64, or_cancel: bool) {
    let b = bucket(word);
    let mut fibers = b.lock.lock().unwrap_or_else(|e| e.into_inner());
    let atomic = unsafe { &*(word as *const AtomicI64) };
    if atomic.load(Ordering::SeqCst) != value
        || (or_cancel && crate::ffi::tasks::vx_cancelled() != 0)
    {
        return;
    }
    let fiber = sched::current_fiber();
    if fiber.is_null() {
        drop(b.cond.wait(fibers).unwrap_or_else(|e| e.into_inner()));
        return;
    }
    sched::will_park(fiber);
    fibers.push(FiberRef(fiber));
    drop(fibers);
    sched::park();
}

/// As `vx_wait_while`, and also return when the block of this thread was cancelled. The cancel
/// is read under the lock of the bucket, and a cancel takes each lock after it sets its mark. So
/// a cancel cannot come between the check and the sleep.
///
/// # Safety
///
/// As for `vx_wait_while`.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while_or_cancel(word: *const i64, value: i64) {
    wait(word, value, true);
}

/// Wake each thread that waits on the word at `word`. Call it after the change to the word.
///
/// # Safety
///
/// `word` is an address that `vx_wait_while` may have been called with. It is not read.
#[no_mangle]
pub unsafe extern "C" fn vx_wake_all(word: *const i64) {
    wake(bucket(word));
}

/// Wake each thread that waits on any word. A cancel uses it: each wait reads its word again,
/// and a wait that can be cancelled then sees the cancel.
pub(crate) fn wake_every_thread() {
    for b in &TABLE {
        wake(b);
    }
}

fn wake(b: &Bucket) {
    let fibers = std::mem::take(&mut *b.lock.lock().unwrap_or_else(|e| e.into_inner()));
    b.cond.notify_all();
    for fiber in fibers {
        sched::wake(fiber);
    }
}

/// Two words for `select`, which waits on several channels at once. Each channel event adds 1 to
/// the first word, and the second word counts the threads in a `select` that sleep on it.
static ANY_CHANNEL: [AtomicI64; 2] = [AtomicI64::new(0), AtomicI64::new(0)];

/// The address of the two words for `select`.
#[no_mangle]
pub extern "C" fn vx_any_channel_words() -> *mut i64 {
    ANY_CHANNEL.as_ptr() as *mut i64
}

/// The send numbers that a cancelled `send` left without a value, with the address of the words
/// of their channel. The `recv` of such a number passes it. A cancel is rare, so one lock for all
/// channels is enough.
static SKIPS: Mutex<Vec<(usize, i64)>> = Mutex::new(Vec::new());

fn skips() -> std::sync::MutexGuard<'static, Vec<(usize, i64)>> {
    SKIPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A cancelled `send` of the channel at `words` left the send number `k` without a value.
#[no_mangle]
pub extern "C" fn vx_channel_skip(words: *mut i64, k: i64) {
    let mut s = skips();
    assert!(
        !s.contains(&(words as usize, k)),
        "a send number was skipped twice"
    );
    s.push((words as usize, k));
}

/// 1 when a cancelled `send` left the send number `k` without a value, and then forget it.
#[no_mangle]
pub extern "C" fn vx_channel_take_skip(words: *mut i64, k: i64) -> i32 {
    let mut s = skips();
    match s.iter().position(|&e| e == (words as usize, k)) {
        Some(i) => {
            s.swap_remove(i);
            1
        }
        None => 0,
    }
}

/// The channel at `words` is freed. Forget each of its numbers that no `recv` passed.
#[no_mangle]
pub extern "C" fn vx_channel_forget_skips(words: *mut i64) {
    skips().retain(|&(w, _)| w != words as usize);
}
