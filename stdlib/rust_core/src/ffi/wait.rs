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
//===----------------------------------------------------------------------===//

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex};

const BUCKETS: usize = 64;

struct Bucket {
    lock: Mutex<()>,
    cond: Condvar,
}

static TABLE: [Bucket; BUCKETS] = [const {
    Bucket {
        lock: Mutex::new(()),
        cond: Condvar::new(),
    }
}; BUCKETS];

fn bucket(word: *const i64) -> &'static Bucket {
    // A word is 8-byte aligned, so the three low bits of its address are always 0.
    &TABLE[(word as usize >> 3) % BUCKETS]
}

/// Sleep while the atomic word at `word` holds `value`. The call can return early, when another
/// word of the same bucket changes. So the caller reads its word again and loops.
///
/// # Safety
///
/// `word` points to a live, 8-byte aligned `i64` that other threads change only atomically.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while(word: *const i64, value: i64) {
    let b = bucket(word);
    let guard = b.lock.lock().unwrap_or_else(|e| e.into_inner());
    let atomic = unsafe { &*(word as *const AtomicI64) };
    if atomic.load(Ordering::SeqCst) == value {
        drop(b.cond.wait(guard).unwrap_or_else(|e| e.into_inner()));
    }
}

/// Wake each thread that waits on the word at `word`. Call it after the change to the word.
///
/// # Safety
///
/// `word` is an address that `vx_wait_while` may have been called with. It is not read.
#[no_mangle]
pub unsafe extern "C" fn vx_wake_all(word: *const i64) {
    let b = bucket(word);
    drop(b.lock.lock().unwrap_or_else(|e| e.into_inner()));
    b.cond.notify_all();
}
