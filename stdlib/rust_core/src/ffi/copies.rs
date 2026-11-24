//===- copies.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The end of the copy of a `transfer_async`, as a source of `select`. The key of a copy is the
// address that it fills.
//
// Only a backend that copies in the background knows when a copy ends. Such a backend exports
// `vx_backend_copy_ended` and `vx_backend_watch_copy`, and this file finds them when the program
// runs. A program with no such backend links nothing new, and each of its copies ended when
// `transfer_async` returned.
//
//===----------------------------------------------------------------------===//

use crate::ffi::wait::{vx_unwatch, vx_wake_watchers, vx_watch};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::OnceLock;

/// 1 when no copy into the address may still run.
type Ended = unsafe extern "C" fn(*const c_void) -> i32;
/// Calls the function once, with the address as its key, after the copy into the address ended.
/// It returns at once when no copy into the address runs.
type Watch = unsafe extern "C" fn(*const c_void, extern "C" fn(*const i64));

#[derive(Clone, Copy)]
struct Backend {
    ended: Ended,
    watch: Watch,
}

static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();

/// The watches of copies that did not end. While one is left, a wake can still come, so the
/// deadlock report waits.
static WATCHES: AtomicI64 = AtomicI64::new(0);

fn backend() -> Option<Backend> {
    *BACKEND.get_or_init(|| {
        let find =
            |name: &std::ffi::CStr| unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        let ended = find(c"vx_backend_copy_ended");
        let watch = find(c"vx_backend_watch_copy");
        if ended.is_null() || watch.is_null() {
            return None;
        }
        Some(unsafe {
            Backend {
                ended: std::mem::transmute::<*mut c_void, Ended>(ended),
                watch: std::mem::transmute::<*mut c_void, Watch>(watch),
            }
        })
    })
}

/// The key of the copy into `at`. Vx cannot turn a pointer into a number.
#[no_mangle]
pub extern "C" fn vx_copy_key(at: *const c_void) -> i64 {
    at as i64
}

/// 1 when no copy into the address `key` may still run.
#[no_mangle]
pub extern "C" fn vx_copy_ended(key: i64) -> i32 {
    match backend() {
        Some(b) => unsafe { (b.ended)(key as *const c_void) },
        None => 1,
    }
}

/// Until `vx_unwatch_copy`, the end of the copy into `key` adds 1 to `waiter` and wakes it.
///
/// # Safety
///
/// `waiter` points to a live `i64` until the `vx_unwatch_copy` of the pair.
#[no_mangle]
pub unsafe extern "C" fn vx_watch_copy(key: i64, waiter: *mut i64) {
    WATCHES.fetch_add(1, Ordering::SeqCst);
    unsafe { vx_watch(key as *const i64, waiter) };
    if let Some(b) = backend() {
        unsafe { (b.watch)(key as *const c_void, vx_wake_watchers) };
    }
}

/// Undo one `vx_watch_copy` of the same pair.
#[no_mangle]
pub extern "C" fn vx_unwatch_copy(key: i64, waiter: *mut i64) {
    vx_unwatch(key as *const i64, waiter);
    WATCHES.fetch_sub(1, Ordering::SeqCst);
}

/// Whether a `select` waits for the end of a copy.
pub(crate) fn pending() -> bool {
    WATCHES.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test process has no backend that copies in the background, so each copy ended. A
    /// watch keeps the deadlock report off until its unwatch.
    #[test]
    fn with_no_backend_a_copy_ended_and_a_watch_is_pending_until_its_unwatch() {
        let at = [0u8; 4];
        let key = vx_copy_key(at.as_ptr().cast());
        assert_eq!(key, at.as_ptr() as i64, "the key is not the address");
        assert_eq!(vx_copy_ended(key), 1, "a copy with no backend did not end");
        let mut waiter = 0i64;
        unsafe { vx_watch_copy(key, &mut waiter) };
        assert!(pending(), "a watch of a copy does not keep the report off");
        vx_wake_watchers(key as *const i64);
        assert_eq!(waiter, 1, "the end of the copy did not add 1 to the waiter");
        vx_unwatch_copy(key, &mut waiter);
        assert!(!pending(), "the unwatch left the copy pending");
    }
}
