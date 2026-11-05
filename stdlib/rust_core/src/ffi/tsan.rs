//===- tsan.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The calls of ThreadSanitizer that the runtime makes when the program runs under it
// (`vxc --sanitize=thread`). This library is not built for ThreadSanitizer, so it does not see
// the futexes under a Rust `Mutex`, or the switch of the shared runtime to the stack of a task.
// These calls tell it. Without ThreadSanitizer, the program has none of these functions, and
// each call does nothing.
//
//===----------------------------------------------------------------------===//

use std::ffi::{c_void, CStr};
use std::sync::OnceLock;

/// `__tsan_switch_to_fiber` does not order the two fibers.
const NO_SYNC: u32 = 1;

type Order = unsafe extern "C" fn(*const c_void);
type Current = unsafe extern "C" fn() -> *mut c_void;
type Create = unsafe extern "C" fn(u32) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Switch = unsafe extern "C" fn(*mut c_void, u32);

struct Hooks {
    acquire: Order,
    release: Order,
    current_fiber: Current,
    create_fiber: Create,
    destroy_fiber: Destroy,
    switch_to_fiber: Switch,
}

fn hooks() -> Option<&'static Hooks> {
    static HOOKS: OnceLock<Option<Hooks>> = OnceLock::new();
    HOOKS
        .get_or_init(|| {
            let find = |name: &CStr| unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
            let all = [
                find(c"__tsan_acquire"),
                find(c"__tsan_release"),
                find(c"__tsan_get_current_fiber"),
                find(c"__tsan_create_fiber"),
                find(c"__tsan_destroy_fiber"),
                find(c"__tsan_switch_to_fiber"),
            ];
            if all.iter().any(|f| f.is_null()) {
                return None;
            }
            unsafe {
                Some(Hooks {
                    acquire: std::mem::transmute::<*mut c_void, Order>(all[0]),
                    release: std::mem::transmute::<*mut c_void, Order>(all[1]),
                    current_fiber: std::mem::transmute::<*mut c_void, Current>(all[2]),
                    create_fiber: std::mem::transmute::<*mut c_void, Create>(all[3]),
                    destroy_fiber: std::mem::transmute::<*mut c_void, Destroy>(all[4]),
                    switch_to_fiber: std::mem::transmute::<*mut c_void, Switch>(all[5]),
                })
            }
        })
        .as_ref()
}

/// What the current thread or fiber did before a `release` at the same address happens before
/// what it does next.
pub(crate) fn acquire<T>(at: &T) {
    if let Some(h) = hooks() {
        unsafe { (h.acquire)(at as *const T as *const c_void) };
    }
}

/// The other half of `acquire`.
pub(crate) fn release<T>(at: &T) {
    if let Some(h) = hooks() {
        unsafe { (h.release)(at as *const T as *const c_void) };
    }
}

/// The fiber of ThreadSanitizer that runs now, or null without ThreadSanitizer.
pub(crate) fn current_fiber() -> *mut c_void {
    hooks().map_or(std::ptr::null_mut(), |h| unsafe { (h.current_fiber)() })
}

/// A new fiber of ThreadSanitizer for a task, or null without ThreadSanitizer. What the caller
/// did before happens before the start of the fiber.
pub(crate) fn create_fiber() -> *mut c_void {
    hooks().map_or(std::ptr::null_mut(), |h| unsafe { (h.create_fiber)(0) })
}

pub(crate) fn destroy_fiber(fiber: *mut c_void) {
    if let Some(h) = hooks() {
        unsafe { (h.destroy_fiber)(fiber) };
    }
}

/// Call it just before the switch of the stack to `fiber`. The switch does not order the two
/// fibers. Two tasks that run one after the other on one worker are not ordered, so a race
/// between them is still found.
pub(crate) fn switch_to_fiber(fiber: *mut c_void) {
    if let Some(h) = hooks() {
        unsafe { (h.switch_to_fiber)(fiber, NO_SYNC) };
    }
}
