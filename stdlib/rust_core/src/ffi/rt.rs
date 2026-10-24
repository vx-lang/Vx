//===- rt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the core runtime initialization routines for Vx.
// It manages standard library state, threading infrastructure, and any required
// setup or teardown steps that must occur before and after the execution of
// a compiled Vx program.
//
//===----------------------------------------------------------------------===//

use std::ffi::{c_char, CStr};
use std::ptr;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
// ============================================================================
// Math & Core Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vx_sigsegv_handler(sig: libc::c_int) {
    // A wild memory access can surface as either SIGSEGV (unmapped page) or
    // SIGBUS (e.g. a far out-of-range address on arm64); catch both so the crash
    // is always reported with a backtrace rather than a silent signal death.
    if sig == libc::SIGBUS {
        println!("Caught SIGBUS: Bus Error!");
    } else {
        println!("Caught SIGSEGV: Segmentation Fault!");
    }
    println!(
        "Backtrace:
{:#?}",
        std::backtrace::Backtrace::force_capture()
    );
    std::process::abort();
}

/// A failed check prints its message with `puts` and then calls `abort`. Standard output is
/// buffered when it is a pipe, and `abort` does not flush it, so the message was lost. Flush,
/// then let the abort go on.
extern "C" fn vx_sigabrt_handler(_sig: libc::c_int) {
    unsafe {
        libc::fflush(ptr::null_mut());
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}

#[no_mangle]
pub extern "C" fn vx_init_signals() {
    unsafe {
        libc::signal(
            libc::SIGABRT,
            vx_sigabrt_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGSEGV,
            vx_sigsegv_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGBUS,
            vx_sigsegv_handler as *const () as libc::sighandler_t,
        );
    }
}

#[no_mangle]
pub extern "C" fn vx_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn vx_expf(x: f32) -> f32 {
    x.exp()
}

#[no_mangle]
pub extern "C" fn vx_cosf(x: f32) -> f32 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn vx_sinf(x: f32) -> f32 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn vx_get_rope_freq(pos: i32, i: i32, head_size: i32) -> f32 {
    let freq = 10000.0f32.powf(-(i as f32) / (head_size as f32));
    (pos as f32) * freq
}

// ============================================================================
// Timing & Benchmarking
// ============================================================================

static mut BENCHMARK_START: Option<Instant> = None;
static mut GLOBAL_START: Option<Instant> = None;

#[no_mangle]
pub extern "C" fn start_benchmark() -> f32 {
    unsafe {
        BENCHMARK_START = Some(Instant::now());
    }
    0.0
}

#[no_mangle]
pub extern "C" fn end_benchmark() {
    unsafe {
        if let Some(start) = BENCHMARK_START {
            let elapsed = start.elapsed().as_secs_f32();
            println!("{}", elapsed);
        }
    }
}

/// Report one benchmark measurement as a single machine-readable line.
///
/// Printed from Rust because `println!` in Vx emits its format string literally and appends the
/// arguments after it (Vx#450), so a Vx-side `println!("{} {}", name, value)` cannot produce a
/// parseable record. The line is what `cargo vx-bench` reads and what CI gates the presence of;
/// the runner adds the commit and the machine, which a program cannot know about itself.
///
/// Shape: `vx-bench <name> <unit> <value>` -- one space-separated record, no interpolation, so it
/// survives being mixed into JIT chatter on stdout. `name` and `unit` must not contain spaces.
#[no_mangle]
pub extern "C" fn vx_bench_report(name: *const c_char, unit: *const c_char, value: f32) -> i32 {
    if name.is_null() || unit.is_null() {
        eprintln!("vx_bench_report: null name or unit");
        return 1;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    let unit = unsafe { CStr::from_ptr(unit) }.to_string_lossy();
    if name.contains(char::is_whitespace) || unit.contains(char::is_whitespace) {
        eprintln!("vx_bench_report: name and unit must not contain whitespace");
        return 1;
    }
    println!("vx-bench {name} {unit} {value}");
    0
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn vx_get_time() -> f32 {
    unsafe {
        if GLOBAL_START.is_none() {
            GLOBAL_START = Some(Instant::now());
        }
        GLOBAL_START.unwrap().elapsed().as_secs_f32()
    }
}

#[no_mangle]
pub extern "C" fn vx_sleep(seconds: f32) -> i32 {
    thread::sleep(Duration::from_secs_f32(seconds));
    0
}

#[no_mangle]
pub extern "C" fn vx_unix_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}
#[no_mangle]
pub extern "C" fn trace_start() {
    println!("[TRACE START] Event ID: 100");
}

#[no_mangle]
pub extern "C" fn trace_end() {
    println!("[TRACE END] Event ID: 100");
}

// ============================================================================
// Memory & Pointers
// ============================================================================

/// A zeroed buffer of `n` f32s, for code that then hands the pointer to
/// something expecting foreign memory -- a tensor view over it, say (#336).
///
/// The runtime already produces such buffers (`vx_load_weights` mmaps a model
/// and returns `*mut f32`); this is the same thing without a file behind it, so
/// the pointer-facing paths can be exercised on their own.
#[no_mangle]
pub extern "C" fn vx_alloc_f32(n: i32) -> *mut f32 {
    if n <= 0 {
        return ptr::null_mut();
    }
    let mut buf = vec![0.0f32; n as usize];
    let p = buf.as_mut_ptr();
    std::mem::forget(buf);
    p
}

/// Free a buffer from `vx_alloc_f32`. The length is required because the
/// allocation is a `Vec`, and reconstructing one needs the capacity it was
/// created with.
#[no_mangle]
pub extern "C" fn vx_free_f32(p: *mut f32, n: i32) {
    if p.is_null() || n <= 0 {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(p, n as usize, n as usize));
    }
}

#[no_mangle]
pub extern "C" fn vx_advance_ptr(p: *mut f32, offset: i32) -> *mut f32 {
    unsafe { p.add(offset as usize) }
}

#[no_mangle]
pub extern "C" fn vx_advance_ptr_const(p: *const f32, offset: i32) -> *const f32 {
    unsafe { p.add(offset as usize) }
}

#[no_mangle]
pub extern "C" fn vx_advance_ptr_f32(p: *mut f32, offset: i32) -> *mut f32 {
    unsafe { p.add(offset as usize) }
}

#[no_mangle]
pub extern "C" fn vx_memcpy(dest: *mut f32, src: *const f32, num_bytes: i32) -> i32 {
    unsafe {
        ptr::copy_nonoverlapping(src as *const u8, dest as *mut u8, num_bytes as usize);
    }
    0
}

#[no_mangle]
pub extern "C-unwind" fn vx_panic() -> i32 {
    println!("Vx runtime panic occurred!");
    panic!("vx_panic");
}

#[no_mangle]
pub extern "C-unwind" fn vx_catch_unwind(f: extern "C-unwind" fn() -> i32) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f()));
    match result {
        Ok(v) => v,
        Err(_) => {
            println!("Caught panic!");
            1
        }
    }
}
