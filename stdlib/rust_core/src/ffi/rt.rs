//===- rt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the core runtime initialization routines for Vx.
// It manages standard library state, threading infrastructure, and any required
// setup or teardown steps that must occur before and after the execution of
// a compiled Vx program.
//
//===----------------------------------------------------------------------===//

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

#[no_mangle]
pub extern "C" fn vx_init_signals() {
    unsafe {
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
