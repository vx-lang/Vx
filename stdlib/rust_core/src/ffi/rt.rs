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

// ============================================================================
// Pseudo-random numbers
// ============================================================================
//
// SplitMix64, which is a multiply-xor-shift over a 64-bit counter. Cheap on
// purpose: this exists so a test can fill a matrix with values that are not all
// zero, and a zero-filled matrix hides indexing and transposition errors that
// varied input catches immediately.
//
// Deterministic and seedable, which matters more here than statistical quality:
// a differential test compares two execution paths on the *same* inputs, so the
// sequence has to repeat exactly. `vx_rand_seed` sets it; the default seed is
// fixed rather than taken from the clock, so a run is reproducible without
// asking for it.
//
// Not cryptographic, and not a substitute for a real generator if distribution
// quality ever matters. SplitMix64 passes BigCrush, which is far more than
// filling test matrices needs.

/// The generator's state. An atomic because a Vx program may fill buffers from
/// more than one thread, and a torn read here would be a data race rather than
/// merely a worse sequence.
static VX_RNG_STATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);

/// One SplitMix64 step: advance the counter, then scramble the value taken.
fn vx_next_u64() -> u64 {
    use std::sync::atomic::Ordering;
    let z = VX_RNG_STATE
        .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Set the seed, so a differential run sees the same sequence twice.
#[no_mangle]
pub extern "C" fn vx_rand_seed(seed: u64) {
    VX_RNG_STATE.store(seed, std::sync::atomic::Ordering::Relaxed);
}

/// A uniform `u64` over the whole range.
#[no_mangle]
pub extern "C" fn vx_rand_u64() -> u64 {
    vx_next_u64()
}

/// A uniform `i32` over the whole range, negatives included.
#[no_mangle]
pub extern "C" fn vx_rand_i32() -> i32 {
    (vx_next_u64() >> 32) as u32 as i32
}

/// A uniform `f32` in [0, 1).
///
/// Built from the top 24 bits rather than by dividing the integer, because f32
/// has 24 bits of mantissa: taking more would round and could return exactly
/// 1.0, which a caller scaling into a half-open range does not expect.
#[no_mangle]
pub extern "C" fn vx_rand_f32() -> f32 {
    ((vx_next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// A uniform `f64` in [0, 1), on the same terms with 53 bits.
#[no_mangle]
pub extern "C" fn vx_rand_f64() -> f64 {
    ((vx_next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

/// A uniform `f32` in [lo, hi), which is what filling a test matrix wants.
#[no_mangle]
pub extern "C" fn vx_rand_range_f32(lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * vx_rand_f32()
}
