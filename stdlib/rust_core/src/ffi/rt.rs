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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
// ============================================================================
// Math & Core Functions
// ============================================================================

/// A failed check prints its message with `puts` and then calls `abort`. Standard output is
/// buffered when it is a pipe, and `abort` does not flush it, so the message was lost. Flush,
/// then say which task stopped, then let the abort go on.
extern "C" fn vx_sigabrt_handler(_sig: libc::c_int) {
    unsafe {
        libc::fflush(ptr::null_mut());
        crate::ffi::tasks::write_site();
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}

#[no_mangle]
pub extern "C" fn vx_init_signals() {
    // A bad value stops the program now, and not at its first `drop`.
    crate::ffi::tasks::drop_wait_cap_ns();
    crate::ffi::crash_report::prepare();
    unsafe {
        // The first `backtrace` call of glibc loads libgcc_s, which allocates. Make it here, so
        // that the call in the SIGSEGV handler does not.
        let mut frame = [ptr::null_mut(); 1];
        libc::backtrace(frame.as_mut_ptr(), 1);
        libc::signal(
            libc::SIGABRT,
            vx_sigabrt_handler as *const () as libc::sighandler_t,
        );
    }
}

// ============================================================================
// Timing & Benchmarking
// ============================================================================

static mut BENCHMARK_START: Option<Instant> = None;

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

/// Sleeps until the clock of `vx_clock_ns` reaches `deadline_ns`. The sleep is a wait, so other
/// tasks can run on the thread of a task that sleeps. A cancel of the `tasks` block, or the limit
/// of `within` or of a `drop`, ends it early. Gives 1 when the clock reached the deadline, and 0
/// when the sleep ended early or the task was cancelled before it.
#[no_mangle]
pub extern "C" fn vx_sleep_until(deadline_ns: i64) -> i32 {
    use crate::ffi::tasks::vx_cancelled;
    use crate::ffi::wait::{vx_clock_ns, vx_wait_while_until};
    // Nothing sets this word.
    let word = std::sync::atomic::AtomicI64::new(0);
    // The cancel comes first, so that a loop of late sleeps in a cancelled task ends.
    loop {
        if vx_cancelled() != 0 {
            return 0;
        }
        if vx_clock_ns() >= deadline_ns {
            return 1;
        }
        unsafe { vx_wait_while_until(word.as_ptr(), 0, deadline_ns, 1) };
    }
}

#[no_mangle]
pub extern "C" fn vx_unix_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}

/// The CPU's counter that ticks at a constant rate: the time stamp counter on x86-64. The rate
/// is constant only on a CPU with an invariant TSC.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub extern "C" fn vx_cycles() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// The CPU's counter that ticks at a constant rate: the virtual count of the generic timer on
/// aarch64. `llvm.readcyclecounter` reads `pmccntr_el0` there, which Linux does not let user
/// code read. Without the `isb`, the CPU can read the counter early, so a later reading can be
/// smaller than an earlier one. Linux puts the same `isb` before its own read.
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub extern "C" fn vx_cycles() -> u64 {
    let count: u64;
    unsafe { std::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) count, options(nostack)) };
    count
}

/// How many ticks of `vx_cycles` there are in one second. On x86-64 the first call measures it
/// against `vx_clock_ns` for 10 ms. When the thread moved to a core whose counter is behind, the
/// counter went back. Then it measures again, at most three times.
#[cfg(target_arch = "x86_64")]
fn cycles_per_second() -> u64 {
    static RATE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RATE.get_or_init(|| {
        for _ in 0..3 {
            let (c0, t0) = cycles_and_clock_ns();
            let (mut c1, mut t1) = (c0, t0);
            while t1 - t0 < 10_000_000 {
                (c1, t1) = cycles_and_clock_ns();
            }
            if c1 > c0 {
                return ((c1 - c0) as u128 * 1_000_000_000 / (t1 - t0) as u128) as u64;
            }
        }
        panic!("vx_cycles: the time stamp counter did not go forward in three tries of 10 ms");
    })
}

/// A reading of `vx_cycles` and of `vx_clock_ns` at the same moment. The counter is read before
/// and after the clock, and the middle is taken. Of three tries, the one with the shortest gap
/// wins, so a thread switch between the reads does not change the rate much.
#[cfg(target_arch = "x86_64")]
fn cycles_and_clock_ns() -> (u64, i64) {
    let mut best = (u64::MAX, 0, 0);
    for _ in 0..3 {
        let before = vx_cycles();
        let ns = crate::ffi::wait::vx_clock_ns();
        // A counter that went back gives a very large gap, so this try loses.
        let gap = vx_cycles().wrapping_sub(before);
        if gap < best.0 {
            best = (gap, before.wrapping_add(gap / 2), ns);
        }
    }
    (best.1, best.2)
}

/// How many ticks of `vx_cycles` there are in one second, as `cntfrq_el0` gives it.
#[cfg(target_arch = "aarch64")]
fn cycles_per_second() -> u64 {
    let rate: u64;
    unsafe { std::arch::asm!("mrs {}, cntfrq_el0", out(reg) rate, options(nomem, nostack)) };
    assert!(
        rate > 0,
        "vx_cycles: cntfrq_el0 is 0, so the counter rate is not known"
    );
    rate
}

/// `cycles` ticks of `vx_cycles` in nanoseconds, or -1 when the answer does not fit in an `i64`.
#[no_mangle]
pub extern "C" fn vx_cycles_to_ns(cycles: u64) -> i64 {
    let ns = cycles as u128 * 1_000_000_000 / cycles_per_second() as u128;
    i64::try_from(ns).unwrap_or(-1)
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
pub extern "C" fn vx_advance_ptr_f32(p: *mut f32, offset: i32) -> *mut f32 {
    unsafe { p.add(offset as usize) }
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

#[cfg(test)]
mod cycle_counter_tests {
    use super::{cycles_per_second, vx_cycles, vx_cycles_to_ns};
    use crate::ffi::wait::vx_clock_ns;

    #[test]
    fn the_cycle_counter_goes_forward() {
        let mut last = vx_cycles();
        for _ in 0..1000 {
            let now = vx_cycles();
            assert!(now >= last, "the counter went back from {last} to {now}");
            last = now;
        }
        let start = vx_cycles();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(vx_cycles() > start, "the counter did not move in 1 ms");
    }

    #[test]
    fn one_second_of_ticks_is_one_billion_nanoseconds() {
        assert_eq!(vx_cycles_to_ns(0), 0);
        assert_eq!(vx_cycles_to_ns(cycles_per_second()), 1_000_000_000);
        assert_eq!(vx_cycles_to_ns(3 * cycles_per_second()), 3_000_000_000);
    }

    // Only a counter slower than about 2 GHz has a `u64` of ticks that is too many nanoseconds
    // for an `i64`. On a faster counter there is nothing to check.
    #[test]
    fn ticks_that_do_not_fit_in_an_i64_of_nanoseconds_give_minus_one() {
        let rate = cycles_per_second() as u128;
        let too_many = ((i64::MAX as u128 + 1) * rate).div_ceil(1_000_000_000);
        if let Ok(too_many) = u64::try_from(too_many) {
            assert_eq!(vx_cycles_to_ns(too_many), -1);
            assert!(vx_cycles_to_ns(too_many - 1) > i64::MAX - 1_000_000_000);
        }
    }

    // Two readings 10 ms apart, in nanoseconds, agree with `vx_clock_ns` to within 1%. A test
    // machine can stop the thread between the two reads of one end, so it gets three tries.
    #[test]
    fn ticks_in_nanoseconds_agree_with_the_clock() {
        let mut errors = Vec::new();
        for _ in 0..3 {
            let (c0, t0) = (vx_cycles(), vx_clock_ns());
            std::thread::sleep(std::time::Duration::from_millis(10));
            let (c1, t1) = (vx_cycles(), vx_clock_ns());
            let by_cycles = vx_cycles_to_ns(c1 - c0);
            let by_clock = t1 - t0;
            if (by_cycles - by_clock).abs() * 100 <= by_clock {
                return;
            }
            errors.push((by_cycles, by_clock));
        }
        panic!("(ns by the counter, ns by the clock) did not agree to 1%: {errors:?}");
    }
}
