//===- stack.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Task stacks for the second runtime, where many tasks share a few OS threads
// (docs/discussions/waiting_tasks_channels_and_task_errors.md, W3).
//
// A stack never moves, because a task can borrow the locals of its parent. So each task that can
// park gets a fixed stack from a pool, with a guard page below it. `switch` saves the registers
// of the running code on its stack, and continues the code that another stack holds.
//
//===----------------------------------------------------------------------===//

use std::sync::Mutex;

fn page_bytes() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// One stack and the guard page below it. The kernel commits a page only when a task touches it.
pub struct Stack {
    /// The start of the mapping. The guard page is the first page.
    map: *mut u8,
    map_bytes: usize,
}

unsafe impl Send for Stack {}

impl Stack {
    /// The lowest address that a task can write.
    pub fn bottom(&self) -> *mut u8 {
        unsafe { self.map.add(page_bytes()) }
    }

    /// The address above the last byte. A stack grows down from here.
    pub fn top(&self) -> *mut u8 {
        unsafe { self.map.add(self.map_bytes) }
    }
}

/// The stacks that no task uses now.
pub struct StackPool {
    stack_bytes: usize,
    free: Mutex<Vec<Stack>>,
}

impl StackPool {
    /// A pool of stacks with room for `stack_bytes` each, rounded up to a page.
    pub fn new(stack_bytes: usize) -> StackPool {
        // The switch does not move a shadow stack yet. With shadow stacks on, the first switch to
        // a new stack would stop the program with a fault that names no cause.
        if shadow_stacks_are_on() {
            eprintln!("vx: the task stacks do not support shadow stacks yet");
            std::process::abort();
        }
        let page = page_bytes();
        StackPool {
            stack_bytes: stack_bytes.div_ceil(page) * page,
            free: Mutex::new(Vec::new()),
        }
    }

    /// A stack from the pool, or a new one. `None` when the kernel gives no more mappings.
    pub fn take(&self) -> Option<Stack> {
        if let Some(stack) = self.free.lock().unwrap().pop() {
            return Some(stack);
        }
        let page = page_bytes();
        let map_bytes = self.stack_bytes + page;
        unsafe {
            let map = libc::mmap(
                std::ptr::null_mut(),
                map_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                MAP_FLAGS,
                -1,
                0,
            );
            if map == libc::MAP_FAILED {
                return None;
            }
            let stack = Stack {
                map: map as *mut u8,
                map_bytes,
            };
            if libc::mprotect(map, page, libc::PROT_NONE) != 0 {
                drop_mapping(&stack);
                return None;
            }
            // A huge page would commit 2 MiB for a stack that uses a few KiB.
            #[cfg(target_os = "linux")]
            libc::madvise(
                stack.bottom() as *mut libc::c_void,
                self.stack_bytes,
                libc::MADV_NOHUGEPAGE,
            );
            Some(stack)
        }
    }

    /// Clears `stack` and keeps it for the next task. The next task cannot read the data of the
    /// last one, and the kernel gets the pages back.
    pub fn give_back(&self, stack: Stack) {
        clear(&stack, self.stack_bytes);
        self.free.lock().unwrap().push(stack);
    }
}

impl Drop for StackPool {
    fn drop(&mut self) {
        for stack in self.free.get_mut().unwrap().drain(..) {
            drop_mapping(&stack);
        }
    }
}

#[cfg(target_os = "linux")]
const MAP_FLAGS: libc::c_int =
    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_STACK;
#[cfg(not(target_os = "linux"))]
const MAP_FLAGS: libc::c_int = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

#[cfg(target_os = "linux")]
fn clear(stack: &Stack, bytes: usize) {
    let r = unsafe {
        libc::madvise(
            stack.bottom() as *mut libc::c_void,
            bytes,
            libc::MADV_DONTNEED,
        )
    };
    assert_eq!(r, 0, "madvise of a task stack failed");
}

/// macOS keeps the data of a page after `MADV_DONTNEED`. A new mapping in the same place clears it.
#[cfg(not(target_os = "linux"))]
fn clear(stack: &Stack, bytes: usize) {
    let map = unsafe {
        libc::mmap(
            stack.bottom() as *mut libc::c_void,
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            MAP_FLAGS | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    assert!(map != libc::MAP_FAILED, "mmap of a task stack failed");
}

fn drop_mapping(stack: &Stack) {
    let r = unsafe { libc::munmap(stack.map as *mut libc::c_void, stack.map_bytes) };
    assert_eq!(r, 0, "munmap of a task stack failed");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn shadow_stacks_are_on() -> bool {
    const ARCH_SHSTK_STATUS: libc::c_long = 0x5005;
    const ARCH_SHSTK_SHSTK: u64 = 1;
    let mut status: u64 = 0;
    // A kernel with no shadow stacks gives an error, and then they are off.
    let r = unsafe {
        libc::syscall(
            libc::SYS_arch_prctl,
            ARCH_SHSTK_STATUS,
            &mut status as *mut u64,
        )
    };
    r == 0 && status & ARCH_SHSTK_SHSTK != 0
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn shadow_stacks_are_on() -> bool {
    const PR_GET_SHADOW_STACK_STATUS: libc::c_int = 74;
    const PR_SHADOW_STACK_ENABLE: u64 = 1;
    let mut status: u64 = 0;
    let r = unsafe { libc::prctl(PR_GET_SHADOW_STACK_STATUS, &mut status as *mut u64, 0, 0, 0) };
    r == 0 && status & PR_SHADOW_STACK_ENABLE != 0
}

#[cfg(not(target_os = "linux"))]
fn shadow_stacks_are_on() -> bool {
    false
}

/// The function that the first switch to a new stack calls. It must never return: it ends with a
/// switch to another stack, and nothing switches back to it.
pub type StackEntry = unsafe extern "C" fn(arg: *mut u8) -> !;

/// Prepares `stack` so that the first `switch` to the result calls `entry(arg)` on it.
///
/// # Safety
/// The stack must stay mapped until `entry` switches away for the last time.
pub unsafe fn start_on(stack: &Stack, entry: StackEntry, arg: *mut u8) -> *mut u8 {
    let top = stack.top() as usize & !15;
    arch::first_frame(top as *mut usize, entry, arg)
}

/// Saves the registers that a call must keep, stores the stack pointer in `*from`, and continues
/// the code whose stack pointer is `to`. The call returns when another switch gives `*from` back.
///
/// # Safety
/// `to` must come from `start_on` or from an earlier `switch`, and no other code may continue it.
pub unsafe fn switch(from: *mut *mut u8, to: *mut u8) {
    arch::vx_stack_switch(from, to)
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use super::StackEntry;

    // The saved frame, from the stack pointer up: MXCSR and the x87 control word, r15, r14, r13,
    // r12, rbx, rbp, and the address that `ret` continues at.
    const SAVED_WORDS: usize = 8;

    pub unsafe fn first_frame(top: *mut usize, entry: StackEntry, arg: *mut u8) -> *mut u8 {
        // After the `ret` into `start`, the stack pointer is a multiple of 16, as a call needs.
        let frame = top.sub(SAVED_WORDS + 2);
        let words: [usize; SAVED_WORDS] = [
            0x037F_0000_1F80, // the default MXCSR, and the default x87 control word above it
            0,
            0,
            arg as usize,
            entry as *const () as usize,
            0,
            0,
            start as *const () as usize,
        ];
        std::ptr::copy_nonoverlapping(words.as_ptr(), frame, SAVED_WORDS);
        frame as *mut u8
    }

    #[unsafe(naked)]
    unsafe extern "C" fn start() {
        core::arch::naked_asm!("mov rdi, r13", "call r12", "ud2")
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn vx_stack_switch(from: *mut *mut u8, to: *mut u8) {
        core::arch::naked_asm!(
            "push rbp",
            "push rbx",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "sub rsp, 8",
            "stmxcsr [rsp]",
            "fnstcw [rsp + 4]",
            "mov [rdi], rsp",
            "mov rsp, rsi",
            "ldmxcsr [rsp]",
            "fldcw [rsp + 4]",
            "add rsp, 8",
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rbx",
            "pop rbp",
            "ret",
        )
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use super::StackEntry;

    // The saved frame, from the stack pointer up: x19 to x28, x29, x30 (the return address),
    // d8 to d15, and FPCR. 22 words, and two more keep the stack pointer a multiple of 16.
    const SAVED_WORDS: usize = 22;

    pub unsafe fn first_frame(top: *mut usize, entry: StackEntry, arg: *mut u8) -> *mut u8 {
        let frame = top.sub(SAVED_WORDS);
        std::ptr::write_bytes(frame, 0, SAVED_WORDS);
        *frame.add(0) = entry as *const () as usize; // x19
        *frame.add(1) = arg as usize; // x20
        *frame.add(11) = start as *const () as usize; // x30
        frame as *mut u8
    }

    #[unsafe(naked)]
    unsafe extern "C" fn start() {
        core::arch::naked_asm!("mov x0, x20", "blr x19", "brk #1")
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn vx_stack_switch(from: *mut *mut u8, to: *mut u8) {
        core::arch::naked_asm!(
            "sub sp, sp, #176",
            "stp x19, x20, [sp, #0]",
            "stp x21, x22, [sp, #16]",
            "stp x23, x24, [sp, #32]",
            "stp x25, x26, [sp, #48]",
            "stp x27, x28, [sp, #64]",
            "stp x29, x30, [sp, #80]",
            "stp d8, d9, [sp, #96]",
            "stp d10, d11, [sp, #112]",
            "stp d12, d13, [sp, #128]",
            "stp d14, d15, [sp, #144]",
            "mrs x9, fpcr",
            "str x9, [sp, #160]",
            "mov x9, sp",
            "str x9, [x0]",
            "mov sp, x1",
            "ldr x9, [sp, #160]",
            "msr fpcr, x9",
            "ldp x19, x20, [sp, #0]",
            "ldp x21, x22, [sp, #16]",
            "ldp x23, x24, [sp, #32]",
            "ldp x25, x26, [sp, #48]",
            "ldp x27, x28, [sp, #64]",
            "ldp x29, x30, [sp, #80]",
            "ldp d8, d9, [sp, #96]",
            "ldp d10, d11, [sp, #112]",
            "ldp d12, d13, [sp, #128]",
            "ldp d14, d15, [sp, #144]",
            "add sp, sp, #176",
            "ret",
        )
    }
}

// To run these tests for aarch64 on an x86_64 machine, install qemu-user and
// gcc-aarch64-linux-gnu, and set:
//   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
//   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
// Then run `cargo test --lib --target aarch64-unknown-linux-gnu stack` in stdlib/rust_core.
#[cfg(test)]
mod tests {
    use super::*;

    const STACK_BYTES: usize = 256 << 10;

    /// What a test task shares with the test.
    struct Switches {
        test: *mut u8,
        task: *mut u8,
        stack_bottom: usize,
        stack_top: usize,
        /// The address of a local of the task, which shows on which stack it ran.
        local_at: usize,
        rounds: u64,
        task_result: u64,
    }

    /// Keeps six integers and eight floats live across each call of `between`. They need more
    /// registers than a call may change, so a register that the switch does not keep changes
    /// the result. The floats need d8 to d15 on aarch64.
    fn live_values(rounds: u64, mut between: impl FnMut()) -> u64 {
        let (mut a, mut b, mut c, mut d, mut e, mut f) = (1u64, 2u64, 3u64, 5u64, 7u64, 11u64);
        let (mut g, mut h, mut i, mut j) = (1.5f64, 2.5f64, 3.5f64, 4.5f64);
        let (mut k, mut l, mut m, mut n) = (5.5f64, 6.5f64, 7.5f64, 8.5f64);
        for r in 0..rounds {
            a = a.wrapping_mul(31).wrapping_add(r);
            b = b.rotate_left(7) ^ a;
            c = c.wrapping_add(b).wrapping_mul(3);
            d ^= c.rotate_right(11);
            e = e.wrapping_sub(d) ^ r;
            f = f.wrapping_mul(5).wrapping_add(e);
            let t = (r % 7) as f64;
            g = g * 0.75 + t;
            h = h * 0.5 + g;
            i = i * 0.25 + h;
            j = j * 0.125 + i;
            k = k * 0.75 - j;
            l = l * 0.5 - k;
            m = m * 0.25 - l;
            n = n * 0.125 - m;
            between();
        }
        let floats = [g, h, i, j, k, l, m, n].map(f64::to_bits);
        [a, b, c, d, e, f]
            .into_iter()
            .chain(floats)
            .fold(0, |hash, v| hash.wrapping_mul(1_000_003) ^ v)
    }

    unsafe extern "C" fn count_rounds(arg: *mut u8) -> ! {
        let s = arg as *mut Switches;
        let local = 0u8;
        (*s).local_at = &local as *const u8 as usize;
        (*s).task_result = live_values((*s).rounds, || switch(&raw mut (*s).task, (*s).test));
        switch(&raw mut (*s).task, (*s).test);
        unreachable!("nothing continues a task that ended");
    }

    #[test]
    fn a_task_runs_on_its_own_stack_and_keeps_its_state_across_switches() {
        let pool = StackPool::new(STACK_BYTES);
        let stack = pool.take().expect("a stack");
        let mut s = Switches {
            test: std::ptr::null_mut(),
            task: std::ptr::null_mut(),
            stack_bottom: stack.bottom() as usize,
            stack_top: stack.top() as usize,
            local_at: 0,
            rounds: 1000,
            task_result: 0,
        };
        let rounds = s.rounds;
        let p = &raw mut s;
        let test_result = unsafe {
            (*p).task = start_on(&stack, count_rounds, p as *mut u8);
            live_values(rounds + 1, || switch(&raw mut (*p).test, (*p).task))
        };
        assert!(
            (s.stack_bottom..s.stack_top).contains(&s.local_at),
            "the task ran on another stack"
        );
        assert_eq!(
            s.task_result,
            live_values(rounds, || ()),
            "the task lost a value"
        );
        assert_eq!(
            test_result,
            live_values(rounds + 1, || ()),
            "the test lost a value"
        );
        pool.give_back(stack);
    }

    #[test]
    fn a_stack_that_goes_back_to_the_pool_is_cleared() {
        let pool = StackPool::new(STACK_BYTES);
        let stack = pool.take().expect("a stack");
        let bottom = stack.bottom();
        unsafe {
            std::ptr::write_bytes(bottom, 0xAB, STACK_BYTES);
        }
        pool.give_back(stack);
        let again = pool.take().expect("a stack");
        assert_eq!(again.bottom(), bottom, "the pool gives the same stack back");
        let bytes = unsafe { std::slice::from_raw_parts(again.bottom(), STACK_BYTES) };
        assert!(
            bytes.iter().all(|&b| b == 0),
            "the stack still holds data of the last task"
        );
        pool.give_back(again);
    }

    #[test]
    fn a_write_below_a_stack_faults_on_its_guard_page() {
        let pool = StackPool::new(STACK_BYTES);
        let stack = pool.take().expect("a stack");
        let below = unsafe { stack.bottom().sub(1) };
        // The write must fault, so it runs in a child process.
        let child = unsafe { libc::fork() };
        assert!(child >= 0, "fork failed");
        if child == 0 {
            unsafe {
                std::ptr::write_volatile(below, 1);
                libc::_exit(0);
            }
        }
        let mut status = 0;
        unsafe { libc::waitpid(child, &mut status, 0) };
        assert!(
            libc::WIFSIGNALED(status),
            "the write below the stack did not fault"
        );
        let signal = libc::WTERMSIG(status);
        assert!(
            signal == libc::SIGSEGV || signal == libc::SIGBUS,
            "signal {signal}"
        );
        pool.give_back(stack);
    }

    /// The rounding mode: x86_64 keeps it in MXCSR, and aarch64 in FPCR.
    #[cfg(target_arch = "x86_64")]
    mod fp {
        pub const DEFAULT: u64 = 0x1F80;
        pub const ROUND_TOWARD_ZERO: u64 = 0x6000;

        pub fn control() -> u64 {
            let mut csr: u32 = 0;
            unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut csr) };
            csr as u64
        }

        pub fn set_control(csr: u64) {
            let csr = csr as u32;
            unsafe { core::arch::asm!("ldmxcsr [{}]", in(reg) &csr) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod fp {
        pub const DEFAULT: u64 = 0;
        pub const ROUND_TOWARD_ZERO: u64 = 0b11 << 22;

        pub fn control() -> u64 {
            let fpcr: u64;
            unsafe { core::arch::asm!("mrs {}, fpcr", out(reg) fpcr) };
            fpcr
        }

        pub fn set_control(fpcr: u64) {
            unsafe { core::arch::asm!("msr fpcr, {}", in(reg) fpcr) };
        }
    }

    /// The rounding mode that a new task saw, and the stack pointers of the switch.
    struct Rounding {
        test: *mut u8,
        task: *mut u8,
        task_saw: u64,
    }

    unsafe extern "C" fn read_rounding(arg: *mut u8) -> ! {
        let r = arg as *mut Rounding;
        (*r).task_saw = fp::control();
        switch(&raw mut (*r).task, (*r).test);
        unreachable!("nothing continues a task that ended");
    }

    /// A task starts with the default rounding mode, and a switch keeps the rounding mode of
    /// each stack.
    #[test]
    fn each_stack_keeps_its_own_rounding_mode() {
        let pool = StackPool::new(STACK_BYTES);
        let stack = pool.take().expect("a stack");
        let mut r = Rounding {
            test: std::ptr::null_mut(),
            task: std::ptr::null_mut(),
            task_saw: 0,
        };
        let p = &raw mut r;
        let before = fp::control();
        fp::set_control(before | fp::ROUND_TOWARD_ZERO);
        unsafe {
            (*p).task = start_on(&stack, read_rounding, p as *mut u8);
            switch(&raw mut (*p).test, (*p).task);
        }
        let after = fp::control();
        fp::set_control(before);
        assert_eq!(
            r.task_saw,
            fp::DEFAULT,
            "a new task did not start with the default rounding mode"
        );
        assert_eq!(
            after,
            before | fp::ROUND_TOWARD_ZERO,
            "the switch back lost the rounding mode"
        );
        pool.give_back(stack);
    }
}
