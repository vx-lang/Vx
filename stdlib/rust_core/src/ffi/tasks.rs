//===- tasks.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OS threads for `tasks { .. }` and `spawn { .. }`
// (docs/discussions/cpu_tasks_send_sync_and_shared_atomics.md, T1 to T7).
//
// A `tasks` block is a group. Each `spawn` starts one OS thread that calls `run(env)`. The
// compiled code makes `env` with `malloc`: it holds what the task copied at its `spawn`, and a
// slot for the result. After `vx_task_join`, the compiled code reads the result from `env`. At
// the end of the block, `vx_tasks_end` joins each task that is not joined yet, and frees each
// `env`.
//
// Each block has a cancel mark (F3 to F7 in waiting_tasks_channels_and_task_errors.md). The
// threads of its tasks, and the thread that runs the code of the block, see it as their current
// mark. A block in a task has the mark of the task's block as its parent, so a cancel reaches it.
// A wait that can be cancelled asks `vx_cancelled`, and a cancel wakes each sleeping thread.
//
// With VX_TASK_RUNTIME=shared, a task is a fiber of the shared runtime in sched.rs, not a
// thread. A join then waits on the word that says that the fiber ended.
//
//===----------------------------------------------------------------------===//

use crate::ffi::sched::{self, Fiber, Scheduler};
use crate::io::lines;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// The function that a task runs.
pub type VxTaskFn = extern "C" fn(*mut u8);

/// The stack of a task. The main thread of a Linux program gets 8 MiB, and a task runs the same
/// kind of code.
pub(crate) const TASK_STACK_BYTES: usize = 8 << 20;

/// The environment of a task, which a thread can take. The compiled code makes sure that only
/// the task uses it until the join.
struct Env(*mut u8);
unsafe impl Send for Env {}

struct Task {
    /// `None` when the task ended and was joined, or when it is a fiber.
    thread: Option<JoinHandle<()>>,
    /// The task in the shared runtime.
    fiber: Option<Box<Fiber>>,
    env: *mut u8,
}

/// The cancel mark of a `tasks` block.
pub(crate) struct CancelMark {
    set: AtomicBool,
    /// The mark of the block that the block is in, when it is in a task or in the code of a block.
    parent: Option<Arc<CancelMark>>,
}

impl CancelMark {
    fn is_set(&self) -> bool {
        self.set.load(Ordering::SeqCst) || self.parent.as_ref().is_some_and(|p| p.is_set())
    }

    fn cancel(&self) {
        self.set.store(true, Ordering::SeqCst);
        crate::ffi::wait::wake_every_thread();
    }
}

thread_local! {
    /// The mark of the innermost block whose task or code this thread runs.
    static CURRENT: RefCell<Option<Arc<CancelMark>>> = const { RefCell::new(None) };
    /// The marks that `vx_cancel_off` put aside, innermost last.
    static PUT_ASIDE: RefCell<Vec<Option<Arc<CancelMark>>>> = const { RefCell::new(Vec::new()) };
}

/// Before a `drop`: this thread runs with no mark until `vx_cancel_on`, so no wait gives
/// `Cancelled` (F6). A `tasks` block in the `drop` gets a mark with no parent.
#[no_mangle]
pub extern "C" fn vx_cancel_off() {
    let mark = CURRENT.with(|c| c.borrow_mut().take());
    PUT_ASIDE.with(|p| p.borrow_mut().push(mark));
}

/// After a `drop`: the mark that `vx_cancel_off` put aside comes back.
#[no_mangle]
pub extern "C" fn vx_cancel_on() {
    let mark = PUT_ASIDE.with(|p| p.borrow_mut().pop());
    let mark = mark.expect("vx_cancel_on comes after vx_cancel_off");
    CURRENT.with(|c| {
        assert!(c.borrow().is_none(), "a tasks block in a drop did not end");
        *c.borrow_mut() = mark;
    });
}

/// Swaps the current mark of this thread with `mark`. The shared runtime gives each fiber its own.
pub(crate) fn swap_current_mark(mark: &mut Option<Arc<CancelMark>>) {
    CURRENT.with(|c| std::mem::swap(&mut *c.borrow_mut(), mark));
}

/// The tasks of one `tasks` block.
pub struct VxTaskGroup {
    tasks: Vec<Task>,
    /// The shared runtime that runs the tasks, or `None` when each task gets its own thread.
    sched: Option<&'static Scheduler>,
    mark: Arc<CancelMark>,
    /// The current mark of this thread before the block, which comes back at its end.
    outer: Option<Arc<CancelMark>>,
}

fn wait(task: &mut Task) {
    if let Some(fiber) = &task.fiber {
        let done = &raw const fiber.done as *const i64;
        while fiber.done.load(Ordering::SeqCst) == 0 {
            unsafe { crate::ffi::wait::vx_wait_while(done, 0) };
        }
    }
    if let Some(thread) = task.thread.take() {
        // A Vx panic stops the program (T6). A Rust panic in a task is a bug of the runtime.
        if thread.join().is_err() {
            eprintln!("vx: a task stopped with a panic of the runtime");
            std::process::abort();
        }
    }
}

/// The start of a `tasks` block.
#[no_mangle]
pub extern "C" fn vx_tasks_begin() -> *mut VxTaskGroup {
    tasks_begin(sched::program_scheduler())
}

pub(crate) fn tasks_begin(sched: Option<&'static Scheduler>) -> *mut VxTaskGroup {
    let outer = CURRENT.with(|c| c.borrow().clone());
    let mark = Arc::new(CancelMark {
        set: AtomicBool::new(false),
        parent: outer.clone(),
    });
    CURRENT.with(|c| *c.borrow_mut() = Some(mark.clone()));
    Box::into_raw(Box::new(VxTaskGroup {
        tasks: Vec::new(),
        sched,
        mark,
        outer,
    }))
}

/// Whether the block of this thread's task or code, or a block around it, was cancelled. A wait
/// that can be cancelled then gives `Cancelled`.
#[no_mangle]
pub extern "C" fn vx_cancelled() -> i32 {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|m| m.is_set())) as i32
}

/// A task or the code of a block failed: cancel its block (F3).
#[no_mangle]
pub extern "C" fn vx_task_failed() {
    if let Some(mark) = CURRENT.with(|c| c.borrow().clone()) {
        mark.cancel();
    }
}

/// A `spawn`: starts `run(env)` on a new thread, and returns the task's number in `group`.
/// The group frees `env` with `free` at the end of its block.
///
/// # Safety
/// `group` comes from `vx_tasks_begin` and its block did not end. `env` comes from `malloc`,
/// and only the task uses it until its join.
#[no_mangle]
pub unsafe extern "C" fn vx_task_start(
    group: *mut VxTaskGroup,
    run: VxTaskFn,
    env: *mut u8,
) -> i64 {
    let group = &mut *group;
    let mark = group.mark.clone();
    lines::task_started();
    if let Some(sched) = group.sched {
        group.tasks.push(Task {
            thread: None,
            fiber: Some(sched.spawn(run, env, Some(mark))),
            env,
        });
        return (group.tasks.len() - 1) as i64;
    }
    let task_env = Env(env);
    let started = thread::Builder::new()
        .stack_size(TASK_STACK_BYTES)
        .spawn(move || {
            let task_env = task_env;
            CURRENT.with(|c| *c.borrow_mut() = Some(mark));
            run(task_env.0);
            lines::task_ended();
        });
    // A task that waits for a later task, on a channel or a lock, would wait forever if it ran
    // here. So with no thread, the program stops (W6 in waiting_tasks_channels_and_task_errors.md).
    let thread = match started {
        Ok(thread) => thread,
        Err(e) => {
            eprintln!("vx: the system could not start a thread for a task: {e}");
            std::process::abort();
        }
    };
    group.tasks.push(Task {
        thread: Some(thread),
        fiber: None,
        env,
    });
    (group.tasks.len() - 1) as i64
}

/// `h.join()`: waits until task `task` of `group` ends. Then its result is in its `env`. The
/// compiled code may join a task more than once. Only the first join waits. A null `group` is a
/// task that did not start, and the join does nothing.
///
/// # Safety
/// `group` is null, or comes from `vx_tasks_begin` and `task` from `vx_task_start` on it.
#[no_mangle]
pub unsafe extern "C" fn vx_task_join(group: *mut VxTaskGroup, task: i64) {
    if group.is_null() {
        return;
    }
    let group = &mut *group;
    let Some(task) = group.tasks.get_mut(task as usize) else {
        eprintln!("vx: a join of a task that is not in this tasks block");
        std::process::abort();
    };
    wait(task);
}

/// A `return`, `break` or `continue` that leaves a `tasks` block: joins each task that is not
/// joined yet, so that the drops before the exit run with no task running. `vx_tasks_end`
/// still ends the block.
///
/// # Safety
/// `group` comes from `vx_tasks_begin`, and its block did not end.
#[no_mangle]
pub unsafe extern "C" fn vx_tasks_wait(group: *mut VxTaskGroup) {
    let group = &mut *group;
    // Leaving the block early cancels its tasks, and then joins them (F7).
    group.mark.cancel();
    for task in &mut group.tasks {
        wait(task);
    }
}

/// The end of a `tasks` block: joins each task that is not joined yet, in the order of their
/// `spawn`s. Then frees each `env`, and `group`.
///
/// # Safety
/// `group` comes from `vx_tasks_begin`, and this is the end of its block.
#[no_mangle]
pub unsafe extern "C" fn vx_tasks_end(group: *mut VxTaskGroup) {
    let mut group = Box::from_raw(group);
    for task in &mut group.tasks {
        wait(task);
    }
    for task in &group.tasks {
        libc::free(task.env as *mut libc::c_void);
    }
    let outer = group.outer.take();
    CURRENT.with(|c| *c.borrow_mut() = outer);
    lines::flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Threads that wait until `need` of them arrived, for at most 10 seconds.
    struct Gate {
        arrived: AtomicUsize,
        need: usize,
    }

    impl Gate {
        /// True when all arrived in time.
        fn arrive_and_wait(&self) -> bool {
            self.arrived.fetch_add(1, Ordering::SeqCst);
            let end = Instant::now() + Duration::from_secs(10);
            while self.arrived.load(Ordering::SeqCst) < self.need {
                if Instant::now() > end {
                    return false;
                }
                thread::yield_now();
            }
            true
        }
    }

    /// The environment that the tests give their tasks, from `malloc` as in compiled code.
    #[repr(C)]
    struct TestEnv {
        input: usize,
        result: usize,
        gate: *const Gate,
        /// Counts the tasks that ended.
        ended: *const AtomicUsize,
        sleep_ms: u64,
    }

    /// The result is `input * 2`, or 0 when the task waited at its gate in vain.
    extern "C" fn double(env: *mut u8) {
        let env = unsafe { &mut *(env as *mut TestEnv) };
        std::thread::sleep(Duration::from_millis(env.sleep_ms));
        let in_time = env.gate.is_null() || unsafe { &*env.gate }.arrive_and_wait();
        env.result = if in_time { env.input * 2 } else { 0 };
        unsafe { &*env.ended }.fetch_add(1, Ordering::SeqCst);
    }

    fn env(input: usize, gate: *const Gate, ended: &AtomicUsize, sleep_ms: u64) -> *mut TestEnv {
        unsafe {
            let e = libc::malloc(std::mem::size_of::<TestEnv>()) as *mut TestEnv;
            e.write(TestEnv {
                input,
                result: 0,
                gate,
                ended,
                sleep_ms,
            });
            e
        }
    }

    #[test]
    fn a_task_runs_at_the_same_time_as_the_code_after_its_spawn() {
        // The task waits until this thread arrives, after the spawn. A task that ran to its end
        // inside `vx_task_start` would wait in vain and give 0.
        let gate = Gate {
            arrived: AtomicUsize::new(0),
            need: 2,
        };
        let ended = AtomicUsize::new(0);
        let group = vx_tasks_begin();
        let e = env(21, &gate, &ended, 0);
        unsafe {
            let task = vx_task_start(group, double, e as *mut u8);
            assert!(gate.arrive_and_wait());
            vx_task_join(group, task);
            assert_eq!((*e).result, 42);
            // A second join does not wait again.
            vx_task_join(group, task);
            vx_tasks_end(group);
        }
        assert_eq!(ended.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_end_of_the_block_waits_for_the_tasks_that_nothing_joined() {
        // Each task sleeps before it ends, so an end that did not wait would see fewer than 8.
        let ended = AtomicUsize::new(0);
        let group = vx_tasks_begin();
        let envs: Vec<*mut TestEnv> = (1..=8)
            .map(|i| env(i, std::ptr::null(), &ended, 50))
            .collect();
        unsafe {
            let tasks: Vec<i64> = envs
                .iter()
                .map(|&e| vx_task_start(group, double, e as *mut u8))
                .collect();
            for &i in &[1usize, 4, 6] {
                vx_task_join(group, tasks[i]);
                assert_eq!((*envs[i]).result, (i + 1) * 2);
            }
            vx_tasks_end(group);
        }
        assert_eq!(ended.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn tasks_in_one_block_run_at_the_same_time() {
        // Each task waits until all four arrived, so four threads must run at once.
        let gate = Gate {
            arrived: AtomicUsize::new(0),
            need: 4,
        };
        let ended = AtomicUsize::new(0);
        let group = vx_tasks_begin();
        let envs: Vec<*mut TestEnv> = (1..=4).map(|i| env(i, &gate, &ended, 0)).collect();
        unsafe {
            let tasks: Vec<i64> = envs
                .iter()
                .map(|&e| vx_task_start(group, double, e as *mut u8))
                .collect();
            for (i, &task) in tasks.iter().enumerate() {
                vx_task_join(group, task);
                assert_eq!((*envs[i]).result, (i + 1) * 2, "task {i} waited in vain");
            }
            vx_tasks_end(group);
        }
    }
}
