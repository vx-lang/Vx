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
// compiled code makes `env`: it holds what the task copied at its `spawn`, and a slot for the
// result. After `vx_task_join`, the compiled code reads the result and frees `env`. At the end
// of the block, `vx_tasks_end` joins each task that the program did not join, and calls
// `drop_result(env)` for it. That function drops the result and frees `env`.
//
//===----------------------------------------------------------------------===//

use std::thread::{self, JoinHandle};

/// The function that a task runs, and the function that drops a result nothing joined.
pub type VxTaskFn = extern "C" fn(*mut u8);

/// The stack of a task. The main thread of a Linux program gets 8 MiB, and a task runs the same
/// kind of code.
const TASK_STACK_BYTES: usize = 8 << 20;

/// The environment of a task, which a thread can take. The compiled code makes sure that only
/// the task uses it until the join.
struct Env(*mut u8);
unsafe impl Send for Env {}

struct Task {
    /// `None` when no thread could start, and the task ran on the thread that started it.
    thread: Option<JoinHandle<()>>,
    env: *mut u8,
    drop_result: VxTaskFn,
}

/// The tasks of one `tasks` block that nothing joined yet.
pub struct VxTaskGroup {
    tasks: Vec<Option<Task>>,
}

fn wait(task: Task) {
    if let Some(thread) = task.thread {
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
    Box::into_raw(Box::new(VxTaskGroup { tasks: Vec::new() }))
}

/// A `spawn`: starts `run(env)` on a new thread, and returns the task's number in `group`.
///
/// # Safety
/// `group` comes from `vx_tasks_begin` and its block did not end. Only the task uses `env`
/// until its join.
#[no_mangle]
pub unsafe extern "C" fn vx_task_start(
    group: *mut VxTaskGroup,
    run: VxTaskFn,
    drop_result: VxTaskFn,
    env: *mut u8,
) -> i64 {
    let group = &mut *group;
    let task_env = Env(env);
    let started = thread::Builder::new()
        .stack_size(TASK_STACK_BYTES)
        .spawn(move || {
            let task_env = task_env;
            run(task_env.0)
        });
    // With no thread, the task runs here, to its end. T7 allows that run.
    let thread = match started {
        Ok(thread) => Some(thread),
        Err(_) => {
            run(env);
            None
        }
    };
    group.tasks.push(Some(Task {
        thread,
        env,
        drop_result,
    }));
    (group.tasks.len() - 1) as i64
}

/// `h.join()`: waits until task `task` of `group` ends. Then its result is in its `env`.
///
/// # Safety
/// `group` comes from `vx_tasks_begin`, and `task` from `vx_task_start` on the same group.
#[no_mangle]
pub unsafe extern "C" fn vx_task_join(group: *mut VxTaskGroup, task: i64) {
    let group = &mut *group;
    let Some(task) = group.tasks.get_mut(task as usize).and_then(Option::take) else {
        eprintln!("vx: a task was joined twice, or it is not in this tasks block");
        std::process::abort();
    };
    wait(task);
}

/// The end of a `tasks` block: joins each task that the program did not join, in the order of
/// their `spawn`s, and drops its result. Then frees `group`.
///
/// # Safety
/// `group` comes from `vx_tasks_begin`, and this is the end of its block.
#[no_mangle]
pub unsafe extern "C" fn vx_tasks_end(group: *mut VxTaskGroup) {
    let group = Box::from_raw(group);
    for task in group.tasks.into_iter().flatten() {
        let (env, drop_result) = (task.env, task.drop_result);
        wait(task);
        drop_result(env);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
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

    /// The environment that the tests give their tasks.
    struct TestEnv {
        input: usize,
        result: usize,
        gate: Option<Arc<Gate>>,
        dropped: Arc<AtomicUsize>,
    }

    /// The result is `input * 2`, or 0 when the task waited at its gate in vain.
    extern "C" fn double(env: *mut u8) {
        let env = unsafe { &mut *(env as *mut TestEnv) };
        let in_time = env.gate.as_ref().is_none_or(|g| g.arrive_and_wait());
        env.result = if in_time { env.input * 2 } else { 0 };
    }

    extern "C" fn drop_env(env: *mut u8) {
        let env = unsafe { Box::from_raw(env as *mut TestEnv) };
        assert_eq!(
            env.result,
            env.input * 2,
            "a task did not end well before its result was dropped"
        );
        env.dropped.fetch_add(1, Ordering::SeqCst);
    }

    fn env(input: usize, gate: Option<Arc<Gate>>, dropped: &Arc<AtomicUsize>) -> *mut u8 {
        Box::into_raw(Box::new(TestEnv {
            input,
            result: 0,
            gate,
            dropped: dropped.clone(),
        })) as *mut u8
    }

    #[test]
    fn a_task_runs_at_the_same_time_as_the_code_after_its_spawn() {
        // The task waits until this thread arrives, after the spawn. A task that ran to its end
        // inside `vx_task_start` would wait in vain and give 0.
        let gate = Arc::new(Gate {
            arrived: AtomicUsize::new(0),
            need: 2,
        });
        let dropped = Arc::new(AtomicUsize::new(0));
        let group = vx_tasks_begin();
        let e = env(21, Some(gate.clone()), &dropped);
        unsafe {
            let task = vx_task_start(group, double, drop_env, e);
            assert!(gate.arrive_and_wait());
            vx_task_join(group, task);
            assert_eq!((*(e as *mut TestEnv)).result, 42);
            drop(Box::from_raw(e as *mut TestEnv));
            vx_tasks_end(group);
        }
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            0,
            "a joined task's result is the program's"
        );
    }

    #[test]
    fn the_end_of_the_block_joins_and_drops_only_the_tasks_nothing_joined() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let group = vx_tasks_begin();
        let envs: Vec<*mut u8> = (1..=8).map(|i| env(i, None, &dropped)).collect();
        unsafe {
            let tasks: Vec<i64> = envs
                .iter()
                .map(|&e| vx_task_start(group, double, drop_env, e))
                .collect();
            for &i in &[1usize, 4, 6] {
                vx_task_join(group, tasks[i]);
                assert_eq!((*(envs[i] as *mut TestEnv)).result, (i + 1) * 2);
                drop(Box::from_raw(envs[i] as *mut TestEnv));
            }
            vx_tasks_end(group);
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn tasks_in_one_block_run_at_the_same_time() {
        // Each task waits until all four arrived, so four threads must run at once.
        let gate = Arc::new(Gate {
            arrived: AtomicUsize::new(0),
            need: 4,
        });
        let dropped = Arc::new(AtomicUsize::new(0));
        let group = vx_tasks_begin();
        unsafe {
            for i in 1..=4 {
                vx_task_start(
                    group,
                    double,
                    drop_env,
                    env(i, Some(gate.clone()), &dropped),
                );
            }
            vx_tasks_end(group);
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 4);
    }
}
