//===- sched.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The shared runtime: many tasks on a few OS threads
// (docs/discussions/waiting_tasks_channels_and_task_errors.md, W2 and W3).
//
// Each task runs on its own stack from the pool in stack.rs. A worker thread takes a ready task
// from the queue, and switches to its stack. The task runs until it waits or ends, and then
// switches back to the worker. A wait parks the task: it goes on the list of the bucket of its
// word in wait.rs, and a wake puts it back on the queue.
//
// The state that the runtime keeps per thread (the cancel mark and the line that `print` keeps)
// belongs to the task. So the worker swaps it in before the switch to a task, and out after.
//
// The runtime is used only when VX_TASK_RUNTIME=shared, with VX_TASK_WORKERS worker threads or
// one for each CPU. It has no monitor yet (W8), so a task that blocks in a foreign call holds
// its worker.
//
//===----------------------------------------------------------------------===//

use crate::ffi::stack::{self, Stack, StackPool};
use crate::ffi::tasks::{CancelMark, VxTaskFn, TASK_STACK_BYTES};
use crate::io::lines;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

// The states of a fiber.
const RUNNING: u8 = 0;
/// The fiber is on the list of a bucket, and it did not switch to its worker yet.
const PARKING: u8 = 1;
const PARKED: u8 = 2;
/// A wake came while the fiber was `PARKING`. Its worker puts it back on the queue.
const WOKEN: u8 = 3;

/// What the runtime keeps per thread, for a task that does not run now.
#[derive(Default)]
pub(crate) struct TaskLocals {
    pub(crate) mark: Option<Arc<CancelMark>>,
    /// The marks that `vx_cancel_off` put aside. A `drop` can park between the off and the on.
    pub(crate) put_aside: Vec<Option<Arc<CancelMark>>>,
    pub(crate) line: Vec<u8>,
    pub(crate) to_stderr: bool,
}

impl TaskLocals {
    /// Swaps these values with the values of this thread.
    fn swap_with_thread(&mut self) {
        crate::ffi::tasks::swap_marks(&mut self.mark, &mut self.put_aside);
        lines::swap_with_thread(&mut self.line, &mut self.to_stderr);
    }
}

/// A task of the shared runtime.
pub(crate) struct Fiber {
    sched: &'static Scheduler,
    stack: Option<Stack>,
    /// The stack pointer of the fiber while it does not run.
    sp: *mut u8,
    run: VxTaskFn,
    env: *mut u8,
    state: AtomicU8,
    locals: TaskLocals,
    /// 0 while the task runs, and 1 after it ended. A join waits on this word.
    pub(crate) done: AtomicI64,
}

/// A fiber that a queue or a bucket holds. A fiber is in at most one of them at once.
#[derive(Clone, Copy)]
pub(crate) struct FiberRef(pub(crate) *mut Fiber);
unsafe impl Send for FiberRef {}

/// Why a fiber switched back to its worker.
#[derive(Clone, Copy, PartialEq)]
enum Why {
    Parked,
    Ended,
}

/// The state of one worker thread.
struct Worker {
    /// The stack pointer of the worker loop while a fiber runs.
    sp: *mut u8,
    running: *mut Fiber,
    why: Why,
}

thread_local! {
    /// The worker that this thread runs, or null on a thread that is not a worker.
    static WORKER: Cell<*mut Worker> = const { Cell::new(std::ptr::null_mut()) };
}

/// The ready queue and its workers.
pub(crate) struct Scheduler {
    ready: Mutex<VecDeque<FiberRef>>,
    has_ready: Condvar,
    stacks: StackPool,
}

impl Scheduler {
    /// A scheduler with `workers` worker threads. It lives until the program ends.
    pub(crate) fn start(workers: usize) -> &'static Scheduler {
        let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
            ready: Mutex::new(VecDeque::new()),
            has_ready: Condvar::new(),
            stacks: StackPool::new(TASK_STACK_BYTES),
        }));
        for i in 0..workers {
            let started = std::thread::Builder::new()
                .name(format!("vx-worker-{i}"))
                .spawn(move || sched.work());
            if let Err(e) = started {
                eprintln!("vx: the system could not start a worker thread: {e}");
                std::process::abort();
            }
        }
        sched
    }

    fn push(&self, fiber: FiberRef) {
        self.ready.lock().unwrap().push_back(fiber);
        self.has_ready.notify_one();
    }

    fn pop(&self) -> FiberRef {
        let mut ready = self.ready.lock().unwrap();
        loop {
            if let Some(fiber) = ready.pop_front() {
                return fiber;
            }
            ready = self.has_ready.wait(ready).unwrap();
        }
    }

    /// The loop of a worker thread.
    fn work(&'static self) {
        let mut worker = Worker {
            sp: std::ptr::null_mut(),
            running: std::ptr::null_mut(),
            why: Why::Ended,
        };
        // The fibers write the worker through this pointer too.
        let worker: *mut Worker = &raw mut worker;
        WORKER.with(|w| w.set(worker));
        loop {
            let FiberRef(fiber) = self.pop();
            unsafe {
                (*fiber).state.store(RUNNING, Ordering::SeqCst);
                (*fiber).locals.swap_with_thread();
                (*worker).running = fiber;
                stack::switch(&raw mut (*worker).sp, (*fiber).sp);
                (*fiber).locals.swap_with_thread();
                match (*worker).why {
                    Why::Parked => {
                        let parked = (*fiber).state.compare_exchange(
                            PARKING,
                            PARKED,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        );
                        // A wake came before the switch, and it left the fiber to this worker.
                        if parked.is_err() {
                            self.push(FiberRef(fiber));
                        }
                    }
                    Why::Ended => {
                        self.stacks.give_back((*fiber).stack.take().unwrap());
                        // After this store, a join can free the fiber. Its address is all that
                        // the wake needs.
                        let done = &raw const (*fiber).done as *const i64;
                        (*fiber).done.store(1, Ordering::SeqCst);
                        crate::ffi::wait::vx_wake_all(done);
                    }
                }
            }
        }
    }

    /// Starts `run(env)` as a fiber with the cancel mark `mark`. The caller frees the fiber after
    /// `done` is 1.
    pub(crate) fn spawn(
        &'static self,
        run: VxTaskFn,
        env: *mut u8,
        mark: Option<Arc<CancelMark>>,
    ) -> Box<Fiber> {
        let Some(stack) = self.stacks.take() else {
            eprintln!("vx: the system could not map a stack for a task");
            std::process::abort();
        };
        let mut fiber = Box::new(Fiber {
            sched: self,
            stack: None,
            sp: std::ptr::null_mut(),
            run,
            env,
            state: AtomicU8::new(RUNNING),
            locals: TaskLocals {
                mark,
                ..Default::default()
            },
            done: AtomicI64::new(0),
        });
        let raw: *mut Fiber = &mut *fiber;
        fiber.sp = unsafe { stack::start_on(&stack, fiber_entry, raw as *mut u8) };
        fiber.stack = Some(stack);
        self.push(FiberRef(raw));
        fiber
    }
}

/// The scheduler of the program, when VX_TASK_RUNTIME=shared. It has VX_TASK_WORKERS worker
/// threads, or one for each CPU.
pub(crate) fn program_scheduler() -> Option<&'static Scheduler> {
    static SCHED: OnceLock<Option<&'static Scheduler>> = OnceLock::new();
    *SCHED.get_or_init(|| {
        let shared = std::env::var("VX_TASK_RUNTIME").is_ok_and(|v| v == "shared");
        shared.then(|| {
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
            let workers = match std::env::var("VX_TASK_WORKERS") {
                Ok(n) => match n.parse::<usize>() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        eprintln!("vx: VX_TASK_WORKERS must be a number above 0, not '{n}'");
                        std::process::abort();
                    }
                },
                Err(_) => cpus,
            };
            Scheduler::start(workers)
        })
    })
}

unsafe extern "C" fn fiber_entry(arg: *mut u8) -> ! {
    let fiber = arg as *mut Fiber;
    ((*fiber).run)((*fiber).env);
    lines::task_ended();
    to_worker(Why::Ended);
    unreachable!("nothing continues a task that ended");
}

/// The fiber that this thread runs, or null on a thread that runs no fiber.
#[inline(never)]
pub(crate) fn current_fiber() -> *mut Fiber {
    let worker = WORKER.with(|w| w.get());
    if worker.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*worker).running }
    }
}

/// Switches from the running fiber back to its worker.
///
/// A parked fiber can continue on another thread. So this function reads the worker before the
/// switch, and reads no thread-local after it: the compiler may keep the address of a
/// thread-local across the switch.
#[inline(never)]
fn to_worker(why: Why) {
    let worker = WORKER.with(|w| w.get());
    unsafe {
        let fiber = (*worker).running;
        (*worker).why = why;
        let worker_sp = (*worker).sp;
        stack::switch(&raw mut (*fiber).sp, worker_sp);
    }
}

/// Parks the running fiber. The caller put it on the list of a bucket, under the lock of the
/// bucket, after it set the state to `PARKING`.
pub(crate) fn park() {
    to_worker(Why::Parked);
}

/// Marks `fiber` as `PARKING`. Call it under the lock of the bucket whose list gets the fiber.
pub(crate) fn will_park(fiber: *mut Fiber) {
    unsafe { (*fiber).state.store(PARKING, Ordering::SeqCst) };
}

/// Puts a fiber that a bucket held back on the ready queue.
pub(crate) fn wake(FiberRef(fiber): FiberRef) {
    unsafe {
        match (*fiber).state.swap(WOKEN, Ordering::SeqCst) {
            PARKED => (*fiber).sched.push(FiberRef(fiber)),
            // The worker of the fiber sees `WOKEN`, and puts it on the queue.
            PARKING => {}
            other => unreachable!("a wake of a fiber in state {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::tasks::{
        tasks_begin, vx_task_join, vx_task_start, vx_tasks_end, vx_tasks_wait,
    };
    use crate::ffi::wait::{vx_wait_while, vx_wait_while_or_cancel, vx_wake_all};
    use std::collections::HashSet;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    /// Waits on `word` until it is not 0.
    fn wait_for(word: &AtomicI64) {
        while word.load(Ordering::SeqCst) == 0 {
            unsafe { vx_wait_while(word.as_ptr(), 0) };
        }
    }

    fn open(word: &AtomicI64) {
        word.store(1, Ordering::SeqCst);
        unsafe { vx_wake_all(word.as_ptr()) };
    }

    /// Stops the test program when a test runs longer than 10 seconds. A broken runtime hangs,
    /// and the message names the test.
    struct Watchdog(Arc<AtomicI64>);

    impl Watchdog {
        fn new(test: &'static str) -> Watchdog {
            let ended = Arc::new(AtomicI64::new(0));
            let seen = ended.clone();
            std::thread::spawn(move || {
                let end = Instant::now() + Duration::from_secs(10);
                while seen.load(Ordering::SeqCst) == 0 {
                    if Instant::now() > end {
                        eprintln!("{test} hangs: a task was not woken, or did not end");
                        std::process::abort();
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            Watchdog(ended)
        }
    }

    impl Drop for Watchdog {
        fn drop(&mut self) {
            self.0.store(1, Ordering::SeqCst);
        }
    }

    /// Polls from a thread that is not a worker, for at most 5 seconds, which is less than the
    /// time of the watchdog.
    fn eventually(what: impl Fn() -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs(5);
        while !what() {
            if Instant::now() > end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    struct Crowd {
        arrived: AtomicUsize,
        gate: AtomicI64,
        ended: AtomicUsize,
        threads: Mutex<HashSet<std::thread::ThreadId>>,
    }

    extern "C" fn arrive_and_wait(env: *mut u8) {
        let crowd = unsafe { &**(env as *const *const Crowd) };
        crowd
            .threads
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        crowd.arrived.fetch_add(1, Ordering::SeqCst);
        wait_for(&crowd.gate);
        crowd.ended.fetch_add(1, Ordering::SeqCst);
    }

    /// The compiled code gives a task an `env` from `malloc`, and the group frees it.
    fn env_with<T>(value: T) -> *mut u8 {
        unsafe {
            let e = libc::malloc(std::mem::size_of::<T>()) as *mut T;
            e.write(value);
            e as *mut u8
        }
    }

    #[test]
    fn a_thousand_tasks_wait_at_the_same_time_on_two_threads() {
        let _watchdog = Watchdog::new("a_thousand_tasks_wait_at_the_same_time_on_two_threads");
        const TASKS: usize = 1000;
        let sched = Scheduler::start(2);
        let crowd = Crowd {
            arrived: AtomicUsize::new(0),
            gate: AtomicI64::new(0),
            ended: AtomicUsize::new(0),
            threads: Mutex::new(HashSet::new()),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            for _ in 0..TASKS {
                let env = env_with(&crowd as *const Crowd);
                vx_task_start(group, arrive_and_wait, env);
            }
            // The gate opens only after each task arrived. A wait that blocked its worker thread
            // would let only two tasks arrive.
            let all_arrived = eventually(|| crowd.arrived.load(Ordering::SeqCst) == TASKS);
            let arrived = crowd.arrived.load(Ordering::SeqCst);
            open(&crowd.gate);
            vx_tasks_end(group);
            assert!(all_arrived, "only {arrived} of {TASKS} tasks arrived");
        }
        assert_eq!(crowd.ended.load(Ordering::SeqCst), TASKS);
        assert!(
            crowd.threads.lock().unwrap().len() <= 2,
            "the tasks ran on more than two threads"
        );
    }

    struct Lines {
        first_parked: AtomicI64,
        second_done: AtomicI64,
        /// What each task kept of its line, after the other task ran on the same thread.
        first_kept: Mutex<Vec<u8>>,
        second_found: Mutex<Vec<u8>>,
    }

    extern "C" fn print_then_park(env: *mut u8) {
        let l = unsafe { &**(env as *const *const Lines) };
        lines::print("a line that is not done");
        open(&l.first_parked);
        wait_for(&l.second_done);
        *l.first_kept.lock().unwrap() = lines::take_line();
    }

    extern "C" fn look_at_the_line(env: *mut u8) {
        let l = unsafe { &**(env as *const *const Lines) };
        wait_for(&l.first_parked);
        *l.second_found.lock().unwrap() = lines::take_line();
        open(&l.second_done);
    }

    #[test]
    fn each_task_keeps_its_own_line_on_one_thread() {
        let _watchdog = Watchdog::new("each_task_keeps_its_own_line_on_one_thread");
        let sched = Scheduler::start(1);
        let l = Lines {
            first_parked: AtomicI64::new(0),
            second_done: AtomicI64::new(0),
            first_kept: Mutex::new(Vec::new()),
            second_found: Mutex::new(Vec::new()),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, print_then_park, env_with(&l as *const Lines));
            vx_task_start(group, look_at_the_line, env_with(&l as *const Lines));
            vx_tasks_end(group);
        }
        assert_eq!(
            *l.second_found.lock().unwrap(),
            b"",
            "a task saw the line of another task"
        );
        assert_eq!(
            *l.first_kept.lock().unwrap(),
            b"a line that is not done",
            "a task lost its line when it parked"
        );
    }

    struct Cancel {
        never: AtomicI64,
        parked: AtomicI64,
        saw_cancel: AtomicI64,
    }

    extern "C" fn wait_until_cancelled(env: *mut u8) {
        let c = unsafe { &**(env as *const *const Cancel) };
        open(&c.parked);
        while c.never.load(Ordering::SeqCst) == 0 && crate::ffi::tasks::vx_cancelled() == 0 {
            unsafe { vx_wait_while_or_cancel(c.never.as_ptr(), 0) };
        }
        c.saw_cancel
            .store(crate::ffi::tasks::vx_cancelled() as i64, Ordering::SeqCst);
    }

    #[test]
    fn a_cancel_wakes_a_parked_task() {
        let _watchdog = Watchdog::new("a_cancel_wakes_a_parked_task");
        let sched = Scheduler::start(1);
        let c = Cancel {
            never: AtomicI64::new(0),
            parked: AtomicI64::new(0),
            saw_cancel: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, wait_until_cancelled, env_with(&c as *const Cancel));
            assert!(eventually(|| c.parked.load(Ordering::SeqCst) == 1));
            // Leaving the block early cancels it, and then joins its task.
            vx_tasks_wait(group);
            vx_tasks_end(group);
        }
        assert_eq!(
            c.saw_cancel.load(Ordering::SeqCst),
            1,
            "the task did not see the cancel"
        );
    }

    struct Pair {
        task_arrived: AtomicI64,
        test_arrived: AtomicI64,
        result: AtomicI64,
    }

    extern "C" fn meet_the_test(env: *mut u8) {
        let p = unsafe { &**(env as *const *const Pair) };
        open(&p.task_arrived);
        wait_for(&p.test_arrived);
        p.result.store(42, Ordering::SeqCst);
    }

    #[test]
    fn a_task_runs_at_the_same_time_as_the_code_after_its_spawn() {
        let _watchdog = Watchdog::new("a_task_runs_at_the_same_time_as_the_code_after_its_spawn");
        let sched = Scheduler::start(1);
        let p = Pair {
            task_arrived: AtomicI64::new(0),
            test_arrived: AtomicI64::new(0),
            result: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            let task = vx_task_start(group, meet_the_test, env_with(&p as *const Pair));
            assert!(eventually(|| p.task_arrived.load(Ordering::SeqCst) == 1));
            open(&p.test_arrived);
            vx_task_join(group, task);
            assert_eq!(p.result.load(Ordering::SeqCst), 42);
            vx_tasks_end(group);
        }
    }

    struct PingPong {
        /// 0 when the first task has the turn, and 1 when the second has it.
        turn: AtomicI64,
        rounds: i64,
    }

    /// Each pass hands the turn to the other task and wakes it. The other task often parks at
    /// the same time, so a wake that comes before its switch to the worker must not be lost.
    fn play(p: &PingPong, mine: i64) {
        for _ in 0..p.rounds {
            while p.turn.load(Ordering::SeqCst) != mine {
                unsafe { vx_wait_while(p.turn.as_ptr(), 1 - mine) };
            }
            p.turn.store(1 - mine, Ordering::SeqCst);
            unsafe { vx_wake_all(p.turn.as_ptr()) };
        }
    }

    extern "C" fn play_first(env: *mut u8) {
        play(unsafe { &**(env as *const *const PingPong) }, 0);
    }

    extern "C" fn play_second(env: *mut u8) {
        play(unsafe { &**(env as *const *const PingPong) }, 1);
    }

    #[test]
    fn two_tasks_pass_a_turn_back_and_forth_without_a_lost_wake() {
        let _watchdog = Watchdog::new("two_tasks_pass_a_turn_back_and_forth_without_a_lost_wake");
        let sched = Scheduler::start(2);
        let p = PingPong {
            turn: AtomicI64::new(0),
            rounds: 20_000,
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, play_first, env_with(&p as *const PingPong));
            vx_task_start(group, play_second, env_with(&p as *const PingPong));
            vx_tasks_end(group);
        }
        assert_eq!(p.turn.load(Ordering::SeqCst), 0);
    }

    struct Drops {
        first_off: AtomicI64,
        second_off: AtomicI64,
        first_on: AtomicI64,
        /// 1 when a task got its own mark back from `vx_cancel_on`.
        first_kept: AtomicI64,
        second_kept: AtomicI64,
    }

    /// The address of the current mark of this thread, or 0.
    fn current_mark() -> usize {
        let (mut mark, mut put_aside) = (None, Vec::new());
        crate::ffi::tasks::swap_marks(&mut mark, &mut put_aside);
        let address = mark.as_ref().map_or(0, |m| Arc::as_ptr(m) as usize);
        crate::ffi::tasks::swap_marks(&mut mark, &mut put_aside);
        address
    }

    extern "C" fn first_drop(env: *mut u8) {
        let d = unsafe { &**(env as *const *const Drops) };
        let mine = current_mark();
        crate::ffi::tasks::vx_cancel_off();
        open(&d.first_off);
        wait_for(&d.second_off);
        crate::ffi::tasks::vx_cancel_on();
        d.first_kept.store(
            (mine != 0 && current_mark() == mine) as i64,
            Ordering::SeqCst,
        );
        open(&d.first_on);
    }

    extern "C" fn second_drop(env: *mut u8) {
        let d = unsafe { &**(env as *const *const Drops) };
        wait_for(&d.first_off);
        let mine = current_mark();
        crate::ffi::tasks::vx_cancel_off();
        open(&d.second_off);
        wait_for(&d.first_on);
        crate::ffi::tasks::vx_cancel_on();
        d.second_kept.store(
            (mine != 0 && current_mark() == mine) as i64,
            Ordering::SeqCst,
        );
    }

    #[test]
    fn each_task_gets_its_own_mark_back_after_a_drop_that_parks() {
        let _watchdog = Watchdog::new("each_task_gets_its_own_mark_back_after_a_drop_that_parks");
        let sched = Scheduler::start(1);
        let d = Drops {
            first_off: AtomicI64::new(0),
            second_off: AtomicI64::new(0),
            first_on: AtomicI64::new(0),
            first_kept: AtomicI64::new(0),
            second_kept: AtomicI64::new(0),
        };
        // Two blocks, so that the two tasks have different marks. Both tasks put their mark
        // aside on the one worker, and then the first one takes its mark back.
        let outer = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(outer, first_drop, env_with(&d as *const Drops));
            let inner = tasks_begin(Some(sched));
            vx_task_start(inner, second_drop, env_with(&d as *const Drops));
            vx_tasks_end(inner);
            vx_tasks_end(outer);
        }
        assert_eq!(
            d.first_kept.load(Ordering::SeqCst),
            1,
            "the first task got another mark back"
        );
        assert_eq!(
            d.second_kept.load(Ordering::SeqCst),
            1,
            "the second task got another mark back"
        );
    }
}
