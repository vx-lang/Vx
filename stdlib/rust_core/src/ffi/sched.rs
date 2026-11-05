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
// Under ThreadSanitizer, each task is a fiber of ThreadSanitizer too, and tsan.rs tells it of
// each switch.
//
// The runtime is used only when VX_TASK_RUNTIME=shared, with VX_TASK_WORKERS worker threads or
// one for each CPU.
//
// A task that blocks in a foreign call, or runs a long loop with no wait, holds its worker
// (W8). So a monitor thread looks at the scheduler on each tick. When tasks are ready, no worker
// is idle, and no worker switched to a task since the last tick, it starts one more worker. The
// workers stay until the program ends. VX_TASK_MAX_WORKERS limits their count.
//
//===----------------------------------------------------------------------===//

use crate::ffi::stack::{self, Stack, StackPool};
use crate::ffi::tasks::{CancelMark, VxTaskFn, TASK_STACK_BYTES};
use crate::ffi::tsan;
use crate::io::lines;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

/// How often the monitor looks at the workers.
const TICK: Duration = Duration::from_millis(10);

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
    pub(crate) put_aside: crate::ffi::tasks::PutAside,
    pub(crate) line: Vec<u8>,
    pub(crate) to_stderr: bool,
    pub(crate) site: Option<crate::ffi::tasks::Site>,
    pub(crate) cancels: u32,
}

impl TaskLocals {
    /// Swaps these values with the values of this thread.
    fn swap_with_thread(&mut self) {
        crate::ffi::tasks::swap_marks(&mut self.mark, &mut self.put_aside);
        lines::swap_with_thread(&mut self.line, &mut self.to_stderr);
        crate::ffi::tasks::swap_site(&mut self.site);
        crate::ffi::tasks::swap_cancels(&mut self.cancels);
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
    /// The fiber of ThreadSanitizer for this task, or null without ThreadSanitizer.
    tsan: *mut std::ffi::c_void,
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
    /// The fiber of ThreadSanitizer of the worker loop, or null without ThreadSanitizer.
    tsan: *mut std::ffi::c_void,
}

thread_local! {
    /// The worker that this thread runs, or null on a thread that is not a worker.
    static WORKER: Cell<*mut Worker> = const { Cell::new(std::ptr::null_mut()) };
    /// The scheduler with a seed whose tasks this thread runs.
    static HERE: Cell<Option<&'static Scheduler>> = const { Cell::new(None) };
    /// The Vx functions that foreign code called on this thread and that did not return yet.
    static CALLBACKS: Cell<u32> = const { Cell::new(0) };
}

/// The ready queue and its workers.
pub(crate) struct Scheduler {
    ready: Mutex<VecDeque<FiberRef>>,
    has_ready: Condvar,
    stacks: StackPool,
    /// The count of switches from a worker to a task. The monitor reads it as the progress.
    switches: AtomicU64,
    /// The workers that wait for a ready task.
    idle: AtomicUsize,
    /// The fibers that a worker runs now. A worker counts its fiber from the `pop` until the
    /// fiber is on a queue or a bucket list again, or ended.
    running: AtomicUsize,
    /// The running fibers that wait on their worker's thread, because foreign code called the Vx
    /// function they are in (W9). Only a wake can let one go on, so `all_parked` counts them.
    blocked: AtomicUsize,
    workers: AtomicUsize,
    max_workers: usize,
    /// The monitor wrote the message about the limit.
    told_limit: AtomicBool,
    /// With a seed, the scheduler has no workers. The thread that waits runs the ready tasks,
    /// and it picks each one at random from this state.
    random: Option<AtomicU64>,
    /// A thread took the scheduler with a seed as its own.
    has_home: AtomicBool,
}

impl Scheduler {
    /// A scheduler with `workers` worker threads, and a monitor that can add workers up to
    /// `max_workers`. It lives until the program ends.
    pub(crate) fn start(workers: usize, max_workers: usize) -> &'static Scheduler {
        assert!(workers > 0 && max_workers >= workers);
        let sched = Scheduler::without_workers(max_workers, None);
        for _ in 0..workers {
            sched.add_worker();
        }
        if max_workers > workers {
            start_thread("vx-monitor".to_string(), move || sched.monitor());
        }
        sched
    }

    fn without_workers(max_workers: usize, seed: Option<u64>) -> &'static Scheduler {
        Box::leak(Box::new(Scheduler {
            ready: Mutex::new(VecDeque::new()),
            has_ready: Condvar::new(),
            stacks: StackPool::new(TASK_STACK_BYTES),
            switches: AtomicU64::new(0),
            idle: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            blocked: AtomicUsize::new(0),
            workers: AtomicUsize::new(0),
            max_workers,
            told_limit: AtomicBool::new(false),
            random: seed.map(AtomicU64::new),
            has_home: AtomicBool::new(false),
        }))
    }

    /// A scheduler with no workers, whose tasks run on the thread that waits for them. The same
    /// seed gives the same order of tasks each time. A task that blocks in a foreign call blocks
    /// all of them.
    /// The first thread that starts a `tasks` block on it runs its tasks.
    pub(crate) fn start_with_seed(seed: u64) -> &'static Scheduler {
        Scheduler::without_workers(0, Some(seed))
    }

    /// With a seed: this thread runs the tasks, when no other thread does.
    pub(crate) fn take_home(&'static self) {
        if self.random.is_some() && !self.has_home.swap(true, Ordering::SeqCst) {
            HERE.with(|h| h.set(Some(self)));
        }
    }

    /// The next number of a splitmix64 sequence.
    fn next_random(random: &AtomicU64) -> u64 {
        let mut z = random.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::SeqCst);
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn add_worker(&'static self) {
        let i = self.workers.fetch_add(1, Ordering::SeqCst);
        start_thread(format!("vx-worker-{i}"), move || self.work());
    }

    /// The loop of the monitor thread.
    fn monitor(&'static self) {
        let mut last = self.switches.load(Ordering::SeqCst);
        loop {
            std::thread::sleep(TICK);
            let now = self.switches.load(Ordering::SeqCst);
            let stuck = now == last
                && self.idle.load(Ordering::SeqCst) == 0
                && !self.ready.lock().unwrap().is_empty();
            last = now;
            if !stuck {
                continue;
            }
            let workers = self.workers.load(Ordering::SeqCst);
            if workers < self.max_workers {
                self.add_worker();
            } else if !self.told_limit.swap(true, Ordering::SeqCst) {
                eprintln!(
                    "vx: tasks wait to run, and all {workers} worker threads are busy. \
                     VX_TASK_MAX_WORKERS sets the limit."
                );
            }
        }
    }

    fn push(&self, fiber: FiberRef) {
        self.ready.lock().unwrap().push_back(fiber);
        self.has_ready.notify_one();
        // The thread that runs the tasks of a seed can wait on any word.
        if self.random.is_some() {
            crate::ffi::wait::wake_every_sleeper();
        }
    }

    fn pop(&self) -> FiberRef {
        let mut ready = self.ready.lock().unwrap();
        loop {
            let next = match &self.random {
                Some(random) if !ready.is_empty() => {
                    (Scheduler::next_random(random) % ready.len() as u64) as usize
                }
                _ => 0,
            };
            if let Some(fiber) = ready.remove(next) {
                self.switches.fetch_add(1, Ordering::SeqCst);
                self.running.fetch_add(1, Ordering::SeqCst);
                return fiber;
            }
            self.idle.fetch_add(1, Ordering::SeqCst);
            ready = self.has_ready.wait(ready).unwrap();
            self.idle.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// The loop of a worker thread.
    fn work(&'static self) {
        let mut worker = Worker {
            sp: std::ptr::null_mut(),
            running: std::ptr::null_mut(),
            why: Why::Ended,
            tsan: tsan::current_fiber(),
        };
        // The fibers write the worker through this pointer too.
        let worker: *mut Worker = &raw mut worker;
        WORKER.with(|w| w.set(worker));
        loop {
            let fiber = self.pop();
            unsafe { self.run(worker, fiber) };
        }
    }

    /// Runs `fiber` on this thread until it parks or ends.
    unsafe fn run(&'static self, worker: *mut Worker, FiberRef(fiber): FiberRef) {
        (*fiber).state.store(RUNNING, Ordering::SeqCst);
        (*fiber).locals.swap_with_thread();
        (*worker).running = fiber;
        tsan::switch_to_fiber((*fiber).tsan);
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
                tsan::destroy_fiber((*fiber).tsan);
                self.stacks.give_back((*fiber).stack.take().unwrap());
                // After this store, a join can free the fiber. Its address is all that the wake
                // needs.
                let done = &raw const (*fiber).done as *const i64;
                (*fiber).done.store(1, Ordering::SeqCst);
                crate::ffi::wait::vx_wake_all(done);
            }
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
    }

    pub(crate) fn has_ready(&self) -> bool {
        !self.ready.lock().unwrap().is_empty()
    }

    /// Runs one ready task on this thread, which is not a worker, until the task parks or ends.
    /// Only a scheduler with a seed does this, and only on its own thread. The caller saw a ready
    /// task.
    pub(crate) fn run_one_here(&'static self) {
        assert!(self.random.is_some() && current_fiber().is_null());
        // Only this thread takes tasks off the queue, so it still has one.
        let fiber = self.pop();
        let mut worker = Worker {
            sp: std::ptr::null_mut(),
            running: std::ptr::null_mut(),
            why: Why::Ended,
            tsan: tsan::current_fiber(),
        };
        let worker: *mut Worker = &raw mut worker;
        WORKER.with(|w| w.set(worker));
        unsafe { self.run(worker, fiber) };
        WORKER.with(|w| w.set(std::ptr::null_mut()));
    }

    /// True when no fiber runs and none is ready. Each fiber that did not end then waits on a
    /// bucket, parked or blocked on its thread, and only a thread that is not a worker can wake
    /// it. With no fibers, it is true.
    pub(crate) fn all_parked(&self) -> bool {
        // `pop` counts a fiber as running under this lock, so no fiber is between the two.
        let ready = self.ready.lock().unwrap();
        ready.is_empty()
            && self.running.load(Ordering::SeqCst) == self.blocked.load(Ordering::SeqCst)
    }

    /// A running fiber starts to wait on its thread, under a callback.
    pub(crate) fn blocked(&self) {
        self.blocked.fetch_add(1, Ordering::SeqCst);
    }

    /// A wake ended the wait of a fiber on its thread, or the wait ended with no wake.
    pub(crate) fn unblocked(&self) {
        self.blocked.fetch_sub(1, Ordering::SeqCst);
    }

    /// Starts `run(env)` as a fiber with the cancel mark `mark`. The caller frees the fiber after
    /// `done` is 1.
    #[cfg(test)]
    pub(crate) fn spawn(
        &'static self,
        run: VxTaskFn,
        env: *mut u8,
        mark: Option<Arc<CancelMark>>,
    ) -> Box<Fiber> {
        self.spawn_at(run, env, mark, None)
    }

    /// `spawn` for a task with the site `site`.
    pub(crate) fn spawn_at(
        &'static self,
        run: VxTaskFn,
        env: *mut u8,
        mark: Option<Arc<CancelMark>>,
        site: Option<crate::ffi::tasks::Site>,
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
                site,
                ..Default::default()
            },
            done: AtomicI64::new(0),
            tsan: tsan::create_fiber(),
        });
        let raw: *mut Fiber = &mut *fiber;
        fiber.sp = unsafe { stack::start_on(&stack, fiber_entry, raw as *mut u8) };
        fiber.stack = Some(stack);
        self.push(FiberRef(raw));
        fiber
    }
}

fn start_thread(name: String, run: impl FnOnce() + Send + 'static) {
    if let Err(e) = std::thread::Builder::new().name(name).spawn(run) {
        eprintln!("vx: the system could not start a thread for the tasks: {e}");
        std::process::abort();
    }
}

/// The count that the environment variable `name` gives, or `default`.
fn count_setting(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(n) => match n.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("vx: {name} must be a number above 0, not '{n}'");
                std::process::abort();
            }
        },
        Err(_) => default,
    }
}

/// The most workers that the monitor starts when no VX_TASK_MAX_WORKERS is set.
const MAX_WORKERS: usize = 256;

/// The scheduler of the program, when VX_TASK_RUNTIME=shared. It starts with VX_TASK_WORKERS
/// worker threads, or one for each CPU. The monitor adds workers up to VX_TASK_MAX_WORKERS.
///
/// With VX_TASK_SEED, the tasks run on the first thread that starts a `tasks` block, in an order
/// that the seed gives. The deadlock watcher can make the scheduler first, so this is not the
/// thread that makes it.
pub(crate) fn program_scheduler() -> Option<&'static Scheduler> {
    static SCHED: OnceLock<Option<&'static Scheduler>> = OnceLock::new();
    *SCHED.get_or_init(|| {
        if let Ok(seed) = std::env::var("VX_TASK_SEED") {
            let Ok(seed) = seed.parse::<u64>() else {
                eprintln!("vx: VX_TASK_SEED must be a number, not '{seed}'");
                std::process::abort();
            };
            if std::env::var("VX_TASK_RUNTIME").is_ok_and(|v| v != "shared") {
                eprintln!("vx: VX_TASK_SEED needs VX_TASK_RUNTIME=shared, or no VX_TASK_RUNTIME");
                std::process::abort();
            }
            for other in ["VX_TASK_WORKERS", "VX_TASK_MAX_WORKERS"] {
                if std::env::var(other).is_ok() {
                    eprintln!(
                        "vx: VX_TASK_SEED runs the tasks on one thread, so {other} must not be set"
                    );
                    std::process::abort();
                }
            }
            return Some(Scheduler::start_with_seed(seed));
        }
        let shared = std::env::var("VX_TASK_RUNTIME").is_ok_and(|v| v == "shared");
        shared.then(|| {
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
            let workers = count_setting("VX_TASK_WORKERS", cpus);
            let max_workers = count_setting("VX_TASK_MAX_WORKERS", MAX_WORKERS.max(workers));
            if max_workers < workers {
                eprintln!(
                    "vx: VX_TASK_MAX_WORKERS ({max_workers}) is less than VX_TASK_WORKERS \
                     ({workers})"
                );
                std::process::abort();
            }
            Scheduler::start(workers, max_workers)
        })
    })
}

unsafe extern "C" fn fiber_entry(arg: *mut u8) -> ! {
    let fiber = arg as *mut Fiber;
    ((*fiber).run)((*fiber).env);
    lines::task_ended();
    // The task happens before the end of its join, which acquires `done`.
    tsan::release(&(*fiber).done);
    to_worker(Why::Ended);
    unreachable!("nothing continues a task that ended");
}

/// Foreign code calls a Vx function. Until `vx_callback_exit`, a wait of the task blocks its
/// thread and does not park. A parked task can continue on another thread, and the foreign
/// frames under the callback can keep state that belongs to this thread.
///
/// The count belongs to the thread, not to the task: a task parks only when the count is 0.
#[no_mangle]
pub extern "C" fn vx_callback_enter() {
    CALLBACKS.with(|c| c.set(c.get() + 1));
}

/// The Vx function that foreign code called returns.
#[no_mangle]
pub extern "C" fn vx_callback_exit() {
    CALLBACKS.with(|c| {
        assert!(
            c.get() > 0,
            "vx_callback_exit comes after vx_callback_enter"
        );
        c.set(c.get() - 1);
    });
}

/// The fiber that can park on this thread now, or null. A fiber under a callback cannot park.
pub(crate) fn fiber_that_can_park() -> *mut Fiber {
    if CALLBACKS.with(|c| c.get()) > 0 {
        std::ptr::null_mut()
    } else {
        current_fiber()
    }
}

/// The scheduler with a seed whose tasks this thread runs when it waits.
pub(crate) fn scheduler_here() -> Option<&'static Scheduler> {
    // A task in a callback waits on the thread, and it cannot run other tasks.
    if current_fiber().is_null() {
        HERE.with(|h| h.get())
    } else {
        None
    }
}

/// The scheduler of the fiber that this thread runs, if any.
pub(crate) fn current_scheduler() -> Option<&'static Scheduler> {
    let fiber = current_fiber();
    (!fiber.is_null()).then(|| unsafe { (*fiber).sched })
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
        tsan::switch_to_fiber((*worker).tsan);
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
        let sched = Scheduler::start(2, 2);
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
        let sched = Scheduler::start(1, 1);
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
        let sched = Scheduler::start(1, 1);
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
        let sched = Scheduler::start(1, 1);
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
        let sched = Scheduler::start(2, 2);
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

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn tasks_park_on_two_workers_with_shadow_stacks_on() {
        const NO_SHADOW_STACKS: i32 = 77;
        // As in the stack tests: the child turns shadow stacks on, and ends with `_exit`. Its
        // workers start after that, so each one gets a shadow stack of its own.
        let child = unsafe { libc::fork() };
        assert!(child >= 0, "fork failed");
        if child == 0 {
            if !crate::ffi::stack::turn_on_shadow_stacks() {
                unsafe { libc::_exit(NO_SHADOW_STACKS) };
            }
            let sched = Scheduler::start(2, 2);
            let p = PingPong {
                turn: AtomicI64::new(0),
                rounds: 2_000,
            };
            let group = tasks_begin(Some(sched));
            unsafe {
                vx_task_start(group, play_first, env_with(&p as *const PingPong));
                vx_task_start(group, play_second, env_with(&p as *const PingPong));
                vx_tasks_end(group);
            }
            let code = if p.turn.load(Ordering::SeqCst) == 0 {
                0
            } else {
                1
            };
            unsafe { libc::_exit(code) };
        }
        let mut status = 0;
        unsafe { libc::waitpid(child, &mut status, 0) };
        if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == NO_SHADOW_STACKS {
            crate::ffi::stack::found_no_shadow_stacks();
            return;
        }
        assert!(
            libc::WIFEXITED(status),
            "the child stopped with signal {}: a switch broke a shadow stack",
            libc::WTERMSIG(status)
        );
        assert_eq!(libc::WEXITSTATUS(status), 0, "a task lost its turn");
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
        crate::ffi::tasks::vx_cancel_off(10.0);
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
        crate::ffi::tasks::vx_cancel_off(10.0);
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
        let sched = Scheduler::start(1, 1);
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

    /// A pipe. A read from it blocks the OS thread until a write, as a foreign call can.
    struct Pipe([libc::c_int; 2]);

    impl Pipe {
        fn new() -> Pipe {
            let mut ends = [0; 2];
            assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0);
            Pipe(ends)
        }

        fn read(&self) {
            let mut byte = 0u8;
            let n = unsafe { libc::read(self.0[0], (&raw mut byte).cast(), 1) };
            assert_eq!(n, 1);
        }

        fn write(&self) {
            let byte = 1u8;
            let n = unsafe { libc::write(self.0[1], (&raw const byte).cast(), 1) };
            assert_eq!(n, 1);
        }
    }

    impl Drop for Pipe {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.0[0]);
                libc::close(self.0[1]);
            }
        }
    }

    struct Blocked {
        pipe: Pipe,
        ran: AtomicI64,
    }

    extern "C" fn read_the_pipe(env: *mut u8) {
        let b = unsafe { &**(env as *const *const Blocked) };
        b.pipe.read();
    }

    extern "C" fn write_the_pipe(env: *mut u8) {
        let b = unsafe { &**(env as *const *const Blocked) };
        b.ran.store(1, Ordering::SeqCst);
        b.pipe.write();
    }

    #[test]
    fn the_monitor_adds_a_worker_when_a_task_blocks_its_thread() {
        let _watchdog = Watchdog::new("the_monitor_adds_a_worker_when_a_task_blocks_its_thread");
        let sched = Scheduler::start(1, 2);
        let b = Blocked {
            pipe: Pipe::new(),
            ran: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, read_the_pipe, env_with(&b as *const Blocked));
            // The second task runs only on a second worker, because the first one blocks in
            // `read` until this task writes.
            vx_task_start(group, write_the_pipe, env_with(&b as *const Blocked));
            vx_tasks_end(group);
        }
        assert_eq!(sched.workers.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_monitor_adds_no_worker_past_the_limit() {
        let _watchdog = Watchdog::new("the_monitor_adds_no_worker_past_the_limit");
        let sched = Scheduler::start(1, 2);
        let (first, second) = (
            Blocked {
                pipe: Pipe::new(),
                ran: AtomicI64::new(0),
            },
            Blocked {
                pipe: Pipe::new(),
                ran: AtomicI64::new(0),
            },
        );
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, read_the_pipe, env_with(&first as *const Blocked));
            vx_task_start(group, read_the_pipe, env_with(&second as *const Blocked));
            // Both workers block, so this task waits to run.
            vx_task_start(group, write_the_pipe, env_with(&first as *const Blocked));
            let told = eventually(|| sched.told_limit.load(Ordering::SeqCst));
            std::thread::sleep(10 * TICK);
            let workers = sched.workers.load(Ordering::SeqCst);
            let ran = first.ran.load(Ordering::SeqCst);
            // The test writes to both pipes. The third task then runs, and it writes too.
            first.pipe.write();
            second.pipe.write();
            vx_tasks_end(group);
            first.pipe.read();
            assert!(told, "the monitor did not tell about the limit");
            assert_eq!(
                workers, 2,
                "the monitor started more workers than the limit"
            );
            assert_eq!(ran, 0, "a task ran with each worker blocked");
        }
    }

    struct Spin {
        gate: AtomicI64,
        waits: AtomicI64,
        spins: AtomicI64,
        stop: AtomicI64,
    }

    extern "C" fn wait_at_the_gate(env: *mut u8) {
        let s = unsafe { &**(env as *const *const Spin) };
        open(&s.waits);
        wait_for(&s.gate);
    }

    extern "C" fn spin_until_stopped(env: *mut u8) {
        let s = unsafe { &**(env as *const *const Spin) };
        open(&s.spins);
        while s.stop.load(Ordering::SeqCst) == 0 {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn all_parked_is_true_only_when_no_task_runs() {
        let _watchdog = Watchdog::new("all_parked_is_true_only_when_no_task_runs");
        let sched = Scheduler::start(2, 2);
        let s = Spin {
            gate: AtomicI64::new(0),
            waits: AtomicI64::new(0),
            spins: AtomicI64::new(0),
            stop: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, wait_at_the_gate, env_with(&s as *const Spin));
            vx_task_start(group, spin_until_stopped, env_with(&s as *const Spin));
            wait_for(&s.waits);
            wait_for(&s.spins);
            let mut seen_while_spinning = false;
            for _ in 0..100 {
                seen_while_spinning |= sched.all_parked();
                std::thread::sleep(Duration::from_millis(1));
            }
            s.stop.store(1, Ordering::SeqCst);
            let seen_after = eventually(|| sched.all_parked());
            open(&s.gate);
            vx_tasks_end(group);
            assert!(!seen_while_spinning, "a task that runs counted as parked");
            assert!(seen_after, "a task that waits did not count as parked");
        }
    }

    #[test]
    fn all_parked_is_false_while_a_task_is_ready() {
        // No worker takes the task off the queue.
        let sched = Scheduler::without_workers(1, None);
        let s = Spin {
            gate: AtomicI64::new(0),
            waits: AtomicI64::new(0),
            spins: AtomicI64::new(0),
            stop: AtomicI64::new(0),
        };
        let fiber = sched.spawn(wait_at_the_gate, env_with(&s as *const Spin), None);
        assert!(
            !sched.all_parked(),
            "a task that is ready counted as parked"
        );
        // The queue still points at the fiber.
        std::mem::forget(fiber);
    }

    struct Callback {
        waits: AtomicI64,
        gate: AtomicI64,
        ran: AtomicI64,
    }

    /// As a Vx function that foreign code calls, and that waits.
    extern "C" fn wait_in_a_callback(env: *mut u8) {
        let c = unsafe { &**(env as *const *const Callback) };
        vx_callback_enter();
        open(&c.waits);
        wait_for(&c.gate);
        vx_callback_exit();
    }

    extern "C" fn wait_after_a_callback(env: *mut u8) {
        let c = unsafe { &**(env as *const *const Callback) };
        vx_callback_enter();
        vx_callback_exit();
        wait_for(&c.gate);
    }

    extern "C" fn run_and_open_the_gate(env: *mut u8) {
        let c = unsafe { &**(env as *const *const Callback) };
        c.ran.store(1, Ordering::SeqCst);
        open(&c.gate);
    }

    #[test]
    fn a_task_in_a_callback_waits_on_its_thread() {
        let _watchdog = Watchdog::new("a_task_in_a_callback_waits_on_its_thread");
        let sched = Scheduler::start(1, 1);
        let c = Callback {
            waits: AtomicI64::new(0),
            gate: AtomicI64::new(0),
            ran: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, wait_in_a_callback, env_with(&c as *const Callback));
            vx_task_start(
                group,
                run_and_open_the_gate,
                env_with(&c as *const Callback),
            );
            wait_for(&c.waits);
            std::thread::sleep(10 * TICK);
            let ran = c.ran.load(Ordering::SeqCst);
            open(&c.gate);
            vx_tasks_end(group);
            assert_eq!(
                ran, 0,
                "a task in a callback parked, and gave its thread away"
            );
        }
    }

    #[test]
    fn a_task_parks_again_after_its_callback_returns() {
        // With one worker, the second task opens the gate only when the first task parks.
        let _watchdog = Watchdog::new("a_task_parks_again_after_its_callback_returns");
        let sched = Scheduler::start(1, 1);
        let c = Callback {
            waits: AtomicI64::new(0),
            gate: AtomicI64::new(0),
            ran: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(
                group,
                wait_after_a_callback,
                env_with(&c as *const Callback),
            );
            vx_task_start(
                group,
                run_and_open_the_gate,
                env_with(&c as *const Callback),
            );
            vx_tasks_end(group);
        }
    }

    struct Order {
        tasks: usize,
        arrived: AtomicUsize,
        gate: &'static AtomicI64,
        seen: Mutex<Vec<usize>>,
        threads: Mutex<HashSet<std::thread::ThreadId>>,
    }

    struct Numbered {
        order: *const Order,
        number: usize,
    }

    /// Notes its number, waits until each task arrived, and notes its number again.
    extern "C" fn arrive_then_note(env: *mut u8) {
        let n = unsafe { &**(env as *const *const Numbered) };
        let o = unsafe { &*n.order };
        o.threads
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        o.seen.lock().unwrap().push(n.number);
        if o.arrived.fetch_add(1, Ordering::SeqCst) + 1 == o.tasks {
            open(o.gate);
        } else {
            wait_for(o.gate);
        }
        o.seen.lock().unwrap().push(100 + n.number);
    }

    /// The order in which 8 tasks run with `seed`, and the threads that they ran on.
    ///
    /// A wake on the address of `gate` puts a waiting task back on the ready queue and changes
    /// the order. Another test can wake an address after it stops using it, so `gate` is a
    /// static that no other test uses.
    fn order_with_seed(
        seed: u64,
        gate: &'static AtomicI64,
    ) -> (Vec<usize>, HashSet<std::thread::ThreadId>) {
        // The scheduler comes from this thread, as the deadlock watcher can make it. The thread
        // that starts the `tasks` block runs the tasks.
        let sched = Scheduler::start_with_seed(seed);
        let order = std::thread::spawn(move || {
            let o = Order {
                tasks: 8,
                arrived: AtomicUsize::new(0),
                gate,
                seen: Mutex::new(Vec::new()),
                threads: Mutex::new(HashSet::new()),
            };
            let numbered: Vec<Numbered> = (0..o.tasks)
                .map(|number| Numbered { order: &o, number })
                .collect();
            let group = tasks_begin(Some(sched));
            unsafe {
                for n in &numbered {
                    vx_task_start(group, arrive_then_note, env_with(n as *const Numbered));
                }
                vx_tasks_end(group);
            }
            let mut threads = o.threads.into_inner().unwrap();
            threads.insert(std::thread::current().id());
            (o.seen.into_inner().unwrap(), threads)
        });
        order.join().unwrap()
    }

    #[test]
    fn a_seed_gives_the_same_order_each_time() {
        let _watchdog = Watchdog::new("a_seed_gives_the_same_order_each_time");
        static GATES: [AtomicI64; 3] = [AtomicI64::new(0), AtomicI64::new(0), AtomicI64::new(0)];
        let (first, first_threads) = order_with_seed(1, &GATES[0]);
        let (again, _) = order_with_seed(1, &GATES[1]);
        let (other, _) = order_with_seed(2, &GATES[2]);
        assert_eq!(first.len(), 16);
        assert_eq!(first, again, "the same seed gave another order");
        assert_ne!(first, other, "two seeds gave the same order");
        assert_eq!(
            first_threads.len(),
            1,
            "a task ran on a thread other than the one that waits"
        );
    }

    extern "C" fn wait_for_the_timer(env: *mut u8) {
        let word = unsafe { &**(env as *const *const AtomicI64) };
        let deadline = crate::ffi::wait::vx_clock_ns() + 20_000_000;
        while crate::ffi::wait::vx_clock_ns() < deadline {
            unsafe { crate::ffi::wait::vx_wait_while_until(word.as_ptr(), 0, deadline, 0) };
        }
    }

    #[test]
    fn a_task_that_the_timer_wakes_runs_with_a_seed() {
        // The timer thread makes the task ready. The thread that waits must see it.
        let _watchdog = Watchdog::new("a_task_that_the_timer_wakes_runs_with_a_seed");
        std::thread::spawn(|| {
            let sched = Scheduler::start_with_seed(1);
            let word = AtomicI64::new(0);
            let group = tasks_begin(Some(sched));
            unsafe {
                vx_task_start(
                    group,
                    wait_for_the_timer,
                    env_with(&word as *const AtomicI64),
                );
                vx_tasks_end(group);
            }
        })
        .join()
        .unwrap();
    }
}
