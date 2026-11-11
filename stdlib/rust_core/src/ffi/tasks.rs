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
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
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
    /// Set to 1, with a wake, as the last step of the task's thread.
    thread_done: Option<Arc<AtomicI64>>,
    /// The task in the shared runtime.
    fiber: Option<Box<Fiber>>,
    env: *mut u8,
    /// Where the task started, so that the deadlock report can name the task that a join waits for.
    site: Option<Site>,
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

/// One scope with a deadline (F6a): a `drop`, with the mark that `vx_cancel_off` put aside, or a
/// call of `within`.
pub(crate) struct Deadline {
    /// The mark put aside, for a `drop`.
    mark: Option<Arc<CancelMark>>,
    /// The earliest deadline of this scope and the scopes around it, on the clock of
    /// `vx_clock_ns`.
    at: i64,
    /// A `drop`, not a call of `within`.
    drop: bool,
}

/// The scopes with a deadline that this task runs in, innermost last.
pub(crate) type PutAside = Vec<Deadline>;

/// Where the task that runs now started: one line for its `spawn`, and then one line for the
/// `spawn` of each task around it. An abort in a task prints it (F9).
pub(crate) type Site = Arc<[u8]>;

thread_local! {
    /// The site of the task that this thread runs, or `None` outside a task.
    static SITE: RefCell<Option<Site>> = const { RefCell::new(None) };
    /// The bytes of `SITE`, for the handler of an abort. A signal handler must not touch a
    /// `RefCell`, so it reads this copy of the pointer.
    static SITE_BYTES: Cell<(*const u8, usize)> = const { Cell::new((std::ptr::null(), 0)) };
    /// How many times `vx_cancelled` gave 1 to the task that this thread runs.
    static CANCELS: Cell<u32> = const { Cell::new(0) };
    /// The mark of the innermost block whose task or code this thread runs.
    static CURRENT: RefCell<Option<Arc<CancelMark>>> = const { RefCell::new(None) };
    static PUT_ASIDE: RefCell<PutAside> = const { RefCell::new(Vec::new()) };
}

/// `VX_DROP_WAIT_SECONDS`, in nanoseconds: a cap on the time limit of each `drop`, for tests and
/// debugging (F6a). `None` when the variable is not set. `vx_init_signals` reads it when the
/// program starts, so a value that is not a number stops the program before it runs.
pub(crate) fn drop_wait_cap_ns() -> Option<i64> {
    static CAP: std::sync::OnceLock<Option<i64>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let text = std::env::var("VX_DROP_WAIT_SECONDS").ok()?;
        match text.parse::<f64>() {
            Ok(seconds) if seconds >= 0.0 && seconds.is_finite() => Some(seconds_to_ns(seconds)),
            _ => {
                eprintln!("vx: VX_DROP_WAIT_SECONDS must be a number of seconds, not '{text}'");
                std::process::abort();
            }
        }
    })
}

/// `seconds` in nanoseconds. A limit below 0, or one that is not a number, is 0, and a limit
/// past the range of `i64` is the end of that range.
fn seconds_to_ns(seconds: f64) -> i64 {
    (seconds.max(0.0) * 1e9) as i64
}

/// The earliest deadline of the `drop`s and the calls of `within` that this thread runs in, if it
/// runs in one. After it, each wait that can be cancelled gives up, as in a cancelled block.
pub(crate) fn drop_deadline() -> Option<i64> {
    PUT_ASIDE.with(|p| p.borrow().last().map(|d| d.at))
}

/// The deadlines that a new task starts with: the earliest deadline of the code that spawns it.
/// So a task in a `drop` or in a call of `within` gives up at the same time as that code.
pub(crate) fn deadlines_for_a_task() -> PutAside {
    let at = drop_deadline();
    at.map(|at| Deadline {
        mark: None,
        at,
        drop: false,
    })
    .into_iter()
    .collect()
}

/// Starts a scope with a deadline `limit_ns` from now. It keeps the deadline of the scope
/// around it when that is earlier.
fn push_deadline(mark: Option<Arc<CancelMark>>, limit_ns: i64, drop: bool) {
    let at = crate::ffi::wait::vx_clock_ns().saturating_add(limit_ns);
    let at = drop_deadline().map_or(at, |outer| outer.min(at));
    PUT_ASIDE.with(|p| p.borrow_mut().push(Deadline { mark, at, drop }));
}

/// Before a `drop`: this thread runs with no mark until `vx_cancel_on`, so no wait gives
/// `Cancelled` (F6), until the deadline of the `drop`. The deadline is `limit_ns` nanoseconds from
/// now, from the type's `Drop::wait_limit`, or `VX_DROP_WAIT_SECONDS` when that is shorter. A
/// `drop` in a `drop` keeps the earlier deadline. A `tasks` block in the `drop` gets a mark
/// with no parent.
#[no_mangle]
pub extern "C" fn vx_cancel_off(limit_ns: i64) {
    let mark = CURRENT.with(|c| c.borrow_mut().take());
    let limit_ns = limit_ns.max(0);
    let limit_ns = drop_wait_cap_ns().map_or(limit_ns, |cap| cap.min(limit_ns));
    push_deadline(mark, limit_ns, true);
}

/// After a `drop`: the mark that `vx_cancel_off` put aside comes back.
#[no_mangle]
pub extern "C" fn vx_cancel_on() {
    let scope = PUT_ASIDE.with(|p| p.borrow_mut().pop());
    let scope = scope.expect("vx_cancel_on comes after vx_cancel_off");
    assert!(scope.drop, "a call of `within` in a drop did not end");
    CURRENT.with(|c| {
        assert!(c.borrow().is_none(), "a tasks block in a drop did not end");
        *c.borrow_mut() = scope.mark;
    });
}

/// The start of a call of `within(limit, body)`: each wait in `body` that can be cancelled gives
/// up `limit_ns` nanoseconds from now, or at the deadline of a scope around it when that is
/// earlier. Outside a `drop`, a cancel still stops the waits. `VX_DROP_WAIT_SECONDS` does not
/// shorten it.
#[no_mangle]
pub extern "C" fn vx_within_start(limit_ns: i64) {
    push_deadline(None, limit_ns.max(0), false);
}

/// The end of a call of `within`, after `body` returns.
#[no_mangle]
pub extern "C" fn vx_within_end() {
    let scope = PUT_ASIDE.with(|p| p.borrow_mut().pop());
    let scope = scope.expect("vx_within_end comes after vx_within_start");
    assert!(!scope.drop, "a drop in a call of `within` did not end");
}

/// Swaps the current mark and the marks put aside of this thread with these. The shared runtime
/// gives each fiber its own.
pub(crate) fn swap_marks(mark: &mut Option<Arc<CancelMark>>, put_aside: &mut PutAside) {
    CURRENT.with(|c| std::mem::swap(&mut *c.borrow_mut(), mark));
    PUT_ASIDE.with(|p| std::mem::swap(&mut *p.borrow_mut(), put_aside));
}

/// Swaps the site of this thread with `site`. The shared runtime gives each fiber its own.
pub(crate) fn swap_site(site: &mut Option<Site>) {
    SITE.with(|s| std::mem::swap(&mut *s.borrow_mut(), site));
    let bytes = SITE.with(|s| s.borrow().as_ref().map(|b| (b.as_ptr(), b.len())));
    SITE_BYTES.with(|b| b.set(bytes.unwrap_or((std::ptr::null(), 0))));
}

/// The site of a task that a `spawn` at `file` and `line` starts on this thread.
fn site_of_spawn(file: *const c_char, line: i64) -> Option<Site> {
    if line <= 0 {
        return None;
    }
    let file = if file.is_null() {
        "an unknown file".into()
    } else {
        unsafe { CStr::from_ptr(file) }.to_string_lossy()
    };
    let mut text = format!("  in the task from the `spawn` at {file}:{line}\n").into_bytes();
    SITE.with(|s| text.extend_from_slice(s.borrow().as_deref().unwrap_or(&[])));
    Some(text.into())
}

/// The file and line of the `spawn` of the task with the site `site`.
pub(crate) fn spawn_line(site: &Site) -> String {
    let text = String::from_utf8_lossy(site);
    let first = text.lines().next().unwrap_or_default();
    first.rsplit(" at ").next().unwrap_or_default().to_string()
}

/// The file and the line of each `spawn` of the site `site`, the innermost first, to sort sites
/// by their lines.
pub(crate) fn site_key(site: &Site) -> Vec<(String, u64)> {
    String::from_utf8_lossy(site)
        .lines()
        .filter_map(|l| {
            let (file, line) = l.rsplit(" at ").next()?.rsplit_once(':')?;
            Some((file.to_string(), line.parse().ok()?))
        })
        .collect()
}

/// The site of the task that this thread runs now.
pub(crate) fn current_site() -> Option<Site> {
    SITE.with(|s| s.borrow().clone())
}

/// Writes the site of this thread to standard error. A signal handler may call it: it uses only
/// a `Cell` and `write`.
pub(crate) fn write_site() {
    let (bytes, len) = SITE_BYTES.with(|b| b.get());
    if !bytes.is_null() {
        unsafe { libc::write(2, bytes as *const libc::c_void, len) };
    }
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
            crate::ffi::wait::waiting_for_join(task.site.clone());
            unsafe { crate::ffi::wait::vx_wait_while(done, 0) };
        }
        crate::ffi::tsan::acquire(&fiber.done);
    }
    if let Some(thread) = task.thread.take() {
        // The wait goes through `wait.rs`, so the deadlock watcher stops counting this thread
        // as waiting at the wake, not when the OS runs it again.
        let thread_done = task
            .thread_done
            .as_ref()
            .expect("a task thread has a done word");
        let done = Arc::as_ptr(thread_done) as *const i64;
        while thread_done.load(Ordering::SeqCst) == 0 {
            crate::ffi::wait::waiting_for_join(task.site.clone());
            unsafe { crate::ffi::wait::vx_wait_while(done, 0) };
        }
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
    if let Some(sched) = sched {
        sched.take_home();
    }
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

/// Whether the block of this thread's task or code, or a block around it, was cancelled, or a
/// `drop` that this thread runs is past its deadline. A wait that can be cancelled then gives
/// `Cancelled`.
///
/// The mark stays, so a task that sees it many times is in a loop that waits again after each
/// `Cancelled` (F4, F9). At the `CANCELS_TO_WARN`th time, a warning says which task it is.
#[no_mangle]
pub extern "C" fn vx_cancelled() -> i32 {
    unsafe { vx_cancelled_at(std::ptr::null(), 0) }
}

/// `vx_cancelled` for the wait at `file` and `line`, which the warning names. `file` is a C string,
/// or null when the place is not known.
///
/// # Safety
/// `file` is null or points to a C string.
#[no_mangle]
pub unsafe extern "C" fn vx_cancelled_at(file: *const c_char, line: i32) -> i32 {
    let set = CURRENT.with(|c| c.borrow().as_ref().is_some_and(|m| m.is_set()));
    if set && CANCELS.with(|n| n.replace(n.get() + 1) + 1) == CANCELS_TO_WARN {
        let wait = if file.is_null() {
            String::new()
        } else {
            let file = unsafe { CStr::from_ptr(file) }.to_string_lossy();
            format!("  at {file}:{line}\n")
        };
        let site = current_site().map(|s| String::from_utf8_lossy(&s).into_owned());
        eprint!(
            "vx: warning: a task saw the cancel of its `tasks` block {CANCELS_TO_WARN} times. A \
             loop may wait again after each `Cancelled`.\n{wait}{}",
            site.unwrap_or_else(|| "  in the main code\n".to_string())
        );
    }
    let late = drop_deadline().is_some_and(|d| crate::ffi::wait::vx_clock_ns() >= d);
    (set || late) as i32
}

/// See `vx_cancelled`.
const CANCELS_TO_WARN: u32 = 1000;

/// Swaps the count of `vx_cancelled` of this thread with `count`. The shared runtime gives each
/// fiber its own.
pub(crate) fn swap_cancels(count: &mut u32) {
    CANCELS.with(|n| *count = n.replace(*count));
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
    vx_task_start_at(group, run, env, std::ptr::null(), 0)
}

/// `vx_task_start` for a `spawn` at line `line` of the file `file`, a C string. An abort in the
/// task prints that line.
///
/// # Safety
/// As for `vx_task_start`. `file` is null or a C string that lives as long as the program.
#[no_mangle]
pub unsafe extern "C" fn vx_task_start_at(
    group: *mut VxTaskGroup,
    run: VxTaskFn,
    env: *mut u8,
    file: *const c_char,
    line: i64,
) -> i64 {
    let group = &mut *group;
    let mark = group.mark.clone();
    let site = site_of_spawn(file, line);
    lines::task_started();
    if let Some(sched) = group.sched {
        group.tasks.push(Task {
            thread: None,
            thread_done: None,
            fiber: Some(sched.spawn_at(run, env, Some(mark), site.clone())),
            env,
            site,
        });
        return (group.tasks.len() - 1) as i64;
    }
    let task_env = Env(env);
    let own_site = site.clone();
    let put_aside = deadlines_for_a_task();
    let thread_done = Arc::new(AtomicI64::new(0));
    let done = thread_done.clone();
    crate::ffi::wait::thread_started();
    let started = thread::Builder::new()
        .stack_size(TASK_STACK_BYTES)
        .spawn(move || {
            let task_env = task_env;
            let mut site = site;
            CURRENT.with(|c| *c.borrow_mut() = Some(mark));
            PUT_ASIDE.with(|p| *p.borrow_mut() = put_aside);
            swap_site(&mut site);
            run(task_env.0);
            lines::task_ended();
            crate::ffi::wait::task_ended();
            // The wake comes before the count of threads goes down. So the joining thread
            // never counts as waiting for a thread that the count no longer has.
            done.store(1, Ordering::SeqCst);
            unsafe { crate::ffi::wait::vx_wake_all(Arc::as_ptr(&done) as *const i64) };
            crate::ffi::wait::thread_ended();
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
        thread_done: Some(thread_done),
        fiber: None,
        env,
        site: own_site,
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

    /// The site that an abort on this thread would print, as text.
    fn site_text() -> String {
        let (bytes, len) = SITE_BYTES.with(|b| b.get());
        if bytes.is_null() {
            return String::new();
        }
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(bytes, len) }).into_owned()
    }

    /// The sites that a task and a task inside it saw.
    #[repr(C)]
    struct SiteEnv {
        outer: String,
        inner: String,
    }

    extern "C" fn record_inner(env: *mut u8) {
        let env = unsafe { &mut *(*(env as *mut *mut SiteEnv)) };
        env.inner = site_text();
    }

    /// Records its site, then starts a task at line 20 that records its own.
    extern "C" fn record_outer(env: *mut u8) {
        let sites = unsafe { *(env as *mut *mut SiteEnv) };
        unsafe { (*sites).outer = site_text() };
        let group = vx_tasks_begin();
        unsafe {
            let e = libc::malloc(std::mem::size_of::<*mut SiteEnv>()) as *mut *mut SiteEnv;
            e.write(sites);
            vx_task_start_at(group, record_inner, e as *mut u8, c"b.vx".as_ptr(), 20);
            vx_tasks_end(group);
        }
    }

    #[test]
    fn a_task_knows_the_spawn_of_each_task_around_it() {
        let mut sites = SiteEnv {
            outer: String::new(),
            inner: String::new(),
        };
        let group = vx_tasks_begin();
        unsafe {
            let e = libc::malloc(std::mem::size_of::<*mut SiteEnv>()) as *mut *mut SiteEnv;
            e.write(&mut sites);
            vx_task_start_at(group, record_outer, e as *mut u8, c"a.vx".as_ptr(), 7);
            vx_tasks_end(group);
        }
        assert_eq!(sites.outer, "  in the task from the `spawn` at a.vx:7\n");
        assert_eq!(
            sites.inner,
            "  in the task from the `spawn` at b.vx:20\n  in the task from the `spawn` at a.vx:7\n"
        );
        // The code outside each task has no site.
        assert_eq!(site_text(), "");
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
