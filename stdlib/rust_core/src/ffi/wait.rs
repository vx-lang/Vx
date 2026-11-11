//===- wait.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A thread waits until an atomic word changes, and another thread wakes it. The channel and
// the `Mutex` of `std` use these two calls to block an OS thread while they wait
// (docs/discussions/waiting_tasks_channels_and_task_errors.md, W2 and M7).
//
// The words live in Vx memory. The runtime keeps a fixed table of buckets, each a lock and a
// condition variable, and a word uses the bucket of its address. A waiter takes the lock, reads
// the word, and sleeps only if the word still holds the value that it saw. A waker changes the
// word before it calls `vx_wake_all`, which takes the same lock. So a wake can never come
// between the read and the sleep, and no wake is lost.
//
// A task of the shared runtime (sched.rs) does not sleep: it parks. Under the lock it goes on
// the list of fibers of the bucket, with the address of its word. A wake of a word puts each
// fiber of that word back on the queue. A fiber of another word of the bucket stays parked, so
// the queue does not depend on which words share a bucket. A cancel wakes only the fibers whose
// wait can be cancelled: another fiber would only park again, and with a seed, its extra turn
// changes the order of the tasks.
//
//===----------------------------------------------------------------------===//

use crate::ffi::sched::{self, FiberRef};
use crate::ffi::tsan;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

const BUCKETS: usize = 64;

struct Bucket {
    lock: Mutex<Waiters>,
    cond: Condvar,
}

struct Waiters {
    /// The fibers that park on a word of this bucket.
    fibers: Vec<Parked>,
    /// The threads that sleep here with no time limit, which the deadlock watcher counts.
    sleepers: Vec<std::thread::ThreadId>,
    /// The scheduler of each fiber that sleeps here on its worker's thread, under a callback.
    /// The scheduler counts it as blocked, not the watcher.
    blocked: Vec<&'static sched::Scheduler>,
    /// Goes up at each wake of the bucket.
    wakes: u64,
    /// What each thread and fiber above waits for, for the deadlock report. A timed sleep of a
    /// thread has no note: it cannot be part of a deadlock.
    notes: Vec<Note>,
}

/// A fiber that parks on a word.
struct Parked {
    fiber: FiberRef,
    /// The address of the word.
    word: usize,
    /// Whether a cancel ends the wait.
    can_cancel: bool,
}

/// What a wait is for, as the library says it with `vx_waiting_for`.
#[derive(Clone)]
enum What {
    Word,
    /// A `send` or a `recv`, with the address of the words of its channel, or 0 when it is not
    /// known.
    Send(usize),
    Recv(usize),
    /// A `lock`, with the address of the holder word of its `Mutex`, or 0 when it is not known.
    Lock(usize),
    Select,
    /// The join of the task with this site, or of a task with no site.
    Join(Option<crate::ffi::tasks::Site>),
}

/// A line of the program: a `Caller` of the library.
#[derive(Clone, Copy)]
struct Place {
    /// A C string that lives as long as the program: a string literal.
    file: usize,
    line: i32,
}

impl Place {
    /// `None` when `file` is null.
    fn new(file: *const libc::c_char, line: i32) -> Option<Place> {
        (!file.is_null()).then_some(Place {
            file: file as usize,
            line,
        })
    }

    fn text(self) -> String {
        let file = unsafe { std::ffi::CStr::from_ptr(self.file as *const libc::c_char) };
        format!("{}:{}", file.to_string_lossy(), self.line)
    }
}

/// What the next wait of a thread is for, and its lines.
#[derive(Clone)]
struct Waiting {
    what: What,
    /// The line of the wait.
    at: Option<Place>,
    /// The line that made the channel or the lock.
    made: Option<Place>,
}

const NOT_SAID: Waiting = Waiting {
    what: What::Word,
    at: None,
    made: None,
};

/// One waiter of a bucket, for the deadlock report.
struct Note {
    /// The thread or the fiber that waits, so that it can take its own note away.
    owner: usize,
    word: usize,
    what: What,
    at: Option<Place>,
    made: Option<Place>,
    /// Who waits, or `None` for a task with no site.
    who: Option<Who>,
}

/// The code that waits, or that holds a lock, for the cycle of waits in the deadlock report.
#[derive(Clone)]
enum Who {
    Main,
    Task(crate::ffi::tasks::Site),
}

impl Who {
    /// Who runs here. `None` for a task with no site, which the report cannot tell apart from
    /// another task.
    fn here() -> Option<Who> {
        match crate::ffi::tasks::current_site() {
            Some(site) => Some(Who::Task(site)),
            None if sched::current_fiber().is_null() && on_the_main_thread() => Some(Who::Main),
            None => None,
        }
    }

    /// Each task has its own site, so the address of the site names the task.
    fn key(&self) -> usize {
        match self {
            Who::Main => HELD_BY_THE_MAIN_CODE as usize,
            Who::Task(site) => std::sync::Arc::as_ptr(site) as *const u8 as usize,
        }
    }

    fn text(&self) -> String {
        match self {
            Who::Main => "the main code".to_string(),
            Who::Task(site) => format!(
                "the task from the `spawn` at {}",
                crate::ffi::tasks::spawn_line(site)
            ),
        }
    }
}

fn on_the_main_thread() -> bool {
    #[cfg(target_os = "linux")]
    return unsafe { libc::gettid() == libc::getpid() };
    #[cfg(target_vendor = "apple")]
    return unsafe { libc::pthread_main_np() != 0 };
    #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
    return false;
}

/// The channel ends that each task moved at its `spawn`: the words of the channel, whether the
/// end is a `Sender`, and the task. The entries of a task go at its end.
static ENDS: Mutex<Vec<(usize, bool, crate::ffi::tasks::Site)>> = Mutex::new(Vec::new());
/// The entries of `ENDS`, so that the end of a task takes no lock when it is empty.
static ENDS_HELD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn ends() -> std::sync::MutexGuard<'static, Vec<(usize, bool, crate::ffi::tasks::Site)>> {
    ENDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The task that runs holds an end of the channel at `words`: a `Sender` when `sender` is 1,
/// the `Receiver` when it is 0. The checker puts a call at the start of a task for each end that
/// the task moves. The report then knows who can end a wait on the channel. An end that moves
/// again after the `spawn` keeps its first holder here.
///
/// # Safety
///
/// `words` is the words of a channel.
#[no_mangle]
pub unsafe extern "C" fn vx_task_holds_an_end(words: *mut i64, sender: i32) {
    if let Some(site) = crate::ffi::tasks::current_site() {
        ends().push((words as usize, sender != 0, site));
        ENDS_HELD.fetch_add(1, Ordering::SeqCst);
    }
}

/// The task that runs ends: it holds no end now.
pub(crate) fn task_ended() {
    if ENDS_HELD.load(Ordering::SeqCst) == 0 {
        return;
    }
    let Some(site) = crate::ffi::tasks::current_site() else {
        return;
    };
    let mut ends = ends();
    let before = ends.len();
    ends.retain(|e| !std::sync::Arc::ptr_eq(&e.2, &site));
    ENDS_HELD.fetch_sub(before - ends.len(), Ordering::SeqCst);
}

/// The one task that holds each end of its kind of the channel at `words` that is not dropped:
/// each `Sender` when `sender`, or the `Receiver`. `None` when the ends that the tasks moved are
/// not all the ends that are not dropped, or when two tasks hold them.
fn only_holder(words: usize, sender: bool) -> Option<crate::ffi::tasks::Site> {
    let word = if sender { SENDERS } else { RECEIVER };
    let live = unsafe { &*(words as *const AtomicI64).add(word) }.load(Ordering::SeqCst);
    let ends = ends();
    let held: Vec<_> = ends
        .iter()
        .filter(|e| e.0 == words && e.1 == sender)
        .collect();
    let first = held.first()?;
    let one_task = held.iter().all(|e| std::sync::Arc::ptr_eq(&e.2, &first.2));
    (held.len() as i64 == live && one_task).then(|| first.2.clone())
}

/// The holder word of a `Mutex` holds 0 when no code holds the lock, or when the report cannot
/// name the holder. It holds 1 for the main code. For a task, it holds the address of the site of
/// the task, and it keeps one count of the site, so that the address stays the task's.
const HELD_BY_THE_MAIN_CODE: i64 = 1;

/// `lock` took the lock of the `Mutex` with the holder word `holder`.
///
/// # Safety
///
/// `holder` is the holder word of a live `Mutex`.
#[no_mangle]
pub unsafe extern "C" fn vx_lock_taken(holder: *mut i64) {
    let held = match Who::here() {
        Some(Who::Main) => HELD_BY_THE_MAIN_CODE,
        Some(Who::Task(site)) => std::sync::Arc::into_raw(site) as *const u8 as i64,
        None => 0,
    };
    unsafe { &*(holder as *const AtomicI64) }.store(held, Ordering::SeqCst);
}

/// The drop of a guard unlocks the `Mutex` with the holder word `holder`. A guard is not `Send`,
/// so the task that took the lock unlocks it.
///
/// # Safety
///
/// `holder` is the holder word of a live `Mutex`.
#[no_mangle]
pub unsafe extern "C" fn vx_lock_released(holder: *mut i64) {
    let held = unsafe { &*(holder as *const AtomicI64) }.swap(0, Ordering::SeqCst);
    if held > HELD_BY_THE_MAIN_CODE {
        let site = crate::ffi::tasks::current_site();
        let site = site.expect("only the task that holds a lock unlocks it");
        let ptr = std::sync::Arc::as_ptr(&site);
        assert_eq!(
            ptr as *const u8 as i64, held,
            "only the task that holds a lock unlocks it"
        );
        unsafe { std::sync::Arc::decrement_strong_count(ptr) };
    }
}

thread_local! {
    /// What the next wait of this thread is for. The wait takes it.
    static WHAT: std::cell::RefCell<Waiting> = const { std::cell::RefCell::new(NOT_SAID) };
}

/// Says what the next wait of this thread is for: 1 for a `send`, 2 for a `recv` and 4 for a
/// `select`. The deadlock report prints it, with the line of the wait at `at_file:at_line`, and
/// the line that made the channel at `made_file:made_line`. A null file means that the line is
/// not known. For a `send` and a `recv`, `channel` is the words of the channel, and the report
/// counts the ends of the channel there. A `lock` uses `vx_waiting_for_lock`.
///
/// # Safety
///
/// Each file that is not null is a C string that lives as long as the program. A `channel` that
/// is not null is the words of a channel that lives until the wait ends.
#[no_mangle]
pub unsafe extern "C" fn vx_waiting_for(
    what: i32,
    at_file: *const libc::c_char,
    at_line: i32,
    made_file: *const libc::c_char,
    made_line: i32,
    channel: *mut i64,
) {
    let what = match what {
        1 => What::Send(channel as usize),
        2 => What::Recv(channel as usize),
        4 => What::Select,
        _ => What::Word,
    };
    let waiting = Waiting {
        what,
        at: Place::new(at_file, at_line),
        made: Place::new(made_file, made_line),
    };
    WHAT.with(|w| *w.borrow_mut() = waiting);
}

/// As `vx_waiting_for` a `lock`, on the `Mutex` with the holder word `holder`.
///
/// # Safety
///
/// As for `vx_waiting_for`, and `holder` is the holder word of a `Mutex` that lives until the
/// wait ends.
#[no_mangle]
pub unsafe extern "C" fn vx_waiting_for_lock(
    at_file: *const libc::c_char,
    at_line: i32,
    made_file: *const libc::c_char,
    made_line: i32,
    holder: *mut i64,
) {
    let waiting = Waiting {
        what: What::Lock(holder as usize),
        at: Place::new(at_file, at_line),
        made: Place::new(made_file, made_line),
    };
    WHAT.with(|w| *w.borrow_mut() = waiting);
}

/// The next wait of this thread is the join of a task with the site `site`.
pub(crate) fn waiting_for_join(site: Option<crate::ffi::tasks::Site>) {
    WHAT.with(|w| {
        *w.borrow_mut() = Waiting {
            what: What::Join(site),
            ..NOT_SAID
        }
    });
}

/// The owner of a note for this thread.
fn this_thread() -> usize {
    WHAT.with(|w| w as *const _ as usize)
}

static TABLE: [Bucket; BUCKETS] = [const {
    Bucket {
        lock: Mutex::new(Waiters {
            fibers: Vec::new(),
            sleepers: Vec::new(),
            blocked: Vec::new(),
            wakes: 0,
            notes: Vec::new(),
        }),
        cond: Condvar::new(),
    }
}; BUCKETS];

/// A held lock of a bucket that ThreadSanitizer can see. This library is not built for it, so it
/// does not see the futex under a `Mutex`. Without these calls, it took a `Vec` that one thread
/// grew and another freed under the lock for a race.
struct Held<'a> {
    guard: Option<std::sync::MutexGuard<'a, Waiters>>,
    b: &'a Bucket,
}

impl<'a> Held<'a> {
    fn new(b: &'a Bucket) -> Held<'a> {
        let guard = b.lock.lock().unwrap_or_else(|e| e.into_inner());
        tsan::acquire(b);
        Held {
            guard: Some(guard),
            b,
        }
    }

    /// `Condvar::wait`, with no time limit or with the limit `left`.
    fn wait(mut self, left: Option<Duration>) -> Held<'a> {
        let guard = self.guard.take().expect("a held lock");
        tsan::release(self.b);
        let guard = match left {
            Some(left) => {
                self.b
                    .cond
                    .wait_timeout(guard, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
            None => self.b.cond.wait(guard).unwrap_or_else(|e| e.into_inner()),
        };
        tsan::acquire(self.b);
        Held {
            guard: Some(guard),
            b: self.b,
        }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if self.guard.is_some() {
            tsan::release(self.b);
        }
    }
}

impl std::ops::Deref for Held<'_> {
    type Target = Waiters;
    fn deref(&self) -> &Waiters {
        self.guard.as_ref().expect("a held lock")
    }
}

impl std::ops::DerefMut for Held<'_> {
    fn deref_mut(&mut self) -> &mut Waiters {
        self.guard.as_mut().expect("a held lock")
    }
}

fn bucket(word: *const i64) -> &'static Bucket {
    // A word is 8-byte aligned, so the three low bits of its address are always 0.
    &TABLE[(word as usize >> 3) % BUCKETS]
}

/// Sleep while the atomic word at `word` holds `value`. A task of the shared runtime parks in
/// place of the sleep. The call can return early, when another word of the same bucket changes.
/// So the caller reads its word again and loops.
///
/// # Safety
///
/// `word` points to a live, 8-byte aligned `i64` that other threads change only atomically.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while(word: *const i64, value: i64) {
    wait(word, value, false, None);
}

/// Sleeps or parks while the word holds `value`, and with `or_cancel`, while the block of this
/// task or thread was not cancelled. With a deadline of `vx_clock_ns`, it also returns at the
/// deadline: a thread sleeps with a time limit, and a fiber asks the timer to wake its bucket.
unsafe fn wait(word: *const i64, value: i64, or_cancel: bool, deadline_ns: Option<i64>) {
    // In a `drop` or a call of `within`, a wait that can be cancelled ends at the earliest deadline
    // of the scopes around it (F6, F6a).
    let deadline_ns = match crate::ffi::tasks::drop_deadline().filter(|_| or_cancel) {
        Some(d) => Some(deadline_ns.map_or(d, |own| own.min(d))),
        None => deadline_ns,
    };
    let b = bucket(word);
    let Waiting { what, at, made } =
        WHAT.with(|w| std::mem::replace(&mut *w.borrow_mut(), NOT_SAID));
    let mut waiters = Held::new(b);
    let atomic = unsafe { &*(word as *const AtomicI64) };
    if atomic.load(Ordering::SeqCst) != value
        || (or_cancel && crate::ffi::tasks::vx_cancelled() != 0)
    {
        return;
    }
    let left = match deadline_ns {
        Some(deadline) => {
            let now = vx_clock_ns();
            if now >= deadline {
                return;
            }
            Some(Duration::from_nanos((deadline - now) as u64))
        }
        None => None,
    };
    let fiber = sched::fiber_that_can_park();
    if left.is_none() {
        deadlock::watch();
    }
    let note = |owner: usize| Note {
        owner,
        word: word as usize,
        what,
        at,
        made,
        who: Who::here(),
    };
    if fiber.is_null() {
        // With a seed, this thread runs the ready tasks. The check is under the lock of the
        // bucket, and a task that gets ready wakes each bucket, so no wake is lost.
        if let Some(sched) = sched::scheduler_here().filter(|s| s.has_ready()) {
            drop(waiters);
            sched.run_one_here();
            return;
        }
        match left {
            Some(left) => drop(waiters.wait(Some(left))),
            None => {
                // The thread counts as waiting until a wake of the bucket, not until the OS
                // runs it again. A woken thread can wait a long time for a CPU, and it must not
                // look stuck then.
                let seen = waiters.wakes;
                // A worker's thread is not a thread that the watcher counts. Its fiber, which
                // cannot park under a callback, counts in its scheduler.
                let blocked_in = sched::current_scheduler();
                match blocked_in {
                    Some(s) => {
                        waiters.blocked.push(s);
                        s.blocked();
                    }
                    None => {
                        waiters.sleepers.push(std::thread::current().id());
                        deadlock::sleep_started();
                    }
                }
                let me = this_thread();
                waiters.notes.push(note(me));
                let mut waiters = waiters.wait(None);
                if waiters.wakes == seen {
                    let i = waiters
                        .notes
                        .iter()
                        .position(|n| n.owner == me)
                        .expect("a waiter has a note in its bucket");
                    waiters.notes.swap_remove(i);
                    match blocked_in {
                        Some(s) => {
                            let i = waiters
                                .blocked
                                .iter()
                                .position(|&x| std::ptr::eq(x, s))
                                .expect("a blocked fiber is on the list of its bucket");
                            waiters.blocked.swap_remove(i);
                            s.unblocked();
                            deadlock::progress();
                        }
                        None => {
                            let me = std::thread::current().id();
                            let i = waiters
                                .sleepers
                                .iter()
                                .position(|&t| t == me)
                                .expect("a sleeping thread is on the list of its bucket");
                            waiters.sleepers.swap_remove(i);
                            deadlock::sleep_ended(1);
                        }
                    }
                }
            }
        }
        return;
    }
    sched::will_park(fiber);
    waiters.fibers.push(Parked {
        fiber: FiberRef(fiber),
        word: word as usize,
        can_cancel: or_cancel,
    });
    waiters.notes.push(note(fiber as usize));
    drop(waiters);
    // A wake of the bucket before the park leaves the fiber `WOKEN`, so the timer can fire now.
    if let Some(deadline) = deadline_ns {
        timer::wake_at(deadline, word);
    }
    sched::park();
}

/// As `vx_wait_while`, and also return when the block of this thread was cancelled. The cancel
/// is read under the lock of the bucket, and a cancel takes each lock after it sets its mark. So
/// a cancel cannot come between the check and the sleep.
///
/// # Safety
///
/// As for `vx_wait_while`.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while_or_cancel(word: *const i64, value: i64) {
    wait(word, value, true, None);
}

/// Nanoseconds since the first call, on a clock that never goes back. A time limit is a
/// deadline on this clock.
#[no_mangle]
pub extern "C" fn vx_clock_ns() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as i64
}

/// `vx_wait_while`, or `vx_wait_while_or_cancel` when `or_cancel` is not 0, that also returns
/// at the deadline `deadline_ns` of `vx_clock_ns`. The caller reads the clock again.
/// It can return a little after the deadline, but never stays parked or asleep past it.
///
/// # Safety
///
/// As for `vx_wait_while`.
#[no_mangle]
pub unsafe extern "C" fn vx_wait_while_until(
    word: *const i64,
    value: i64,
    deadline_ns: i64,
    or_cancel: i32,
) {
    wait(word, value, or_cancel != 0, Some(deadline_ns));
}

/// Wake each thread that waits on the word at `word`. Call it after the change to the word.
///
/// # Safety
///
/// `word` is an address that `vx_wait_while` may have been called with. It is not read.
#[no_mangle]
pub unsafe extern "C" fn vx_wake_all(word: *const i64) {
    wake(bucket(word), Fibers::Of(word as usize));
}

/// Wake each thread that waits on any word, and each fiber whose wait a cancel ends. A cancel
/// uses it: each woken wait reads its word again, and a wait that can be cancelled sees the cancel.
pub(crate) fn wake_every_thread() {
    for b in &TABLE {
        wake(b, Fibers::ThatCanCancel);
    }
}

/// Wakes each thread that sleeps on any word, and no fiber. The thread that runs the tasks of
/// a seed sleeps on any word, and a fiber that wakes for no reason changes the order of a seed.
pub(crate) fn wake_every_sleeper() {
    for b in &TABLE {
        wake(b, Fibers::None);
    }
}

/// The fibers of a bucket that a wake puts back on the queue.
#[derive(Clone, Copy)]
enum Fibers {
    /// The fibers that park on the word at this address.
    Of(usize),
    /// The fibers whose wait a cancel ends.
    ThatCanCancel,
    None,
}

/// Takes the fibers that `which` names off the list of the bucket.
fn take_fibers(waiters: &mut Waiters, which: Fibers) -> Vec<FiberRef> {
    let (taken, kept) = std::mem::take(&mut waiters.fibers)
        .into_iter()
        .partition(|p: &Parked| match which {
            Fibers::Of(word) => p.word == word,
            Fibers::ThatCanCancel => p.can_cancel,
            Fibers::None => false,
        });
    waiters.fibers = kept;
    taken.into_iter().map(|p| p.fiber).collect()
}

/// Drops the notes of the waiters that a wake took. A wake takes every thread, but a fiber of
/// another word stays parked, and it keeps its note for the deadlock report.
fn keep_the_notes_of_parked_fibers(waiters: &mut Waiters) {
    let parked: Vec<usize> = waiters.fibers.iter().map(|p| p.fiber.0 as usize).collect();
    waiters.notes.retain(|n| parked.contains(&n.owner));
}

/// Wakes each thread of the bucket, and the fibers that `which` names.
fn wake(b: &Bucket, which: Fibers) {
    deadlock::progress();
    let mut waiters = Held::new(b);
    let fibers = wake_under_the_lock(&mut waiters, which);
    drop(waiters);
    b.cond.notify_all();
    for fiber in fibers {
        sched::wake(fiber);
    }
}

/// The part of a wake that runs under the lock of the bucket. It takes the waiters off the
/// counts, and returns the fibers to put back on the queue.
fn wake_under_the_lock(waiters: &mut Waiters, which: Fibers) -> Vec<FiberRef> {
    let fibers = take_fibers(waiters, which);
    keep_the_notes_of_parked_fibers(waiters);
    deadlock::sleep_ended(std::mem::take(&mut waiters.sleepers).len() as i64);
    for s in std::mem::take(&mut waiters.blocked) {
        s.unblocked();
    }
    waiters.wakes = waiters.wakes.wrapping_add(1);
    fibers
}

/// One thread that wakes the bucket of a word at a deadline, for a fiber that parks with a time
/// limit. The wake puts each fiber of the word back on the queue. Each caller of a wait loops,
/// so a fiber that the deadline did not end parks again.
mod timer {
    use super::{bucket, vx_clock_ns, wake, Fibers};
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::sync::{Condvar, Mutex, OnceLock};
    use std::time::Duration;

    struct Timer {
        deadlines: Mutex<Deadlines>,
        cond: Condvar,
    }

    struct Deadlines {
        /// The deadlines, the earliest first, with the address of the word of each.
        heap: BinaryHeap<Reverse<(i64, usize)>>,
        /// The timer took a deadline off the heap and did not end its wake yet.
        firing: bool,
    }

    impl Deadlines {
        /// Whether a wake of the timer is still to come. Between the pop of a deadline and its
        /// wake, the heap can be empty, and no progress shows. So the deadline that the timer
        /// fires counts too.
        fn pending(&self) -> bool {
            !self.heap.is_empty() || self.firing
        }
    }

    static TIMER: OnceLock<&'static Timer> = OnceLock::new();

    /// Whether a deadline is still to come. Its wake can let a parked fiber go on.
    pub(super) fn pending() -> bool {
        TIMER.get().is_some_and(|t| {
            t.deadlines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pending()
        })
    }

    fn timer() -> &'static Timer {
        TIMER.get_or_init(|| {
            let t: &'static Timer = Box::leak(Box::new(Timer {
                deadlines: Mutex::new(Deadlines {
                    heap: BinaryHeap::new(),
                    firing: false,
                }),
                cond: Condvar::new(),
            }));
            std::thread::Builder::new()
                .name("vx-timer".into())
                .spawn(move || run(t))
                .expect("the system could not start the timer thread");
            t
        })
    }

    /// Wakes the bucket of `word` at `deadline_ns`. The word is not read, so it can be freed.
    pub(super) fn wake_at(deadline_ns: i64, word: *const i64) {
        let t = timer();
        let mut d = t.deadlines.lock().unwrap_or_else(|e| e.into_inner());
        d.heap.push(Reverse((deadline_ns, word as usize)));
        t.cond.notify_one();
    }

    fn run(t: &'static Timer) {
        let mut d = t.deadlines.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let Some(&Reverse((deadline, word))) = d.heap.peek() else {
                d = t.cond.wait(d).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            let now = vx_clock_ns();
            if now < deadline {
                let left = Duration::from_nanos((deadline - now) as u64);
                d = t
                    .cond
                    .wait_timeout(d, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                continue;
            }
            d.heap.pop();
            d.firing = true;
            drop(d);
            wake(bucket(word as *const i64), Fibers::Of(word));
            d = t.deadlines.lock().unwrap_or_else(|e| e.into_inner());
            d.firing = false;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_deadline_that_the_timer_fires_is_pending_until_its_wake() {
            let mut d = Deadlines {
                heap: BinaryHeap::new(),
                firing: false,
            };
            assert!(!d.pending(), "a timer with no deadline is pending");
            d.heap.push(Reverse((1, 8)));
            assert!(d.pending(), "a deadline on the heap is not pending");
            d.heap.pop();
            d.firing = true;
            assert!(
                d.pending(),
                "a deadline off the heap is not pending before its wake"
            );
        }
    }
}

/// One wait of the report: who waits, its word, what it waits for, its line, and the line that
/// made its channel or lock.
type Wait = (Option<Who>, usize, What, Option<Place>, Option<Place>);

/// One line for each waiter in the buckets: what it waits for, and then the lines of the `spawn`
/// of its task. The main code comes first, then the tasks in the order of their lines. Two
/// waits on one channel or one lock name the same number. Then each cycle of waits, when the
/// report knows what each wait in it waits for: the task of a join, the holder of a lock, or the
/// one task that holds the other ends of a channel.
fn report() -> String {
    let mut waits = Vec::new();
    for b in &TABLE {
        let w = Held::new(b);
        for n in &w.notes {
            waits.push((n.who.clone(), n.word, n.what.clone(), n.at, n.made));
        }
    }
    let site_of = |who: &Option<Who>| match who {
        Some(Who::Task(site)) => Some(site.clone()),
        _ => None,
    };
    waits.sort_by_key(|w| site_of(&w.0).as_ref().map(crate::ffi::tasks::site_key));
    // The waiter with the key `key`.
    let waiter = |key: usize| {
        waits
            .iter()
            .position(|w| w.0.as_ref().is_some_and(|who| who.key() == key))
    };
    // The waiter that each waiter waits for, when the report knows it.
    let waits_for: Vec<Option<usize>> = waits
        .iter()
        .map(|w| match &w.2 {
            What::Lock(holder) if *holder != 0 => {
                let held = unsafe { &*(*holder as *const AtomicI64) }.load(Ordering::SeqCst);
                waiter(held as usize).filter(|_| held != 0)
            }
            What::Join(Some(site)) => waiter(std::sync::Arc::as_ptr(site) as *const u8 as usize),
            // A `recv` waits for a `Sender`, and a `send` on a full channel for the `Receiver`.
            What::Recv(words) | What::Send(words) if *words != 0 => {
                let site = only_holder(*words, matches!(w.2, What::Recv(_)))?;
                waiter(std::sync::Arc::as_ptr(&site) as *const u8 as usize)
            }
            _ => None,
        })
        .collect();
    let (mut channels, mut locks) = (Vec::new(), Vec::new());
    let number = |list: &mut Vec<usize>, word: usize| match list.iter().position(|&w| w == word) {
        Some(i) => i + 1,
        None => {
            list.push(word);
            list.len()
        }
    };
    // `, which only the task from the `spawn` at main.vx:20 can send on`.
    let only = |i: usize, verb: &str| {
        (waits_for[i].and_then(|h| waits[h].0.as_ref()))
            .map(|who| format!(", which only {} can {verb}", who.text()))
            .unwrap_or_default()
    };
    let mut doing = Vec::new();
    for (i, (_, word, what, at, made)) in waits.iter().enumerate() {
        // ` at main.vx:15` and `, made at main.vx:11`, or nothing when the line is not known.
        let at = at.map(|p| format!(" at {}", p.text())).unwrap_or_default();
        let made = made
            .map(|p| format!(", made at {}", p.text()))
            .unwrap_or_default();
        doing.push(match what {
            What::Send(_) => format!(
                "waits in `send`{at} on channel {}{made}{}",
                number(&mut channels, *word),
                only(i, "receive from")
            ),
            What::Recv(_) => format!(
                "waits in `recv`{at} on channel {}{made}{}",
                number(&mut channels, *word),
                only(i, "send on")
            ),
            What::Lock(_) => {
                let held = waits_for[i]
                    .and_then(|h| waits[h].0.as_ref())
                    .map(|who| format!(", held by {}", who.text()))
                    .unwrap_or_default();
                format!(
                    "waits in `lock`{at} on lock {}{made}{held}",
                    number(&mut locks, *word)
                )
            }
            What::Select => format!("waits in `select`{at}"),
            What::Join(Some(task)) => format!(
                "waits for the end of the task from the `spawn` at {}",
                crate::ffi::tasks::spawn_line(task)
            ),
            What::Join(None) => "waits for the end of a task".to_string(),
            What::Word => "waits on an atomic word".to_string(),
        });
    }
    let mut out = String::new();
    for (w, doing) in waits.iter().zip(&doing) {
        let who = match &w.0 {
            Some(Who::Main) => "the main code",
            _ => "a task",
        };
        out += &format!("  {who} {doing}\n");
        for line in site_of(&w.0)
            .as_deref()
            .map(String::from_utf8_lossy)
            .unwrap_or_default()
            .lines()
        {
            out += &format!("  {line}\n");
        }
    }
    for cycle in cycles(&waits_for) {
        out += "  the waits form a cycle:\n";
        for i in cycle {
            let who = waits[i]
                .0
                .as_ref()
                .expect("a wait in a cycle has a known waiter");
            out += &format!("    {} {}\n", who.text(), doing[i]);
        }
    }
    if !channels.is_empty() {
        out += "  the channels of these waits:\n";
    }
    for (i, &word) in channels.iter().enumerate() {
        out += &format!("    channel {}{}\n", i + 1, channel_text(&waits, word));
    }
    out
}

/// The index of the count of `Sender`s, and of the word that is 1 until the `Receiver` is
/// dropped, in the words of a channel, as stdlib/std/sync/channel.vx lays them out.
const SENDERS: usize = 2;
const RECEIVER: usize = 3;

/// The waits on the channel whose waits sleep on `word`, and its `Sender`s that are not dropped:
/// `, made at main.vx:43: 1 wait in `recv` and 0 in `send`. 1 `Sender` is not dropped.` A wait
/// holds the `Receiver`, or waits only while it is not dropped, so the text does not name it.
fn channel_text(waits: &[Wait], word: usize) -> String {
    let on_it = |w: &&Wait| w.1 == word;
    let recvs = (waits.iter().filter(on_it))
        .filter(|w| matches!(w.2, What::Recv(_)))
        .count();
    let sends = (waits.iter().filter(on_it))
        .filter(|w| matches!(w.2, What::Send(_)))
        .count();
    let (words, made) = (waits.iter().filter(on_it))
        .find_map(|w| match w.2 {
            What::Send(words) | What::Recv(words) => Some((words, w.4)),
            _ => None,
        })
        .expect("a channel of the report has a wait on it");
    let made = made
        .map(|p| format!(", made at {}", p.text()))
        .unwrap_or_default();
    let plural = if recvs == 1 { "wait" } else { "waits" };
    let mut text = format!("{made}: {recvs} {plural} in `recv` and {sends} in `send`.");
    if words != 0 {
        let senders = unsafe { &*(words as *const AtomicI64).add(SENDERS) };
        text += &match senders.load(Ordering::SeqCst) {
            0 => " Each `Sender` is dropped.".to_string(),
            1 => " 1 `Sender` is not dropped.".to_string(),
            n => format!(" {n} `Sender`s are not dropped."),
        };
    }
    text
}

/// Each cycle in a graph where node `i` has at most one edge, to `next[i]`. A cycle starts at
/// its lowest node.
fn cycles(next: &[Option<usize>]) -> Vec<Vec<usize>> {
    const NEW: u8 = 0;
    const ON_THE_PATH: u8 = 1;
    const DONE: u8 = 2;
    let mut state = vec![NEW; next.len()];
    let mut found = Vec::new();
    for start in 0..next.len() {
        let mut path = Vec::new();
        let mut i = Some(start);
        while let Some(at) = i {
            match state[at] {
                NEW => {
                    state[at] = ON_THE_PATH;
                    path.push(at);
                    i = next[at];
                }
                ON_THE_PATH => {
                    let first = path.iter().position(|&p| p == at).expect("on the path");
                    let mut cycle = path[first..].to_vec();
                    let lowest = (0..cycle.len()).min_by_key(|&k| cycle[k]).expect("a cycle");
                    cycle.rotate_left(lowest);
                    found.push(cycle);
                    break;
                }
                _ => break,
            }
        }
        for p in path {
            state[p] = DONE;
        }
    }
    found
}

/// The deadlock report (F9 in docs/discussions/waiting_tasks_channels_and_task_errors.md). A
/// watcher thread starts at the first wait with no time limit. On each tick, it looks for a
/// program that cannot go on:
///
/// - no fiber of the shared runtime runs or is ready,
/// - each Vx thread that is not a worker waits in this file, as the join of a task thread does,
/// - no deadline of the timer is to come,
/// - no fiber waits for a socket (io.rs),
/// - and no wake, and no end of a wait, came since the last tick.
///
/// A thread in a foreign call or in I/O is not in a wait here, so it never looks stuck. When
/// two ticks in a row see all of this, the program stops with a report.
mod deadlock {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
    use std::sync::Once;
    use std::time::Duration;

    /// The Vx threads that are not workers: the main thread, and each task thread of the
    /// runtime with one thread for each task. The monitor and the timer are not Vx threads.
    static THREADS: AtomicI64 = AtomicI64::new(1);
    /// The Vx threads that wait with no time limit.
    static WAITING: AtomicI64 = AtomicI64::new(0);
    /// Goes up at each wake and at each end of a wait.
    static PROGRESS: AtomicU64 = AtomicU64::new(0);
    static WATCHER: Once = Once::new();

    const TICK: Duration = Duration::from_millis(100);

    pub(crate) fn thread_started() {
        THREADS.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn thread_ended() {
        THREADS.fetch_sub(1, Ordering::SeqCst);
        progress();
    }

    pub(crate) fn progress() {
        PROGRESS.fetch_add(1, Ordering::SeqCst);
    }

    /// A thread starts to sleep with no time limit.
    pub(super) fn sleep_started() {
        WAITING.fetch_add(1, Ordering::SeqCst);
    }

    /// A wake ended the sleep of `n` threads, or one thread's sleep ended with no wake.
    pub(super) fn sleep_ended(n: i64) {
        if n > 0 {
            WAITING.fetch_sub(n, Ordering::SeqCst);
            progress();
        }
    }

    /// Starts the watcher, once. The unit tests of this crate run on threads that the count
    /// does not know, so they get no watcher.
    pub(super) fn watch() {
        if cfg!(test) {
            return;
        }
        WATCHER.call_once(|| {
            let started = std::thread::Builder::new()
                .name("vx-deadlock".into())
                .spawn(run);
            if let Err(e) = started {
                eprintln!("vx: the system could not start the deadlock watcher: {e}");
                std::process::abort();
            }
        });
    }

    fn stuck() -> bool {
        let fibers_parked = crate::ffi::sched::program_scheduler().is_none_or(|s| s.all_parked());
        fibers_parked
            && WAITING.load(Ordering::SeqCst) == THREADS.load(Ordering::SeqCst)
            && !super::timer::pending()
            && !crate::ffi::io::pending()
    }

    /// One check of the watcher. `seen` is the progress count of the last check, if that check
    /// found the program stuck. Returns the value of `seen` for the next check, and true to report.
    fn check_once(seen: Option<u64>, stuck: bool, progress: u64) -> (Option<u64>, bool) {
        let now = stuck.then_some(progress);
        (now, now.is_some() && now == seen)
    }

    fn run() {
        let mut seen = None;
        loop {
            std::thread::sleep(TICK);
            let progress = PROGRESS.load(Ordering::SeqCst);
            let (now, report) = check_once(seen, stuck(), progress);
            if report {
                eprint!(
                    "vx: deadlock: each task of the program waits, and nothing can wake one. \
                     The program stops.\n{}",
                    super::report()
                );
                std::process::abort();
            }
            seen = now;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::check_once;

        /// Runs the checks of the watcher in order, each a pair (stuck, progress). Returns the
        /// number of the check that reports, if one does.
        fn report_at(checks: &[(bool, u64)]) -> Option<usize> {
            let mut seen = None;
            for (i, &(stuck, progress)) in checks.iter().enumerate() {
                let (now, report) = check_once(seen, stuck, progress);
                if report {
                    return Some(i);
                }
                seen = now;
            }
            None
        }

        #[test]
        fn one_stuck_check_does_not_report() {
            assert_eq!(report_at(&[(true, 5)]), None);
        }

        #[test]
        fn two_stuck_checks_with_no_progress_report_at_the_second() {
            assert_eq!(report_at(&[(false, 1), (true, 5), (true, 5)]), Some(2));
        }

        /// The timer thread or the io thread woke a task between the two checks. The task can
        /// go on, so no report.
        #[test]
        fn a_wake_between_two_stuck_checks_stops_the_report() {
            assert_eq!(report_at(&[(true, 5), (true, 6)]), None);
        }

        /// A check that does not find the program stuck starts the count again, even with no
        /// progress. For example, a deadline of the timer was still to come at the middle check.
        #[test]
        fn a_check_that_does_not_find_the_program_stuck_starts_the_count_again() {
            assert_eq!(report_at(&[(true, 5), (false, 5), (true, 5)]), None);
            assert_eq!(
                report_at(&[(true, 5), (false, 5), (true, 5), (true, 5)]),
                Some(3)
            );
        }
    }
}

pub(crate) use deadlock::{thread_ended, thread_started};

/// Two words for `select`, which waits on several channels at once. Each channel event adds 1 to
/// the first word, and the second word counts the threads in a `select` that sleep on it.
static ANY_CHANNEL: [AtomicI64; 2] = [AtomicI64::new(0), AtomicI64::new(0)];

/// The address of the two words for `select`.
#[no_mangle]
pub extern "C" fn vx_any_channel_words() -> *mut i64 {
    ANY_CHANNEL.as_ptr() as *mut i64
}

/// The send numbers that a cancelled `send` left without a value, with the address of the words
/// of their channel. The `recv` of such a number passes it. A cancel is rare, so one lock for all
/// channels is enough.
static SKIPS: Mutex<Vec<(usize, i64)>> = Mutex::new(Vec::new());

fn skips() -> std::sync::MutexGuard<'static, Vec<(usize, i64)>> {
    SKIPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A cancelled `send` of the channel at `words` left the send number `k` without a value.
#[no_mangle]
pub extern "C" fn vx_channel_skip(words: *mut i64, k: i64) {
    let mut s = skips();
    assert!(
        !s.contains(&(words as usize, k)),
        "a send number was skipped twice"
    );
    s.push((words as usize, k));
}

/// 1 when a cancelled `send` left the send number `k` without a value, and then forget it.
#[no_mangle]
pub extern "C" fn vx_channel_take_skip(words: *mut i64, k: i64) -> i32 {
    let mut s = skips();
    match s.iter().position(|&e| e == (words as usize, k)) {
        Some(i) => {
            s.swap_remove(i);
            1
        }
        None => 0,
    }
}

/// The channel at `words` is freed. Forget each of its numbers that no `recv` passed.
#[no_mangle]
pub extern "C" fn vx_channel_forget_skips(words: *mut i64) {
    skips().retain(|&(w, _)| w != words as usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread::ThreadId;

    // Other tests can wait on words of the same bucket at the same time. So each test counts
    // only its own thread or its own scheduler.

    /// How many times the thread `t` is on the list of sleepers of `b`.
    fn sleeps(b: &Bucket, t: ThreadId) -> usize {
        b.lock
            .lock()
            .unwrap()
            .sleepers
            .iter()
            .filter(|&&x| x == t)
            .count()
    }

    /// How many fibers of `s` are on the list of blocked fibers of `b`.
    fn blocked_in(w: &Waiters, s: &sched::Scheduler) -> usize {
        w.blocked.iter().filter(|&&x| std::ptr::eq(x, s)).count()
    }

    /// A thread that waits on `word` until it is not 0, and counts each wait in `rounds`.
    fn sleeper(word: &Arc<AtomicI64>, rounds: &Arc<AtomicI64>) -> std::thread::JoinHandle<()> {
        let (w, r) = (word.clone(), rounds.clone());
        std::thread::spawn(move || {
            while w.load(Ordering::SeqCst) == 0 {
                r.fetch_add(1, Ordering::SeqCst);
                unsafe { vx_wait_while(w.as_ptr(), 0) };
            }
        })
    }

    #[test]
    fn a_wait_has_a_note_with_what_it_waits_for_until_its_wake() {
        let word: &'static AtomicI64 = Box::leak(Box::new(AtomicI64::new(0)));
        let at = word.as_ptr() as usize;
        let waiter = std::thread::spawn(move || {
            while word.load(Ordering::SeqCst) == 0 {
                unsafe {
                    vx_waiting_for(
                        2,
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        0,
                        std::ptr::null_mut(),
                    )
                };
                unsafe { vx_wait_while(at as *const i64, 0) };
            }
        });
        let b = bucket(word.as_ptr());
        let notes_of_the_word = |recv_only: bool| {
            b.lock
                .lock()
                .unwrap()
                .notes
                .iter()
                .filter(|n| n.word == at && (!recv_only || matches!(n.what, What::Recv(_))))
                .count()
        };
        while notes_of_the_word(true) == 0 {
            std::thread::yield_now();
        }
        word.store(1, Ordering::SeqCst);
        wake(b, Fibers::Of(at));
        waiter.join().unwrap();
        assert_eq!(notes_of_the_word(false), 0, "a woken wait kept its note");
    }

    #[test]
    fn a_wake_stops_counting_a_thread_before_the_thread_runs_again() {
        let word = Arc::new(AtomicI64::new(0));
        let rounds = Arc::new(AtomicI64::new(0));
        let sleeper = sleeper(&word, &rounds);
        let me = sleeper.thread().id();
        let b = bucket(word.as_ptr());
        while sleeps(b, me) == 0 {
            std::thread::yield_now();
        }
        word.store(1, Ordering::SeqCst);
        wake(b, Fibers::None);
        // The sleeper needs this lock to come back from its wait, so it is still in it here.
        let counted = sleeps(b, me);
        sleeper.join().unwrap();
        assert_eq!(counted, 0, "a woken thread still counts as waiting");
    }

    #[test]
    fn a_thread_that_wakes_with_no_wake_counts_once() {
        let word = Arc::new(AtomicI64::new(0));
        let rounds = Arc::new(AtomicI64::new(0));
        let sleeper = sleeper(&word, &rounds);
        let me = sleeper.thread().id();
        let b = bucket(word.as_ptr());
        // The count is the one that ended the wait. The system can return the sleeper with no
        // wake at any time, so a second read can give 0.
        let asleep_in_round = |n| loop {
            if rounds.load(Ordering::SeqCst) >= n {
                let count = sleeps(b, me);
                if count != 0 {
                    return count;
                }
            }
            std::thread::yield_now();
        };
        asleep_in_round(1);
        // A return of the condition variable with no wake, as the system may give.
        drop(b.lock.lock().unwrap());
        b.cond.notify_all();
        let counted = asleep_in_round(2);
        word.store(1, Ordering::SeqCst);
        wake(b, Fibers::Of(word.as_ptr() as usize));
        sleeper.join().unwrap();
        assert_eq!(
            counted, 1,
            "a thread counted twice after a return with no wake"
        );
    }

    fn parked(word: usize, can_cancel: bool) -> Parked {
        Parked {
            fiber: FiberRef(std::ptr::null_mut()),
            word,
            can_cancel,
        }
    }

    /// A wake of a word must not take a fiber that parks on another word of the same bucket, and
    /// a cancel must not take a fiber whose wait a cancel does not end. With a seed, such a wake
    /// changes the order of the tasks. The fibers here never run.
    #[test]
    fn a_wake_takes_only_the_fibers_that_it_names() {
        let mut waiters = Waiters {
            fibers: vec![
                parked(8, false),
                parked(16, true),
                parked(8, true),
                parked(24, false),
            ],
            sleepers: Vec::new(),
            blocked: Vec::new(),
            notes: Vec::new(),
            wakes: 0,
        };
        assert_eq!(
            take_fibers(&mut waiters, Fibers::None).len(),
            0,
            "a wake of the sleepers took a fiber"
        );
        assert_eq!(
            take_fibers(&mut waiters, Fibers::Of(8)).len(),
            2,
            "a wake of a word did not take each fiber of the word"
        );
        let words = |w: &Waiters| w.fibers.iter().map(|p| p.word).collect::<Vec<usize>>();
        assert_eq!(
            words(&waiters),
            [16, 24],
            "a wake of a word took a fiber of another word"
        );
        take_fibers(&mut waiters, Fibers::ThatCanCancel);
        let left = words(&waiters);
        assert!(
            left.contains(&24),
            "a cancel took a fiber whose wait it does not end"
        );
        assert!(
            !left.contains(&16),
            "a cancel did not take the fiber whose wait it ends"
        );
    }

    /// The report names the holder of the `Sender`s of a channel only when one task moved each
    /// `Sender` that is not dropped. The words here are those of a channel that nothing uses.
    #[test]
    fn only_one_task_that_moved_each_sender_is_the_only_holder() {
        let words: [AtomicI64; 4] = std::array::from_fn(|_| AtomicI64::new(0));
        let at = words.as_ptr() as usize;
        let (a, b): (crate::ffi::tasks::Site, crate::ffi::tasks::Site) = (
            b"  in the task from the `spawn` at a.vx:1\n"[..].into(),
            b"  in the task from the `spawn` at a.vx:2\n"[..].into(),
        );
        let is = |found: Option<crate::ffi::tasks::Site>, site: &crate::ffi::tasks::Site| {
            found.is_some_and(|f| std::sync::Arc::ptr_eq(&f, site))
        };
        words[SENDERS].store(2, Ordering::SeqCst);
        ends().push((at, true, a.clone()));
        assert!(
            only_holder(at, true).is_none(),
            "one of two `Sender`s is placed"
        );
        ends().push((at, true, b.clone()));
        assert!(
            only_holder(at, true).is_none(),
            "two tasks hold the `Sender`s"
        );
        ends().retain(|e| e.0 != at || !std::sync::Arc::ptr_eq(&e.2, &b));
        ends().push((at, true, a.clone()));
        assert!(
            is(only_holder(at, true), &a),
            "one task holds both `Sender`s"
        );
        assert!(
            only_holder(at, false).is_none(),
            "no task holds the `Receiver`"
        );
        ends().retain(|e| e.0 != at);
    }

    /// Waiter 0 waits for the cycle 1, 2, 3 and is not in it. Waiter 4 waits for itself, and
    /// waiter 5 waits for nothing.
    #[test]
    fn each_cycle_of_waits_is_found_once_from_its_lowest_waiter() {
        let next = [Some(2), Some(2), Some(3), Some(1), Some(4), None];
        assert_eq!(cycles(&next), vec![vec![1, 2, 3], vec![4]]);
    }

    /// A fiber that a wake leaves parked keeps its note for the deadlock report. The waiters
    /// that the wake took lose theirs. The fibers here never run.
    #[test]
    fn a_wake_keeps_the_notes_of_the_fibers_that_stay_parked() {
        let on_a = FiberRef(std::ptr::without_provenance_mut(8));
        let on_b = FiberRef(std::ptr::without_provenance_mut(16));
        let note = |owner: usize, word: usize| Note {
            owner,
            word,
            what: What::Recv(0),
            at: None,
            made: None,
            who: None,
        };
        let mut waiters = Waiters {
            fibers: vec![
                Parked {
                    fiber: on_a,
                    word: 100,
                    can_cancel: false,
                },
                Parked {
                    fiber: on_b,
                    word: 200,
                    can_cancel: false,
                },
            ],
            sleepers: Vec::new(),
            blocked: Vec::new(),
            // A thread that sleeps on word 100 has the owner 24.
            notes: vec![note(8, 100), note(16, 200), note(24, 100)],
            wakes: 0,
        };
        take_fibers(&mut waiters, Fibers::Of(100));
        keep_the_notes_of_parked_fibers(&mut waiters);
        let owners: Vec<usize> = waiters.notes.iter().map(|n| n.owner).collect();
        assert_eq!(
            owners,
            [16],
            "after a wake of a word, the notes are not those of the fibers that stay parked"
        );
    }

    struct Gate {
        rounds: AtomicI64,
        gate: AtomicI64,
        thread: Mutex<Option<ThreadId>>,
    }

    /// As a Vx function that foreign code calls, and that waits for the gate.
    extern "C" fn wait_for_the_gate_in_a_callback(env: *mut u8) {
        // The runtime frees the environment, which holds the address of the gate.
        let g = unsafe { &**(env as *const *const Gate) };
        *g.thread.lock().unwrap() = Some(std::thread::current().id());
        sched::vx_callback_enter();
        while g.gate.load(Ordering::SeqCst) == 0 {
            g.rounds.fetch_add(1, Ordering::SeqCst);
            unsafe { vx_wait_while(g.gate.as_ptr(), 0) };
        }
        sched::vx_callback_exit();
    }

    /// Starts one task on its own scheduler with one worker. The task waits for the gate under
    /// a callback.
    fn start_a_task_in_a_callback(
        g: &Gate,
    ) -> (
        &'static sched::Scheduler,
        *mut crate::ffi::tasks::VxTaskGroup,
    ) {
        let sched = sched::Scheduler::start(1, 1);
        let group = crate::ffi::tasks::tasks_begin(Some(sched));
        unsafe {
            let e = libc::malloc(std::mem::size_of::<*const Gate>()) as *mut *const Gate;
            e.write(g);
            crate::ffi::tasks::vx_task_start(group, wait_for_the_gate_in_a_callback, e as *mut u8);
        }
        (sched, group)
    }

    fn new_gate() -> Gate {
        Gate {
            rounds: AtomicI64::new(0),
            gate: AtomicI64::new(0),
            thread: Mutex::new(None),
        }
    }

    #[test]
    fn a_task_that_waits_in_a_callback_counts_as_parked_until_its_wake() {
        let g = new_gate();
        let (sched, group) = start_a_task_in_a_callback(&g);
        let b = bucket(g.gate.as_ptr());
        // Under one hold of the bucket lock, so that a wake from another test that uses the
        // same bucket cannot come between the two reads.
        let parked_while_waiting = loop {
            let w = b.lock.lock().unwrap();
            if blocked_in(&w, sched) != 0 {
                break sched.all_parked();
            }
            drop(w);
            std::thread::yield_now();
        };
        let worker = g.thread.lock().unwrap().expect("the task ran");
        let worker_sleeps = sleeps(b, worker);
        g.gate.store(1, Ordering::SeqCst);
        // The task needs the bucket lock to come back from its wait. The test holds the lock
        // from the wake to the read, so the task is still in its wait at the read.
        let mut w = b.lock.lock().unwrap();
        let fibers = wake_under_the_lock(&mut w, Fibers::Of(g.gate.as_ptr() as usize));
        let parked_after_the_wake = sched.all_parked();
        drop(w);
        b.cond.notify_all();
        assert!(fibers.is_empty(), "a task in a callback parked as a fiber");
        unsafe { crate::ffi::tasks::vx_tasks_end(group) };
        assert!(
            parked_while_waiting,
            "a task that waits in a callback did not count as parked"
        );
        assert_eq!(worker_sleeps, 0, "the watcher counted a worker's thread");
        assert!(
            !parked_after_the_wake,
            "a woken task in a callback still counted as parked"
        );
    }

    #[test]
    fn a_task_in_a_callback_that_wakes_with_no_wake_counts_once() {
        let g = new_gate();
        let (sched, group) = start_a_task_in_a_callback(&g);
        let b = bucket(g.gate.as_ptr());
        let asleep_in_round = |n| {
            while g.rounds.load(Ordering::SeqCst) < n
                || blocked_in(&b.lock.lock().unwrap(), sched) == 0
            {
                std::thread::yield_now();
            }
        };
        asleep_in_round(1);
        // A return of the condition variable with no wake, as the system may give.
        drop(b.lock.lock().unwrap());
        b.cond.notify_all();
        asleep_in_round(2);
        // The bucket lock keeps a wake from another test out while this test reads.
        let w = b.lock.lock().unwrap();
        let entries = blocked_in(&w, sched);
        let parked = sched.all_parked();
        drop(w);
        g.gate.store(1, Ordering::SeqCst);
        wake(b, Fibers::Of(g.gate.as_ptr() as usize));
        unsafe { crate::ffi::tasks::vx_tasks_end(group) };
        assert_eq!(entries, 1, "a task in a callback is on the list twice");
        assert!(parked, "a task in a callback counted as blocked twice");
    }
}
