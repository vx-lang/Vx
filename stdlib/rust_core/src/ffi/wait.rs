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
// the queue does not depend on which words share a bucket.
//
//===----------------------------------------------------------------------===//

use crate::ffi::sched::{self, FiberRef};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

const BUCKETS: usize = 64;

struct Bucket {
    lock: Mutex<Waiters>,
    cond: Condvar,
}

struct Waiters {
    /// The fibers that park on a word of this bucket, with the address of the word.
    fibers: Vec<(FiberRef, usize)>,
    /// The threads that sleep here with no time limit, which the deadlock watcher counts.
    sleepers: i64,
    /// Goes up at each wake of the bucket.
    wakes: u64,
}

static TABLE: [Bucket; BUCKETS] = [const {
    Bucket {
        lock: Mutex::new(Waiters {
            fibers: Vec::new(),
            sleepers: 0,
            wakes: 0,
        }),
        cond: Condvar::new(),
    }
}; BUCKETS];

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
    // In a `drop`, a wait that can be cancelled ends at the deadline of the `drop` (F6).
    let deadline_ns = match crate::ffi::tasks::drop_deadline().filter(|_| or_cancel) {
        Some(d) => Some(deadline_ns.map_or(d, |own| own.min(d))),
        None => deadline_ns,
    };
    let b = bucket(word);
    let mut waiters = b.lock.lock().unwrap_or_else(|e| e.into_inner());
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
    if fiber.is_null() {
        // With a seed, this thread runs the ready tasks. The check is under the lock of the
        // bucket, and a task that gets ready wakes each bucket, so no wake is lost.
        if let Some(sched) = sched::scheduler_here().filter(|s| s.has_ready()) {
            drop(waiters);
            sched.run_one_here();
            return;
        }
        match left {
            Some(left) => drop(
                b.cond
                    .wait_timeout(waiters, left)
                    .unwrap_or_else(|e| e.into_inner()),
            ),
            None => {
                // The thread counts as waiting until a wake of the bucket, not until the OS
                // runs it again. A woken thread can wait a long time for a CPU, and it must not
                // look stuck then.
                let seen = waiters.wakes;
                waiters.sleepers += 1;
                deadlock::sleep_started();
                let mut waiters = b.cond.wait(waiters).unwrap_or_else(|e| e.into_inner());
                if waiters.wakes == seen {
                    waiters.sleepers -= 1;
                    deadlock::sleep_ended(1);
                }
            }
        }
        return;
    }
    sched::will_park(fiber);
    waiters.fibers.push((FiberRef(fiber), word as usize));
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

/// Wake each thread that waits on any word. A cancel uses it: each wait reads its word again,
/// and a wait that can be cancelled then sees the cancel.
pub(crate) fn wake_every_thread() {
    for b in &TABLE {
        wake(b, Fibers::All);
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
    All,
    None,
}

/// Takes the fibers that `which` names off the list of the bucket.
fn take_fibers(waiters: &mut Waiters, which: Fibers) -> Vec<FiberRef> {
    let (taken, kept) = std::mem::take(&mut waiters.fibers)
        .into_iter()
        .partition(|&(_, w)| match which {
            Fibers::Of(word) => w == word,
            Fibers::All => true,
            Fibers::None => false,
        });
    waiters.fibers = kept;
    taken.into_iter().map(|(fiber, _)| fiber).collect()
}

/// Wakes each thread of the bucket, and the fibers that `which` names.
fn wake(b: &Bucket, which: Fibers) {
    deadlock::progress();
    let mut waiters = b.lock.lock().unwrap_or_else(|e| e.into_inner());
    let fibers = take_fibers(&mut waiters, which);
    deadlock::sleep_ended(std::mem::take(&mut waiters.sleepers));
    waiters.wakes = waiters.wakes.wrapping_add(1);
    drop(waiters);
    b.cond.notify_all();
    for fiber in fibers {
        sched::wake(fiber);
    }
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
        /// The deadlines, the earliest first, with the address of the word of each.
        deadlines: Mutex<BinaryHeap<Reverse<(i64, usize)>>>,
        cond: Condvar,
    }

    static TIMER: OnceLock<&'static Timer> = OnceLock::new();

    /// Whether a deadline is still to come. Its wake can let a parked fiber go on.
    pub(super) fn pending() -> bool {
        TIMER.get().is_some_and(|t| {
            !t.deadlines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        })
    }

    fn timer() -> &'static Timer {
        TIMER.get_or_init(|| {
            let t: &'static Timer = Box::leak(Box::new(Timer {
                deadlines: Mutex::new(BinaryHeap::new()),
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
        d.push(Reverse((deadline_ns, word as usize)));
        t.cond.notify_one();
    }

    fn run(t: &'static Timer) {
        let mut d = t.deadlines.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let Some(&Reverse((deadline, word))) = d.peek() else {
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
            d.pop();
            drop(d);
            wake(bucket(word as *const i64), Fibers::Of(word));
            d = t.deadlines.lock().unwrap_or_else(|e| e.into_inner());
        }
    }
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

    fn run() {
        let mut seen = None;
        loop {
            std::thread::sleep(TICK);
            let progress = PROGRESS.load(Ordering::SeqCst);
            let now = stuck().then_some(progress);
            if now.is_some() && now == seen {
                eprintln!(
                    "vx: deadlock: each task of the program waits, and nothing can wake one. \
                     The program stops."
                );
                std::process::abort();
            }
            seen = now;
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

    #[test]
    fn a_wake_stops_counting_a_thread_before_the_thread_runs_again() {
        let word = Arc::new(AtomicI64::new(0));
        let w = word.clone();
        let sleeper = std::thread::spawn(move || {
            while w.load(Ordering::SeqCst) == 0 {
                unsafe { vx_wait_while(Arc::as_ptr(&w) as *const i64, 0) };
            }
        });
        let b = bucket(Arc::as_ptr(&word) as *const i64);
        while b.lock.lock().unwrap().sleepers == 0 {
            std::thread::yield_now();
        }
        word.store(1, Ordering::SeqCst);
        wake(b, Fibers::All);
        // The sleeper needs this lock to come back from its wait, so it is still in it here.
        let waiters = b.lock.lock().unwrap();
        assert_eq!(
            waiters.sleepers, 0,
            "a woken thread still counts as waiting"
        );
        drop(waiters);
        sleeper.join().unwrap();
    }

    /// A wake of a word must not take a fiber that parks on another word of the same bucket.
    /// With a seed, such a wake changes the order of the tasks. The fibers here never run.
    #[test]
    fn a_wake_takes_only_the_fibers_that_it_names() {
        let fiber = || FiberRef(std::ptr::null_mut());
        let mut waiters = Waiters {
            fibers: vec![(fiber(), 8), (fiber(), 16), (fiber(), 8)],
            sleepers: 0,
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
        let words: Vec<usize> = waiters.fibers.iter().map(|&(_, w)| w).collect();
        assert_eq!(words, [16], "a wake of a word took a fiber of another word");
        assert_eq!(take_fibers(&mut waiters, Fibers::All).len(), 1);
        assert!(waiters.fibers.is_empty());
    }
}
