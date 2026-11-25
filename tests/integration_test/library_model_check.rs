//===- library_model_check.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A model check of the channel, `select`, `Mutex` and `Semaphore` in `stdlib/std/sync`. Each
// access to a word there is a `SeqCst` atomic, so each run of the code is some interleaving of its atomic steps.
// This file writes the steps of each operation as a small program, copied by hand from the Vx
// code, and visits every interleaving of a few threads. `wait_while` blocks only while the word
// holds the value, and `wake_all` wakes only the threads that wait on that word. `wake_one` wakes
// one of them, and the search visits a wake of each one. The runtime can also wake a thread
// early, which never hides a lost wake, so the model leaves that out.
//
// The search keeps a 128-bit hash of each state that it saw, not the state (`fingerprint`).
//
// A state in which no thread can move and some thread is not done is a lost wake or a deadlock.
// Each value must end exactly once: received, given back by `send`, or dropped by the last end.
// The values of each sender arrive in order. A slot is never written while it holds a value.
//
// The model must change with the Vx code. The last tests break the model in the ways that the
// Vx code was broken while it was written, and check that the search finds each one.
//
//===----------------------------------------------------------------------===//

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

/// A value that a step computes from the registers of its thread.
#[derive(Clone, Debug)]
enum E {
    C(i64),
    R(usize),
    Add(Box<E>, Box<E>),
    Mul(i64, Box<E>),
    Mod(Box<E>, i64),
}

fn c(v: i64) -> E {
    E::C(v)
}
fn r(i: usize) -> E {
    E::R(i)
}
fn add(a: E, b: E) -> E {
    E::Add(Box::new(a), Box::new(b))
}
fn mul(k: i64, a: E) -> E {
    E::Mul(k, Box::new(a))
}
fn modulo(a: E, k: i64) -> E {
    E::Mod(Box::new(a), k)
}

impl E {
    fn eval(&self, regs: &[i64]) -> i64 {
        match self {
            E::C(v) => *v,
            E::R(i) => regs[*i],
            E::Add(a, b) => a.eval(regs) + b.eval(regs),
            E::Mul(k, a) => k * a.eval(regs),
            E::Mod(a, k) => a.eval(regs).rem_euclid(*k),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Cmp {
    Eq,
    Ne,
    Gt,
}

/// What happened to a value at its end.
#[derive(Clone, Copy, Debug, PartialEq)]
enum End {
    Received,
    GivenBack,
    Dropped,
}

/// One atomic step. `Jump` and `Set` touch no shared state, but each still counts as a step.
#[derive(Clone, Debug)]
enum Step {
    FetchAdd {
        dst: usize,
        word: E,
        delta: E,
    },
    Load {
        dst: usize,
        word: E,
    },
    Store {
        word: E,
        val: E,
    },
    /// Block while `word` holds `val`. The check and the block are one step, as under the lock of
    /// the bucket in the runtime. With `or_cancel`, also do not block when that cancel mark is
    /// set, as `vx_wait_while_or_cancel` does.
    WaitWhile {
        word: E,
        val: E,
        or_cancel: Option<i64>,
    },
    /// Put the send number in register `k` into the skip set of the channel at `chan`, as
    /// `vx_channel_skip` does under its lock.
    SkipAdd {
        chan: i64,
        k: usize,
    },
    /// Take the send number in register `k` out of the skip set. `dst` is 1 when it was there.
    SkipTake {
        dst: usize,
        chan: i64,
        k: usize,
    },
    /// Remove each number of the channel at `chan` from the skip set.
    SkipForget {
        chan: i64,
    },
    WakeAll {
        word: E,
    },
    /// Wake one thread that waits on `word`, as `vx_wake_one` does. The search visits a wake of
    /// each such thread, so a model holds for each thread that the runtime can take.
    WakeOne {
        word: E,
    },
    /// The time limit of thread `thread` ends its wait. No other thread wakes.
    EndWait {
        thread: usize,
    },
    /// Wake each thread that waits, on any word, as a cancel does.
    WakeEvery,
    /// Wait until thread `thread` is done, as the join of its task does.
    Join {
        thread: usize,
    },
    /// Write a value into a slot of a ring. The slot must be empty.
    Put {
        slot: E,
        val: E,
    },
    /// Take the value out of a slot of a ring into `dst`. The slot must hold one.
    Take {
        dst: usize,
        slot: E,
    },
    /// The value in register `val` ends.
    Ends {
        val: usize,
        how: End,
    },
    /// The value `val` ends without having been put in a ring, as when `send` gives it back.
    EndsNow {
        val: E,
        how: End,
    },
    /// Compare-and-swap: when `word` holds `expect`, store `new` and set `dst` to 1. Else set
    /// `dst` to 0.
    Cas {
        dst: usize,
        word: E,
        expect: E,
        new: E,
    },
    /// Copy the value in a slot into `dst`, as a receiver of a broadcast clones it. The slot must
    /// hold one.
    Peek {
        dst: usize,
        slot: E,
    },
    /// The broadcast receiver `recv` got the value in register `val`. It must be the value after
    /// the last one that this receiver got.
    Got {
        recv: usize,
        val: usize,
    },
    /// The broadcast receiver `recv` starts after `val` values.
    StartAt {
        recv: usize,
        val: E,
    },
    /// The broadcast receiver `recv` gives `Closed`. It must have got each value that was sent:
    /// `sends` is the word that counts them.
    ReceiverClosed {
        recv: usize,
        sends: i64,
    },
    /// A receiver of the work queue `chan` gives `Closed`. No value that no receiver took may be
    /// in a slot.
    QueueClosed {
        chan: Chan,
    },
    /// Enter or leave the part of the code that the `Mutex` guards.
    Enter,
    Leave,
    Set {
        dst: usize,
        val: E,
    },
    If {
        a: E,
        cmp: Cmp,
        b: E,
        to: usize,
    },
    Jump {
        to: usize,
    },
    Done,
}

/// A program under construction, with labels that are resolved at the end.
struct Asm {
    steps: Vec<Step>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    fn new() -> Self {
        Asm {
            steps: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }
    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }
    fn here(&mut self, l: usize) {
        self.labels[l] = Some(self.steps.len());
    }
    fn push(&mut self, s: Step) {
        self.steps.push(s);
    }
    fn jump_if(&mut self, a: E, cmp: Cmp, b: E, l: usize) {
        self.fixups.push((self.steps.len(), l));
        self.steps.push(Step::If { a, cmp, b, to: 0 });
    }
    fn jump(&mut self, l: usize) {
        self.fixups.push((self.steps.len(), l));
        self.steps.push(Step::Jump { to: 0 });
    }
    fn finish(mut self) -> Vec<Step> {
        for (at, l) in self.fixups {
            let target = self.labels[l].expect("each label is placed");
            match &mut self.steps[at] {
                Step::If { to, .. } | Step::Jump { to } => *to = target,
                _ => unreachable!("only jumps have fixups"),
            }
        }
        self.steps
    }
}

/// The ways the tests below break the model, each a bug that the Vx code once had or could have.
#[derive(Clone, Copy, Default)]
struct Breaks {
    /// Vyukov's numbering: a slot says "full" with `k + 1` and "free" with `k + n`.
    vyukov_numbering: bool,
    /// An event on a channel does not change the word that `select` sleeps on.
    no_select_event: bool,
    /// `recv` reads `SENDERS` after the slot, not before.
    senders_after_slot: bool,
    /// The drop of the `MutexGuard` does not wake the threads that wait.
    unlock_wakes_none: bool,
    /// The wait of a `recv` does not read the cancel under the lock of its bucket.
    wait_ignores_cancel: bool,
    /// A cancel does not stop a `send` that has its place, as before the skip set.
    send_keeps_its_place: bool,
    /// `recv` does not look in the skip set.
    recv_ignores_skips: bool,
    /// `recv` passes a skipped number but does not free its slot for the `send` `cap` later.
    skip_keeps_the_slot: bool,
    /// `pass_a_skip` frees the slot before the `recv` `cap` places earlier took its value.
    pass_does_not_wait_for_the_slot: bool,
    /// A `WorkReceiver` takes its receive number with a plain store, as the `Receiver` does.
    queue_takes_with_a_store: bool,
    /// A `WorkReceiver` gives `Closed` when each `Sender` was dropped, and does not check that
    /// each send number had a `recv`.
    queue_closed_without_the_count: bool,
    /// The clone of a broadcast receiver does not add to the counts of the values that it did
    /// not read.
    clone_skips_the_counts: bool,
    /// The clone of a broadcast receiver does not take the lock.
    clone_without_the_lock: bool,
    /// The drop of a broadcast receiver does not take the lock.
    drop_without_the_lock: bool,
    /// The drop of a broadcast receiver does not take 1 from the counts of the values that it
    /// did not read.
    drop_keeps_its_counts: bool,
    /// An `acquire` of the `Semaphore` that gives up does not wake the next sleeper.
    acquire_gives_up_and_wakes_none: bool,
    /// The drop of a `Permit` wakes a sleeper only when two tasks sleep.
    permit_drop_needs_two_sleepers: bool,
}

// The words of a channel, as in `channel.vx`.
const SENDS: i64 = 0;
const RECVS: i64 = 1;
const SENDERS: i64 = 2;
const RECEIVER: i64 = 3;
const ENDS: i64 = 4;
const WAKE: i64 = 5;
const SLEEPERS: i64 = 6;
const SKIPS: i64 = 7;
const SLOTS: i64 = 8;

// The two words for `select`, after the words of every channel.
const ANY_EVENTS: i64 = 40;
const ANY_SLEEPERS: i64 = 41;

// The cancel mark of the `tasks` block of most threads. `vx_cancelled` reads it.
const CANCEL: i64 = 42;
// The cancel mark of a second block, which no thread cancels.
const OTHER_BLOCK: i64 = 43;
// The count of words before the ring.
const WORDS: usize = 44;

/// One channel: where its words start, where its slots start in the ring memory, and its room.
#[derive(Clone, Copy, Debug)]
struct Chan {
    base: i64,
    ring: i64,
    cap: i64,
}

impl Chan {
    fn w(&self, i: i64) -> E {
        c(self.base + i)
    }
    /// The word of the slot for the number in register `k`.
    fn slot_word(&self, k: usize) -> E {
        add(c(self.base + SLOTS), modulo(r(k), self.cap))
    }
    fn slot(&self, k: usize) -> E {
        add(c(self.ring), modulo(r(k), self.cap))
    }
    /// The number that lets the `send` with the number in `k` write its slot.
    fn free(&self, k: usize, b: Breaks) -> E {
        if b.vyukov_numbering {
            r(k)
        } else {
            mul(2, r(k))
        }
    }
    /// The number that a `send` stores when its value is in the slot.
    fn full(&self, k: usize, b: Breaks) -> E {
        if b.vyukov_numbering {
            add(r(k), c(1))
        } else {
            add(mul(2, r(k)), c(1))
        }
    }
    /// The number that a `recv` stores when it took the value: free for the `send` `cap` later.
    fn taken(&self, k: usize, b: Breaks) -> E {
        if b.vyukov_numbering {
            add(r(k), c(self.cap))
        } else {
            mul(2, add(r(k), c(self.cap)))
        }
    }
}

// Registers. Each program uses the same names.
const J: usize = 0; // the count of values sent, or of locks taken
const K: usize = 1; // the send or receive number
const SEEN: usize = 2; // the value of the wake word before a check
const T: usize = 3; // a word that was just read
const CLOSED: usize = 4; // the value of `SENDERS` that `recv` read
const V: usize = 5; // a value that was just taken
const OPEN: usize = 6; // the channels that `select` found open
const T2: usize = 7; // a second word that was just read
const TICKET: usize = 8; // the ticket of a lock
const NOW: usize = 9; // the ticket that a lock serves now

/// `notify` in `channel.vx`.
fn notify(a: &mut Asm, ch: Chan, b: Breaks) {
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(WAKE),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: T,
        word: ch.w(SLEEPERS),
    });
    let skip = a.label();
    a.jump_if(r(T), Cmp::Eq, c(0), skip);
    a.push(Step::WakeAll { word: ch.w(WAKE) });
    a.here(skip);
    if !b.no_select_event {
        a.push(Step::FetchAdd {
            dst: T,
            word: c(ANY_EVENTS),
            delta: c(1),
        });
    }
    a.push(Step::Load {
        dst: T,
        word: c(ANY_SLEEPERS),
    });
    let skip = a.label();
    a.jump_if(r(T), Cmp::Eq, c(0), skip);
    a.push(Step::WakeAll {
        word: c(ANY_EVENTS),
    });
    a.here(skip);
}

/// `wait_for_event` in `channel.vx`: wait for an event after `SEEN`, and read `WAKE` again. With
/// `can_cancel`, a cancel also ends the wait.
fn sleep(a: &mut Asm, ch: Chan, mark: Option<i64>, b: Breaks) {
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SLEEPERS),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: ch.w(WAKE),
        val: r(SEEN),
        or_cancel: if b.wait_ignores_cancel { None } else { mark },
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SLEEPERS),
        delta: c(-1),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
}

/// `release` in `channel.vx`: the last end drops the values that no `recv` took.
fn release(a: &mut Asm, ch: Chan, b: Breaks) {
    let (done, not_last) = (a.label(), a.label());
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(ENDS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), not_last);
    a.push(Step::Load {
        dst: K,
        word: ch.w(RECVS),
    });
    a.push(Step::Load {
        dst: T2,
        word: ch.w(SENDS),
    });
    let top = a.label();
    let next = a.label();
    a.here(top);
    // `while k < end`.
    a.jump_if(r(K), Cmp::Eq, r(T2), done);
    a.jump_if(r(K), Cmp::Gt, r(T2), done);
    a.push(Step::Load {
        dst: T,
        word: ch.slot_word(K),
    });
    a.jump_if(r(T), Cmp::Ne, ch.full(K, b), next);
    a.push(Step::Take {
        dst: V,
        slot: ch.slot(K),
    });
    a.push(Step::Ends {
        val: V,
        how: End::Dropped,
    });
    a.here(next);
    a.push(Step::Set {
        dst: K,
        val: add(r(K), c(1)),
    });
    a.jump(top);
    a.here(done);
    // The numbers of cancelled sends that no `recv` passed.
    let none_left = a.label();
    a.push(Step::Load {
        dst: T,
        word: ch.w(SKIPS),
    });
    a.jump_if(r(T), Cmp::Eq, c(0), none_left);
    a.push(Step::SkipForget { chan: ch.base });
    a.here(none_left);
    a.here(not_last);
}

/// A task in the block with the cancel mark `mark` that sends `count` values with its own
/// `Sender`, and then drops it. Sender `id` sends the values `id * 100 + 1` and on.
fn sender(ch: Chan, id: i64, count: i64, mark: i64, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, retry, sleeping, gave_back, next, drop) = (
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
    );
    a.push(Step::Set { dst: J, val: c(1) });
    a.here(top);
    a.jump_if(r(J), Cmp::Gt, c(count), drop);
    a.push(Step::Load {
        dst: T,
        word: c(mark),
    });
    a.jump_if(r(T), Cmp::Ne, c(0), gave_back);
    a.push(Step::FetchAdd {
        dst: K,
        word: ch.w(SENDS),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
    a.here(retry);
    a.push(Step::Load {
        dst: T,
        word: ch.w(RECEIVER),
    });
    // A `work_queue` has many receivers, so the send counts them.
    a.jump_if(r(T), Cmp::Eq, c(0), gave_back);
    a.push(Step::Load {
        dst: T,
        word: ch.slot_word(K),
    });
    a.jump_if(r(T), Cmp::Ne, ch.free(K, b), sleeping);
    a.push(Step::Put {
        slot: ch.slot(K),
        val: add(c(id * 100), r(J)),
    });
    a.push(Step::Store {
        word: ch.slot_word(K),
        val: ch.full(K, b),
    });
    notify(&mut a, ch, b);
    a.jump(next);
    a.here(sleeping);
    if b.send_keeps_its_place {
        sleep(&mut a, ch, None, b);
    } else {
        // A cancelled `send` with a place puts its number in the skip set, so that the `recv`
        // passes it. A `send_for` whose time ran out takes the same path.
        let wait = a.label();
        a.push(Step::Load {
            dst: T,
            word: c(mark),
        });
        a.jump_if(r(T), Cmp::Eq, c(0), wait);
        a.push(Step::SkipAdd {
            chan: ch.base,
            k: K,
        });
        a.push(Step::FetchAdd {
            dst: T,
            word: ch.w(SKIPS),
            delta: c(1),
        });
        notify(&mut a, ch, b);
        a.jump(gave_back);
        a.here(wait);
        sleep(&mut a, ch, Some(mark), b);
    }
    a.jump(retry);
    a.here(gave_back);
    a.push(Step::EndsNow {
        val: add(c(id * 100), r(J)),
        how: End::GivenBack,
    });
    a.here(next);
    a.push(Step::Set {
        dst: J,
        val: add(r(J), c(1)),
    });
    a.jump(top);
    a.here(drop);
    let no_notify = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SENDERS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), no_notify);
    notify(&mut a, ch, b);
    a.here(no_notify);
    release(&mut a, ch, b);
    a.push(Step::Done);
    a.finish()
}

/// The drop of the `Receiver`.
fn drop_receiver(a: &mut Asm, ch: Chan, b: Breaks) {
    a.push(Step::Store {
        word: ch.w(RECEIVER),
        val: c(0),
    });
    notify(a, ch, b);
    release(a, ch, b);
}

/// The part of `recv` and `try_recv` that passes the number in `K` when a cancelled `send` put
/// it in the skip set. It then goes to `next`, which reads the receive number again.
fn pass_a_skip(a: &mut Asm, ch: Chan, next: usize, b: Breaks) {
    let no_skip = a.label();
    a.push(Step::Load {
        dst: T,
        word: ch.w(SKIPS),
    });
    a.jump_if(r(T), Cmp::Eq, c(0), no_skip);
    if !b.pass_does_not_wait_for_the_slot {
        // With many receivers, the `recv` `cap` places before can still read the slot.
        a.push(Step::Load {
            dst: T,
            word: ch.slot_word(K),
        });
        a.jump_if(r(T), Cmp::Ne, ch.free(K, b), no_skip);
    }
    a.push(Step::SkipTake {
        dst: T,
        chan: ch.base,
        k: K,
    });
    a.jump_if(r(T), Cmp::Eq, c(0), no_skip);
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SKIPS),
        delta: c(-1),
    });
    a.push(Step::Store {
        word: ch.w(RECVS),
        val: add(r(K), c(1)),
    });
    if !b.skip_keeps_the_slot {
        a.push(Step::Store {
            word: ch.slot_word(K),
            val: ch.taken(K, b),
        });
    }
    notify(a, ch, b);
    a.jump(next);
    a.here(no_skip);
}

/// A task in the block with the cancel mark `mark` that calls `recv` until it gives `Closed` or
/// `Cancelled`, or `limit` times, and then drops the `Receiver`. `recv` takes its receive number
/// only when it takes a value or passes a cancelled send.
fn receiver(ch: Chan, limit: i64, mark: i64, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, retry, not_full, closed) = (a.label(), a.label(), a.label(), a.label());
    a.push(Step::Set { dst: J, val: c(0) });
    a.here(top);
    a.jump_if(r(J), Cmp::Eq, c(limit), closed);
    a.push(Step::Load {
        dst: K,
        word: ch.w(RECVS),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
    a.here(retry);
    a.push(Step::Load {
        dst: T,
        word: c(mark),
    });
    a.jump_if(r(T), Cmp::Ne, c(0), closed);
    if !b.senders_after_slot {
        a.push(Step::Load {
            dst: CLOSED,
            word: ch.w(SENDERS),
        });
    }
    a.push(Step::Load {
        dst: T,
        word: ch.slot_word(K),
    });
    if b.senders_after_slot {
        a.push(Step::Load {
            dst: CLOSED,
            word: ch.w(SENDERS),
        });
    }
    a.jump_if(r(T), Cmp::Ne, ch.full(K, b), not_full);
    a.push(Step::Store {
        word: ch.w(RECVS),
        val: add(r(K), c(1)),
    });
    a.push(Step::Take {
        dst: V,
        slot: ch.slot(K),
    });
    a.push(Step::Store {
        word: ch.slot_word(K),
        val: ch.taken(K, b),
    });
    notify(&mut a, ch, b);
    a.push(Step::Ends {
        val: V,
        how: End::Received,
    });
    a.push(Step::Set {
        dst: J,
        val: add(r(J), c(1)),
    });
    a.jump(top);
    a.here(not_full);
    if !b.recv_ignores_skips {
        pass_a_skip(&mut a, ch, top, b);
    }
    a.jump_if(r(CLOSED), Cmp::Eq, c(0), closed);
    sleep(&mut a, ch, Some(mark), b);
    a.jump(retry);
    a.here(closed);
    drop_receiver(&mut a, ch, b);
    a.push(Step::Done);
    a.finish()
}

/// A task that calls `select` on two channels until it gives `Closed`, and then drops both
/// `Receiver`s. Each `try_recv` is as in `channel.vx`.
fn selector(chans: [Chan; 2], b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, done) = (a.label(), a.label());
    a.here(top);
    a.push(Step::Load {
        dst: SEEN,
        word: c(ANY_EVENTS),
    });
    a.push(Step::Set {
        dst: OPEN,
        val: c(0),
    });
    for ch in chans {
        let (again, empty, after) = (a.label(), a.label(), a.label());
        a.here(again);
        a.push(Step::Load {
            dst: K,
            word: ch.w(RECVS),
        });
        a.push(Step::Load {
            dst: CLOSED,
            word: ch.w(SENDERS),
        });
        a.push(Step::Load {
            dst: T,
            word: ch.slot_word(K),
        });
        a.jump_if(r(T), Cmp::Ne, ch.full(K, b), empty);
        a.push(Step::Store {
            word: ch.w(RECVS),
            val: add(r(K), c(1)),
        });
        a.push(Step::Take {
            dst: V,
            slot: ch.slot(K),
        });
        a.push(Step::Store {
            word: ch.slot_word(K),
            val: ch.taken(K, b),
        });
        notify(&mut a, ch, b);
        a.push(Step::Ends {
            val: V,
            how: End::Received,
        });
        a.jump(top);
        a.here(empty);
        if !b.recv_ignores_skips {
            pass_a_skip(&mut a, ch, again, b);
        }
        a.jump_if(r(CLOSED), Cmp::Eq, c(0), after);
        a.push(Step::Set {
            dst: OPEN,
            val: add(r(OPEN), c(1)),
        });
        a.here(after);
    }
    a.jump_if(r(OPEN), Cmp::Eq, c(0), done);
    a.push(Step::FetchAdd {
        dst: T,
        word: c(ANY_SLEEPERS),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: c(ANY_EVENTS),
        val: r(SEEN),
        or_cancel: None,
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: c(ANY_SLEEPERS),
        delta: c(-1),
    });
    a.jump(top);
    a.here(done);
    for ch in chans {
        drop_receiver(&mut a, ch, b);
    }
    a.push(Step::Done);
    a.finish()
}

// The words of a `Mutex`, as in `mutex.vx`.
const NEXT: i64 = 0;
const SERVING: i64 = 1;
const M_SLEEPERS: i64 = 2;

/// A task that locks the `Mutex` `count` times, and unlocks it each time.
fn locker(count: i64, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, wait, got, done) = (a.label(), a.label(), a.label(), a.label());
    a.push(Step::Set { dst: J, val: c(0) });
    a.here(top);
    a.jump_if(r(J), Cmp::Eq, c(count), done);
    // `lock`: K is the ticket, SEEN the value of `serving`.
    a.push(Step::FetchAdd {
        dst: K,
        word: c(NEXT),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: c(SERVING),
    });
    a.here(wait);
    a.jump_if(r(SEEN), Cmp::Eq, r(K), got);
    a.push(Step::FetchAdd {
        dst: T,
        word: c(M_SLEEPERS),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: c(SERVING),
        val: r(SEEN),
        or_cancel: None,
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: c(M_SLEEPERS),
        delta: c(-1),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: c(SERVING),
    });
    a.jump(wait);
    a.here(got);
    a.push(Step::Enter);
    a.push(Step::Leave);
    // The drop of the `MutexGuard`.
    a.push(Step::FetchAdd {
        dst: T,
        word: c(SERVING),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: T,
        word: c(M_SLEEPERS),
    });
    let skip = a.label();
    a.jump_if(r(T), Cmp::Eq, c(0), skip);
    if !b.unlock_wakes_none {
        a.push(Step::WakeAll { word: c(SERVING) });
    }
    a.here(skip);
    a.push(Step::Set {
        dst: J,
        val: add(r(J), c(1)),
    });
    a.jump(top);
    a.here(done);
    a.push(Step::Done);
    a.finish()
}

/// A task with a `WorkReceiver` of `work_queue.vx`, in the block with the cancel mark `mark`. It
/// calls `recv` until it gives `Closed` or `Cancelled`, and then drops the `WorkReceiver`. A
/// `recv` takes the receive number `k` with a compare-and-swap, and only when slot `k` holds
/// its value.
fn work_receiver(ch: Chan, mark: i64, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (retry, again, not_full, empty, done) =
        (a.label(), a.label(), a.label(), a.label(), a.label());
    a.here(retry);
    a.push(Step::Load {
        dst: T,
        word: c(mark),
    });
    a.jump_if(r(T), Cmp::Ne, c(0), done);
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
    // `take`.
    a.here(again);
    a.push(Step::Load {
        dst: CLOSED,
        word: ch.w(SENDERS),
    });
    a.push(Step::Load {
        dst: K,
        word: ch.w(RECVS),
    });
    a.push(Step::Load {
        dst: T,
        word: ch.slot_word(K),
    });
    a.jump_if(r(T), Cmp::Ne, ch.full(K, b), not_full);
    if b.queue_takes_with_a_store {
        a.push(Step::Store {
            word: ch.w(RECVS),
            val: add(r(K), c(1)),
        });
    } else {
        a.push(Step::Cas {
            dst: T,
            word: ch.w(RECVS),
            expect: r(K),
            new: add(r(K), c(1)),
        });
        // Another `WorkReceiver` took the value.
        a.jump_if(r(T), Cmp::Eq, c(0), again);
    }
    a.push(Step::Take {
        dst: V,
        slot: ch.slot(K),
    });
    a.push(Step::Store {
        word: ch.slot_word(K),
        val: ch.taken(K, b),
    });
    notify(&mut a, ch, b);
    a.push(Step::Ends {
        val: V,
        how: End::Received,
    });
    a.jump(retry);
    a.here(not_full);
    pass_a_skip(&mut a, ch, again, b);
    a.jump_if(r(CLOSED), Cmp::Ne, c(0), empty);
    if !b.queue_closed_without_the_count {
        a.push(Step::Load {
            dst: T,
            word: ch.w(RECVS),
        });
        a.push(Step::Load {
            dst: T2,
            word: ch.w(SENDS),
        });
        a.jump_if(r(T2), Cmp::Gt, r(T), empty);
    }
    a.push(Step::QueueClosed { chan: ch });
    a.jump(done);
    a.here(empty);
    sleep(&mut a, ch, Some(mark), b);
    a.jump(retry);
    a.here(done);
    // The drop of the `WorkReceiver`.
    let no_notify = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(RECEIVER),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), no_notify);
    notify(&mut a, ch, b);
    a.here(no_notify);
    release(&mut a, ch, b);
    a.push(Step::Done);
    a.finish()
}

// The words of a broadcast channel, as in `broadcast.vx`, after the words of `channel`.
const B_LOCK: i64 = 8;
const B_SLOTS: i64 = 11;
// The word of each receiver that holds the number of its next value, by receiver id.
const B_NEXT: i64 = 30;
// 1 when the clone of a receiver is ready for the task that uses it.
const HANDOFF: i64 = 36;

/// The number of the broadcast slot for the number in register `k`.
fn turn(cap: i64, k: usize) -> E {
    add(c(B_SLOTS), mul(2, modulo(r(k), cap)))
}

/// The count of receivers that did not read the broadcast slot for the number in register `k`.
fn readers(cap: i64, k: usize) -> E {
    add(c(B_SLOTS + 1), mul(2, modulo(r(k), cap)))
}

/// `lock_channel` in `channel.vx`: a ticket lock.
fn lock_channel(a: &mut Asm) {
    let (wait, got) = (a.label(), a.label());
    a.push(Step::FetchAdd {
        dst: TICKET,
        word: c(B_LOCK),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: NOW,
        word: c(B_LOCK + 1),
    });
    a.here(wait);
    a.jump_if(r(NOW), Cmp::Eq, r(TICKET), got);
    a.push(Step::FetchAdd {
        dst: T,
        word: c(B_LOCK + 2),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: c(B_LOCK + 1),
        val: r(NOW),
        or_cancel: None,
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: c(B_LOCK + 2),
        delta: c(-1),
    });
    a.push(Step::Load {
        dst: NOW,
        word: c(B_LOCK + 1),
    });
    a.jump(wait);
    a.here(got);
}

/// `unlock_channel` in `channel.vx`.
fn unlock_channel(a: &mut Asm) {
    let skip = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: c(B_LOCK + 1),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: T,
        word: c(B_LOCK + 2),
    });
    a.jump_if(r(T), Cmp::Eq, c(0), skip);
    a.push(Step::WakeAll {
        word: c(B_LOCK + 1),
    });
    a.here(skip);
}

/// `broadcast_free`: drop the value with the number in register `k`, and free its slot.
fn broadcast_free(a: &mut Asm, ch: Chan, k: usize, b: Breaks) {
    a.push(Step::Take {
        dst: V,
        slot: ch.slot(k),
    });
    a.push(Step::Ends {
        val: V,
        how: End::Dropped,
    });
    a.push(Step::Store {
        word: turn(ch.cap, k),
        val: ch.taken(k, b),
    });
    notify(a, ch, b);
}

/// `broadcast_release`: the last end drops each value that is still in a slot.
fn broadcast_release(a: &mut Asm, ch: Chan) {
    let not_last = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(ENDS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), not_last);
    for i in 0..ch.cap {
        let empty = a.label();
        a.push(Step::Load {
            dst: T,
            word: c(B_SLOTS + 2 * i),
        });
        a.jump_if(modulo(r(T), 2), Cmp::Ne, c(1), empty);
        a.push(Step::Take {
            dst: V,
            slot: c(ch.ring + i),
        });
        a.push(Step::Ends {
            val: V,
            how: End::Dropped,
        });
        a.here(empty);
    }
    a.here(not_last);
}

/// A task with the `BroadcastSender`. It sends the values 101 to `100 + count`, and then drops
/// the sender. A `send` takes the lock, and writes the value when its slot is free.
fn broadcast_sender(ch: Chan, count: i64, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, retry, have, full, next, drop) = (
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
    );
    a.push(Step::Set { dst: J, val: c(1) });
    a.here(top);
    a.jump_if(r(J), Cmp::Gt, c(count), drop);
    a.here(retry);
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
    lock_channel(&mut a);
    a.push(Step::Load {
        dst: T,
        word: ch.w(RECEIVER),
    });
    a.jump_if(r(T), Cmp::Ne, c(0), have);
    unlock_channel(&mut a);
    a.push(Step::EndsNow {
        val: add(c(100), r(J)),
        how: End::GivenBack,
    });
    a.jump(next);
    a.here(have);
    // T holds the count of receivers.
    a.push(Step::Load {
        dst: K,
        word: ch.w(SENDS),
    });
    a.push(Step::Load {
        dst: T2,
        word: turn(ch.cap, K),
    });
    a.jump_if(r(T2), Cmp::Ne, ch.free(K, b), full);
    a.push(Step::Put {
        slot: ch.slot(K),
        val: add(c(100), r(J)),
    });
    a.push(Step::Store {
        word: readers(ch.cap, K),
        val: r(T),
    });
    a.push(Step::Store {
        word: ch.w(SENDS),
        val: add(r(K), c(1)),
    });
    a.push(Step::Store {
        word: turn(ch.cap, K),
        val: ch.full(K, b),
    });
    unlock_channel(&mut a);
    notify(&mut a, ch, b);
    a.jump(next);
    a.here(full);
    unlock_channel(&mut a);
    sleep(&mut a, ch, None, b);
    a.jump(retry);
    a.here(next);
    a.push(Step::Set {
        dst: J,
        val: add(r(J), c(1)),
    });
    a.jump(top);
    a.here(drop);
    let no_notify = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SENDERS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), no_notify);
    notify(&mut a, ch, b);
    a.here(no_notify);
    broadcast_release(&mut a, ch);
    a.push(Step::Done);
    a.finish()
}

/// What a broadcast receiver task does besides `recv`.
#[derive(Clone, Copy)]
struct BroadcastTask {
    /// The id of the receiver. Its next number is in the word `B_NEXT + id`.
    id: i64,
    /// The task waits until another task cloned the receiver for it.
    cloned: bool,
    /// After this many values, clone the receiver for the task of receiver `child`.
    clone_after: Option<(i64, i64)>,
    /// Drop the receiver after this many values, or at `Closed`.
    limit: i64,
}

/// The clone of a broadcast receiver: it adds 1 to the count of each value that the receiver
/// did not read, under the lock. The new receiver starts at the same place.
fn clone_receiver(a: &mut Asm, ch: Chan, id: i64, child: i64, b: Breaks) {
    if !b.clone_without_the_lock {
        lock_channel(a);
    }
    a.push(Step::Load {
        dst: K,
        word: c(B_NEXT + id),
    });
    a.push(Step::Load {
        dst: T2,
        word: ch.w(SENDS),
    });
    if !b.clone_skips_the_counts {
        let (top, done) = (a.label(), a.label());
        a.push(Step::Set { dst: V, val: r(K) });
        a.here(top);
        a.jump_if(r(V), Cmp::Eq, r(T2), done);
        a.push(Step::FetchAdd {
            dst: T,
            word: readers(ch.cap, V),
            delta: c(1),
        });
        a.push(Step::Set {
            dst: V,
            val: add(r(V), c(1)),
        });
        a.jump(top);
        a.here(done);
    }
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(RECEIVER),
        delta: c(1),
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(ENDS),
        delta: c(1),
    });
    if !b.clone_without_the_lock {
        unlock_channel(a);
    }
    a.push(Step::StartAt {
        recv: child as usize,
        val: r(K),
    });
    a.push(Step::Store {
        word: c(B_NEXT + child),
        val: r(K),
    });
    a.push(Step::Store {
        word: c(HANDOFF),
        val: c(1),
    });
    a.push(Step::WakeAll { word: c(HANDOFF) });
}

/// A task with a `BroadcastReceiver`. Its `try_recv` reads with no lock. When the count of the
/// value is 1, it moves the value out. Else it clones the value, and takes 1 from the count. The
/// receiver that takes the count to 0 drops the value. Each frees the slot for the next `send`.
fn broadcast_receiver(ch: Chan, task: BroadcastTask, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let next = c(B_NEXT + task.id);
    let id = task.id as usize;
    if task.cloned {
        let (wait, go) = (a.label(), a.label());
        a.here(wait);
        a.push(Step::Load {
            dst: T,
            word: c(HANDOFF),
        });
        a.jump_if(r(T), Cmp::Ne, c(0), go);
        a.push(Step::WaitWhile {
            word: c(HANDOFF),
            val: c(0),
            or_cancel: None,
        });
        a.jump(wait);
        a.here(go);
    }
    let (top, retry, not_there, share, counted, drop) = (
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
    );
    a.push(Step::Set { dst: J, val: c(0) });
    a.push(Step::Set {
        dst: OPEN,
        val: c(0),
    });
    a.here(top);
    if let Some((after, child)) = task.clone_after {
        let no_clone = a.label();
        a.jump_if(r(J), Cmp::Ne, c(after), no_clone);
        a.jump_if(r(OPEN), Cmp::Ne, c(0), no_clone);
        clone_receiver(&mut a, ch, task.id, child, b);
        a.push(Step::Set {
            dst: OPEN,
            val: c(1),
        });
        a.here(no_clone);
    }
    a.jump_if(r(J), Cmp::Eq, c(task.limit), drop);
    a.here(retry);
    a.push(Step::Load {
        dst: SEEN,
        word: ch.w(WAKE),
    });
    // `try_recv`.
    a.push(Step::Load {
        dst: CLOSED,
        word: ch.w(SENDERS),
    });
    a.push(Step::Load {
        dst: K,
        word: next.clone(),
    });
    a.push(Step::Load {
        dst: T,
        word: turn(ch.cap, K),
    });
    a.jump_if(r(T), Cmp::Ne, ch.full(K, b), not_there);
    a.push(Step::Store {
        word: next.clone(),
        val: add(r(K), c(1)),
    });
    a.push(Step::Load {
        dst: T,
        word: readers(ch.cap, K),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), share);
    // Each other receiver read the value, so this one moves it out.
    a.push(Step::Store {
        word: readers(ch.cap, K),
        val: c(0),
    });
    a.push(Step::Take {
        dst: V,
        slot: ch.slot(K),
    });
    a.push(Step::Store {
        word: turn(ch.cap, K),
        val: ch.taken(K, b),
    });
    notify(&mut a, ch, b);
    a.push(Step::Got { recv: id, val: V });
    a.push(Step::Ends {
        val: V,
        how: End::Received,
    });
    a.jump(counted);
    a.here(share);
    a.push(Step::Peek {
        dst: V,
        slot: ch.slot(K),
    });
    a.push(Step::Got { recv: id, val: V });
    a.push(Step::FetchAdd {
        dst: T,
        word: readers(ch.cap, K),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), counted);
    broadcast_free(&mut a, ch, K, b);
    a.here(counted);
    a.push(Step::Set {
        dst: J,
        val: add(r(J), c(1)),
    });
    a.jump(top);
    a.here(not_there);
    let wait = a.label();
    a.jump_if(r(CLOSED), Cmp::Ne, c(0), wait);
    a.push(Step::ReceiverClosed {
        recv: id,
        sends: ch.base + SENDS,
    });
    a.jump(drop);
    a.here(wait);
    sleep(&mut a, ch, None, b);
    a.jump(retry);
    a.here(drop);
    // The drop of the receiver takes 1 from the count of each value that it did not read.
    if !b.drop_without_the_lock {
        lock_channel(&mut a);
    }
    a.push(Step::Load {
        dst: T2,
        word: ch.w(SENDS),
    });
    a.push(Step::Load { dst: K, word: next });
    let (each, step, done) = (a.label(), a.label(), a.label());
    a.here(each);
    a.jump_if(r(K), Cmp::Eq, r(T2), done);
    if !b.drop_keeps_its_counts {
        a.push(Step::FetchAdd {
            dst: T,
            word: readers(ch.cap, K),
            delta: c(-1),
        });
        a.jump_if(r(T), Cmp::Ne, c(1), step);
        broadcast_free(&mut a, ch, K, b);
    }
    a.here(step);
    a.push(Step::Set {
        dst: K,
        val: add(r(K), c(1)),
    });
    a.jump(each);
    a.here(done);
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(RECEIVER),
        delta: c(-1),
    });
    if !b.drop_without_the_lock {
        unlock_channel(&mut a);
    }
    notify(&mut a, ch, b);
    broadcast_release(&mut a, ch);
    a.push(Step::Done);
    a.finish()
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Thread {
    pc: usize,
    regs: [i64; 10],
    /// The word that the thread waits on, when it waits.
    waits_on: Option<i64>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct State {
    /// The shared words. Each value is small, and an `i16` keeps a state small: the search
    /// stores millions of them.
    words: Vec<i16>,
    ring: Vec<Option<i64>>,
    threads: Vec<Thread>,
    /// One bit for each value that ended.
    ended: u64,
    /// The last value received from each sender, by sender id.
    last: [i64; 4],
    /// The count of values that each receiver of a broadcast got, by receiver id.
    got: [i64; 4],
    in_lock: u8,
    /// The skip set of each channel: (where its words start, a send number), in order.
    skips: Vec<(i64, i64)>,
}

/// A bug that the search found, and where each thread was in the state before it.
#[derive(Debug)]
struct Found {
    what: String,
    threads: String,
}

fn found(what: String, s: &State) -> Found {
    let threads = s
        .threads
        .iter()
        .enumerate()
        .map(|(i, t)| match t.waits_on {
            Some(w) => format!("thread {i} at step {} waits on word {w}", t.pc),
            None => format!("thread {i} at step {}", t.pc),
        })
        .collect::<Vec<_>>()
        .join(", ");
    Found { what, threads }
}

struct Model {
    programs: Vec<Vec<Step>>,
    /// The values that must end, by sender id and count.
    senders: Vec<(i64, i64)>,
    /// How a value may end in this model.
    allowed: Vec<End>,
    /// The words at the start, as `channel` and `Mutex::new` set them.
    first_words: Vec<i64>,
    ring: usize,
    /// The values of each sender must arrive in order. With many receivers, a later value can
    /// end before an earlier one, so the work queue does not check this.
    in_order: bool,
}

/// A 128-bit hash of `s`. The search keeps this in place of each state that it saw, so that
/// millions of states fit in the memory of a test machine. Two states with the same hash would
/// hide one of them, but with a few million states the chance is about 1 in 10^25.
/// The most states that one search may visit. Each takes about 40 bytes of memory, so this is
/// under 1 GB, and a few searches at once fit on a CI runner.
const MAX_STATES: usize = 20_000_000;

fn fingerprint(s: &impl Hash) -> u128 {
    let half = |salt: u8| {
        let mut h = DefaultHasher::new();
        salt.hash(&mut h);
        s.hash(&mut h);
        h.finish() as u128
    };
    (half(0) << 64) | half(1)
}

/// `v` as a shared word.
fn narrow(v: i64) -> i16 {
    i16::try_from(v).expect("a shared word of the model fits in an i16")
}

/// The bit of the value `v`, from sender `v / 100`.
fn bit(v: i64) -> u64 {
    1u64 << ((v / 100) * 16 + v % 100)
}

impl Model {
    fn start(&self) -> State {
        State {
            words: self.first_words.iter().map(|&w| narrow(w)).collect(),
            ring: vec![None; self.ring],
            threads: self
                .programs
                .iter()
                .map(|_| Thread {
                    pc: 0,
                    regs: [0; 10],
                    waits_on: None,
                })
                .collect(),
            ended: 0,
            last: [0; 4],
            got: [0; 4],
            in_lock: 0,
            skips: Vec::new(),
        }
    }

    /// The count of ways that the next step of thread `t` can go. A `WakeOne` has one way for
    /// each thread that waits on its word.
    fn choices(&self, s: &State, t: usize) -> usize {
        let th = &s.threads[t];
        match &self.programs[t][th.pc] {
            Step::WakeOne { word } => {
                let w = word.eval(&th.regs);
                let waiters = s.threads.iter().filter(|o| o.waits_on == Some(w));
                waiters.count().max(1)
            }
            _ => 1,
        }
    }

    /// Run one step of thread `t`, and then the steps after it that touch only its registers.
    /// No other thread can see those, so running them at once hides no interleaving.
    /// `choice` is a number below `choices`. `Ok(false)` when the thread cannot move.
    fn step(&self, s: &mut State, t: usize, choice: usize) -> Result<bool, String> {
        let th = &s.threads[t];
        if th.waits_on.is_some() || matches!(self.programs[t][th.pc], Step::Done) {
            return Ok(false);
        }
        if let Step::Join { thread } = self.programs[t][th.pc] {
            let other = &s.threads[thread];
            if !matches!(self.programs[thread][other.pc], Step::Done) {
                return Ok(false);
            }
        }
        self.act(s, t, choice)?;
        while s.threads[t].waits_on.is_none()
            && matches!(
                self.programs[t][s.threads[t].pc],
                Step::Set { .. } | Step::If { .. } | Step::Jump { .. }
            )
        {
            self.act(s, t, 0)?;
        }
        Ok(true)
    }

    fn act(&self, s: &mut State, t: usize, choice: usize) -> Result<(), String> {
        let th = &s.threads[t];
        let regs = th.regs;
        let mut pc = th.pc + 1;
        match &self.programs[t][th.pc] {
            Step::Done => unreachable!("a thread that is done does not move"),
            Step::FetchAdd { dst, word, delta } => {
                let w = word.eval(&regs) as usize;
                let old = s.words[w] as i64;
                s.words[w] = narrow(old + delta.eval(&regs));
                s.threads[t].regs[*dst] = old;
            }
            Step::Load { dst, word } => {
                s.threads[t].regs[*dst] = s.words[word.eval(&regs) as usize] as i64;
            }
            Step::Store { word, val } => {
                s.words[word.eval(&regs) as usize] = narrow(val.eval(&regs));
            }
            Step::WaitWhile {
                word,
                val,
                or_cancel,
            } => {
                let w = word.eval(&regs);
                let cancelled = or_cancel.is_some_and(|m| s.words[m as usize] != 0);
                if s.words[w as usize] as i64 == val.eval(&regs) && !cancelled {
                    s.threads[t].waits_on = Some(w);
                }
            }
            Step::WakeOne { word } => {
                let w = word.eval(&regs);
                let mut waiters = s.threads.iter_mut().filter(|o| o.waits_on == Some(w));
                if let Some(other) = waiters.nth(choice) {
                    other.waits_on = None;
                }
            }
            Step::EndWait { thread } => s.threads[*thread].waits_on = None,
            Step::WakeEvery => {
                for other in s.threads.iter_mut() {
                    other.waits_on = None;
                }
            }
            Step::Join { .. } => {}
            Step::WakeAll { word } => {
                let w = word.eval(&regs);
                for other in s.threads.iter_mut() {
                    if other.waits_on == Some(w) {
                        other.waits_on = None;
                    }
                }
            }
            Step::Put { slot, val } => {
                let i = slot.eval(&regs) as usize;
                if let Some(old) = s.ring[i] {
                    return Err(format!(
                        "a send wrote over the value {old}, which no recv took"
                    ));
                }
                s.ring[i] = Some(val.eval(&regs));
            }
            Step::Take { dst, slot } => {
                let i = slot.eval(&regs) as usize;
                match s.ring[i].take() {
                    Some(v) => s.threads[t].regs[*dst] = v,
                    None => return Err("a recv took a value from an empty slot".to_string()),
                }
            }
            Step::Ends { val, how } => self.end(s, regs[*val], *how)?,
            Step::EndsNow { val, how } => self.end(s, val.eval(&regs), *how)?,
            Step::Cas {
                dst,
                word,
                expect,
                new,
            } => {
                let w = word.eval(&regs) as usize;
                let hit = s.words[w] as i64 == expect.eval(&regs);
                if hit {
                    s.words[w] = narrow(new.eval(&regs));
                }
                s.threads[t].regs[*dst] = hit as i64;
            }
            Step::Peek { dst, slot } => match s.ring[slot.eval(&regs) as usize] {
                Some(v) => s.threads[t].regs[*dst] = v,
                None => return Err("a receiver cloned a value from an empty slot".to_string()),
            },
            Step::Got { recv, val } => {
                let n = regs[*val] % 100;
                if n != s.got[*recv] + 1 {
                    return Err(format!(
                        "receiver {recv} got the value number {n} after {} values",
                        s.got[*recv]
                    ));
                }
                s.got[*recv] = n;
            }
            Step::StartAt { recv, val } => s.got[*recv] = val.eval(&regs),
            Step::ReceiverClosed { recv, sends } => {
                if s.got[*recv] != s.words[*sends as usize] as i64 {
                    return Err(format!(
                        "receiver {recv} gave Closed before it got each value"
                    ));
                }
            }
            Step::QueueClosed { chan } => {
                let recvs = s.words[(chan.base + RECVS) as usize] as i64;
                for i in 0..chan.cap {
                    let n = s.words[(chan.base + SLOTS + i) as usize] as i64;
                    let full = n % 2 == 1 && (n - 1) / 2 >= recvs;
                    if full && s.ring[(chan.ring + i) as usize].is_some() {
                        return Err(
                            "a recv gave Closed while a value that no recv took was in the queue"
                                .to_string(),
                        );
                    }
                }
            }
            Step::Enter => {
                s.in_lock += 1;
                if s.in_lock > 1 {
                    return Err("two threads hold the Mutex at the same time".to_string());
                }
            }
            Step::Leave => s.in_lock -= 1,
            Step::SkipAdd { chan, k } => {
                let e = (*chan, regs[*k]);
                if s.skips.contains(&e) {
                    return Err(format!("the send number {} was skipped twice", e.1));
                }
                s.skips.push(e);
                s.skips.sort();
            }
            Step::SkipTake { dst, chan, k } => {
                let e = (*chan, regs[*k]);
                let had = s.skips.contains(&e);
                s.skips.retain(|x| *x != e);
                s.threads[t].regs[*dst] = had as i64;
            }
            Step::SkipForget { chan } => s.skips.retain(|x| x.0 != *chan),
            Step::Set { dst, val } => s.threads[t].regs[*dst] = val.eval(&regs),
            Step::If { a, cmp, b, to } => {
                let (a, b) = (a.eval(&regs), b.eval(&regs));
                let taken = match cmp {
                    Cmp::Eq => a == b,
                    Cmp::Ne => a != b,
                    Cmp::Gt => a > b,
                };
                if taken {
                    pc = *to;
                }
            }
            Step::Jump { to } => pc = *to,
        }
        s.threads[t].pc = pc;
        Ok(())
    }

    fn end(&self, s: &mut State, v: i64, how: End) -> Result<(), String> {
        if !self.allowed.contains(&how) {
            return Err(format!(
                "the value {v} ended as {how:?}, which this run must not do"
            ));
        }
        if s.ended & bit(v) != 0 {
            return Err(format!("the value {v} ended twice"));
        }
        s.ended |= bit(v);
        if how == End::Received && self.in_order {
            let (from, n) = ((v / 100) as usize, v % 100);
            if n <= s.last[from] {
                return Err(format!(
                    "the value {v} came after a later value of its sender"
                ));
            }
            s.last[from] = n;
        }
        Ok(())
    }

    /// At the end of a run: each value ended, and the ring is empty.
    fn check_end(&self, s: &State) -> Result<(), String> {
        for &(id, count) in &self.senders {
            for n in 1..=count {
                let v = id * 100 + n;
                if s.ended & bit(v) == 0 {
                    return Err(format!("the value {v} never ended: it was lost"));
                }
            }
        }
        if s.ring.iter().any(|v| v.is_some()) {
            return Err("the channel was freed with a value still in it".to_string());
        }
        if !s.skips.is_empty() {
            return Err("the channel was freed with a number still in its skip set".to_string());
        }
        Ok(())
    }

    /// Visit every interleaving. Answers the count of states.
    fn check(&self) -> Result<usize, Found> {
        self.search(None, None)
    }

    /// Visit each interleaving that switches away from a thread that can still move at most
    /// `k` times. A model with many threads has too many states for `check`, and most bugs of
    /// concurrent code need only a few such switches. Answers the count of states.
    fn check_within(&self, k: u8) -> Result<usize, Found> {
        self.search(Some(k), None)
    }

    /// `check_within(k)`, but a failure whose text does not contain `what` ends only its own
    /// run, and the search goes on. So the failure it answers is the one that a test looks for.
    fn find_within(&self, k: u8, what: &str) -> Result<usize, Found> {
        self.search(Some(k), Some(what))
    }

    fn search(&self, bound: Option<u8>, want: Option<&str>) -> Result<usize, Found> {
        let other = |what: &str| want.is_some_and(|w| !what.contains(w));
        // With a bound, a state counts again when a run reaches it with fewer switches, so the
        // key holds the last thread and the count of switches too.
        let key = |s: &State, last: usize, used: u8| match bound {
            None => fingerprint(s),
            Some(_) => fingerprint(&(s, last, used)),
        };
        let mut seen = HashSet::new();
        let start = self.start();
        seen.insert(key(&start, usize::MAX, 0));
        let mut stack = vec![(start, usize::MAX, 0u8)];
        while let Some((s, last, used)) = stack.pop() {
            // Each move is a thread and the state after its step.
            let mut moves = Vec::new();
            let mut cut = false;
            for t in 0..s.threads.len() {
                for choice in 0..self.choices(&s, t) {
                    let mut next = s.clone();
                    match self.step(&mut next, t, choice) {
                        Ok(false) => {}
                        Ok(true) => moves.push((t, next)),
                        Err(what) if other(&what) => cut = true,
                        Err(what) => return Err(found(what, &s)),
                    }
                }
            }
            let last_can_move = moves.iter().any(|m| m.0 == last);
            let moved = !moves.is_empty();
            for (t, next) in moves {
                let used = used + u8::from(last_can_move && t != last);
                if bound.is_some_and(|k| used > k) {
                    continue;
                }
                if seen.insert(key(&next, t, used)) {
                    assert!(
                        seen.len() <= MAX_STATES,
                        "the model has more than {MAX_STATES} states, so it needs too much \
                         memory: make it smaller"
                    );
                    stack.push((next, t, used));
                }
            }
            if moved || cut {
                continue;
            }
            let stuck = "each thread that is not done waits, and nothing can wake it".to_string();
            let end = if s.threads.iter().any(|t| t.waits_on.is_some()) {
                Err(stuck)
            } else {
                self.check_end(&s)
            };
            match end {
                Err(what) if !other(&what) => return Err(found(what, &s)),
                _ => {}
            }
        }
        Ok(seen.len())
    }
}

/// The code of a `tasks` block that holds a `Sender` and leaves the block early. As
/// `vx_tasks_wait`, it cancels the block and joins the task `receiver`. Its own drops come after
/// the join, so it drops the `Sender` last.
fn leaving_block(ch: Chan, receiver: usize, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    a.push(Step::Store {
        word: c(CANCEL),
        val: c(1),
    });
    a.push(Step::WakeEvery);
    a.push(Step::Join { thread: receiver });
    let no_notify = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SENDERS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), no_notify);
    notify(&mut a, ch, b);
    a.here(no_notify);
    release(&mut a, ch, b);
    a.push(Step::Done);
    a.finish()
}

/// A channel with room for `cap`, `senders` tasks that each send `count` values, and one task
/// that receives `limit` values or until `Closed`.
fn channel_model(cap: i64, senders: i64, count: i64, limit: i64, b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap,
    };
    let mut programs: Vec<Vec<Step>> = (1..=senders)
        .map(|id| sender(ch, id, count, CANCEL, b))
        .collect();
    programs.push(receiver(ch, limit, CANCEL, b));
    let mut words = vec![0; WORDS];
    words[SENDERS as usize] = senders;
    words[RECEIVER as usize] = 1;
    words[ENDS as usize] = senders + 1;
    for i in 0..cap {
        // The first number of slot `i`: free for the send with number `i`.
        words[(SLOTS + i) as usize] = if b.vyukov_numbering { i } else { 2 * i };
    }
    Model {
        programs,
        senders: (1..=senders).map(|id| (id, count)).collect(),
        allowed: if limit == i64::MAX {
            vec![End::Received]
        } else {
            vec![End::Received, End::GivenBack, End::Dropped]
        },
        first_words: words,
        ring: cap as usize,
        in_order: true,
    }
}

/// Two channels with room for `cap`, one sender task on each, and one task in a `select` loop.
fn select_model(cap: i64, count: i64, b: Breaks) -> Model {
    let chans = [
        Chan {
            base: 0,
            ring: 0,
            cap,
        },
        Chan {
            base: 20,
            ring: cap,
            cap,
        },
    ];
    let mut programs = vec![
        sender(chans[0], 1, count, CANCEL, b),
        sender(chans[1], 2, count, CANCEL, b),
    ];
    programs.push(selector(chans, b));
    let mut words = vec![0; WORDS];
    for ch in chans {
        words[(ch.base + SENDERS) as usize] = 1;
        words[(ch.base + RECEIVER) as usize] = 1;
        words[(ch.base + ENDS) as usize] = 2;
        for i in 0..cap {
            words[(ch.base + SLOTS + i) as usize] = 2 * i;
        }
    }
    Model {
        programs,
        senders: vec![(1, count), (2, count)],
        allowed: vec![End::Received],
        first_words: words,
        ring: 2 * cap as usize,
        in_order: true,
    }
}

fn mutex_model(threads: usize, count: i64, b: Breaks) -> Model {
    Model {
        programs: (0..threads).map(|_| locker(count, b)).collect(),
        senders: Vec::new(),
        allowed: Vec::new(),
        first_words: vec![0; 3],
        ring: 0,
        in_order: true,
    }
}

fn holds(name: &str, model: Model) {
    match model.check() {
        Ok(states) => println!("{name}: holds in {states} states"),
        Err(f) => panic!("{name}: {}. {}", f.what, f.threads),
    }
}

fn finds(name: &str, model: Model, what: &str) {
    match model.check() {
        Ok(states) => panic!("{name}: the search visited {states} states and found no bug"),
        Err(f) => {
            println!("{name}: found \"{}\". {}", f.what, f.threads);
            assert!(f.what.contains(what), "{name}: expected \"{what}\"");
        }
    }
}

/// `holds`, with the search of `check_within(k)`.
fn holds_within(name: &str, k: u8, model: Model) {
    match model.check_within(k) {
        Ok(states) => println!("{name}: holds within {k} switches, in {states} states"),
        Err(f) => panic!("{name}: {}. {}", f.what, f.threads),
    }
}

/// `finds`, with the search of `find_within(k, what)`.
fn finds_within(name: &str, k: u8, model: Model, what: &str) {
    match model.find_within(k, what) {
        Ok(states) => panic!("{name}: the search visited {states} states and found no bug"),
        Err(f) => {
            println!("{name}: found \"{}\". {}", f.what, f.threads);
            assert!(f.what.contains(what), "{name}: expected \"{what}\"");
        }
    }
}

const NONE: Breaks = Breaks {
    vyukov_numbering: false,
    no_select_event: false,
    senders_after_slot: false,
    unlock_wakes_none: false,
    wait_ignores_cancel: false,
    send_keeps_its_place: false,
    recv_ignores_skips: false,
    skip_keeps_the_slot: false,
    pass_does_not_wait_for_the_slot: false,
    queue_takes_with_a_store: false,
    queue_closed_without_the_count: false,
    clone_skips_the_counts: false,
    clone_without_the_lock: false,
    drop_without_the_lock: false,
    drop_keeps_its_counts: false,
    acquire_gives_up_and_wakes_none: false,
    permit_drop_needs_two_sleepers: false,
};

#[test]
fn channel_with_room_for_one() {
    holds(
        "2 senders, 2 values each, room 1",
        channel_model(1, 2, 2, i64::MAX, NONE),
    );
}

#[test]
fn channel_with_room_for_two() {
    holds(
        "2 senders, 2 values each, room 2",
        channel_model(2, 2, 2, i64::MAX, NONE),
    );
}

#[test]
fn channel_whose_receiver_stops_early() {
    // The receiver takes one value and drops its end. Each other value is given back by its
    // `send` or dropped by the last end.
    holds(
        "2 senders, 2 values each, room 1, 1 recv",
        channel_model(1, 2, 2, 1, NONE),
    );
    holds(
        "1 sender, 3 values, room 2, 1 recv",
        channel_model(2, 1, 3, 1, NONE),
    );
}

#[test]
fn select_on_two_channels() {
    holds(
        "select, 2 values on each of 2 channels, room 1",
        select_model(1, 2, NONE),
    );
}

#[test]
fn mutex_with_three_threads() {
    holds("3 threads, 2 locks each", mutex_model(3, 2, NONE));
}

// The words of a `Semaphore`, as in `semaphore.vx`.
const PERMITS: i64 = 0;
const S_SLEEPERS: i64 = 1;
// Not 0 when the time limit of the first task passed. `vx_cancelled_at` then gives true in that
// task, as in a call of `within`. No cancel comes with it, so no other task wakes.
const LATE: i64 = 2;

/// A task that calls `acquire`, runs the code that the permit guards, and drops the `Permit`.
/// With `late`, the task has a time limit: when that word is set, its wait ends and its `acquire`
/// gives up. With `one_permit`, the permit works as a lock, and `Enter` checks that one task at
/// a time holds it.
fn acquirer(late: Option<i64>, one_permit: bool, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    let (top, sleep, got, wake, gave_up, done) = (
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
    );
    // `acquire`: SEEN is the value of `permits`.
    a.push(Step::Load {
        dst: SEEN,
        word: c(PERMITS),
    });
    a.here(top);
    if let Some(late) = late {
        a.push(Step::Load {
            dst: T,
            word: c(late),
        });
        a.jump_if(r(T), Cmp::Ne, c(0), gave_up);
    }
    a.jump_if(r(SEEN), Cmp::Eq, c(0), sleep);
    a.push(Step::Cas {
        dst: T,
        word: c(PERMITS),
        expect: r(SEEN),
        new: add(r(SEEN), c(-1)),
    });
    a.jump_if(r(T), Cmp::Eq, c(1), got);
    a.push(Step::Load {
        dst: SEEN,
        word: c(PERMITS),
    });
    a.jump(top);
    a.here(sleep);
    a.push(Step::FetchAdd {
        dst: T,
        word: c(S_SLEEPERS),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: c(PERMITS),
        val: c(0),
        or_cancel: late,
    });
    a.push(Step::FetchAdd {
        dst: T,
        word: c(S_SLEEPERS),
        delta: c(-1),
    });
    a.push(Step::Load {
        dst: SEEN,
        word: c(PERMITS),
    });
    a.jump(top);
    a.here(got);
    if one_permit {
        a.push(Step::Enter);
        a.push(Step::Leave);
    }
    // The drop of the `Permit`.
    a.push(Step::FetchAdd {
        dst: T,
        word: c(PERMITS),
        delta: c(1),
    });
    a.push(Step::Load {
        dst: T,
        word: c(S_SLEEPERS),
    });
    let enough = if b.permit_drop_needs_two_sleepers {
        1
    } else {
        0
    };
    a.jump_if(r(T), Cmp::Gt, c(enough), wake);
    a.jump(done);
    // `wake_a_sleeper`, after an `acquire` that gives up.
    a.here(gave_up);
    if b.acquire_gives_up_and_wakes_none {
        a.jump(done);
    }
    a.push(Step::Load {
        dst: T,
        word: c(PERMITS),
    });
    a.jump_if(r(T), Cmp::Eq, c(0), done);
    a.push(Step::Load {
        dst: T,
        word: c(S_SLEEPERS),
    });
    a.jump_if(r(T), Cmp::Eq, c(0), done);
    a.here(wake);
    a.push(Step::WakeOne { word: c(PERMITS) });
    a.here(done);
    a.push(Step::Done);
    a.finish()
}

/// A `Semaphore` with `permits` permits and `tasks` tasks that call `acquire`. The first `timed`
/// tasks have a time limit, and the last thread is the clock: at any point, the limit passes and
/// the wait of each of these tasks ends.
fn semaphore_model(permits: i64, tasks: usize, timed: usize, b: Breaks) -> Model {
    let mut clock = Asm::new();
    clock.push(Step::Store {
        word: c(LATE),
        val: c(1),
    });
    for thread in 0..timed {
        clock.push(Step::EndWait { thread });
    }
    clock.push(Step::Done);
    let late = |task: usize| (task < timed).then_some(LATE);
    let mut programs: Vec<_> = (0..tasks)
        .map(|task| acquirer(late(task), permits == 1, b))
        .collect();
    programs.push(clock.finish());
    Model {
        programs,
        senders: Vec::new(),
        allowed: Vec::new(),
        first_words: vec![permits, 0, 0],
        ring: 0,
        in_order: true,
    }
}

#[test]
fn semaphore_with_three_tasks_and_one_that_gives_up() {
    holds(
        "1 permit, 3 tasks, and the time limit of one task passes",
        semaphore_model(1, 3, 1, NONE),
    );
}

/// The first task gives up and wakes the second one, which gives up too.
#[test]
fn semaphore_with_two_tasks_that_give_up() {
    holds(
        "1 permit, 3 tasks, and the time limit of two tasks passes",
        semaphore_model(1, 3, 2, NONE),
    );
}

/// With 2 permits and 4 tasks, two tasks can sleep while two drops give permits back. Each drop
/// must wake a sleeper of its own.
#[test]
fn semaphore_with_two_permits_and_four_tasks() {
    holds(
        "2 permits, 4 tasks, and the time limit of one task passes",
        semaphore_model(2, 4, 1, NONE),
    );
}

#[test]
fn the_search_finds_a_lost_wake_with_two_permits() {
    let b = Breaks {
        permit_drop_needs_two_sleepers: true,
        ..NONE
    };
    finds(
        "2 permits, and the drop of a permit wakes only when two tasks sleep",
        semaphore_model(2, 4, 1, b),
        "nothing can wake it",
    );
}

/// The drop of a `Permit` wakes one sleeper. When that sleeper gives up, it must wake the next
/// one, or the permit stays free while a task sleeps.
#[test]
fn the_search_finds_an_acquire_that_gives_up_and_wakes_no_one() {
    let b = Breaks {
        acquire_gives_up_and_wakes_none: true,
        ..NONE
    };
    finds(
        "an acquire that gives up wakes no one",
        semaphore_model(1, 3, 1, b),
        "nothing can wake it",
    );
}

#[test]
fn the_search_finds_a_permit_drop_that_wakes_only_when_two_tasks_sleep() {
    let b = Breaks {
        permit_drop_needs_two_sleepers: true,
        ..NONE
    };
    finds(
        "the drop of a permit wakes only when two tasks sleep",
        semaphore_model(1, 3, 1, b),
        "nothing can wake it",
    );
}

#[test]
fn the_search_finds_vyukov_numbering_with_room_for_one() {
    let b = Breaks {
        vyukov_numbering: true,
        ..NONE
    };
    finds(
        "Vyukov numbering, room 1",
        channel_model(1, 1, 2, i64::MAX, b),
        "was lost",
    );
}

#[test]
fn the_search_finds_a_select_that_misses_an_event() {
    let b = Breaks {
        no_select_event: true,
        ..NONE
    };
    finds(
        "no select event",
        select_model(1, 1, b),
        "nothing can wake it",
    );
}

#[test]
fn the_search_finds_a_recv_that_reads_senders_after_the_slot() {
    let b = Breaks {
        senders_after_slot: true,
        ..NONE
    };
    finds(
        "SENDERS after the slot",
        channel_model(1, 1, 1, i64::MAX, b),
        "ended as Dropped",
    );
}

#[test]
fn the_search_finds_an_unlock_that_wakes_no_one() {
    let b = Breaks {
        unlock_wakes_none: true,
        ..NONE
    };
    finds(
        "unlock wakes no one",
        mutex_model(2, 1, b),
        "nothing can wake it",
    );
}

/// `channel_model`, and the code of the block holds one more `Sender` and leaves the block early,
/// at any point. Each value is then received, given back by its `send`, or dropped by the last
/// end, and the `recv` must see the cancel.
fn cancelled_channel_model(b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap: 1,
    };
    let mut model = channel_model(1, 1, 2, 1000, b);
    model.first_words[SENDERS as usize] += 1;
    model.first_words[ENDS as usize] += 1;
    // The threads are the sender, the receiver, and then the code of the block.
    model.programs.push(leaving_block(ch, 1, b));
    model
}

#[test]
fn channel_with_a_cancel() {
    holds(
        "1 sender, 2 values, room 1, and a cancel",
        cancelled_channel_model(NONE),
    );
}

#[test]
fn the_search_finds_a_wait_that_does_not_read_the_cancel() {
    let b = Breaks {
        wait_ignores_cancel: true,
        ..NONE
    };
    finds(
        "the wait does not read the cancel",
        cancelled_channel_model(b),
        "nothing can wake it",
    );
}

/// The code of a `tasks` block that holds the `Receiver` and leaves the block early: it cancels,
/// joins the task `sender`, and drops the `Receiver` last.
fn leaving_block_with_the_receiver(ch: Chan, sender: usize, b: Breaks) -> Vec<Step> {
    let mut a = Asm::new();
    a.push(Step::Store {
        word: c(CANCEL),
        val: c(1),
    });
    a.push(Step::WakeEvery);
    a.push(Step::Join { thread: sender });
    drop_receiver(&mut a, ch, b);
    a.push(Step::Done);
    a.finish()
}

/// One sender task sends 2 values with room 1. The code of the block holds the `Receiver` and
/// leaves the block at any point: it cancels, joins the sender, and drops the `Receiver` last.
fn block_with_the_receiver_model(b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap: 1,
    };
    let mut words = vec![0; WORDS];
    words[SENDERS as usize] = 1;
    words[RECEIVER as usize] = 1;
    words[ENDS as usize] = 2;
    Model {
        programs: vec![
            sender(ch, 1, 2, CANCEL, b),
            leaving_block_with_the_receiver(ch, 0, b),
        ],
        senders: vec![(1, 2)],
        allowed: vec![End::GivenBack, End::Dropped],
        first_words: words,
        ring: 1,
        in_order: true,
    }
}

#[test]
fn a_cancel_stops_a_send_that_has_its_place() {
    holds(
        "a send that has its place, and a block with the Receiver that leaves",
        block_with_the_receiver_model(NONE),
    );
}

#[test]
fn the_search_finds_a_send_that_keeps_its_place_after_a_cancel() {
    let b = Breaks {
        send_keeps_its_place: true,
        ..NONE
    };
    finds(
        "the send keeps its place",
        block_with_the_receiver_model(b),
        "nothing can wake it",
    );
}

/// Room 1. Sender 1 sends 2 values in a block that its code cancels at any point. Sender 2 sends
/// 1 value, and the receiver receives until `Closed`, both in another block. So the `recv`
/// must pass each place that a cancelled `send` left, and free its slot for the next `send`.
fn skip_model(b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap: 1,
    };
    let mut cancel = Asm::new();
    cancel.push(Step::Store {
        word: c(CANCEL),
        val: c(1),
    });
    cancel.push(Step::WakeEvery);
    cancel.push(Step::Join { thread: 0 });
    cancel.push(Step::Done);
    let mut words = vec![0; WORDS];
    words[SENDERS as usize] = 2;
    words[RECEIVER as usize] = 1;
    words[ENDS as usize] = 3;
    Model {
        programs: vec![
            sender(ch, 1, 2, CANCEL, b),
            sender(ch, 2, 1, OTHER_BLOCK, b),
            receiver(ch, i64::MAX, OTHER_BLOCK, b),
            cancel.finish(),
        ],
        senders: vec![(1, 2), (2, 1)],
        allowed: vec![End::Received, End::GivenBack],
        first_words: words,
        ring: 1,
        in_order: true,
    }
}

#[test]
fn a_recv_passes_the_place_of_a_cancelled_send() {
    holds(
        "a cancelled send and a recv in another block",
        skip_model(NONE),
    );
}

#[test]
fn the_search_finds_a_recv_that_does_not_look_in_the_skip_set() {
    let b = Breaks {
        recv_ignores_skips: true,
        ..NONE
    };
    finds("recv ignores skips", skip_model(b), "nothing can wake it");
}

#[test]
fn the_search_finds_a_skip_that_does_not_free_its_slot() {
    let b = Breaks {
        skip_keeps_the_slot: true,
        ..NONE
    };
    finds("skip keeps the slot", skip_model(b), "nothing can wake it");
}

/// A work queue with room `cap` and two `WorkReceiver`s that receive until `Closed`. Sender 1
/// sends `first` values in a block that its code cancels at any point, when `cancel` is true.
/// Sender 2 sends `second` values in another block, with the receivers. So with a cancel, a
/// receiver must pass each place that a cancelled `send` left, while the other receiver takes
/// values.
fn work_queue_model(cap: i64, first: i64, second: i64, cancel: bool, b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap,
    };
    let mut programs = vec![
        sender(ch, 1, first, CANCEL, b),
        sender(ch, 2, second, OTHER_BLOCK, b),
        work_receiver(ch, OTHER_BLOCK, b),
        work_receiver(ch, OTHER_BLOCK, b),
    ];
    if cancel {
        let mut a = Asm::new();
        a.push(Step::Store {
            word: c(CANCEL),
            val: c(1),
        });
        a.push(Step::WakeEvery);
        a.push(Step::Join { thread: 0 });
        a.push(Step::Done);
        programs.push(a.finish());
    }
    let mut words = vec![0; WORDS];
    words[SENDERS as usize] = 2;
    words[RECEIVER as usize] = 2;
    words[ENDS as usize] = 4;
    for i in 0..cap {
        words[(SLOTS + i) as usize] = 2 * i;
    }
    Model {
        programs,
        senders: vec![(1, first), (2, second)],
        allowed: if cancel {
            vec![End::Received, End::GivenBack]
        } else {
            vec![End::Received]
        },
        first_words: words,
        ring: cap as usize,
        in_order: false,
    }
}

#[test]
fn work_queue_with_two_receivers() {
    holds_within(
        "work queue, 2 senders, 2 values each, 2 receivers, room 1",
        2,
        work_queue_model(1, 2, 2, false, NONE),
    );
    holds_within(
        "work queue, 2 senders, 2 values each, 2 receivers, room 2",
        2,
        work_queue_model(2, 2, 2, false, NONE),
    );
}

#[test]
fn work_queue_receivers_pass_the_places_of_cancelled_sends() {
    holds_within(
        "work queue, a cancelled send, 2 receivers, room 1",
        1,
        work_queue_model(1, 2, 1, true, NONE),
    );
    holds_within(
        "work queue, a cancelled send, 2 receivers, room 2",
        1,
        work_queue_model(2, 2, 2, true, NONE),
    );
}

#[test]
fn the_search_finds_a_pass_that_frees_a_slot_still_in_use() {
    // A receiver takes the number `k - 1` and has not yet read the value. The other receiver
    // passes the cancelled send `k` and frees the slot. The send `k + 1` then writes over the
    // value.
    let b = Breaks {
        pass_does_not_wait_for_the_slot: true,
        ..NONE
    };
    finds_within(
        "the pass does not wait for the slot",
        1,
        work_queue_model(1, 2, 1, true, b),
        "wrote over the value",
    );
}

#[test]
fn the_search_finds_two_receivers_that_take_one_number() {
    let b = Breaks {
        queue_takes_with_a_store: true,
        ..NONE
    };
    finds_within(
        "the work queue takes its number with a store",
        2,
        work_queue_model(1, 2, 2, false, b),
        "empty slot",
    );
}

#[test]
fn the_search_finds_a_work_queue_closed_too_early() {
    let b = Breaks {
        queue_closed_without_the_count: true,
        ..NONE
    };
    finds_within(
        "the work queue is closed without the count",
        2,
        work_queue_model(2, 2, 2, true, b),
        "no recv took was in the queue",
    );
}

/// A broadcast channel with room `cap`, and a sender task that sends `count` values. Receivers
/// 0 and 1 are there from the start. Receiver 0 clones itself for receiver 2 after
/// `clone_after` values. Receiver 1 drops its end after `limit` values. Receivers 0 and 2
/// receive until `Closed`.
fn broadcast_model(cap: i64, count: i64, clone_after: i64, limit: i64, b: Breaks) -> Model {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap,
    };
    let programs = vec![
        broadcast_sender(ch, count, b),
        broadcast_receiver(
            ch,
            BroadcastTask {
                id: 0,
                cloned: false,
                clone_after: Some((clone_after, 2)),
                limit: i64::MAX,
            },
            b,
        ),
        broadcast_receiver(
            ch,
            BroadcastTask {
                id: 1,
                cloned: false,
                clone_after: None,
                limit,
            },
            b,
        ),
        broadcast_receiver(
            ch,
            BroadcastTask {
                id: 2,
                cloned: true,
                clone_after: None,
                limit: i64::MAX,
            },
            b,
        ),
    ];
    let mut words = vec![0; WORDS];
    words[SENDERS as usize] = 1;
    words[RECEIVER as usize] = 2;
    words[ENDS as usize] = 3;
    for i in 0..cap {
        words[(B_SLOTS + 2 * i) as usize] = 2 * i;
    }
    Model {
        programs,
        senders: vec![(1, count)],
        allowed: vec![End::Received, End::Dropped],
        first_words: words,
        ring: cap as usize,
        in_order: false,
    }
}

#[test]
fn broadcast_with_a_clone_of_a_receiver() {
    holds_within(
        "broadcast, 2 values, room 1, a clone before the first value",
        2,
        broadcast_model(1, 2, 0, i64::MAX, NONE),
    );
    holds_within(
        "broadcast, 2 values, room 2, a clone after one value",
        2,
        broadcast_model(2, 2, 1, i64::MAX, NONE),
    );
}

#[test]
fn broadcast_with_a_receiver_that_drops_early() {
    holds_within(
        "broadcast, 2 values, room 1, a clone and a drop after one value",
        2,
        broadcast_model(1, 2, 1, 1, NONE),
    );
    holds_within(
        "broadcast, 3 values, room 2, a clone after one value, a drop at the start",
        2,
        broadcast_model(2, 3, 1, 0, NONE),
    );
}

#[test]
fn the_search_finds_a_clone_that_does_not_count_the_unread_values() {
    let b = Breaks {
        clone_skips_the_counts: true,
        ..NONE
    };
    finds_within(
        "the clone skips the counts",
        2,
        broadcast_model(2, 2, 1, i64::MAX, b),
        "empty slot",
    );
}

#[test]
fn the_search_finds_a_clone_without_the_lock() {
    let b = Breaks {
        clone_without_the_lock: true,
        ..NONE
    };
    finds_within(
        "the clone does not take the lock",
        2,
        broadcast_model(1, 2, 0, i64::MAX, b),
        "empty slot",
    );
}

#[test]
fn the_search_finds_a_drop_without_the_lock() {
    let b = Breaks {
        drop_without_the_lock: true,
        ..NONE
    };
    finds_within(
        "the drop does not take the lock",
        2,
        broadcast_model(1, 2, 1, 0, b),
        "nothing can wake it",
    );
}

#[test]
fn the_search_finds_a_drop_that_keeps_its_counts() {
    let b = Breaks {
        drop_keeps_its_counts: true,
        ..NONE
    };
    finds_within(
        "the drop keeps its counts",
        2,
        broadcast_model(1, 3, 1, 1, b),
        "nothing can wake it",
    );
}
