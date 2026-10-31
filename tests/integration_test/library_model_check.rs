//===- library_model_check.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A model check of the channel, `select` and `Mutex` in `stdlib/std/sync`. Each access to a word
// there is a `SeqCst` atomic, so each run of the code is some interleaving of its atomic steps.
// This file writes the steps of each operation as a small program, copied by hand from the Vx
// code, and visits every interleaving of a few threads. `wait_while` blocks only while the word
// holds the value, and `wake_all` wakes only the threads that wait on that word. The runtime can
// also wake a thread early, which never hides a lost wake, so the model leaves that out.
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
    /// the bucket in the runtime. With `or_cancel`, also do not block when `CANCEL` is set, as
    /// `vx_wait_while_or_cancel` does.
    WaitWhile {
        word: E,
        val: E,
        or_cancel: bool,
    },
    WakeAll {
        word: E,
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
}

// The words of a channel, as in `channel.vx`.
const SENDS: i64 = 0;
const RECVS: i64 = 1;
const SENDERS: i64 = 2;
const RECEIVER: i64 = 3;
const ENDS: i64 = 4;
const WAKE: i64 = 5;
const SLEEPERS: i64 = 6;
const SLOTS: i64 = 7;

// The two words for `select`, after the words of every channel.
const ANY_EVENTS: i64 = 40;
const ANY_SLEEPERS: i64 = 41;

// The cancel mark of the `tasks` block of the threads. In the runtime it is per block, and
// `vx_cancelled` reads it.
const CANCEL: i64 = 42;

/// One channel: where its words start, where its slots start in the ring memory, and its room.
#[derive(Clone, Copy)]
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

/// `sleep` in `channel.vx`: wait for an event after `SEEN`, and read `WAKE` again. With
/// `can_cancel`, a cancel also ends the wait.
fn sleep(a: &mut Asm, ch: Chan, can_cancel: bool, b: Breaks) {
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(SLEEPERS),
        delta: c(1),
    });
    a.push(Step::WaitWhile {
        word: ch.w(WAKE),
        val: r(SEEN),
        or_cancel: can_cancel && !b.wait_ignores_cancel,
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
    let done = a.label();
    a.push(Step::FetchAdd {
        dst: T,
        word: ch.w(ENDS),
        delta: c(-1),
    });
    a.jump_if(r(T), Cmp::Ne, c(1), done);
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
}

/// A task that sends `count` values with its own `Sender`, and then drops it. Sender `id`
/// sends the values `id * 100 + 1` and on.
fn sender(ch: Chan, id: i64, count: i64, b: Breaks) -> Vec<Step> {
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
    // A cancel stops a `send` only before it takes its place.
    a.push(Step::Load {
        dst: T,
        word: c(CANCEL),
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
    a.jump_if(r(T), Cmp::Ne, c(1), gave_back);
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
    sleep(&mut a, ch, false, b);
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

/// A task that calls `recv` until it gives `Closed` or `Cancelled`, or `limit` times, and then
/// drops the `Receiver`. `recv` takes its receive number only when it takes a value.
fn receiver(ch: Chan, limit: i64, b: Breaks) -> Vec<Step> {
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
        word: c(CANCEL),
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
    a.jump_if(r(CLOSED), Cmp::Eq, c(0), closed);
    sleep(&mut a, ch, true, b);
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
        let (empty, after) = (a.label(), a.label());
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
        or_cancel: false,
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
        or_cancel: false,
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

#[derive(Clone, PartialEq, Eq, Hash)]
struct Thread {
    pc: usize,
    regs: [i64; 8],
    /// The word that the thread waits on, when it waits.
    waits_on: Option<i64>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct State {
    words: Vec<i64>,
    ring: Vec<Option<i64>>,
    threads: Vec<Thread>,
    /// One bit for each value that ended.
    ended: u64,
    /// The last value received from each sender, by sender id.
    last: [i64; 4],
    in_lock: u8,
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
}

/// The bit of the value `v`, from sender `v / 100`.
fn bit(v: i64) -> u64 {
    1u64 << ((v / 100) * 16 + v % 100)
}

impl Model {
    fn start(&self) -> State {
        State {
            words: self.first_words.clone(),
            ring: vec![None; self.ring],
            threads: self
                .programs
                .iter()
                .map(|_| Thread {
                    pc: 0,
                    regs: [0; 8],
                    waits_on: None,
                })
                .collect(),
            ended: 0,
            last: [0; 4],
            in_lock: 0,
        }
    }

    /// Run one step of thread `t`, and then the steps after it that touch only its registers.
    /// No other thread can see those, so running them at once hides no interleaving.
    /// `Ok(false)` when the thread cannot move.
    fn step(&self, s: &mut State, t: usize) -> Result<bool, String> {
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
        self.act(s, t)?;
        while s.threads[t].waits_on.is_none()
            && matches!(
                self.programs[t][s.threads[t].pc],
                Step::Set { .. } | Step::If { .. } | Step::Jump { .. }
            )
        {
            self.act(s, t)?;
        }
        Ok(true)
    }

    fn act(&self, s: &mut State, t: usize) -> Result<(), String> {
        let th = &s.threads[t];
        let regs = th.regs;
        let mut pc = th.pc + 1;
        match &self.programs[t][th.pc] {
            Step::Done => unreachable!("a thread that is done does not move"),
            Step::FetchAdd { dst, word, delta } => {
                let w = word.eval(&regs) as usize;
                let old = s.words[w];
                s.words[w] = old + delta.eval(&regs);
                s.threads[t].regs[*dst] = old;
            }
            Step::Load { dst, word } => {
                s.threads[t].regs[*dst] = s.words[word.eval(&regs) as usize];
            }
            Step::Store { word, val } => {
                s.words[word.eval(&regs) as usize] = val.eval(&regs);
            }
            Step::WaitWhile {
                word,
                val,
                or_cancel,
            } => {
                let w = word.eval(&regs);
                let cancelled = *or_cancel && s.words[CANCEL as usize] != 0;
                if s.words[w as usize] == val.eval(&regs) && !cancelled {
                    s.threads[t].waits_on = Some(w);
                }
            }
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
            Step::Enter => {
                s.in_lock += 1;
                if s.in_lock > 1 {
                    return Err("two threads hold the Mutex at the same time".to_string());
                }
            }
            Step::Leave => s.in_lock -= 1,
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
        if how == End::Received {
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
        Ok(())
    }

    /// Visit every interleaving. Answers the count of states.
    fn check(&self) -> Result<usize, Found> {
        let mut seen = HashSet::new();
        let start = self.start();
        seen.insert(start.clone());
        let mut stack = vec![start];
        while let Some(s) = stack.pop() {
            let mut moved = false;
            for t in 0..s.threads.len() {
                let mut next = s.clone();
                match self.step(&mut next, t) {
                    Ok(false) => continue,
                    Ok(true) => moved = true,
                    Err(what) => return Err(found(what, &s)),
                }
                if seen.insert(next.clone()) {
                    stack.push(next);
                }
            }
            if moved {
                continue;
            }
            let fail = |what: String| found(what, &s);
            if s.threads.iter().any(|t| t.waits_on.is_some()) {
                return Err(fail(
                    "each thread that is not done waits, and nothing can wake it".to_string(),
                ));
            }
            self.check_end(&s).map_err(fail)?;
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
    let mut programs: Vec<Vec<Step>> = (1..=senders).map(|id| sender(ch, id, count, b)).collect();
    programs.push(receiver(ch, limit, b));
    let mut words = vec![0; (CANCEL + 1) as usize];
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
    let mut programs = vec![sender(chans[0], 1, count, b), sender(chans[1], 2, count, b)];
    programs.push(selector(chans, b));
    let mut words = vec![0; (CANCEL + 1) as usize];
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
    }
}

fn mutex_model(threads: usize, count: i64, b: Breaks) -> Model {
    Model {
        programs: (0..threads).map(|_| locker(count, b)).collect(),
        senders: Vec::new(),
        allowed: Vec::new(),
        first_words: vec![0; 3],
        ring: 0,
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

const NONE: Breaks = Breaks {
    vyukov_numbering: false,
    no_select_event: false,
    senders_after_slot: false,
    unlock_wakes_none: false,
    wait_ignores_cancel: false,
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

/// Not fixed yet. A `send` that has its place waits for room, and a cancel does not stop it. When
/// the code of the block holds the `Receiver` and leaves the block, it joins the sender first and
/// drops the `Receiver` after. Then neither can move. This test passes while the hang is there.
#[test]
fn a_send_that_has_its_place_still_hangs_when_the_block_with_the_receiver_leaves() {
    let ch = Chan {
        base: 0,
        ring: 0,
        cap: 1,
    };
    let mut words = vec![0; (CANCEL + 1) as usize];
    words[SENDERS as usize] = 1;
    words[RECEIVER as usize] = 1;
    words[ENDS as usize] = 2;
    let model = Model {
        programs: vec![
            sender(ch, 1, 2, NONE),
            leaving_block_with_the_receiver(ch, 0, NONE),
        ],
        senders: vec![(1, 2)],
        allowed: vec![End::Received, End::GivenBack, End::Dropped],
        first_words: words,
        ring: 1,
    };
    finds(
        "a send that has its place, and a block that leaves",
        model,
        "nothing can wake it",
    );
}
