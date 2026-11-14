# One `select` for channels, sockets, timers and tokens

Issue #3414, part of #3413. This is a design for review. No code comes before the review.

## 1. The problem

`Selector<T>` (`stdlib/std/sync/channel.vx`) waits on channels of one element type. A server must
often wait for the first of several kinds of events:

- a value on a channel, of any element type
- a socket that is ready to read, or a new connection on a listener
- a timer or a deadline
- a `transfer_async` token that is done
- the new sources of #3415 to #3418: a buffered socket, a oneshot, broadcast, watch or MPMC
  channel, an `Interval`, and a semaphore

Today a task can wait for only one of these at a time. This is what the user code should look like:

<!-- vx-doctest: skip (proposed API) -->

```vx
// A server: take jobs from a channel, take new connections, and flush the log once a second.
let mut tick = Interval::every(Duration::from_secs(1));
let mut jobs_open = true;
loop {
  match select3(&when(jobs_open, &jobs), &listener, &tick) {
    Result<i32, SelectError>::Ok(0) => {
      match jobs.try_recv() {
        Result<Job, TryRecvError>::Ok(job) => { run(job); }
        Result<Job, TryRecvError>::Err(_) => { jobs_open = false; }
      }
    }
    Result<i32, SelectError>::Ok(1) => { serve(listener.accept()?); }
    Result<i32, SelectError>::Ok(_) => { tick.next(); flush_log(); }
    Result<i32, SelectError>::Err(_) => { break; } // the `tasks` block was cancelled
  }
}
```

A wait for "a value, or give up after 2 seconds of silence" uses a `Deadline` as a source:

<!-- vx-doctest: skip (proposed API) -->

```vx
let quiet = Deadline::after(Duration::from_secs(2));
match select2(&replies, &quiet) { ... }
```

## 2. How waits work today

Every wait in Vx uses one call: "sleep while this atomic word holds this value"
(`vx_wait_while_until` in `stdlib/rust_core/src/ffi/wait.rs`). The runtime hashes the word to one
of 64 buckets. On the default runtime (one OS thread for each task), the thread sleeps on the
condition variable of the bucket. On the shared runtime (`VX_TASK_RUNTIME=shared`), the task
parks: it goes on the list of the bucket with the address of its word, and the worker runs another
task.

- **Channels.** Each channel has a `WAKE` word. Each event adds 1 to it and wakes the sleepers.
  `notify` also adds 1 to one global word, `vx_any_channel_words()`, and `Selector::select` sleeps
  on that word. So each event on any channel of the program wakes each `select`.
- **Sockets.** A socket wait puts `(fd, events, address of a word)` on the list of one poller thread
  (`ffi/io.rs`). The poller calls `poll(2)`. When the socket is ready, it sets the word under the
  lock of the list, and wakes it. A cancelled caller removes its entry from the list, so the poller
  never writes to a word after its wait returned.
- **Time.** `sleep` waits on a word that nothing sets, with a deadline. A parked task asks the
  `vx-timer` thread to wake the bucket of its word at the deadline.
- **Cancel and `within`.** A cancel wakes each wait that can be cancelled. In `within` or a `drop`,
  each wait that can be cancelled ends at the earliest deadline around it (F6a).
- **Tokens.** `transfer_async` gives a `Token<T>`, a type that the compiler knows. Today the copy
  ends before `transfer_async` returns, so a token is always done.
- **The deadlock report.** A wait with no time limit leaves a note in its bucket. A program in
  which each thread and task sleeps with no limit, with no timer and no socket pending, stops with
  a report.

Two facts shape the design:

1. A fiber can be on the list of only one bucket. `sched::wake` panics on a fiber that a second
   wake finds `WOKEN`. So a task must park on **one** word, not on one word for each source.
1. The poller and the timer already wake any word that a caller gives them. So a socket and a
   deadline can wake a word that the `select` owns. The poller needs one small change: it must add
   1 to the word, not store 1. A `select` waits many times on one word, and a second store of 1
   does not change the word, so the wake would be lost.

## 3. Other languages

| | Rust tokio `select!` | Go `select` | Kotlin `select` | OCaml 5 Eio `Fiber.first` |
|---|---|---|---|---|
| Form | macro over futures | statement | builder DSL | function over closures |
| Colored functions | yes, `async fn` | no | yes, `suspend fun` | no |
| The losing branches | dropped. A future that did part of its work loses data ("cancel safety") | the op never started | the op never started | the fiber gets a cancel and must clean up |
| Order of the search | random, or `biased;` | random | first clause, or `selectUnbiased` | first to finish |
| Time limit | `sleep` branch | `time.After` channel | `onTimeout` | `Fiber.first` with a sleep |
| Deadlock report | no | only "all goroutines are asleep" | no | no |

Java has no `select` over `BlockingQueue`s. NIO's `Selector` waits only on sockets.

Vx can be strictly better than each of these:

- **No colors.** `select3` is a normal function. A function that calls it looks like any other
  function (W1).
- **A losing branch loses nothing.** The `select` waits only until a source is ready. It starts
  no operation, so a losing branch has nothing to undo. Tokio's `select!` can drop a half-done
  `read_line`. Vx cannot.
- **Cancel and `within` need no branch.** The cancel of the `tasks` block and the limit of `within`
  end the `select`, as they end each other wait.
- **A deadlock report names each source.** The report says "waits in `select` at a.vx:12 on the
  channel made at a.vx:3 and the channel made at a.vx:4". Go reports no lines.
- **Fairness with a bound.** Go and tokio pick at random. Vx searches round robin, so a source that
  stays ready is picked within `n` calls of a loop with one `select` of `n` sources.

## 4. The proposed API

A library, not syntax. The language needs no new feature.

<!-- vx-doctest: skip (proposed API) -->

```vx
// std::sync::select

/// Something that a task can wait on in `select`.
trait WaitSource {
  /// True when the operation of the source can continue now with no wait. It must not wait, and it
  /// must not change the source.
  fn is_ready(self : &Self) -> bool;
  /// Wake `w` at each later change that can make the source ready.
  fn watch(self : &Self, w : &Waiter) -> void;
  /// Undo `watch`. After it returns, the source does not touch `w` again.
  fn unwatch(self : &Self, w : &Waiter) -> void;
  /// The time at which the source gets ready with no event, or `Instant::never()`.
  fn deadline(self : &Self) -> Instant {
    return Instant::never();
  }
}

enum SelectError {
  // The `tasks` block of this task was cancelled, or the limit of `within` ran out.
  Cancelled,
}

fn select2<A : WaitSource, B : WaitSource>(a : &A, b : &B, at : Caller) -> Result<i32, SelectError>;
// select3 to select6 in the same way.
fn select_list<A : WaitSource>(sources : &Vec<A>, at : Caller) -> Result<i32, SelectError>;

/// A source that is never ready when `active` is false.
fn when<A : WaitSource>(active : bool, source : &A) -> When<A>;

/// A source that gets ready at a time.
struct Deadline { at : Instant }
```

The sources in `std`: `Receiver<T>` (ready when a value is in the channel, or the channel is
closed), `TcpListener`, `TcpStream` and `UdpSocket` (ready to read), and `Deadline`. The sources of
#3415 to #3418 implement the same trait. `Interval` gives its next tick in `deadline`.

**The result is an index, not a value.** The caller then does the operation: `try_recv`, `accept`,
`tick.next()`. This is the readiness model of `poll(2)`. It is the reason that a library is
enough:

- The sources have different types, and Vx has no `dyn`. Generic functions with a fixed count of
  sources make one copy for each set of types. They need no dynamic dispatch.
- The code of each branch is a `match` arm of the caller. Two arms can change the same local, and
  the borrow checker sees a plain `match`. A closure for each branch would fail there (M3).
- No branch commits an operation, so a losing branch needs no undo.

**Most sources stay ready.** A `Receiver` has one owner, so after `select` says it is ready,
`try_recv` gives a value or `Closed`. The same holds for a socket that one task owns, a timer, a
token, and a broadcast or watch receiver. An MPMC receiver and a semaphore are shared: another task
can take the value between `select` and `try_recv`. Then `try_recv` gives `Empty`, and the loop
calls `select` again. The doc of each shared source must say this.

**A closed channel is ready.** `try_recv` then gives `Closed`, as in Go. A loop disables that branch
with `when`, as the first example does. A `select` in which no source can get ready waits for
a cancel. If nothing else can wake the program, the deadlock report stops it.

**Priority.** A caller who wants one source first checks it first:
`if urgent.is_ready() { ... } else { select2(...) }`. Vx needs no biased variant.

`Selector<T>` stays. It is `select_list` on its receivers, and then `try_recv` on the one picked.

## 5. The runtime mechanism

**The waiter.** `Waiter` is one `AtomicI64` in the frame of the `select`. A source wakes it: it
adds 1 to the word, and calls `wake_all` on it. The `select` waits on that word with
`wait_while_until`. So the task parks once, on one word, in one bucket, on both runtimes.

**The watch list.** The Rust core keeps a table from a source key to the waiters that watch it.
The key is the address of a word that the source owns, such as the words of a channel. It has three
calls:

- `vx_watch(key, waiter)` and `vx_unwatch(key, waiter)`, under the lock of a shard of the table
- `vx_wake_watchers(key)`: under the same lock, add 1 to each waiter of the key. After the lock,
  call `wake_all` on each. `wake_all` reads only the address, so the word can be gone by then.

A waiter changes only under the lock, so after `vx_unwatch` returns, no source writes to it. This
is the rule that the poller already follows.

**How each source watches.**

| Source | `is_ready` | `watch` |
|---|---|---|
| `Receiver<T>` and the channels of #3416 | the slot of the next receive number is full, or no `Sender` is left | add 1 to a new word `WATCHERS` of the channel, then `vx_watch`. `notify` calls `vx_wake_watchers` only when `WATCHERS > 0`. |
| sockets, and #3415 | the buffer has bytes, or `poll` with a time limit of 0 says ready | put `(fd, events, waiter)` on the list of the poller, which adds 1 to the waiter (section 2) |
| `Deadline`, `Interval` | the clock passed the time | nothing: `select` waits with the earliest `deadline()` |
| semaphore (#3418) | a permit is free | as a channel |
| `Token<T>` | the copy ended | today: always ready. With a background copy: the end of the copy calls `vx_wake_watchers` |

A new source implements the four methods of the trait. The three runtime calls are its whole
interface to the runtime.

**One `select` call.**

1. Search the sources once, round robin, with `is_ready`. If one is ready, return its index. This
   path makes no runtime call and takes no lock.
1. Read the waiter word into `seen`. Call `watch` on each source.
1. Search again with `is_ready`. If one is ready, call `unwatch` on each, and return its index.
1. Wait while the word holds `seen`, until the earliest `deadline()`, and with cancel on. Then call
   `unwatch` on each, check for a cancel, and go to step 1.

Step 3 closes the race between the first search and the watch. A source that gets ready after
step 2 adds 1 to the word, and the wait of step 4 returns at once. The pair "the source changes
its state and then reads `WATCHERS`", and "the select adds 1 to `WATCHERS` and then reads the
state", is the store-buffering pattern. Both sides need `SeqCst`, as the channel needs today for
`SLEEPERS` (see the header of `channel.vx`). A herd7 test checks it.

**Two sources get ready at once.** Both add 1 to the word. The fiber is on the list of one
bucket, so the first `wake_all` removes it from the list, and the second finds nothing. The search
after the wake picks one source in round robin order. The other one stays ready for the next
call.

**Losers.** Step 4 and step 3 call `unwatch` on each source, on each exit. A losing branch started
no operation, so this is all of the clean-up. `select` does not return until each `unwatch`
returned, so no source holds the address of the frame after it.

**Fairness.** `select2` has no state of its own, so the runtime keeps one number for each task:
the index after the last choice. Each `select` starts its search there, modulo `n`. In a loop with
one `select`, this is the rule of M3, and a source that stays ready is picked within `n` calls.
The progress model of `formal_proofs_for_tasks_channels_and_errors.md` states this bound for M3.
Two `select` calls in one loop share the number, so the bound is weaker there.

**The global word goes.** `notify` no longer adds 1 to `vx_any_channel_words()`. Today each event
on any channel writes that one cache line, and wakes each `select` of the program. After the
change, an event on a channel that no `select` watches costs one more load (`WATCHERS`), and a
`select` wakes only for its own sources.

## 6. The two runtimes, cancel, `within` and the deadlock report

- **Default runtime.** Step 4 sleeps the OS thread of the task on the bucket of the waiter word.
  The poller and the watch list wake it with `wake_all`, as they wake any wait.
- **Shared runtime.** Step 4 parks the fiber on the same word. A deadline goes to the `vx-timer`
  thread, as for `sleep`. `sched.rs` needs no change.
- **Cancel.** The wait of step 4 can be cancelled. A cancel wakes it, step 4 calls `unwatch`, and
  `select` gives `Cancelled`.
- **`within` and `drop`.** `wait` in `wait.rs` already takes the earliest deadline of the scopes
  around a wait that can be cancelled. So `select` gives up at the limit of `within`, with
  `Cancelled`, as `Selector::select` does today. A time limit for one wait is a `Deadline` source,
  so `select` needs no `TimedOut` error.
- **The deadlock report.** `watch` gives the runtime the key of each source and the line that made
  it. The note of the `select` lists them: "waits in `select` at a.vx:12 on the channel made at
  a.vx:3 and on the channel made at a.vx:4". A `select` with a socket or a deadline never counts as
  stuck, because the poller or the timer is pending. The report prints no cycle through a `select`
  in the first version: a `select` is stuck only when each of its sources is stuck, and the report
  must never print a cycle that is not real (#2932).

## 7. Costs

- A ready source: `n` calls of `is_ready`, with no lock and no system call. A socket's
  `is_ready` makes one `poll` call with a time limit of 0.
- A wait: one `watch` and one `unwatch` for each source. A channel watch takes the lock of one
  shard. A socket watch takes the lock of the poller and writes one byte to its pipe. Programs
  that wait on thousands of sockets need epoll or kqueue (section 6 of
  `waiting_tasks_channels_and_task_errors.md`). That change is in the poller, not in `select`.
- Memory: one word in the frame, and one entry in a table for each watched source.
- Each event on a channel: one load of `WATCHERS` in place of the add to the global word.
- Code size: one copy of `select3` for each set of three source types that a program uses.

## 8. The parallel compiler

The design adds no language rule. `WaitSource`, `select2` to `select6` and `when` are library
code with written signatures. A new set of source types is a new generic instance, the one
exception that the rules allow (`language_rules_for_a_parallel_compiler.md`). No fact about a
signature comes from a body.

## 9. Plan

Each stage is one PR with its own tests.

1. **The core.** The watch list in `wait.rs`. `WaitSource`, `Waiter`, `select2` to `select6`,
   `select_list`, `when` and `Deadline`. `Receiver<T>` as a source. `Selector<T>` on top of it.
   Remove the global word.
1. **Sockets.** `TcpListener`, `TcpStream` and `UdpSocket` as sources, through the poller.
1. **The new sources.** The buffered socket (#3415), the channels (#3416), `Interval` (#3417) and
   the semaphore (#3418). Each one is a small PR after its parent PR and stage 1 merged.
1. **The deadlock report** lists the sources of a `select`.
1. **Tokens**, when a background copy exists (#244).

**Tests.** Each Vx fixture in `tests/backend/pass/` runs on the AST code generator, the flat
code generator, and the shared runtime with one worker, as the `select` tests do today.

- `select` picks a channel of `i64`, a channel of `String`, a listener and a deadline, in one call.
- A source that loses keeps its value: the next `try_recv` on it gets the value.
- A channel that sends at each pass does not starve a slower channel (the round robin bound).
- A cancel of the `tasks` block, and the limit of `within`, end a `select` that waits.
- A `Deadline` source ends a `select` on channels that stay empty.
- A stress test: two tasks send at the moment the `select` watches, many times. No wake is lost.
  `VX_TEST_TIMER_PAUSE_MS` holds the timer, to test a deadline that fires late.
- A `select` returns, and the task uses its frame again at once: no source writes into it. This runs
  under ASan and TSan.
- The deadlock report of a `select` on two channels names both channels.
- Rust unit tests for the watch list, and a herd7 test for the watch and the check (section 5).

Before you trust a new test, break the code under each of its checks, one at a time, and read
which check fails.

## 10. Open questions

Each question has a recommended answer.

1. **A library or syntax?** The block expression of M3 commits the operation, and can have `send`
   arms. It needs work in the checker and in both code generators. Recommended: the library of
   section 4. It needs no new feature. M3 can still come later, on the same watch list.
1. **An index, or the value?** A `select` that gives the value must do the operation, which needs
   one result type for all sources. Recommended: the index (the readiness model). Shared sources
   (MPMC, semaphore) then may lose a race, and the loop calls `select` again.
1. **How many sources in one call?** Vx has no variadic generics. Recommended: `select2` to
   `select6`, and `select_list` for many sources of one type. A program with more sources gives
   some of them to a task that sends on a channel.
1. **Fairness: round robin or random?** Recommended: round robin, from the index after the last
   choice of the task. It gives a bound, and the same order on each run with a seed. No biased
   variant: the caller checks `is_ready` on its urgent source first.
1. **A closed channel: ready, or left out?** Recommended: ready, as in Go, with `when` to disable
   the branch. A source that leaves itself out would hide the close from the caller.
1. **Remove the global word now?** Recommended: yes, in stage 1. `Selector<T>` moves to the watch
   list in the same PR, so no code needs the global word.
1. **A `Sender` as a source** (ready when the channel has room)? A channel has many `Sender`s, so
   it is a shared source. Recommended: not in stage 1. Add it with the channels of #3416.
1. **Tokens.** `Token<T>` is a type that the compiler knows, so a library `impl` for it needs a
   change in the checker, or a library struct for the token. Recommended: wait for the background
   copy of #244. Until then, a token is always ready, and a `select` on it has no use.
