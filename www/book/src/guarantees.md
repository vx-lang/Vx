# What Vx guarantees about concurrency

This page lists what you can rely on when tasks run at the same time, and when work runs on other
machines. For each promise, it says where the promise stops, and how we check it.

Most of the checks are proofs or model checks of a *model*: a small version of the language, the
library or the runtime. A model leaves out details, so a bug in the compiler or in the runtime can
still break a promise. Tests and checks of the real compiled code make that gap smaller, and this
page says which promise has them.

The models and their proofs are in
[`formal/`](https://github.com/vx-lang/Vx/tree/main/formal). Issue
[#2170](https://github.com/vx-lang/Vx/issues/2170) tracks the work.

## No data races

**A program that compiles has no data race outside `unsafe` code.** A data race is two accesses
to one place at the same time, where at least one is a write and not both are atomic.

- Code in `unsafe` and each `unsafe impl Send` or `unsafe impl Sync` make claims that the compiler
  cannot check. The promise assumes that those claims are true. To find a race in such code, run
  the program with `--sanitize=thread` (see [Tasks and channels](tasks.md)).
- **The check:** a proof in Lean, on a small typed language with tasks, references, raw pointers,
  channels and `Mutex`. The proof shows that each program that its checker accepts has no data
  race (`Typed.lean`, `TypedChan.lean`). The checker is a model of the checker in `vxc`.

## A program with no data race runs as if its tasks took turns

**If a program has no data race, each run of it gives a result that some order of its steps,
one task at a time, also gives.** This is *DRF-SC*. Such a program cannot see how the hardware
reorders memory accesses.

- This holds when each atomic uses `SeqCst` and the program has no fences. With weaker
  [memory orders](tasks.md#memory-orders), the result follows the RC11 model, which C++ and Rust
  also use.
- **The checks:**
  - A proof in Lean, for executions of any size (`Program.drf_sc`).
  - The model in herd7, run on one small test for each case of the rules. Lean and herd7 agree on
    these tests.
  - A comparison with RC11 and with AArch64 in memalloy, for executions of up to 6 events.

## `Mutex`, channels and `Arc` order what they must

**After a `lock`, a task sees each write that the last holder of the lock made. After a `recv`,
a task sees each write made before the `send` of that value. The last drop of an `Arc` sees each
write made through the other holders.**

- **The checks:**
  - A proof in Lean for each such pair of accesses, with a small test for herd7.
  - GenMC runs the LLVM IR that `vxc` generates for three small programs in every execution that
    RC11 allows. It reports a data race, a failed assertion, a use of freed memory, and a wait
    that never ends. This checks the compiled code, not a model of it.
- GenMC does not check a cancel or a time limit.

## Tasks make progress

**A task that is ready to run does not wait forever.** The runtime that runs many tasks on a few
OS threads (`VX_TASK_RUNTIME=shared`) keeps its ready tasks in one first-in, first-out queue.

- This assumes that a worker thread takes a task within some time while the queue is not empty.
- **The check:** a proof in Lean (`Scheduler.lean`). It also shows that a queue where a task
  goes to the front can make a task wait forever.

**A task finishes if each of its waits can continue at some point.**

- The proof covers code with no loops, on a model of a CPU (`Finish.lean`).

**`select` does not starve a channel.** Take `select()` on `n` channels. If one channel has a
value at each of `n` calls in a row, one of those calls takes a value from it.

- **The check:** a proof in Lean (`Select.lean`). It also shows that the rule that `select` used
  before could starve a channel.

## The runtime reports each deadlock, and only a deadlock

**When each task waits and nothing can wake one, the runtime stops the program with
`vx: deadlock: ...`. It never stops a program that can still continue.**

- **The checks:**
  - A proof in Lean that a deadlock stays a deadlock, so that a report is never too early
    (`CpuDeadlock.lean`, `Deadlock.lean`).
  - TLC, a model checker for TLA+, on a model of the waits and wakes of the runtime. It includes
    the timer, the io thread and the thread that watches for a deadlock (`WaitWake.tla`). TLC
    checks every order of the steps for a few tasks and threads.

## Work on other machines

These promises are about a host program that sends work to workers on other machines.

**Two host programs that use the same workers do not wait for each other forever.** A
worker serves its hosts in turn, one request at a time.

**A host stops with an error when it loses a worker.** It does not wait forever.

- A worker process that stops closes its connections, and the host sees that at once.
- A lost machine or a lost link closes nothing. So the host checks the link, and stops after 30
  seconds with no answer. `VX_REMOTE_LINK_TIMEOUT` sets this time in seconds. A long kernel on
  the worker does not trip this check, because the machine of the worker still answers.
- A worker drops a host that sends part of a request and then stops, or that stops reading a
  reply. It then serves its other hosts. `vx-worker --io-timeout` sets this time (30 seconds by
  default).

**The ranks of a job join for the collectives, or stop with an error.** No rank reads half of the
id that rank 0 writes, and an old id file stops rank 0.

Where these promises stop:

- A worker that is alive but stuck still answers the link check, so its host waits.
- An error stops the whole program. It is not yet an error that the program can handle.

**The checks:** TLC on models of the protocol between hosts and workers (`Agent.tla`) and of
the join (`Rendezvous.tla`). Each model also keeps a list of past or likely mistakes. The check
puts each mistake into the model and makes sure that TLC finds it. Tests run real workers on one
machine.

## Atomics on a GPU

**The memory orders of atomics mean the same on a GPU.** Every atomic in `core::sync::atomic`
has system scope.

- **The checks:** a model of the scopes of atomics and fences, run in herd7. Small programs in Vx
  run the same tests on a GPU, to check the code that `vxc` makes for it.

## Checked by tests only

These facts of the runtime have tests, but no proof or model:

- The stack of each task, and the guard page that stops a stack overflow.
- A foreign call that blocks its OS thread does not stop the other tasks. A monitor thread then
  starts another OS thread for the ready tasks.
- A task can continue on another OS thread after a wait. The state that belongs to one OS thread,
  such as the current CUDA device, must still be correct for the task.
- A cancelled wait for a file or a socket on io_uring returns only after the kernel finishes with
  its buffer.
