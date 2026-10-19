# A plan for formal proofs of tasks, channels and errors

Status: proposed. Tracking: #1870, #1850. Designs: `cpu_tasks_send_sync_and_shared_atomics.md`
(T1 to T8, S1 to S6, U1 to U4) and `waiting_tasks_channels_and_task_errors.md` (W1 to W9, M1 to
M7, F1 to F9). Model: `formal/memory_model/lean/`.

This note plans the proofs. It contains no proofs.

## What the model proves today

- **DRF-SC** (`Program.drf_sc`, `Prog.drf_sc`): if no interleaving of a program has a data race,
  each execution that the model allows is an interleaving. It needs `sc` atomics and no fences.
- **Fork and join** (`Tasks.lean`): the code before a `spawn` happens before the task
  (`fork_hb`). The task happens before the code after its `join` (`join_hb`).

The gaps:

- No proof gives race freedom. `drf_sc` takes it as the hypothesis `hP`.
- "Interleaving" means that `ord` has no cycle. No machine with a global state exists, so
  properties of library code, such as FIFO order or mutual exclusion, are hard to state.
- Each place is a constant. So the model has no arrays, no ring buffers and no `spawn` in a loop.
- A task cannot start a task.
- An execution is a finite prefix, so no progress property (T7, W7) can be stated.

## The main idea: two layers

1. **The weak memory layer** exists. It has RC11, DRF-SC, and the edges from release and acquire.
1. **The interleaving layer** is new. It is a machine with one memory and the code of each
   thread. The proofs of the library (channels, `Mutex`, cancellation) use this layer.

DRF-SC moves a property of the second layer to each consistent execution. This holds only for
library code that uses `sc` atomics and has no race. Library code with release and acquire
needs a proof on the weak layer for each weaker access, as `hb_of_signal` does. Each such
access also gets a herd7 litmus test.

## Phase 0: what all the other phases need

- **Adequacy.** A finite execution of a `Prog` with no `ord` cycle is a run of the interleaving
  machine. A run of the machine also gives such an execution.
- **Computed addresses.** A load and a store whose address is an `Expr`. The three properties of
  `Program` must still hold, also receptiveness. Ring buffers, arrays and a `spawn` in a loop need
  them.

## Phase 1: race freedom from the borrow rules (C3, T5, U4, S5 without types)

Each task gets a static footprint: the places that it reads, the places that it writes, and the
places that it uses only through atomics. The rule:

- The writes of a task do not overlap the accesses of its sibling tasks.
- They also do not overlap the accesses of the parent between the `spawn` and the `join`.
- Two tasks share a place only through `sc` atomics.

**Theorem.** If the footprints follow the rule, each SC execution has no race. This gives `hP`,
so DRF-SC becomes a property of the code. It is the core of S6, and it gives the most value for
the work.

## Phase 2: complete structured tasks

The README lists these limits:

- Tasks in a tree: a task can start tasks.
- A `spawn` in a loop starts a new task on each pass, with a new task id.
- The joins at the end of a `tasks` block are a lowering in the model. Today the model takes the
  program after the compiler adds them.
- Captured values (E1) follow from `fork_hb`.

**Theorem `tasks_block_hb`.** Each access of each task that a block started, at any depth,
happens before each event after the block. F7 (an early exit cancels and then joins) comes when
the model has `break`.

## Phase 3: channels (M1, M2, M4)

The channel is code of the core language, in steps:

1. One slot, one sender.
1. A ring of `n` slots.
1. Many senders, with a counter for `clone` and `drop` (`fetchAdd`).
1. `Closed`.

The theorems:

- **FIFO.** Each `recv` gets the value of one `send`, in order. No value is lost, and no value
  arrives twice.
- **The three edges of M4.**
- **`Closed`** comes only after the program dropped each `Sender`, and only when the channel is
  empty.

Each edge gets a litmus test. `run_lean.py` checks that Lean and herd7 agree on it.

## Phase 4: `Mutex` (M7)

- Two tasks never hold the lock at the same time.
- Each `unlock` happens before the next `lock`.

## Phase 5: errors and cancellation (F1 to F4, T2)

These proofs use the interleaving machine.

- If a task returns `Err`, the block returns `Err`. No error is lost.
- The `Cancelled` of a sibling is never the value of the block when a real error exists.
- The cancel mark never goes away.

## Phase 6: progress (W6, W7, M3, F9)

The current model cannot say "finishes". This phase needs a new model: a scheduler over tasks
that are ready, parked or done. The code between two wait points is one step that ends.

- The scheduler on one thread (W6) stops only when each task is done or each task is parked.
- **F9 is sound.** The runtime reports a deadlock only when no scheduler can move a task.
- **W7.** With a FIFO queue of ready tasks, no ready task waits forever.
- **M3.** A branch that is ready at `k` `select` calls in a row, for `k` branches, is chosen in
  one of them.
- Research, optional: Kahn determinism. If each channel has one sender, the result and a deadlock
  do not depend on the schedule.

## Phase 7: a typed layer

The language gets references, `&` and `&mut`, structs, raw pointers, `SharedCell`, `Send` and
`Sync` (S1 to S4), M5 (no reference in a channel) and T3 (a `Task<T>` stays in its block).

**Theorem.** A well-typed program follows the footprint rule of Phase 1. This is a syntactic
proof, not a semantic model as in RustBelt. It is the largest phase.

## What Lean does not cover

These are facts of the runtime. Tests or a model checker check them:

- W3: the stacks and the guard pages.
- W8: the monitor thread.
- W9: a task stays on its thread.
- F5: io_uring.
- A dispatch runtime that is safe from many threads.

A Lean proof is about the Lean copy of the channel, not about `stdlib`. Litmus tests can link
the two. GenMC checks LLVM code against RC11, so it may check the compiled library. Nobody
tried it on the LLVM code of Vx yet.

## Order and size

The sizes are rough guesses.

| Step | Size |
| --- | --- |
| Adequacy (Phase 0) | M |
| Footprints for today's `TaskProg` (Phase 1) | M to L |
| Computed addresses and Phase 2 | M |
| Channels (Phase 3) | L |
| `Mutex` (Phase 4) | S |
| Cancellation (Phase 5) | M |
| Progress (Phase 6, new model) | L |
| Types (Phase 7) | XL |

For each new theorem, break each hypothesis one at a time, and record which proof fails. The
commits of the current model do this.

## Questions for the waiting-tasks note

A proof cannot state these claims as the note writes them:

1. **M3.** "A new index each time" gives a bound only for a round-robin start. A random start
   gives only a probability.
1. **F3.** "The first error" needs a definition in a weak memory model. A proposal: the error
   whose `cas` on the cancel flag succeeded.
1. **M4.** With many senders, "the `k`-th `send`" needs an order from the channel, for example the
   order of the `cas` on the tail.
1. **M4.** Each `Sender` drop must happen before a `recv` that returns `Closed`, not only the last
   drop. This needs an `acq_rel` counter and the release sequence of the model. It needs its own
   test.
1. **W7 and T7** are promises of progress. They wait for the model of Phase 6.
