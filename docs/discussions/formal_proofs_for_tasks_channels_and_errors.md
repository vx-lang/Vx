# A plan for formal proofs of tasks, channels and errors

Status: agreed on 2026-10-19. Tracking: #1870, #1850. Designs:
`cpu_tasks_send_sync_and_shared_atomics.md` (T1 to T8, S1 to S6, U1 to U4) and
`waiting_tasks_channels_and_task_errors.md` (W1 to W9, M1 to M7, F1 to F9). Model:
`formal/memory_model/lean/`.

This note plans the proofs and records the decisions. It contains no proofs.

## What the model proves today

- **DRF-SC** (`Program.drf_sc`, `Prog.drf_sc`): if no interleaving of a program has a data race,
  each execution that the model allows is an interleaving. It needs `sc` atomics and no fences.
- **Fork and join** (`Tasks.lean`): the code before a `spawn` happens before the task
  (`fork_hb`). The task happens before the code after its `join` (`join_hb`).

The gaps:

- No proof gives race freedom. `drf_sc` takes it as the hypothesis `hP`.
- Each place is a constant. So the model has no arrays, no ring buffers and no `spawn` in a loop.
- A task cannot start a task.
- An execution is a finite prefix, so no progress property (T7, W7) can be stated.

## How the proofs work

Each new construct is a lowering to `sc` atomics on places that the program code cannot use, as
`Tasks.lean` does for `spawn` and `join`. Then:

- DRF-SC follows from `Prog.drf_sc`, with no new proof.
- Each edge of M4 is a theorem about the lowering, on the weak memory layer (happens-before and
  coherence).
- The model gets no new kind of event, so `Rc11.lean`, `DrfSc.lean` and `Program.lean` stay the
  same.

## The steps

Each step is one PR from `main`. A step that needs the step before it goes in a stack.

### Step 0: computed addresses

`Thread.lean` gets `Stmt.loadAt` and `Stmt.storeAt`, whose address is an `Expr`. A `spawn` in a
loop and a ring of `n` slots need them. `Trace.receptive`, `Run.prefix_closed`, `Run.any_co` and
`Run.receptive` must still hold.

### Step 1: a tree of tasks (#1867)

- A task can start tasks. Each task has a parent, and `TaskProg.Ok` allows `spawn` in a task.
- A `spawn` in a loop takes its task id from a counter. Its flags are at `start id` and `done id`.
- `Blocks.lean`: `Stmt.tasksBlock`, and the lowering `addJoins`. It joins each task that the block
  did not join (T2). Today `Tasks.lean` takes the program after the compiler adds these joins.

The theorems:

- `fork_hb` and `join_hb` for a task at any depth of the tree.
- `block_hb`: each access of each task that a block started happens before the code after the
  block.
- `addJoins_ok`: the lowering keeps `TaskProg.Ok`.
- `drf_sc`.

### Step 2: channels, by lowering (`Channel.lean`)

`send` and `recv` lower to `sc` atomics. The theorems:

- `send_hb_recv`: a `send` happens before the `recv` that gets its value.
- `recv_hb_send_capacity`: for a channel with room for `n` values, the `k`-th `recv` happens
  before the `(k + n)`-th `send` ends. The counter of the channel gives the order. A `send` ends at
  its last event.
- `drop_hb_closed`: each `Sender` drop happens before a `recv` that returns `Closed`. This is
  stronger than "the last drop" in M4. The counter of senders is a `fetchAdd`, so the release
  sequence gives it.
- `ChanProg.drf_sc`.

### Step 3: `Mutex`, by lowering (`Mutex.lean`)

- `unlock_hb_lock`: an `unlock` happens before the next `lock`.
- `MutexProg.drf_sc`.

### Step 4: library code for the channel and the lock

`ChannelImpl.lean` and `LockImpl.lean` hold a bounded channel and a lock in the core language, on
`sc` atomics. This step shows that real code gives the edges that steps 2 and 3 define. #1864 asks
for this check. The theorems:

- `ring_send_hb_recv`, `ring_recv_hb_send` and `ring_drop_hb_closed`: the edges of step 2.
- `ring_fifo`: each `recv` gets the value of one `send`, in order. No value is lost, and no value
  arrives twice.
- `lock_excl`: two tasks never hold the lock at the same time.
- `lock_unlock_hb_lock`: the edge of step 3.
- DRF-SC for a program that uses this library.

Each edge also gets a herd7 litmus test, and `run_lean.py` checks that Lean and herd7 agree.

DRF-SC for a client holds only with this `sc` library. A later step can make an access release or
acquire. Each such access then needs its own edge proof and a litmus test.

### Step 5: cancellation, safety only (`Cancel.lean`, F3)

The model:

- A failure slot. A failure writes it by a CAS from 0.
- These write the slot by the same CAS: a task that returns `Err`, a `?` in the code of the block
  (F2), and an `Err` that is the last value of the block.
- The cancel flag is a release write, after the write of the slot in program order.
- Each wait that can be cancelled reads the flag with acquire.

The theorems:

- `one_failure_wins`: exactly one CAS succeeds. This follows from the atomicity of a
  read-modify-write in `Consistent`.
- `block_value_is_failure`: after the joins, the block reads the value of that CAS. A task or the
  code of the block produced it.
- `block_value_not_cancelled`: a task that sees the flag also sees a slot that is not 0. So its CAS
  fails, and `Cancelled` never reaches the slot.

Sabotage, one at a time: set the flag before the slot, and make the write of the flag relaxed.
Each one must break `block_value_not_cancelled`. Nested `tasks` blocks come later.

### Step 6: race freedom from footprints (C3, T5, U4, S5 without types)

Each task gets a static footprint: the places that it reads, the places that it writes, and the
places that it uses only through atomics. The rule:

- The writes of a task do not overlap the accesses of its sibling tasks.
- They also do not overlap the accesses of the parent between the `spawn` and the `join`.
- Two tasks share a place only through a channel, the lock or `sc` atomics.

**Theorem.** If the footprints follow the rule, each SC execution has no race. This gives `hP`,
so DRF-SC becomes a property of the code. This is the core of S6. It comes after steps 2 to 4,
because tasks then share places only through the channel and the lock.

## Later

- **Progress (W6, W7, M3, F9).** The current model cannot say "finishes". This needs a new model:
  a scheduler over tasks that are ready, parked or done. It also needs an interleaving machine
  with one memory, and a proof that its runs match the executions with no `ord` cycle.
  - The scheduler on one thread (W6) stops only when each task is done or each task is parked.
  - F9 is sound: the runtime reports a deadlock only when no scheduler can move a task.
  - With a FIFO queue of ready tasks, no ready task waits forever (W7).
  - A `select` branch that is ready at `k` calls in a row, for `k` branches, is chosen in one of
    them (M3).
- **A typed layer.** References, `&` and `&mut`, structs, raw pointers, `SharedCell`, `Send` and
  `Sync` (S1 to S4), M5 and T3. The theorem: a well-typed program follows the footprint rule of
  step 6. It is a syntactic proof, not a semantic model as in RustBelt.
- **Release and acquire in the library**, one access at a time (step 4).

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

## Rules for each proof

- No `sorry`, and no axiom that the model does not already have.
- Break the model on purpose before you trust a theorem, for example make a release relaxed.
  Break each theorem separately, and record which proof fails.

## Decisions

We decided these on 2026-10-19.

1. **The order.** Steps 0 to 6 as above. Progress and the typed layer come later.
1. **Channels and the lock are lowerings**, not new kinds of event. So the existing DRF-SC proof
   covers them.
1. **The library uses `sc` atomics first.** Release and acquire come one access at a time, each
   with an edge proof and a litmus test.
1. **`drop_hb_closed` covers each `Sender` drop**, not only the last one.
1. **The first error is the failure whose CAS on the slot succeeded** (F3).
1. **The interleaving machine waits** until the progress model needs it. Steps 0 to 5 use the
   weak memory layer.

## Open questions for the waiting-tasks note

1. **M3.** "A new index each time" gives a bound only for a round-robin start. A random start
   gives only a probability.
1. **W7 and T7** are promises of progress. They wait for the progress model.
