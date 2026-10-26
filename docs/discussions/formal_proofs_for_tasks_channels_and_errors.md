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

Step 1 is two PRs in a stack.

Step 1a:

- A task can start tasks, and `TaskProg.Ok` allows `spawn` in a task.
- `spawnAt` and `joinAt` compute the task id. A `spawn` in a loop takes its task id from a
  counter. Its flags are at `start id` and `done id`. `joinAt` lowers to `waitAt`.
- A task handle cannot go into an array or a struct yet. So a join in a loop is in the same
  pass as its `spawn`, and needs no array of handles.
- The theorems: `fork_hb` and `join_hb` for a task at any depth of the tree, and `drf_sc`.

Step 1b:

- `Stmt.tasksBlock`. Each `spawn` sets a register, the mark of its task. The block clears the
  marks at its start. At its end, it joins each task that has its mark set (T2).
- The marks are registers, not places in memory. The model has no initial values and no total
  `co`, so a thread can read an old write of its own to a place.
- `block_hb` (`Blocks.lean`): each access of each task that a block started happens before the
  code after the block. `block_hb_nested` allows a body that holds other `tasks` blocks, when each
  inner block starts only tasks that the blocks around it do not start (#2243).
- `TaskProg.Ok` allows `tasksBlock`, and `fork_hb`, `join_hb` and `drf_sc` cover its lowering. So
  `addJoins_ok` is not a separate theorem.

The model has no captured values. When a later step adds them, it must follow T4: a task copies
each `Copy` variable that it only reads, at its `spawn`. It borrows a variable in place only when
it writes it, or when the variable is not `Copy`.

### Step 2: channels, by lowering (`Channel.lean`)

`send` and `recv` lower to `sc` atomics. Each `send` gets its number from a counter, with a
`fetchAdd`. `Counter.lean` shows that no two `fetchAdd`s get the same number. That needs the rule
of RC11 that `co` orders the writes to each place. `Consistent` does not have that rule, so a
theorem that needs it takes `CoOrder` as a hypothesis. Step 5 needs it too: without it, two CAS
operations on the failure slot can both succeed.

The theorems:

- `send_hb_recv`: a `send` happens before the `recv` that gets its value.
- `recv_hb_send_capacity`: for a channel with room for `n` values, the `k`-th `recv` happens
  before the `(k + n)`-th `send` ends. The counter of the channel gives the order. A `send` ends at
  its last event.
- `drop_hb_closed`: each `Sender` drop happens before a `recv` that returns `Closed`. This is
  stronger than "the last drop" in M4. The counter of senders is a `fetchAdd`, so the release
  sequence gives it.
- `ChanProg.drf_sc`.

`Channel.lean` has the four theorems. A drop adds 1 to a third counter of the channel, and a
`recv` on a closed channel gets no value. Each channel carries at most `msgs` values in a run, so
the number of places stays finite. The number of senders of a channel is fixed, so a `Sender`
cannot be cloned yet.

### Step 3: `Mutex`, by lowering (`Mutex.lean`)

- `unlock_hb_lock`: an `unlock` happens before the next `lock`.
- `MutexProg.drf_sc`.

The lock went into `Channel.lean`, so one program can use channels and locks, and
`ChanProg.drf_sc` covers both. The lock is a ticket lock. A `lock` takes ticket `k` from the
counter `next`, then reads the counter `serving` until it is `k`. An `unlock` adds 1 to `serving`.
`unlock_hb_lock` shows that the `unlock` that moved `serving` past `j` happens before each read of
`serving` that sees a larger value. `MutexExample.lean` shows a run where the lock orders a plain
write and a plain read of two tasks.

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

Steps 2 and 3 already lower the channel and the lock to `sc` atomics in the core language. So the
edges of steps 2 and 3 are already theorems about real code, and step 4 has no separate copy of
the library. `ChanProg.drf_sc` is the DRF-SC theorem. Step 4 adds the two theorems about values:

- `lock_excl` (`Lock.lean`). A lock is now a block, `locked m s`, so each `unlock` has its
  `lock`. `unlock_ticket` shows that the `unlock` of the ticket `k` reads `serving` as `k`. Then
  each event of the block of ticket `k` happens before each read of `serving` that sees more than
  `k`. The `lock` of a later ticket ends with such a read.
- `ring_fifo` (`Fifo.lean`). Only a `send` writes an element of the values, and only a `recv`
  reads one. Each access of element `k` comes after the `fetchAdd` that gave its thread the number
  `k`. So the `recv` with the number `k` gets a value that the `send` with the number `k` wrote.
  `ring_once` shows that two reads of the same write belong to the same `recv`.
  `ring_move_once` shows that each operation accesses its element once.

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
- `each_error_once` (F3a, #1937): the error whose CAS succeeded moves into the value of the block,
  and the block drops each other error. The block returns or drops each error exactly once, and
  `join()` never gives one to the code of the block. The ownership model of #1937
  (`TaskErrorOwnership.lean`) checks the same property, with `join` giving the caller nothing.

`Cancel.lean` has the first three theorems. `fail v`, `checkCancel` and `result r` are
statements of `Channel.lean`, so a program with failures can also use channels and locks. A
program has one failure slot and one cancel flag, after the flags of the tasks. The writes at the
start set both to 0. A task that sees the flag fails with `cancelled`, so its CAS shows what a
cancelled task would write. `one_failure_wins` says that at most one CAS writes the slot.
`block_value_is_failure` takes the edge from the winning CAS to the read of the slot as a
hypothesis: the joins of the block give it. `block_value_from_fail` shows that a value other than 0
comes from the CAS of a failure. `ChanProg.drf_sc` now asks for code with no failures, because
the flag uses release and acquire.

`each_error_once` follows each error by the CAS of its failure. The error of at most one failure
moves into the slot. A CAS that read the slot set does not write it, so its failure drops its
error. Other than a CAS, only the code of the block reads the slot. A task has no `result`, so
`join` gives no error to its caller. A value other than 0 that the block reads is the error that
moved.

Sabotage, one at a time: set the flag before the slot, and make the write of the flag relaxed.
Each one must break `block_value_not_cancelled`. Let `join()` give the error to its caller: that
must break `each_error_once`. It breaks `fclean_lower`, which `each_error_once` uses for the code
of a task. Nested `tasks` blocks come later.

### Step 6: race freedom from footprints (C3, T5, U4, S5 without types)

Each task gets a static footprint: the places that it reads, the places that it writes, and the
places that it uses only through atomics. The rule:

- The writes of a task do not overlap the accesses of its sibling tasks.
- They also do not overlap the accesses of the parent between the `spawn` and the `join`.
- Two tasks share a place only through a channel, the lock or `sc` atomics.

**Theorem.** If the footprints follow the rule, each SC execution has no race. This gives `hP`,
so DRF-SC becomes a property of the code. This is the core of S6. It comes after steps 2 to 4,
because tasks then share places only through the channel and the lock.

#2107 has the design. The rule applies to each `tasks` block, so the proof needs only the edges of
`fork_hb`, `join_hb` and `block_hb`. `Footprint.lean` has the first part, for `TaskProg`. The main
code is a list of phases, and each phase is code that starts no task or a block. `raceFree` shows
that each execution of a program that follows the rule has no data race. Tasks do not start
tasks yet. `ChanFootprint.lean` has the second part, for programs with channels. The places of the
channels and the locks never race (`chan_no_race` in `Recv.lean`), so the rule speaks only of the
places of the code. `LockFootprint.lean` has the third part: a place that the rule gives a lock
is accessed only under that lock, and `lock_excl` orders two such accesses. The fourth part
covers failures. Each access to the failure slot and the cancel flag is atomic (`fail_atomic` in
`Cancel.lean`), so a program with failures that follows the rule has no data race. But DRF-SC
does not hold for it. The cancel flag uses release and acquire, so a task can miss a failure that
an interleaving would show it (`cancel_flag_read_before_the_failure.litmus`). So
`drf_sc_of_footprints` still asks for code with no failures. The fifth part lets a task start
tasks. The code of each task is a list of phases too, and the rule counts the accesses of the
descendants of a task as its own (`Desc`). One thread starts each task, and a task starts only
tasks with larger ids. The proof finds where two threads are in the tree of tasks: one starts the
other at some depth (`anc`), or two tasks of one block start them (`sib`). The sixth part gives
the same tree of tasks to `ChanFootprint.lean`, so a task with channels, locks or failures can
also start tasks. The seventh part makes the lock rule less strict. The main code outside its
blocks accesses a place of a lock without the lock, because no task runs then (`main_task`). The
eighth part lets a block run in a loop. Each pass joins its tasks, so the rule of the block holds
for each pass (`Runs.loop_pass`). In `Footprint.lean` the loop can also run code that starts no
task before and after the block (#2241). Two tasks that one `spawn` of the block starts need not
be apart, because one pass starts at most one of them (#2242).

## Later

- **Progress (W6, W7, M3, F9).** The current model cannot say "finishes". This needs a new model:
  a scheduler over tasks that are ready, parked or done. It also needs an interleaving machine
  with one memory, and a proof that its runs match the executions with no `ord` cycle.
  - The scheduler on one thread (W6) stops only when each task is done or each task is parked.
  - F9 is sound: the runtime reports a deadlock only when no scheduler can move a task.
  - With a FIFO queue of ready tasks, no ready task waits forever (W7).
  - M3: `select` starts its search at the index after the one that it chose last time. A branch
    that can continue at each of `k` calls in a row, for `k` branches, is chosen in one of them.
- **A typed layer.** References, `&` and `&mut`, structs, raw pointers, `SharedCell`, `Send` and
  `Sync` (S1 to S4), M5 and T3. The theorem: a well-typed program follows the footprint rule of
  step 6. It is a syntactic proof, not a semantic model as in RustBelt.
  #2244 has the design. `Typed.lean` has the first part: locals, structs, cells, atomics, and
  captures by `move`, `ref` and `mut`.
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
1. **`select` is round robin** (M3, #1898). Its bound is a theorem of the progress model.

## Open questions for the waiting-tasks note

1. **W7 and T7** are promises of progress. They wait for the progress model.
