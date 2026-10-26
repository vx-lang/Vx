# The Vx memory model

This directory is an executable model of the rules for code that runs at the same time: C0 to
C7 in `docs/discussions/borrow_check_sequence_points_for_calls.md`. Tracking issue: #1850.

[herd7](https://github.com/herd/herdtools7) reads a model and a small concurrent program (a
*litmus test*), and lists every result that the model allows. Each test here says which result
it expects.

| File | What it is |
| --- | --- |
| `vx.cat` | The model: RC11 (Lahav et al., PLDI 2017), with barriers and atomics of a scope. |
| `vx_events.cat` | The tags, the scopes and the barriers. `vx.cat` and `sequential.cat` share it. |
| `sequential.cat` | Sequential consistency with barriers: every run is one interleaving. |
| `vx.bell` | The tags that a test can put on an access or a fence. |
| `run_litmus.py` | Runs every test and compares each result with its `Expect:` line. |
| `run_lean.py` | Checks that the Lean model and `vx.cat` agree on the tests for CPUs. |
| `tests/` | One test for each case of the rules. |
| `memalloy/` | The CPU part of the model, compared with RC11 and with AArch64 by memalloy. |
| `lean/` | The CPU part of the model in Lean 4, with proofs for executions of any size. |

## Running the tests

```bash
opam install herdtools7        # version 7.58 or later
python3 formal/memory_model/run_litmus.py
```

The script also checks *DRF-SC*. A test without a data race, whose accesses are plain or
sequentially consistent, must give the same final states on `vx.cat` as on `sequential.cat`.
Such a program cannot see how the hardware reorders memory accesses.

CI (`.github/workflows/formal.yml`) runs this script, `run_lean.py` and the Lean build on each
pull request that changes `formal/`. memalloy needs local patches, so CI does not run it.

## Comparing with RC11 and AArch64

[memalloy](https://github.com/johnwickerson/memalloy) searches for a small execution that one
model allows and another forbids. It reads a smaller dialect of cat, so `memalloy/vx_cpu.cat`
is the CPU part of `vx.cat` in that dialect, without scopes and barriers.

```bash
MEMALLOY=~/memalloy python3 formal/memory_model/memalloy/run_memalloy.py   # about 3 minutes
```

The script checks, up to 5 or 6 events:

- `vx_cpu.cat` and RC11 allow the same executions, and find the same data races.
- The mapping of `vx_to_aarch64.als` gives no AArch64 result that `vx_cpu.cat` forbids.
- Without the dependency after a relaxed load, the mapping gives such a result. This is the price
  of the rule against values out of thin air.

To build memalloy with OCaml 4.14 and Python 3, add `let compare = Stdlib.compare` after `open List` in `src/myList.ml`, and port the `print` and `string.replace` calls in `etc/` to Python 3.
Its model of PTX is from 2013, older than the PTX model of Volta, so it cannot check a mapping to
current GPUs.

## Proofs in Lean

herd7 and memalloy check small executions. `lean/` proves these theorems for executions of any
size, on the CPU part of the model:

- `consistent_of_seqConsistent`: the model allows every interleaving of the threads.
- `seqConsistent_of_raceFree`: take an execution that the model allows. If it has no data race,
  no fences, and only sequentially consistent atomics, then it is an interleaving (DRF-SC).
- `Program.drf_sc`: DRF-SC for programs. If no interleaving of a program has a data race, then
  every execution of the program that the model allows is an interleaving.
- `Prog.drf_sc`: the same theorem for the programs of a small core language.

For `Program.drf_sc`, a program is the set of its executions, with three properties that any
language with threads has. A prefix of an execution is an execution. The coherence order is any
order. A read at the end of its thread can read any write to its place. The theorem also needs
sequentially consistent atomics and no fences. A place can have both atomic and plain accesses:
`atomic_read_of_a_plain_write.litmus` shows such a race.

The core language in `Thread.lean` and `Language.lean` has registers, loads, stores, `fetchAdd`,
`cas`, fences, `if` and loops. Each access names its memory order. `loadAt` and `storeAt` access
an element of an array at an index that the code computes. An index outside the array stops the
thread, as a failed bounds check does. A read-modify-write is a read
and then a write, so a thread can stop between them. All threads start together. `Language.lean`
shows that the executions of these programs have the three properties. `Prog.drf_sc` then needs
one rule on the code, which a type checker can check: `Prog.SeqCstOnly` (only `sc` atomics, no
fences).

`Tasks.lean` adds `spawn` and `join`, with the meaning that #1860 chose for `tasks { .. }`. A
`spawn` is an `sc` write of a flag, and the task waits for that flag before it starts. A task
sets another flag at its end, and `join` waits for it. A task can start tasks, and `spawnAt`
computes the task id, as a `spawn` in a loop does. `fork_hb` and `join_hb` show that in each run,
the code before a `spawn` comes before the task, and the task comes before the code after its
`join`, at any depth of the tree of tasks. `LanguageExample.lean` builds an execution of message
passing with release and acquire, and one that passes a message through an array. With plain
data, it also builds one where the main code starts two tasks and joins them, one where a task
starts a task, and one with a `spawn` and a `join` in each pass of a loop.

`Blocks.lean` adds `tasks { .. }`. Each `spawn` sets a register, the mark of its task. At its end,
the block joins each task that has its mark set, so it waits only for the tasks that it started.
`block_hb` shows that each task that a block started happens before the code after the block.
`BlocksExample.lean` applies `block_hb` to a run, and shows a block that starts no task and does
not wait.

`Footprint.lean` proves race freedom from a rule on the code. The code of each thread is a list
of phases: code that starts no task, or a `tasks` block. A block can also run again in a loop,
with code that starts no task before and after it. Each pass joins its tasks, so the rule of one
block holds for each pass (`Runs.loop_pass`). The code around the block can access any place
(`block_pos`). In a
block, two tasks make no accesses
that can race, and the code of the block makes no access that can race with a task of the block.
The accesses of a task include the accesses of the tasks that it starts, at each depth (`Desc`).
Code outside the blocks can access any place. `raceFree` shows that each execution of such a
program has no data race, so `drf_sc_of_footprints` needs no hypothesis on its executions. The
proof finds where two threads are in the tree of tasks. One thread starts the other at some depth
(`anc`), or two tasks of one block start them (`sib`). `FootprintExample.lean` shows that the
tasks of `forkJoin` in a block follow the rule, also when the block runs in a loop and starts
one task in each pass and reads the output of the task after the block (`loopFoot`). It also has a task that starts two tasks in its own block
(`nestFoot`).

`ChanFootprint.lean` gives the same rule to programs with channels and locks. The rule speaks only
of the places of the code of the program, below `places`. `Recv.lean` shows that the places of the
channels and the locks never race (`chan_no_race`). Their counters and flags have only `sc`
accesses. A `recv` reads its value only after it reads the flag `full k` as set (`recv_full`), so
the `send` happens before it (`data_hb`). The rule can also give a place a lock. The tasks and the
blocks of the main code then access the place only in `locked` blocks of that lock, and two tasks
of a block can share it. `LockFootprint.lean` shows that each such access comes after the thread
entered the lock and before it let the lock go (`guard_access`, `guard_block`). So `lock_excl`
orders two of them (`guard_hb`). The main code outside its blocks needs no lock: no task runs
then, so the spawns and the joins order its accesses with each task (`main_task`).
The code can also fail. Each access to the failure slot and the cancel flag is atomic
(`fail_atomic`), so these places never race. `drf_sc_of_footprints` still needs code with no
failures: `cancel_flag_read_before_the_failure.litmus` has a result that no interleaving gives.
A task can also start tasks, with the same tree of tasks as in `Footprint.lean`.
`ChanFootprintExample.lean` has a block where one task sends a value to the other, a block
where two tasks add 1 to one place under a lock, and a block where a task fails. In
`nestChanFoot`, a task starts two tasks, and one sends a value to the other. The edges of
channels need a finite execution whose `co` orders the writes to each place. So these theorems use `drf_sc_co`, which gives the race-freedom hypothesis
only such executions.

`Typed.lean` proves the footprint rule from types. A small typed language has locals whose types
are trees of leaves: a number, a cell or an atomic. A cell is not `Sync`. A task captures each
local by `move`, `ref` or `mut`. The checker gives the task and the code of its block a
permission for each local. Two permissions fit when one of them is none, or both are `ref` and
the type is `Sync`. `WellTyped.footprints` shows that the compiled program follows the rule of
`Footprint.lean`. So a well-typed program has no data race (`WellTyped.raceFree`).
`TypedExample.lean` checks a block where two tasks add to one atomic and one task writes a
struct. It also shows that the checker rejects a task that borrows a cell with `ref`.

```bash
lake -d formal/memory_model/lean build    # Lean 4.35, no other libraries
```

`Counter.lean` shows that two `fetchAdd`s of a counter never read the same value
(`Counter.tickets`). A channel gives each `send` its number this way. The proof needs the rule of
RC11 that `co` orders the writes to each place (`CoOrder`). `Consistent` does not have that rule,
so the theorems that need it take `CoOrder` as a hypothesis. `CounterExample.lean` shows an
execution that `Consistent` allows without it, where both `fetchAdd`s read 0.

`Channel.lean` adds `send` and `recv`, and lowers them to `sc` atomics on places that the code of
the program cannot use. Each `send` and each `recv` takes its number from a counter of the
channel. A `send` waits for the flag `taken k`, writes its value, and sets `full k`. A `recv` waits
for `full k`, reads the value, and sets `taken (k + n)`, for a channel with room for `n` values.
By `Counter.tickets`, each flag has one writer. So `send_hb_recv` shows that a `send` happens
before the `recv` that gets its value. `recv_hb_send_capacity` shows that the `recv` with the
number `k` happens before the `send` with the number `k + n` ends. A drop of a sender adds 1 to a
third counter. A `recv` waits in a loop that reads this counter, then `full k`. It gets no value
when all senders dropped and `full k` is not set. `drop_hb_closed` shows that each drop happens
before each read that sees all the drops. `Counter.hb_of_lt` gives this edge: a `fetchAdd` happens
before each `sc` read of a larger value. `ChanProg.drf_sc` follows from `TaskProg.drf_sc`.
A lock is a ticket lock with the counters `next` and `serving`. The code holds it in a block,
`locked m s`. The block takes ticket `k` from `next`, reads `serving` until it is `k`, runs `s`,
and adds 1 to `serving`. `unlock_hb_lock` gives the edge from each `unlock` to each later `lock`,
also by `Counter.hb_of_lt`. `Lock.lean` shows that the `unlock` of the ticket `k` reads `serving`
as `k` (`unlock_ticket`). So `lock_excl` shows that two tasks never hold a lock at the same time:
each event of the block of ticket `k` happens before each event of the block of a later ticket.
`Fifo.lean` shows that each `recv` gets the value of one `send`, in order (`ring_fifo`). The
`recv` with the number `k` reads element `k` of the values, and only the `send` with the number
`k` writes it. By `ring_once`, two reads of the same write belong to the same `recv`, so no value
arrives twice. `ring_move_once` shows that each operation accesses its element once. Right after
the access, the operation sets its flag, and its number is used only once.
`Cancel.lean` gives a program a failure slot and a cancel flag. `fail v` writes the error `v` to
the slot with a CAS from 0. If the CAS wrote the slot, it sets the flag with a release write.
`checkCancel` reads the flag with acquire, and fails with `cancelled` if the flag is set.
`one_failure_wins` shows that at most one CAS writes the slot. `block_value_is_failure` shows
that a read of the slot that the winning CAS happens before gets its value.
`block_value_not_cancelled` shows that `cancelled` never reaches the slot. `each_error_once`
shows that the error of at most one failure moves into the slot. The CAS of each other failure
does not write, so that failure drops its error. A task with no `result` never reads the slot, so
`join` gives no error to its caller.
`ChannelExample.lean` builds a run where the main code sends 7 and drops its sender, and a task
makes two `recv`s. The first gets 7, and the second finds the channel closed. `check` finds that
the model allows this run with no data race. `MutexExample.lean` builds a run where a lock orders
a plain write of the main code before a plain read of a task.

The definitions in `lean/VxMemoryModel/Rc11.lean` follow `memalloy/vx_cpu.cat` line by line.
`run_lean.py` checks that Lean and `vx.cat` agree on the 13 tests without scopes and barriers:

```bash
python3 formal/memory_model/run_lean.py    # needs lake and herd7
```

For each test, the script lists every candidate execution, and Lean decides which ones the
model allows and which have a data race. `lean/VxMemoryModel/Check.lean` decides the
definitions themselves, so a wrong answer cannot come from a second copy of the model. The
final states, the observation and the race flag must be the same as those of herd7.

## Reading a test

```
{ data = 0; flag = 0; }
 P0                  | P1                   ;
 w[plain] data 1     | r[acq] r0 flag       ;
 w[rel] flag 1       | r[plain] r1 data     ;
scopes: (system P0 P1)
filter (1:r0 = 1)
exists (1:r1 = 0)
```

- `P0` and `P1` are threads. Each column is the code of one thread.
- `w[rel] flag 1` writes 1 to `flag` with release order. `r[acq] r0 flag` reads `flag` into
  `r0` with acquire order. `plain` is an access that is not atomic. The orders are `plain`,
  `rlx`, `acq`, `rel`, `acq_rel` and `sc`.
- `r[acq,block] r0 flag` is an atomic of block scope. An atomic without a scope has system scope.
- `f[barrier,block]` is a barrier of block scope. The scopes are `subgroup`, `block`, `device`
  and `system`.
- `scopes:` puts the threads in subgroups, blocks and devices. A CPU test writes
  `(system P0 P1)`.
- `filter` keeps only the runs where its condition holds. A test uses it for a wait: a join, or
  a spin on a flag.
- `exists` is the result that the test asks about. herd7 answers *never*, *sometimes* or
  *always*, and says whether a run has a data race.

A fork or a join is a release write and an acquire read of a flag, as in the example.

## Two choices in the model

- **No value comes out of thin air.** The model keeps the rule of RC11: no cycle of program order
  and reads, also for relaxed atomics. AArch64 allows such a cycle, so a relaxed load needs a
  dependency on AArch64, as RC11 says. The other choice would leave the problem open, as in C++.
- **Two atomics race unless each one reaches the thread of the other.** This follows the PTX
  model. `atomic_of_block_scope_across_blocks.litmus` shows the case.

## Limits

- A test may have at most one barrier in each thread. The model matches barriers by scope, not
  by count.
- The rule for sequentially consistent atomics ignores scopes.
- herd7 checks each test completely, but only the tests. memalloy checks every execution up to 5
  or 6 events.
- The core language has no functions and no references.
- `block_hb` needs a block whose body holds no other `tasks` block. A task can hold its own
  blocks.
- The Lean model has no scopes and no barriers.
- The model covers only C0 to C7. The rules for sequential code, E1 to E7 and A1 to A6, are not in
  it.
- `Consistent` does not ask that `co` orders the writes to each place. A theorem that needs it
  takes `CoOrder` as a hypothesis.
- A channel carries at most `msgs` values in a run. Its number of senders is fixed, so a sender
  cannot be cloned yet.
- `drop_hb_closed` speaks of the reads of the counter of drops. That a `recv` returns no value
  only after such a read, and that it then loses no value, follows from its code. No theorem shows
  it yet.
- The code holds a lock for one block. So a guard of a `Mutex` that moves out of its block has no
  model yet. A block of a lock cannot hold another block of the same lock.
- `lock_excl` speaks of the reads of `serving` that let a later ticket in. That the `lock` of a
  ticket ends with such a read follows from `Hold` in `Lock.lean`. No theorem states it on its
  own.
- A program has one failure slot, so only one `tasks` block can fail. A task that sees the cancel
  flag does not stop: the model shows only that its CAS fails. `ChanProg.drf_sc` needs code with
  no failures, because the flag uses release and acquire. A program with failures has no data
  race when it follows the footprint rule, but its executions are not always interleavings.
  `run_lean.py` has no read-modify-writes,
  so no herd7 test has the CAS of a failure. `message_passing_release_acquire` checks the edge of
  the flag.
- `each_error_once` follows an error by the CAS of its failure, not by a value with an owner. A
  CAS that loses still reads the error that won, but only to see that the slot is set.
- A task accesses a place that a lock guards only under that lock, even outside its own blocks,
  because its sibling tasks can run then. Only the main code outside its blocks needs no lock.
- In `Footprint.lean`, the body of a loop is one block with code around it. In
  `ChanFootprint.lean`, it is only the block. Two tasks of a block in a loop must be apart, also
  when they run in two passes. The rule is stricter than needed: two blocks that
  run one after the other can share places, but two tasks of one block cannot.
- In both rules, one thread starts each task, and a task starts only tasks with larger ids.
