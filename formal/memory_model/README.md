# The Vx memory model

The memory model says which values a read can see when code runs at the same time. This
directory holds the model in three forms, and tools that check the forms against each other. The
rules are C0 to C7 in `docs/discussions/borrow_check_sequence_points_for_calls.md`. Tracking
issue: #1850.

| Form | Tool | What it checks |
| --- | --- | --- |
| `vx.cat` | [herd7](https://github.com/herd/herdtools7) | Each litmus test in `tests/` gives the result that it expects. |
| `memalloy/vx_cpu.cat` | [memalloy](https://github.com/johnwickerson/memalloy) | The CPU part agrees with RC11, and the mapping to AArch64 is correct, up to 6 events. |
| `lean/` | Lean 4 | Proofs for programs of any size. |

A *litmus test* is a small concurrent program and a question about its result. For example: can
the reader see the flag set but the data not set?

## Files

| File | What it is |
| --- | --- |
| `vx.cat` | The model: RC11 (Lahav et al., PLDI 2017), with GPU scopes and barriers. |
| `vx_events.cat` | The tags, the scopes and the barriers. `vx.cat` and `sequential.cat` use it. |
| `sequential.cat` | Sequential consistency: each run is one interleaving of the threads. |
| `vx.bell` | The tags that a test can put on an access or a fence. |
| `tests/` | One litmus test for each case of the rules. |
| `run_litmus.py` | Runs each test in herd7 and compares the result with its `Expect:` line. |
| `run_lean.py` | Checks that Lean and `vx.cat` give the same answer on each test. |
| `memalloy/` | The CPU part of the model in the dialect of memalloy, and the AArch64 mapping. |
| `lean/` | The model and the proofs in Lean 4. |
| `gpu/` | Litmus tests written as Vx programs, to run on a real GPU. |

## Running the checks

```bash
opam install herdtools7                                   # 7.58 or later
python3 formal/memory_model/run_litmus.py                 # the litmus tests
lake -d formal/memory_model/lean build                    # the Lean proofs, Lean 4.35
python3 formal/memory_model/run_lean.py                   # Lean against herd7
MEMALLOY=~/memalloy python3 formal/memory_model/memalloy/run_memalloy.py   # about 3 minutes
```

CI (`.github/workflows/formal.yml`) runs `run_litmus.py`, the Lean build and `run_lean.py`. It
runs them on each pull request that changes `formal/`, and each Monday. memalloy needs local patches, so CI does not run it.

`run_litmus.py` also checks *DRF-SC*: a program with no data race acts as if the threads take
turns. So a test with no data race, and only plain or `sc` accesses, must give the same results
on `vx.cat` as on `sequential.cat`.

`run_lean.py` makes herd7 list each candidate execution of a test. Lean then decides which ones
the model allows and which have a data race, and the answer must match herd7. It uses
`Check.lean` for CPU tests and `ScopedCheck.lean` for tests with scopes. A CPU test goes through
both, so a wrong answer cannot hide in a second copy of the model. The check runs as a compiled
program, `vxcheck` (`lean/CheckMain.lean`), about 70 times faster than `#eval`.

To build memalloy with OCaml 4.14 and Python 3, make two changes:

1. Add `let compare = Stdlib.compare` after `open List` in `src/myList.ml`.
1. Port the `print` and `string.replace` calls in `etc/` to Python 3.

## What memalloy checks

memalloy searches for a small execution that one model allows and another forbids. It checks
these three facts, up to 5 or 6 events:

- `vx_cpu.cat` and RC11 allow the same executions and find the same data races.
- The mapping in `vx_to_aarch64.als` gives no AArch64 result that `vx_cpu.cat` forbids.
- If the mapping drops the dependency after a relaxed load, it does give such a result. This is
  the cost of the rule against values out of thin air (see below).

Its model of PTX is from 2013, older than the PTX model of Volta. So it cannot check a mapping to
current GPUs.

## What the Lean proofs show

Each file in `lean/VxMemoryModel/` starts with a comment that explains it. The tables give the
main result of each file. A file named `...Example.lean` builds small runs for the file of the
same name, and shows why each rule is there.

**The model**

| File | What it shows |
| --- | --- |
| `Execution.lean`, `Rc11.lean` | An execution is a graph of the accesses of one run. `Rc11.lean` follows `memalloy/vx_cpu.cat` line by line. |
| `Scopes.lean` | Adds the GPU scopes and barriers of `vx.cat`. With every scope `system`, it equals `Rc11.lean`, so the CPU proofs still hold. |
| `ScIsRc11.lean` | The model allows each interleaving of the threads. |
| `DrfSc.lean` | An execution with no data race, no fences and only `sc` atomics is an interleaving (DRF-SC). |
| `Program.lean` | DRF-SC for a whole program: if no interleaving has a data race, each execution is an interleaving. |

**A small language with tasks**

| File | What it shows |
| --- | --- |
| `Thread.lean`, `Language.lean` | A core language with loads, stores, `fetchAdd`, `cas`, fences, `if`, loops and arrays. DRF-SC holds for its programs that use only `sc` atomics. |
| `Tasks.lean` | `spawn` and `join`: the code before a `spawn` happens before the task, and the task happens before the code after its `join`. |
| `Blocks.lean` | A `tasks { .. }` block waits for each task that it started, also with blocks inside blocks. |
| `Footprint.lean` | A rule on the code: two tasks of one block touch different places. A program that follows it has no data race. |
| `ChanFootprint.lean`, `LockFootprint.lean` | The same rule for programs with channels and locks. Two tasks can share a place that a lock guards. |
| `Typed.lean` | A type checker in the style of Rust: `Send`, `Sync`, borrows, and raw pointers with `unsafe impl`. A program that it accepts follows the rule of `Footprint.lean`, so it has no data race. |
| `TypedChan.lean` | The same for a typed language with channels and mutexes. |

**Channels, locks and the library**

| File | What it shows |
| --- | --- |
| `Counter.lean` | Two `fetchAdd`s on one counter never read the same value. |
| `Channel.lean` | A `send` happens before the `recv` that gets its value. Each drop of a sender happens before a `recv` sees the channel closed. |
| `Recv.lean` | The places inside a channel or a lock never race. |
| `Fifo.lean` | Each `recv` gets the value of one `send`, in order, and no value arrives twice. |
| `Lock.lean` | Two tasks never hold a ticket lock at the same time. |
| `MutexOrders.lean` | The same for the memory orders that `Mutex` in the library uses. |
| `Arc.lean` | The memory orders of `std::sync::Arc` are correct: each use of the value happens before the last drop frees it. |
| `ChannelOrders.lean` | The memory orders of the slots in `std::sync::channel` are correct. |
| `ReleaseSequence.lean` | Lemmas that `Arc.lean`, `MutexOrders.lean` and `ChannelOrders.lean` share. |
| `Cancel.lean` | The failure design of #1893: at most one task error becomes the error of the block. The compiler does not have this design yet. |
| `LastArriver.lean` | The last-arriver pattern of #2139 on a GPU: the last block reads each partial result, when the fences have device scope. |

**Progress: does the program finish?**

| File | What it shows |
| --- | --- |
| `Machine.lean`, `Interleave.lean` | A fleet of nodes that pass messages, and a CPU node where one thread moves at each step. Each run of a CPU node gives an execution of the model. |
| `InterleaveComplete.lean` | The other direction: for a program with no data race, each finite execution of the model comes from a run. |
| `Progress.lean` | The WF1 rule of TLA. In a fair run, each message in flight arrives. |
| `Deadlock.lean`, `CpuDeadlock.lean` | A fleet where each node waits and no message is in flight stays stuck forever. A snapshot of each node can detect this. |
| `Finish.lean` | A thread with no loop finishes in a fair run, if each of its waits can end. A wait across nodes needs a time limit. |
| `Scheduler.lean` | The ready queue of `sched.rs` is first in, first out. So a ready task runs within a bound, if the workers keep taking tasks. A queue that pushes to the front can starve a task. |
| `Select.lean` | `select` in `channel.vx` picks a receiver that stays ready within `n` calls. The old rule could starve it. |

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
  `r0` with acquire order. The orders are `plain` (not atomic), `rlx`, `acq`, `rel`, `acq_rel`
  and `sc`.
- `r[acq,block] r0 flag` is an atomic of block scope. `f[rel,device]` is a fence of device
  scope. With no scope, an atomic or a fence has system scope.
- `rmw[rlx,device] r0 (add r0 1) count` adds 1 to `count` and reads the old value into `r0`.
- `f[barrier,block]` is a barrier of block scope. The scopes are `subgroup`, `block`, `device`
  and `system`.
- `scopes:` puts the threads in subgroups, blocks and devices. A CPU test writes
  `(system P0 P1)`.
- `filter` keeps only the runs where its condition holds. A test uses it for a wait, such as a
  join or a spin on a flag.
- `exists` is the question. herd7 answers *never*, *sometimes* or *always*, and says whether a
  run has a data race.

A fork or a join is a release write and an acquire read of a flag, as in the example.

## Four choices in the model

- **No value comes out of thin air.** The model keeps the rule of RC11: no cycle of program order
  and reads, also for relaxed atomics. PTX and AArch64 allow such a cycle, so vxc makes each
  relaxed atomic read an acquire read. `-fweak-relaxed-atomics` keeps relaxed reads relaxed. A
  program compiled with it can show `load_buffering_relaxed.litmus`.
- **Two atomics race unless each one reaches the thread of the other.** This follows the PTX
  model. See `atomic_of_block_scope_across_blocks.litmus`.
- **A release and an acquire synchronize only when they reach each other.** The same holds for
  the write and the read that link them. With fences, these are two pairs, as in PTX. See
  `last_arriver_with_fences_of_block_scope.litmus`.
- **A read-modify-write continues a release sequence only when it and the write it reads reach
  each other.** This is the "observation order" of the PTX model. See
  `release_sequence_broken_by_an_add_of_block_scope.litmus`.

## Limits

**The model**

- The model matches barriers by scope, not by count. So a thread can run two barriers of one
  scope only when no other thread is in its group of that scope
  (`two_barriers_in_a_block_of_one_thread.litmus`). A test with two such barriers in two threads
  can get a wrong answer.
- A barrier of system scope has no effect (`barrier_of_system_scope_orders_nothing.litmus`).
- The rule for `sc` atomics ignores scopes.
- The model covers only C0 to C7. The rules for sequential code, E1 to E7 and A1 to A6, are not in
  it.

**The checks**

- herd7 checks each test completely, but only the tests. memalloy checks every execution, but
  only up to 5 or 6 events.
- `run_lean.py` skips the tests with a branch. It has no read-modify-writes, so no test has the
  CAS of a failure. `message_passing_release_acquire` checks the edge of the cancel flag.
- `vxcheck` copies each execution into arrays before it checks it (`fastData`). No theorem shows
  that the copy equals `Data.ofLists`. On the tests, its answers are the same as those of `#eval`.

**The Lean proofs**

- The core language has no functions and no references.
- `Consistent` does not require that `co` orders the writes to each place. A theorem that needs
  this takes `CoOrder` as a hypothesis.
- The footprint rule is stricter than needed. Two tasks from two `spawn`s of one block cannot
  share a place, even when the two `spawn`s run one after the other.
- The rule compares each task of an inner block with all the code of the outer block, also the
  code after the inner block joined it.
- One thread starts each task, and a task starts only tasks with larger ids.
- In `Footprint.lean`, the body of a loop is a block with code around it. In
  `ChanFootprint.lean`, it is only the block.
- A task touches a place that a lock guards only under that lock, also outside its own blocks,
  because its sibling tasks can run then. Only the main code outside its blocks needs no lock.

**Channels and locks**

- A channel carries at most `msgs` values in a run. Its number of senders is fixed, so a sender
  cannot be cloned yet.
- `drop_hb_closed` speaks of the reads of the count of drops. No theorem shows yet that a `recv`
  returns no value only after such a read, and that it then loses no value. This follows from its
  code.
- The code holds a lock for one block. So a `Mutex` guard that moves out of its block has no
  model yet. A block of a lock cannot hold another block of the same lock.
- No theorem states on its own that the `lock` of a ticket ends with a read of `serving` that lets
  it in. This follows from `Hold` in `Lock.lean`.

**Failures**

- `Cancel.lean` models #1893, not the rule of the compiler today, where the first failed join in
  the code wins. No file models that rule yet.
- A program has one failure slot, so only one `tasks` block can fail. A task that sees the cancel
  flag does not stop. The model shows only that its CAS fails.
- `ChanProg.drf_sc` needs code with no failures, because the cancel flag uses release and
  acquire. A program with failures that follows the footprint rule has no data race. But its
  executions are not always interleavings (`cancel_flag_read_before_the_failure.litmus`).
- `each_error_once` follows an error by the CAS of its failure, not by a value with an owner. A
  CAS that loses still reads the error that won, but only to see that the slot is set.

**Progress**

- A CPU node sends and takes no messages yet. A read of a place that no write wrote has no step.

- An execution can end between the read and the write of a read-modify-write, but a run of a CPU
  node cannot. So in `Cpu.complete`, the run can lack the last read of a thread.

- `Cpu.complete` needs code with no `rmwStore` (`Stmt.NoRmw`). Only a `fetchAdd` or a `cas` makes
  one. Without this rule, a `load` and then an `rmwStore` give an execution that no run makes.

- `snapshot` assumes a cut that no message crosses. The marker protocol that gives such a cut is
  not modeled.

- A thread that a failed bounds check stopped has no step. `Cpu.stops` cannot tell it from a
  thread that is done.

- `Cpu.finishes` takes its three conditions as hypotheses on the run. No proof yet shows that the
  waits of `tasks` and of channels meet them. A thread with a `loop` is not covered.

- `fifo_fair` takes three facts of the runtime as hypotheses, and no proof covers them:

  - The workers take a fiber within `B` points.
  - A wake puts a fiber that can continue on the queue.
  - A fiber that a worker takes makes a step.

  `formal/runtime/WaitWake.tla` checks the second fact at a small size, with TLC.
