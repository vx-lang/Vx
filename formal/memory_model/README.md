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

herd7 and memalloy check small executions. `lean/` proves two theorems for executions of any
size, on the CPU part of the model:

- `consistent_of_seqConsistent`: the model allows every interleaving of the threads.
- `seqConsistent_of_raceFree`: take an execution that the model allows. If it has no data race,
  no fences, and only sequentially consistent atomics, then it is an interleaving (DRF-SC).

```bash
lake -d formal/memory_model/lean build    # Lean 4.35, no other libraries
```

The definitions in `lean/VxMemoryModel/Rc11.lean` follow `memalloy/vx_cpu.cat` line by line.
`run_lean.py` checks that Lean and `vx.cat` agree on the 12 tests without scopes and barriers:

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
- The Lean DRF-SC theorem is about one execution. The theorem for programs needs a semantics of
  programs: if no interleaving of a program has a data race, then every run is an interleaving.
- The Lean model has no scopes and no barriers.
- The model covers only C0 to C7. The rules for sequential code, E1 to E7 and A1 to A6, are not in
  it.
