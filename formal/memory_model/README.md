# The Vx memory model in herd7

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
| `tests/` | One test for each case of the rules. |

## Running the tests

```bash
opam install herdtools7        # version 7.58 or later
python3 formal/memory_model/run_litmus.py
```

The script also checks *DRF-SC*. A test without a data race, whose accesses are plain or
sequentially consistent, must give the same final states on `vx.cat` as on `sequential.cat`.
Such a program cannot see how the hardware reorders memory accesses.

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
- herd7 checks each test completely, but only the tests. A proof for all programs belongs to the
  Lean model of #1850.
- The model covers only C0 to C7. The rules for sequential code, E1 to E7 and A1 to A6, are not in
  it.
