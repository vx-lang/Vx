# `Vec::get` and `Vec::set` in a loop, before and after #3841

#3841 lets LLVM inline `Vec::get` and `Vec::set` (#3268). This measures four loops over a
`Vec<i64>` of 50 million values, compiled by `main` and by the PR.

## Result

Milliseconds for one pass over 50 million values. Median of 5 runs, with the fastest and the slowest
run in brackets. Lower is better.

| Loop | `main` (a488cc3d9) | PR #3841 (a355127ef) | `main` / PR |
|---|---|---|---|
| `sum_get`: sum with `a.get(i)` | 45.4 (44.8–46.0) | 6.9 (6.9–7.7) | 6.59× |
| `fill_set`: fill with `a.set(i, v)` | 45.9 (45.1–48.5) | 6.4 (6.2–8.9) | 7.14× |
| `sum_raw`: sum through a raw pointer | 6.9 (6.7–12.2) | 6.8 (6.7–8.0) | 1.01× |
| `fill_raw`: fill through a raw pointer | 6.3 (6.2–6.4) | 6.6 (6.2–7.1) | 0.96× |

- With the PR, `sum_get` and `fill_set` are as fast as the raw pointer loops. LLVM inlines `get` and
  `set` and takes the index check out of the loop.
- The raw pointer loops are the controls. The PR does not change their code, and their times did not
  change beyond the noise.
- All 10 runs printed the same results: the last value 49999999 and the sum 1249999975000000.

## The first write after `push`

An earlier version of this benchmark had no untimed pass. Its first timed loop, `fill_set`, was 3 to
5 times slower than later passes, and it varied from 18 to 78 ms. So it reported `fill_set` as 4
times slower than `fill_raw`, which was wrong. Tests on the PR's compiler showed this:

- The first loop that writes the vector after `push` is slow, whichever loop it is. When `fill_raw`
  ran first, it took 22 to 34 ms, and `fill_set` after it took 6.3 ms.
- Later passes of either loop take about 6.3 ms.
- It is not page faults: `getrusage` counted 0 minor and 0 major faults in the slow pass, and about
  one fault for each 16 KB page during `push`.
- It is not the values: the pass was slow also when `push` wrote non-zero values, and when each pass
  wrote different values.
- It is not the speed of the CPU: 400 ms of arithmetic before the pass did not make it faster.
- All four quarters of the vector are equally slow, about 1 µs more for each 16 KB page.
- After a pause of 3 s, the next pass was sometimes slow again (7 to 20 ms).

The machine was under memory pressure (see Limits). The cost may come from how macOS tracks the use
of each page. I could not confirm this without kernel tracing, which needs root. The program now
writes the whole vector once before the timed loops, so this cost is in no measured loop.

## Setup

What both sides share:

| | |
|---|---|
| Machine | Apple M2 Pro (Mac14,9), 6 performance and 4 efficiency cores, 16 GB, on AC power |
| OS | macOS 27.2.1 (26B131) |
| LLVM | 22.1.8 |
| Program | `vec_loops.vx`, compiled with `-O3` and the default code generator, run as a native program |
| Data | `Vec<i64>` of 50 million values, filled with `push` and then written once more before the timed loops |
| Timer | `std::time::Instant` inside the program, so compile time is not counted |

What differs:

| | `main` | PR #3841 |
|---|---|---|
| vxc commit | a488cc3d9 | a355127ef |
| Build | `cargo build --release` | `cargo build --release` |

`env.txt` has the full list.

## How it ran

- 5 rounds. Each round runs both compilers. The first one alternates between rounds.
- Each run compiles the program and runs it once. Each loop runs once in each run.
- The script waits until the 1-minute load average is below 6 before each run. A virtual machine used
  about 2 cores during all runs, so the load stayed between 5.7 and 6.0. `runs.csv` has the load of
  each run.
- `runs.csv` has every run. `logs/` has the output of each run. No run was left out.

## Limits

- One machine, one size (50 million values) and one element type (`i64`).
- The machine was not idle: a virtual machine used about 2 of its 10 cores.
- The machine was under memory pressure: about 90 MB of memory was free, and the compressor held
  about 11.8 GB. Another machine may not show the first-write cost above.
- Each loop runs once in each run, so the times include cold caches and branch predictors.

## Commands

From the root of a Vx checkout with `source config.local`, with release builds of both commits:

```sh
docs/benchmarks/2026-11-23-vec-get-set-inlining/run.sh <vxc of main> <vxc of the PR> 5
```

The script runs `vxc -O3 vec_loops.vx` for each compiler.
