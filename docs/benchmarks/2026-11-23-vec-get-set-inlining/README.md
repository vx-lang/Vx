# `Vec::get` and `Vec::set` in a loop, before and after #3841

#3841 lets LLVM inline `Vec::get` and `Vec::set` (#3268). This measures four loops over a
`Vec<i64>` of 50 million values, compiled by `main` and by the PR.

## Result

Milliseconds for one pass over 50 million values. Median of 5 runs, with the fastest and the slowest
run in brackets. Lower is better.

| Loop | `main` (a488cc3d9) | PR #3841 (a355127ef) | `main` / PR |
|---|---|---|---|
| `sum_get`: sum with `a.get(i)` | 45.0 (44.5–45.3) | 7.3 (6.9–7.5) | 6.18× |
| `fill_set`: fill with `a.set(i, v)` | 66.9 (52.5–77.4) | 26.5 (18.6–37.8) | 2.52× |
| `sum_raw`: sum through a raw pointer | 6.7 (6.7–6.9) | 6.9 (6.8–7.4) | 0.98× |
| `fill_raw`: fill through a raw pointer | 6.5 (6.2–6.8) | 6.4 (6.3–8.4) | 1.00× |

- `sum_get` is now as fast as `sum_raw`. LLVM inlines `get` and vectorizes the loop.
- `fill_set` is 2.5× faster, but still about 4× slower than `fill_raw`. LLVM inlines `set`, unrolls
  the loop by 4, and moves the index check out of the loop. It does not vectorize the loop
  ("the cost-model indicates that vectorization is not beneficial"). I did not look further.
- The raw pointer loops are the controls. The PR does not change their code, and their times did not
  change beyond the noise.
- All 10 runs printed the same results: the last value 49999999 and the sum 1249999975000000.

## Setup

What both sides share:

| | |
|---|---|
| Machine | Apple M2 Pro (Mac14,9), 6 performance and 4 efficiency cores, 16 GB, on AC power |
| OS | macOS 27.2.1 (26B131) |
| LLVM | 22.1.8 |
| Program | `vec_loops.vx`, compiled with `-O3` and the default code generator, run as a native program |
| Data | `Vec<i64>` of 50 million values, filled with `push` before the timed loops |
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
- `fill_set` varies a lot between runs on both sides (18.6 to 37.8 ms on the PR). I did not find
  out why. It is the first timed loop in each run.
- Each loop runs once in each run, so the times include cold caches and branch predictors.

## Commands

From the root of a Vx checkout with `source config.local`, with release builds of both commits:

```sh
docs/benchmarks/2026-11-23-vec-get-set-inlining/run.sh <vxc of main> <vxc of the PR> 5
```

The script runs `vxc -O3 vec_loops.vx` for each compiler.
