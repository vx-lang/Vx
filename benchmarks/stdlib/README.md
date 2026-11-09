# Standard-library benchmarks: Vx against C++

Each directory here is one benchmark, written twice: `<name>/<name>.vx` and
`<name>/<name>.cpp`. `cargo vx-bench compare` builds both, runs them, and compares them.

```
cargo build --release          # vxc and the runtime library the harness links against
cargo vx-bench compare --cpu 3
cargo vx-bench compare powi --rounds 20 --json results.json
```

## What the harness does

- Builds the Vx program with `vxc --action emit-obj -O3`, linked against the same libraries as
  `vxc --run`.
- Builds the C++ program with each compiler in `--cxx` (default `clang++,g++`), with
  `-O3 -march=native`, once as is and once with `-ffast-math` (unless `--no-fast-math`).
- Runs every program `--rounds` times (default 10), one program after another in each round, so a
  slow stretch of the machine is spread over all of them. `--cpu N` keeps every run on one core.
- Prints one table per quantity, sorted by median time, with each row's time relative to the
  fastest Vx implementation (`vs vx`, above 1 is slower than Vx), and each row's answer.
- Marks an answer that differs from the fastest Vx answer by more than `--tolerance`. That is
  not a failure: a floating-point sum taken in another order is a different number.

## Writing a benchmark

A program reports two lines per implementation, which the harness reads:

```
vx-bench <quantity>/<implementation> seconds <fastest repetition>
vx-bench <quantity>/<implementation> result  <sum of every repetition's answer>
```

In Vx, `bench_report` from `std::time` prints them; in C++, `vxbench::measure` from `bench.hpp`
does. Rows with the same quantity (the part before `/`) go in one table, and their answers
should agree.

Two kinds of comment line in the `.vx` file change how the harness builds and runs it:

```
// vx-bench flags: --host default --machine fleet/m4-uma.vx
// vx-bench env: VX_METAL=0
```

- The harness gives the flags of each `vx-bench flags:` line to vxc. So does the test that
  compiles every benchmark (`every_shipped_example_and_benchmark_compiles`).
- For each `vx-bench env:` line, the harness runs the Vx program one more time, with these
  environment variables. Its build column shows them, for example `vx VX_METAL=0`. The `vs vx`
  column compares with the run that has no such variables.

The compiler removes work whose result it can prove unused or unchanged. Each of these rules
once produced a number that measured nothing:

- **Every repetition must compute something new.** Pass the repetition number in (a seed added
  to the inputs, or a length shortened by a little), because a repeated pure call is computed
  once. A result near zero seconds means this happened.
- **Every answer must be used.** Add each repetition's answer into the reported result.
- **In C++, the result must exist before the clock is read.** `vxbench::measure` does this with
  `vxbench::keep`; call `vxbench::escape` on arrays the timed code only reads. `noinline` alone
  was not enough: clang moved the call past the second clock read.
- **Compute the same thing on both sides**, with the same inputs, so the answers can be compared.

A Vx benchmark reads the clock with `Instant::now()` from `std::time`. The difference of two
readings is a `Duration`. `bench_report` takes seconds as an `f32`, so the benchmark converts the
`Duration` with `as_secs_f32()`. A Vx result therefore carries about seven significant digits.
