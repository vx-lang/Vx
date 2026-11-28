# `find_byte` and `compare_bytes` against C's `memchr` and `memcmp`

`bytes.vx` measures four routines and two byte loops on 16 bytes to 64 MiB:

- `find_byte` and `compare_bytes` from `core::ptr`, written in Vx.
- The C library's `memchr` and `memcmp`. On Linux this is glibc, which picks a version for the CPU
  when the program starts.
- A byte loop for each, as a program writes it by hand. LLVM does not make vector code from these
  loops, because each loop can stop at any byte.

Each call reads the whole range but one byte. The byte to find, and the one byte that differs,
are the last but one. The ranges start 16 bytes after the start of a page, as a block from
`malloc` does. The program checks the result of each routine and prints `wrong result` when one
is wrong.

## Run it

```bash
source config.local
cargo build --release --bin vxc
LOAD_MAX=2 benchmarks/find_byte_compare_bytes/run.sh ./target/release/vxc 5 /tmp/bytes-bench
```

`run.sh` runs the program 5 times. Each run has 3 rounds, and each round runs the six routines in
the other order. Before each run, `run.sh` waits until the 1-minute load average is below
`LOAD_MAX`. It writes:

- `setup.txt`: the date, the CPU, the OS and C library, and the versions of vxc and LLVM.
- `runs.csv`: one line for each run, round, size and routine, with the load at the start of the run.
- `summary.txt`: the median, the slowest and the fastest rate of each routine at each size.

## What it does not measure

- vxc compiles the Vx routines into the program, and LLVM can inline them. The program calls the C
  routines through the dynamic linker.
- Each size runs many calls on the same bytes, so the branches of each routine are always predicted
  and a range that fits in the cache stays there. A program that calls a routine once on new data
  pays for the cache misses and the branch misses.
- The byte to find is always found, and the ranges always differ at the end. A range that holds the
  byte early, or differs early, stops early in every routine.
