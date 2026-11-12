# The compiler

## Actions

`vxc` runs one action per invocation. The default is `run-jit`.

| Flag | What it does |
| --- | --- |
| `--run` | Compile and execute immediately. A program that returns a value other than 0 makes `vxc` exit with status 1 |
| `-c` | Compile to an object file |
| `--emit-mlir` | Emit the MLIR representation |
| `--emit-llvm` | Emit LLVM IR |
| `--print-ast` | Parse and typecheck, then print the AST |
| `--parse-only` | Lex and parse only |
| `--emit-interface` | Serialize this module's import interface to a `.vxlib` |
| `--audit` | Parse and typecheck, then print each fact that the compiler trusts without a proof |
| `--emit-ptx` | Write the GPU kernels of the program as one PTX file. A `.json` file beside it gives the parameters of each kernel |
| `--emit-msl` | Write the Apple GPU kernels of the program as one Metal Shading Language file, and a `.json` file beside it |

```bash
vxc --run program.vx
vxc -c program.vx -o program.o
vxc --emit-mlir program.vx
```

Optimization is `-O0` through `-O3`, defaulting to `-O0`.

| Flag | What it does |
| --- | --- |
| `-fweak-relaxed-atomics` | A relaxed atomic read stays relaxed, so the program can see load buffering. The default, `-fno-weak-relaxed-atomics`, makes each relaxed read an acquire read |
| `-fauto-schedule-levels` | An `across` loop on a chip that cannot launch its levels runs as a plain `for` loop. The default, `-fno-auto-schedule-levels`, makes such a loop an error |
| `--legacy-codegen` | Generate code with the second code generator, which walks the syntax tree. The tests check that both code generators print the same |

For each `-f` feature, the last flag on the command line wins.

## Hardware and machine models

| Flag | What it does |
| --- | --- |
| `--machine <FILE>` | Compile against a declared machine — see [machine files](machine-files.md) |
| `--host <FILE\|default>` | Declare the host the program runs on |
| `--diagnostics-json [PATH]` | Write the admission verdict as one structured JSON record |
| `--verify-seams` | Discharge asynchronous-visibility obligations with z3 |

`--diagnostics-json` is the one to reach for in a build script or a CI job. The schema is versioned,
and a capacity rejection carries the space, the requirement, the availability and the margin as
fields rather than as prose you would have to parse out of a message.

Note that with `--verify-seams` and no solver on `PATH`, the compile *fails* rather than certifying
seams it could not check. `VX_ALLOW_UNVERIFIED=1` downgrades that to a warning.

## Where imports come from

An import that starts with a package name reads only that package (see below). `vxc` looks for any
other import in two places, in this order:

1. The standard library. An installed toolchain keeps it in `stdlib/` beside `bin/`, and a
   development build uses the `stdlib/` of the source tree it was built from. `--std-path DIR`
   replaces it. Give the flag once for each directory.
1. The directory of the file you compile, so `import helper;` reads the `helper.vx` beside it.

`vxc` never searches the current directory, so a command gives the same result wherever you run it.

## Packages

`--package name=dir` makes a library in another directory importable by name. With
`--package textkit=../textkit`, `import textkit::runs` reads `../textkit/runs.vx`. Imports that start
with `textkit` are read from that directory only, and the stdlib paths stay as they are. Give the
flag once for each package.

```bash
vxc --package textkit=../textkit main.vx
```

## Separate compilation

`--emit-interface` writes a `.vxlib`: the module's frozen registry and portable flat-HIR bodies.
A downstream compile consumes it with `--link-interface` and resolves calls into that module without
parsing its source.

```bash
vxc --emit-interface lib.vx -o lib.vxlib
vxc --link-interface lib.vxlib main.vx
```

The `.vxlib` format carries a version tag, and a compiler rejects artifacts written by a different
one. Regenerate them when you upgrade the toolchain rather than keeping them in a cache.

## Version numbers

`vx-api diff` compares the interfaces of two versions of a library and prints the smallest safe
version bump. An added item needs a minor release. A removed item, or a change that breaks a caller,
needs a major one. Anything else is a patch.

```bash
vx-api diff old/lib.vxlib new/lib.vxlib
```

Every item in a `.vxlib` counts as public, because a program can call any of them. The tool cannot
see traits, data-carrying enums, constants or the bounds of generic parameters, because a `.vxlib`
does not hold them.

## The other tools

| Tool | Purpose |
| --- | --- |
| `vx-format` | The canonical source formatter |
| `vx-opt` | MLIR pass driver for the Vx dialect |
| `vx-analyzer` | Language server |
| `vx-api` | Compares the interfaces of two versions of a library and reports the version bump |
| `cargo vx-bench` | Runs the programs in `benchmarks/manifest.txt` and collects the times they report; `cargo vx-bench compare` runs the Vx-against-C++ suite in `benchmarks/stdlib` |

`vx-format` has no options worth learning: there is one canonical style, and it applies it.

```bash
vx-format src/*.vx
```

## The standard library

The library has two parts. `core::<name>` holds the language's basic types and traits, and needs no
operating system. `std::<name>` holds the rest. Every type and function is listed in the
[standard library reference](stdlib-reference.md), generated from the sources.

| | |
| --- | --- |
| **core** | `option`, `result`, `iter`, `ops`, `cmp`, `clone`, `convert`, `default`, `hash`, `marker`, `mem`, `fmt`, `num`, `cell`, `ptr`, `tuple`, `time` |
| **Collections** | `std::vec`, `std::string`, `std::box`, `std::alloc` |
| **Numerics** | `std::tensor`, `std::simd`, `std::rand`, `std::nn` |
| **Tasks** | `std::sync::channel`, `std::sync::mutex`, `std::sync::arc`, `std::sync::wait`, `std::collective` |
| **System** | `std::io`, `std::fs`, `std::net`, `std::mmap`, `std::time`, `std::libc` |
| **Testing** | `std::googletest` |

`std::hash_map` and `std::hash_set` have only C functions for `i32` keys, with no `HashMap` type yet.

Beyond `std` the toolchain ships `graph`, imported as `graph::traversal` and friends.

The repository also carries `examples/llama.vx`, a Llama 2 inference port, and an early
`packages/vx_linalg`. Several other package directories exist under `packages/` but are still
empty placeholders — do not plan around them yet.

## Writing tests

A test in Vx is a program. It runs the code under test, compares each result with the value that
it expects, and returns 0 when each comparison passes. `expect_eq(actual, expected)` from
`std::googletest` does one comparison. When the two values are equal, the program continues.
When they are different, `expect_eq` prints both values and stops the program with `panic`. So the
test ends with an exit code that is not 0, and a build script or a CI job sees the failure.

This test checks two functions (`book_tooling_expect_eq.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tooling_expect_eq.vx:program}}
```

It prints `square passed`, and then `half passed`.

If `square(12)` gave 145, the program would stop at that line and print:

```text
expect_eq failed: expected 144, actual 145
panic: expect_eq failed
  at tests/backend/pass/book_tooling_expect_eq.vx:30
```

The last line gives the file and the line of the call to `expect_eq`, not a line in
`std::googletest`. `expect_eq` gets this place from its last parameter, `at : Caller` (see
[Stopping a program](tour.md#stopping-a-program)).

Today `expect_eq` compares only `i32` and `f32` values. A call with an `i64` gets the error
`Type 'i64' does not implement trait 'GoogletestEq'`. Two `f32` values must be exactly equal, so
a result that is one rounding step away fails. For a value of another type, use `assert(cond)`.

To run a test, compile and run it with `vxc --run test.vx`. When a check fails, `vxc` exits with
status 1, so a build script or a CI job sees the failure.

## Environment variables

| Variable | Effect |
| --- | --- |
| `VX_RUNTIME_LIB_DIR` | Where to find the Vx runtime library |
| `LLVM_CONFIG_PATH`, `MLIR_TRANSLATE_PATH`, `OPT_PATH`, `LLC_PATH`, `CLANG_PATH` | Absolute paths to the LLVM tools, if they are not on `PATH` |
| `VX_DISPATCH_LIB` | The accelerator dispatch backend to load |
| `ENZYME_LIB` | The Enzyme plugin, for autodiff |
| `VX_ALLOW_UNVERIFIED` | Downgrade an undischarged seam obligation to a warning |
| `VX_TASK_RUNTIME` | `shared` runs many tasks on a few OS threads. `threads`, the default, gives each task an OS thread of its own. Any other value stops the program. See [How tasks run](tasks.md#how-tasks-run) |
| `VX_TASK_WORKERS` | With `VX_TASK_RUNTIME=shared`, the number of threads at the start. The default is one thread for each CPU |
| `VX_TASK_MAX_WORKERS` | With `VX_TASK_RUNTIME=shared`, the largest number of threads. The default is 256, or `VX_TASK_WORKERS` when that is larger |
| `VX_TASK_SEED` | Runs all tasks on one thread, in an order that the number gives. It cannot go with `VX_TASK_WORKERS` or `VX_TASK_MAX_WORKERS` |
| `VX_DROP_WAIT_SECONDS` | Makes the time limit of each `drop` shorter, never longer. The value is a number of seconds. See [Tasks and channels](tasks.md) |
| `VX_REMOTE_LINK_TIMEOUT` | The seconds that a host waits for an answer from the machine of a remote worker. The default is 30 |
| `VX_HOST_THREADS` | The number of CPU threads that run the CPU copy of a GPU kernel. The default is each CPU that the thread can run on |
| `VX_NUMA_NO_AFFINITY` | When set, a dispatch does not pin its thread to the NUMA node of its data. See [NUMA](numa.md) |
| `VX_METAL` | `0` runs the CPU copy of each kernel and not the Apple GPU. The default uses the GPU |
| `VX_CUDA_SYNC` | When set, the program waits after each CUDA kernel launch, so an error names the kernel that failed |
| `VX_GPU_CHIP` | Read by the compiler: the NVIDIA chip to compile kernels for, such as `sm_90`. The default is the `chip:` of the machine file, or else `sm_80` |
| `VX_PRINT_DROPS` | Read by the compiler: when set, it prints the place where each function frees each tensor that it owns |

An installed toolchain sets the first several of these for you through a wrapper script, so you
normally need none of them.
