# The compiler

## Actions

`vxc` runs one action per invocation. The default is `run-jit`.

| Flag | What it does |
| --- | --- |
| `--run` | Compile and execute immediately, propagating the program's exit code |
| `-c` | Compile to an object file |
| `--emit-mlir` | Emit the MLIR representation |
| `--emit-llvm` | Emit LLVM IR |
| `--print-ast` | Parse and typecheck, then print the AST |
| `--parse-only` | Lex and parse only |
| `--emit-interface` | Serialize this module's import interface to a `.vxlib` |

```bash
vxc --run program.vx
vxc -c program.vx -o program.o
vxc --emit-mlir program.vx
```

Optimization is `-O0` through `-O3`, defaulting to `-O0`.

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

## Separate compilation

`--emit-interface` writes a `.vxlib`: the module's frozen registry and portable flat-HIR bodies.
A downstream compile consumes it with `--link-interface` and resolves calls into that module without
parsing its source.

```bash
vxc --emit-interface lib.vx -o lib.vxlib
vxc --link-interface lib.vxlib main.vx -o main
```

The `.vxlib` format carries a version tag, and a compiler rejects artifacts written by a different
one. Regenerate them when you upgrade the toolchain rather than keeping them in a cache.

## The other tools

| Tool | Purpose |
| --- | --- |
| `vx-format` | The canonical source formatter |
| `vx-opt` | MLIR pass driver for the Vx dialect |
| `vx-analyzer` | Language server |
| `cargo vx-bench` | Benchmark harness that injects timing into the AST to measure real hardware execution time |

`vx-format` has no options worth learning: there is one canonical style, and it applies it.

```bash
vx-format src/*.vx
```

## The standard library

21 modules, imported as `std::<name>`. Every type and function is listed in the
[standard library reference](stdlib-reference.md), generated from the sources.

| | |
| --- | --- |
| **Core** | `option`, `result`, `box`, `alloc`, `closure`, `iter` |
| **Collections** | `vec`, `hash_map`, `hash_set`, `string` |
| **Numerics** | `math`, `simd`, `tensor` |
| **System** | `io`, `fs`, `net`, `mmap`, `time`, `libc` |
| **Testing** | `googletest` |

Beyond `std` the toolchain ships `graph`, imported as `graph::traversal` and friends.

The repository also carries `examples/llama.vx`, a Llama 2 inference port, and an early
`packages/vx_linalg`. Several other package directories exist under `packages/` but are still
empty placeholders — do not plan around them yet.

## Environment variables

| Variable | Effect |
| --- | --- |
| `VX_STD_PATH` | Library search path. A `PATH`-style list, not a single directory |
| `VX_RUNTIME_LIB_DIR` | Where to find the Vx runtime library |
| `LLVM_CONFIG_PATH`, `MLIR_TRANSLATE_PATH`, `OPT_PATH`, `LLC_PATH`, `CLANG_PATH` | Absolute paths to the LLVM tools, if they are not on `PATH` |
| `VX_DISPATCH_LIB` | The accelerator dispatch backend to load |
| `ENZYME_LIB` | The Enzyme plugin, for autodiff |
| `VX_ALLOW_UNVERIFIED` | Downgrade an undischarged seam obligation to a warning |

An installed toolchain sets the first several of these for you through a wrapper script, so you
normally need none of them.
