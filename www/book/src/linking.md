# Linking an object file into an executable

`vxc -c hello.vx -o hello.o` writes an **object file**, not an executable. The object needs the
Vx runtime libraries around it before anything can run it. This page shows the verified way to
turn one into a program you can run or hand to someone else.

## What `emit-obj` gives you

`-c` is the shorthand for `--action emit-obj`. It lowers your program through MLIR to a native
object file, exactly as far as the toolchain goes on its own. An object file is not a program:
it names the functions your code defines and the ones it calls, and something else has to fill
the calls in.

A program compiled this way calls three kinds of function your object does not define:

| Symbol | Comes from |
| --- | --- |
| `vx_init_signals`, `print_str`, and the rest of the Vx runtime | `libvx_std_core` (shipped in `lib/` next to the compiler) |
| `printMemrefF32` and the other MLIR memref printers | `libmlir_runner_utils` and `libmlir_c_runner_utils` from LLVM 22 |
| `vx_plugin_dispatch_async`, only in a program with `spawn on` | the dispatch backend named by `VX_DISPATCH_LIB` |

An installed toolchain points the compiler at all of these through wrapper scripts, so the
commands below work as written there. In a source checkout, `source config.local` first.

## The link line

`vxc --run` links a temporary object itself, and the same line works for one you made:

```bash
vxc -c hello.vx -O2 -o hello.o
clang hello.o -o hello \
    "$VX_RUNTIME_LIB_DIR/libvx_std_core.dylib" \
    -L"$(dirname "$(llvm-config --libdir)")/../lib" \
    -lmlir_runner_utils -lmlir_c_runner_utils \
    -Wl,-rpath,"$(llvm-config --libdir)" \
    -Wl,-rpath,"$VX_RUNTIME_LIB_DIR" \
    -lm
./hello
```

Two details matter and a plain `clang hello.o -o hello` gets both wrong:

1. The MLIR runner libraries and the Vx runtime must be named at link time, not found later.
   Without them the link fails on `printMemrefF32`, `print_str` or `vx_init_signals` — the
   undefined references are the runtime the object expects, not a compiler bug.
2. macOS records where a dynamic library was at link time, so pass `-rpath` for each directory
   a library lives in. An executable that linked cleanly and then fails at startup with a
   "library not loaded" message is missing an rpath, not a symbol.

On Linux, add `-rdynamic` if the program contains a non-CPU `spawn on`: the dispatch runtime
finds outlined kernels in the executable's own symbol table, and ELF does not export those
unless asked.

A program with `spawn on` on the GPU also needs the dispatch backend named at link time, the
same file `VX_DISPATCH_LIB` names. See `gpu-norms/build.sh` in the Vx-starters repository for a
working GPU link line.

## Asking the compiler to do it

You do not have to write the link line. `vxc --run hello.vx` compiles and links exactly this way
in a temporary directory and executes the result. It is the same pipeline:

```
MLIR  --mlir-translate-->  LLVM IR  --opt-->  --llc-->  object  --clang-->  executable
```

`--run` stops at "executable" and runs it; `emit-obj` stops at "object" and hands the step to
you. The bench harness in `benchmarks/stdlib` is a worked example of linking an `emit-obj`
output against the same libraries `--run` uses — see `build_vx` in `src/bin/cargo-vx-bench.rs`.

If your goal is a shared library rather than a program — calling Vx from C, or loading it from
Python's `ctypes` — the object file is all you need:

```bash
vxc add_one.vx --action emit-obj -o add_one.o
cc -shared -o libadd_one.so add_one.o
```

That works because a shared library carries no `main` and resolves its callers' symbols when it
is loaded. See [Unsafe and FFI](unsafe-and-ffi.md) for what to export and how.
