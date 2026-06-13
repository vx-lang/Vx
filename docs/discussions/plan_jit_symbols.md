# JIT Stack Frame Symbol Resolution (`<unknown>`)

Currently, when the `RunJit` action is executed (which is the default when running `vxc` without emitting object files), `vx` lowers the code to LLVM IR and then executes it using LLVM's `lli` tool as a subprocess.

## The Root Cause of `<unknown>`

The Rust standard library and the `backtrace` crate use platform-specific APIs to resolve instruction pointers to symbol names. On macOS, this is done using `dladdr()`, which queries the dynamic linker (`dyld`) for loaded Mach-O images.
However, `lli` executes the code by mapping machine code directly into memory (JIT). This in-memory code is never registered with the dynamic linker as a standard library image, meaning `dladdr()` cannot find the symbols and returns `<unknown>`.
Even if we register the JIT memory with the unwinder (`__register_frame`), this only fixes the *unwinding* of the stack (allowing the stack trace to not abort prematurely), but it does **not** provide the *symbol names* to `dladdr`.

## User Review Required

> [!IMPORTANT]
> To seamlessly resolve the JIT stack frames to their proper function names (e.g., `do_test_chain`, `f1`, etc.) without writing a highly complex custom JIT symbolizer, we should change the `execute_mlir` function in `jit.rs` to **compile the optimized LLVM IR into a temporary native executable and run that executable**, rather than using `lli`.

### Why this is the best approach:

1. **Perfect Stack Traces**: `backtrace-rs` will natively understand the temporary executable on disk, read its DWARF debug info, and perfectly resolve all function names and line numbers.
1. **Native Debugger Support**: You'll be able to attach `lldb` or `gdb` natively without relying on GDB JIT integration.
1. **No Overhead Difference**: We already write the MLIR/LLVM IR to temporary files and invoke `opt` and `lli` as subprocesses. Invoking `llc` (to generate `.o`) and `clang` (to link) takes roughly the same amount of time as `lli`'s JIT compilation and execution.

## Proposed Changes

### `src/jit.rs`

#### [MODIFY] \[jit.rs\](file:///Users/adityak/go/Vx/src/jit.rs)

- Instead of calling `lli`, use `llc` to compile `temp_opt_{uid}.ll` to `temp_opt_{uid}.o`.
- Use `clang` to link `temp_opt_{uid}.o` along with the required libraries (`libmlir_c_runner_utils`, `libvx_std_core.dylib`, etc.) into an executable `temp_{uid}.out`.
- Execute `temp_{uid}.out` and capture its output (same as we did for `lli`).

### `tests/backend/pass/unwind.vx`

#### [MODIFY] \[unwind.vx\](file:///Users/adityak/go/Vx/tests/backend/pass/unwind.vx)

- Update the `// EXPECT:` lines to expect the actual function names (`f1`, `f2`, `do_test_chain`, etc.) instead of `<unknown>`.

## Verification Plan

### Automated Tests

- Run `cargo run --bin vxc -- tests/backend/pass/unwind.vx` and verify that the output perfectly matches the new `EXPECT` checks without any `<unknown>` frames.
