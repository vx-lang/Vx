# JIT Stack Trace Symbol Resolution

## The Problem

The JIT compiler was failing to resolve the names of functions in the execution stack trace when panics occurred, printing `<unknown>` instead. This occurred because `vxc` relied on LLVM's `lli` as a subprocess for executing LLVM IR. The `lli` tool executes machine code dynamically within in-memory buffers without registering the memory segments with the standard system dynamic loader (`dyld` on macOS, or `/proc/self/maps` on Linux). As a result, the `dladdr()` call used internally by `std::backtrace` or `backtrace-rs` to discover symbol names was returning no matches.

## The Solution

Rather than patching `backtrace-rs` or attempting complex custom JIT event listener architectures in-memory (which still wouldn't interface nicely with standard tools), we altered the execution methodology of the `RunJit` backend in `vxc`.

### Native Compilation Sub-process

In `src/jit.rs`, we replaced the `lli` execution path with:

1. `llc`: Compiles the optimized LLVM IR (`temp_opt_{uid}.ll`) into a native object file (`temp_opt_{uid}.o`).
1. `clang`: Links the object file with necessary libraries (`libmlir_c_runner_utils`, `libvx_std_core`, etc.) into a temporary native executable (`temp_{uid}.out`).
1. Executes the native binary, ensuring that standard unwinding and debug trace features work seamlessly, exactly as they do in natively compiled binaries.

### Backtrace Fallback

In addition, we ensured that `RUST_BACKTRACE=1` is automatically set in the executable environment if it wasn't already provided by the user, so `vx_panic` calls will automatically trigger the backtrace traceout.

## Verification

- We updated `tests/backend/pass/unwind.vx` to assert on actual function names (`f1`, `f2`, `do_test_chain`, `do_test_lambda`, etc.) rather than `<unknown>`.
- We successfully ran the full `cargo test` suite with all 13 `compile_test` integrations passing perfectly.
- Lints and code formatting rules were successfully executed and everything was securely committed into the repository under the `01f3bfe` commit hash.
