# JIT Stack Trace Symbol Resolution

## The Problem

The JIT compiler was failing to resolve the names of functions in the execution stack trace when panics occurred, printing `<unknown>` instead. This occurred because `vxc` relied on LLVM's `lli` as a subprocess for executing LLVM IR. The `lli` tool executes machine code dynamically within in-memory buffers without registering the memory segments with the standard system dynamic loader (`dyld` on macOS, or `/proc/self/maps` on Linux). As a result, the `dladdr()` call used internally by `std::backtrace` or `backtrace-rs` to discover symbol names was returning no matches.

## The Solution

Rather than patching `backtrace-rs` or attempting complex custom JIT event listener architectures in-memory (which still wouldn't interface nicely with standard tools), we altered the execution methodology of the `RunJit` backend in `vxc`.

### 1. DWARF Debugging Support in JIT Execution

We replaced the execution pipeline in `src/jit.rs` which used LLVM's `lli` with a direct compile, link, and execute pipeline using `llc` and `clang` to produce a full Mach-O executable containing DWARF sections. This successfully restored our ability to see resolved function names and line numbers instead of `<unknown>` when `std::panic::catch_unwind` kicks in!

### 2. Runtime Panic Handlers

We implemented proper runtime aborts, `vx_catch_unwind`, and `vx_panic` handling routines in `stdlib/rust_core/src/ffi/rt.rs`.

### 3. JIT Exit Code Propagation

We discovered that tests written to intentionally crash with segmentation faults were incorrectly succeeding, because the JIT compilation driver was inadvertently converting the binary's `SIGSEGV` exit codes into a `Ok(())` Result, causing `vxc` to exit with `0`. We modified `src/jit.rs` to propagate non-zero exit codes from the JIT-executed binary via `Err()`, correctly causing `vxc` to return a non-zero exit code.

### 4. Hardware Safety Negative Testing

We created three new negative test cases inside `tests/backend/fail/` leveraging `// RUN: not vxc %s`:

- `null_deref.vx`: Intentional `nullptr` dereference.
- `box_oob.vx`: Out of bounds dereference against memory initialized via `Box::new()`.
- `vec_oob.vx`: Out of bounds dereference against heap memory behind a `Vec<T>`.

We verified that by using very large indices (e.g. `100000000`) for the out of bounds tests, we hit unmapped virtual memory and guarantee a `SIGSEGV`, causing the tests to pass successfully in our `RUN: not` assertion framework.

### Backtrace Fallback

In addition, we ensured that `RUST_BACKTRACE=1` is automatically set in the executable environment if it wasn't already provided by the user, so `vx_panic` calls will automatically trigger the backtrace traceout.

## Verification

- We updated `tests/backend/pass/unwind.vx` to assert on actual function names (`f1`, `f2`, `do_test_chain`, `do_test_lambda`, etc.) rather than `<unknown>`.
- We successfully ran the full `cargo test` suite with all 13 `compile_test` integrations passing perfectly.
- Lints and code formatting rules were successfully executed and everything was securely committed into the repository under the `01f3bfe` commit hash.

## `lower.rs` Structural Refactoring

The massive ~4800-line monolithic `lower.rs` file was split into a more manageable module structure:

1. `src/codegen/lower/expr.rs`
1. `src/codegen/lower/stmt.rs`
1. `src/codegen/lower/control_flow.rs`
1. `src/codegen/lower/tensors.rs`
1. `src/codegen/lower.rs` (Now acts as the module boundary and houses the core `LowerToMelior` trait).

We also cleaned up all the unused imports (`std::collections::HashMap`, `ArrayAttribute`, etc.) and fixed an ambiguous glob re-export in `src/codegen/mod.rs` by making the submodules private. All unit, integration, and lint tests successfully passed, maintaining full functional parity while greatly improving the maintainability of the compiler's backend lowering phase.
