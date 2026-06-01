# Implementation of `std::env::args` in Vx

This walkthrough summarizes the implementation for providing native Rust-like `std::env::args()` capability inside the Vx language.

## 1. Native FFI standard library functions
In the `vx_std_core` standard library, we implemented `vx_env_args_count()` and `vx_env_arg()` as `extern "C"` functions.
- These query `std::env::args()` natively to get real command line arguments.
- **JIT vs AOT unified architecture**: We created intelligent runtime filtering that detects if execution is happening via the JIT engine (`lli`). If executed via JIT, it filters out `lli`, internal flags, and the compiled `.ll` module, mapping the remaining arguments transparently.
- Memory handling properly wraps strings as `CString` dynamically and safely exposes the `*mut i8` to the Vx frontend.

## 2. Vx standard library interface
We created a new standard library module `tests/modules/env.vx`:
- It exposes clean, safe `args_count()` and `arg(idx)` functions so that developers don't have to interact with `unsafe` pointer bindings directly for environment reading.

## 3. Compiler driver integration
We exposed trailing arguments logic within the `vxc` compiler via `clap` (`#[arg(last = true)]`).
These arguments are seamlessly passed through the `execute_mlir` JIT boundary natively into `lli`.

## Verification
A new script `test_args.vx` loops over all arguments using our `env` module and successfully printed them during manual testing `vxc test_args.vx -- --hello world`.
