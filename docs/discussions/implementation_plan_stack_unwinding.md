# Stack Unwinding Implementation Plan

Implement stack unwinding to safely and deterministically clean up resources (e.g. `drop` calls) and surface error states when a panic or failure occurs during runtime execution.

## Proposed Changes

We have two primary options for implementing stack unwinding in a custom MLIR-based compiler targeting LLVM:

1. **Zero-Cost DWARF Unwinding (`libunwind`)**
   - Emit `llvm.invoke` instead of `llvm.call` for function invocations.
   - Emit `llvm.landingpad` to catch exceptions.
   - Use `libunwind` to throw exceptions (like `__cxa_throw` in C++ or `panic!` in Rust).
   - This provides "zero-cost" exceptions (no overhead on the happy path) but is more complex to wire up.

2. **`setjmp` / `longjmp` Based Unwinding**
   - Push to a global thread-local "unwind stack" containing `jmp_buf` and closure pointers for `drop()` calls.
   - Upon panic, we pop closures and call them, eventually calling `longjmp` back to the catch handler.
   - This is easier to implement directly in the compiler frontend but has a slight overhead for registering handlers at runtime.

### Recommended Approach
Given that we want to safely clean up resources and call `.drop()` explicitly in `lower.rs`, I recommend using **LLVM's native exception handling (`llvm.invoke` / `llvm.landingpad`)**. 

- **`src/codegen/lower.rs`**: 
  - Change `func.call` to `llvm.invoke` for functions that might panic.
  - Generate a `landingpad` block that calls `.drop()` on locally allocated variables and then resumes the unwind using `llvm.resume`.
- **`stdlib/rust_core/src/ffi/rt.rs`**:
  - Implement a `vx_panic` FFI function that uses `libunwind` (`_Unwind_RaiseException`) to trigger the unwind process.

## Open Questions

> [!IMPORTANT]
> **Unwinding Strategy Decision**
> Should we implement this via native LLVM DWARF Unwinding (`llvm.invoke` / `llvm.landingpad` + `libunwind`), or would you prefer a simpler, explicit `setjmp/longjmp` strategy managed within the Vx AST/Frontend? 

> [!WARNING]
> **Standard Library Support**
> If we use `libunwind`, it might require linking against the system's C++ standard library (`libc++abi` / `libgcc_s`). Are you okay with adding this linkage to the build?

## Verification Plan

### Automated Tests
- Create a `tests/backend/pass/unwind.vx` test.
- The test will allocate a struct with a custom `drop()` method that prints to the console or increments a global counter.
- It will call a function that triggers a `panic!`.
- We will verify that the custom `drop()` method is called during the unwind process.
