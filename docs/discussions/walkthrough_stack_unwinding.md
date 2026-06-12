# Walkthrough: Implementing JIT Stack Unwinding

## Objective
Implement stack unwinding for the JIT backend to properly propagate runtime panics from JIT-compiled MLIR code back to the native Rust environment.

## Changes Made
1. **Disabled `mcjit` for `lli` Execution**:
   - `mcjit` on macOS prevented the unwinder (`libunwind`) from locating dynamic `eh_frame` information. We switched to the default ORC JIT, which uses standard dynamic frame registration.
   
2. **Introduced `vx_catch_unwind`**:
   - The native FFI panic function `vx_panic` (which utilizes `std::panic::panic_any`) correctly begins the unwinding process (Phase 1). However, since there was no native catch frame wrapping the JIT invocation, `libunwind` naturally escalated the unwound exception up the process stack, ultimately crashing since it went uncaught.
   - We introduced `vx_catch_unwind` in `stdlib/rust_core/src/ffi/rt.rs` to serve as an AOT-compiled Rust frame that uses `std::panic::catch_unwind`. This securely catches exceptions crossing the JIT-to-Native boundary.

3. **Reverted `llvm.invoke` logic**:
   - While investigating MLIR's exception handling, we realized that since automatic `Drop` closures are not yet implemented in AST lowering, generating `llvm.landingpad` nodes that only yield `llvm.resume` is semantically redundant.
   - We returned `lower.rs` to generate standard LLVM `call` instructions. Because the target generates `uwtable` (unwind tables) globally, the unwinder transparently walks past JIT frames that lack specific landing pads and successfully bubbles the exception straight up to the `vx_catch_unwind` boundary.

## Verification
- Wrote `tests/backend/pass/unwind.vx` to simulate a panic inside a JIT function and confirmed the `catch_unwind` mechanism traps it cleanly without a Phase 1 fatal error (`error 5`).
- Passed all tests cleanly (`cargo test --test compile_test`).

## Follow-up Work
When automatic resource tracking (`drop`) logic is implemented, we can reintroduce `llvm.invoke` and `llvm.landingpad` into `lower.rs` to selectively run `.drop()` routines during unwind passes.
