# Walkthrough: Strict Tensor Initialization

## Summary of Changes

We addressed the issue where `Tensor` initialization silently defaulted its inner generic element type to `f32` when none was provided. This behavior was suppressing potentially critical type mismatch errors and making it harder to track down bugs involving missing types.

1. **Strict Type Enforcement in Sema**:

   - Modified `check_functioncall_expr` in `src/sema/expr.rs` to stop silently defaulting `Tensor` initialization to `ElementType::F32`.
   - Now, if generic arguments are missing or invalid, we push a formal semantic error using `self.errors.push(...)`.
   - Reverted the experimental `Result<Type, ()>` refactor on the signature of `check_functioncall_expr` to maintain the compiler's resilience, choosing to push errors and continue semantic analysis instead of hard-failing the type checking of the entire expression immediately. This allows the compiler to gather multiple errors simultaneously.

1. **Complex Generic Argument Parsing**:

   - Fixed the generic argument parsing logic within `src/sema/expr.rs` to properly handle methods called on generic types.
   - For example, `Tensor<f32>::zeros` string slices were originally misparsed. The logic was adjusted to find the last `>` using `.rfind('>')` instead of the first `>`, properly handling the `base_name` separation.

1. **Updating the Codebase & Tests**:

   - Migrated all test `.vx` programs (under `tests/frontend/pass/`, etc.) from using `Tensor(...)` to `Tensor<f32>(...)` using a python script.
   - Fixed `tests/integration_test.rs` which had hardcoded strings using `Tensor([10])` and `Tensor(2, 2)` to `Tensor<f32>([10])` and `Tensor<f32>(2, 2)`.

## Verification

- Ran the full test suite via `cargo test`.
- All tests in `tests/compile_test.rs`, `tests/integration_test.rs`, and unit tests within `src/sema/expr.rs` now pass flawlessly.
- Pre-commit formatting and clippy lints run successfully.
