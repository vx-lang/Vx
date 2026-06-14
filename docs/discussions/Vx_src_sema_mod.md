# Code Review: `src/sema/mod.rs`

This file serves as the module declaration for the semantic analysis phase. It exports:

- `env` (`GlobalAstEnv`, unification logic, type representations)
- `expr` (expression type checking, generic instantiation, hardware topology tracking)
- `stmt` (statement checking, SMT-based contract verification, borrow tracking)
- `prover` (the `SmtProver` wrapper for theorem proving)

## Noteworthy Tests

The module itself contains integration-level tests for semantic edge cases:

- `test_sema_distributed_matmul`: Tests the `spawn on` syntax and cross-topology assignments, verifying that valid transfers are accepted by the `TransferCostGraph`.
- `test_sema_type_mismatch`: Standard error path testing for function return types.
- `test_sema_struct_and_pointers`: Verifies that pointers and references (`*mut`, `&mut`) can be safely created and dereferenced inside `unsafe` blocks.
- `test_sema_extern_unsafe`: Verifies the fundamental rule that `extern` FFI functions (`malloc`) are intrinsically unsafe and calling them without an `unsafe` block triggers a compile-time error.
- `test_sema_as_ptr_and_len`: Validates intrinsic properties on Tensors (`as_ptr()`, `as_mut_ptr()`, `len()`).

## Design Critique

The module is straightforward and clean. No immediate refactoring is required. It efficiently segregates tests related to module-level interactions (cross-file tests) from localized unit tests.
