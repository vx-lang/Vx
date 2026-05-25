# Verified<T> Testing and Directory Restructuring

This plan outlines the next steps to ensure `Verified<T>` is robust against edge cases and formal verification tests are properly organized.

## Open Questions

> [!IMPORTANT]
> You mentioned "move all the formal verification tests into a separate directory just like autograd". Currently, we don't have a directory named `autograd`, but we have `tests/backend/pass/autodiff` and the formal verification tests are already separated in `tests/backend/pass/formal_verification`.
> Do you want me to:
>
> 1. Move them to a top-level `tests/formal_verification/` (with `pass` and `fail` subdirectories)?
> 1. Create `tests/backend/fail/formal_verification/` alongside the existing `pass` directory to store the new negative edge-cases?
>
> My proposal (detailed below) is to use option 2, keeping it consistent with the `backend/pass/` and `backend/fail/` structures. Please let me know if you prefer option 1!

## Proposed Changes

### 1. Robust Testing for `Verified<T>` Edge Cases

We will introduce a comprehensive suite of ~10-12 tests to guarantee the formal verification type logic is bulletproof.

#### [NEW] `tests/backend/fail/formal_verification/verified_assignment.vx`

Tests that an unverified `Tensor<T>` cannot be assigned to a `Verified<Tensor<T>>`.

#### [NEW] `tests/backend/fail/formal_verification/verified_stripping.vx`

Tests that `Verified<Tensor<T>>` does not implicitly coerce down to `Tensor<T>` when passed into strict functions (unless explicitly unwrapped or expected by design).

#### [NEW] `tests/backend/fail/formal_verification/smt_generic_mismatch.vx`

Tests that generic formal parameters that are instantiated but mismatch the pre-conditions/post-conditions trigger static compilation errors.

#### [NEW] `tests/backend/fail/formal_verification/failed_assertion.vx`

Tests that an explicitly incorrect `assert` within a function body properly halts the compiler during the formal verification pass.

#### [NEW] `tests/backend/pass/formal_verification/verified_coercion.vx`

Positive tests ensuring that valid operations on `Verified<T>` values propagate the verified status correctly.

### 2. Directory Migration & Test Runner Updates

If we stick with `backend/pass` and `backend/fail`:

#### [NEW] `tests/backend/fail/formal_verification/`

We will create this directory to house all the new failure-case tests.

#### [MODIFY] `tests/compile_test.rs`

Update the Rust test runner to explicitly traverse and run the `fail` tests for `formal_verification`.

```rust
#[test]
fn test_backend_fail_formal_verification() {
    // Scaffold test runner for tests/backend/fail/formal_verification
}
```

## Verification Plan

### Automated Tests

- Run `cargo test test_backend_pass_formal_verification` to ensure existing passes work.
- Run `cargo test test_backend_fail_formal_verification` to ensure all edge cases are properly caught and flagged by the compiler.
- Validate that MLIR is not emitted when formal verification constraints fail.
