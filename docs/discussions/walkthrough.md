# Formal Verification & `Verified<T>` Testing Walkthrough

We have successfully restructured our formal verification tests and significantly improved the compiler's resilience against edge cases for `Verified<T>` semantics.

## What Changed?

- We created a dedicated failure testing suite for formal verification within `tests/backend/fail/formal_verification/`.
- We added `tests/backend/pass/formal_verification/` to explicitly handle formal verification capabilities in edge-case compiler passes.
- Modified the `tests/compile_test.rs` compiler runner to traverse, catch, and assert panics seamlessly within the formal verification failure directory.

## Testing Suite Highlights

> [!TIP]
> The formal verification engine uses MLIR dialect translations to emit specific SMT constraints. These tests assert that the translations enforce standard compiler invariants prior to code-gen.

We added the following comprehensive test cases to protect against regressions:

1. **`verified_assignment.vx`**: Asserts that an unverified `Tensor<f32>` cannot be assigned to a `Verified<Tensor<f32>>` strictly.
1. **`verified_stripping.vx`**: Validates that functions expecting raw tensors reject `Verified` structs, guaranteeing that `Verified` wrappers must be explicitly unpacked, preventing implicit coercion pitfalls.
1. **`smt_generic_mismatch.vx`**: Exercises the formal verification generic pass to ensure that SMT constraints correctly capture generic shape logic (e.g. `n == m` assert failing).
1. **`failed_assertion.vx`**: Confirms that standard logical asserts within the code (e.g., `assert(x > y)`) are effectively evaluated and halt compilation if statistically unprovable prior to instantiation.
1. **`verified_coercion.vx`**: Asserts that successfully verified tensor wrappers *can* be correctly passed into another verified variable natively without issue.

## Validation Results

All tests were executed natively via `cargo test`, correctly capturing compiler type-checking assertions prior to JIT execution:

```rust
test test_backend_fail_formal_verification ... ok
test test_backend_pass_formal_verification ... ok
test test_frontend_pass_modules ... ok
```

The entire workflow, including `autodiff` structure mapping and `Tensor<T>` generic testing for MLIR generation, is complete.
