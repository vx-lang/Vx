# Refactor Test Runner to Collect Errors

The test runner in `tests/compile_test.rs` currently iterates over `.vx` files using Rayon's parallel iterator `into_par_iter().for_each(...)`. Inside this loop, it uses `assert!` or `panic!` to flag failures. However, panicking inside a Rayon iterator aborts the iteration, preventing the rest of the tests in that directory from running.

We need to collect all test failures and report them at the end.

## Proposed Changes

### `tests/compile_test.rs`

We will refactor all test directory loops in `compile_test.rs`.

For each test function (e.g., `test_frontend_pass`, `test_frontend_fail`, `test_backend_fail`, etc.):

1. Change `entries.into_par_iter().for_each(|entry| { ... })` to `entries.into_par_iter().filter_map(|entry| -> Option<String> { ... }).collect::<Vec<_>>()`
1. Wrap the invocation of `run_frontend_test`, `run_backend_test`, `run_optimization_test`, etc., inside `std::panic::catch_unwind`.
1. For tests that "shell out" (execute `sh -c`), replace the `assert!(status.success())` with returning a failure string instead.
1. Replace `panic!("Test with no RUN line...")` with returning a failure string.
1. After the `filter_map` collects all failures, assert that the error list is empty:
   ```rust
   if !errors.is_empty() {
       panic!("The following tests failed:\n{}", errors.join("\n\n"));
   }
   ```

#### [MODIFY] \[compile_test.rs\](file:///Users/adityak/go/Vx/tests/compile_test.rs)

- Refactor `test_frontend_pass`
- Refactor `test_frontend_fail`
- Refactor `test_frontend_pass_formal_verification`
- Refactor `test_frontend_fail_formal_verification`
- Refactor `test_frontend_fail_unimplemented_smt`
- Refactor `test_middle_end`
- Refactor `test_middle_end_fail`
- Refactor `test_backend`
- Refactor `test_backend_fail`
- Refactor `test_optimizations`
- Refactor `test_backend_pass_autodiff`

## Open Questions

None. This strictly implements the user's request.

## Verification Plan

We will intentionally introduce a syntax error in two unrelated `.vx` files inside `tests/frontend/fail/` to ensure that `cargo test` runs both of them and reports both failures in the final panic message, proving that execution does not abort on the first failure.
