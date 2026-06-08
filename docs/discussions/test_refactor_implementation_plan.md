# Test Runner Error Aggregation

The testing framework currently halts immediately upon encountering the first test failure, obscuring other potential failures in the same suite. Additionally, failures are generated via `panic!` which creates messy output and stops the parallel thread execution rather than cleanly accumulating errors. The goal is to accumulate all failures and present them in a single comprehensive output block at the end of the test directory run.

## Proposed Changes

### 1. Refactor Shell Tests

Extract the duplicate shell execution loop inside tests like `test_backend_fail` and `test_frontend_fail` into a shared `run_shell_tests` function. Change the `assert!(status.success())` block to instead return `Err` capturing the stderr/stdout.

### 2. Refactor Runner Helpers

Update `run_frontend_test`, `run_backend_test`, `run_middle_end_test`, and `run_optimization_test` to change their signatures to `Result<(), String>`.

### 3. Implement `run_directory_tests`

Create a high-order function `run_directory_tests` that:

- Uses `rayon`'s `into_par_iter()`
- Executes the provided closure returning `Result<(), String>`
- Catches any unexpected panics using `std::panic::catch_unwind`
- Collects all error messages and panics at the very end with a unified summary.

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure all existing tests pass and no functionality is broken.
- Introduce intentional syntax errors in tests to verify that the runner collects multiple errors instead of short-circuiting.
