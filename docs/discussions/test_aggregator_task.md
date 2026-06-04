# Task Tracker: Test Aggregator Refactoring

- `[x]` Introduce `run_directory_tests` helper function.
- `[x]` Introduce `run_shell_tests` helper function.
- `[x]` Refactor `test_frontend_pass` to use `run_directory_tests`.
- `[x]` Refactor `test_frontend_fail` to use `run_directory_tests` + `run_shell_tests`.
- `[x]` Refactor `test_frontend_pass_formal_verification` to use `run_directory_tests`.
- `[x]` Refactor `test_frontend_fail_formal_verification` to use `run_directory_tests` + `run_shell_tests`.
- `[x]` Refactor `test_frontend_fail_unimplemented_smt` to use `run_directory_tests` + `run_shell_tests`.
- `[x]` Refactor `test_middle_end` to use `run_directory_tests`.
- `[x]` Refactor `test_middle_end_fail` to use `run_directory_tests`.
- `[x]` Refactor `test_backend` to use `run_directory_tests`.
- `[x]` Refactor `test_backend_fail` to use `run_directory_tests` + `run_shell_tests`.
- `[x]` Refactor `test_optimizations` to use `run_directory_tests`.
- `[x]` Refactor `test_backend_pass_autodiff` to use `run_directory_tests`.
- `[x]` Run `cargo test` to verify normal functionality.
- `[x]` Intentionally introduce an error to ensure `cargo test` collects all failures and does not abort early.
