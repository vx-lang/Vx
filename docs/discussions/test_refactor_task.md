# Test Aggregator Refactoring Tasks

- [x] Analyze current test runner setup (`run_frontend_test`, `run_backend_test`, `run_shell_tests`, `run_optimization_test`, etc.)
- [x] Create `run_directory_tests` helper in `tests/compile_test.rs`
- [x] Integrate Rayon's `into_par_iter` to parallelize test execution within `run_directory_tests`
- [x] Add robust failure aggregation using `Result<(), String>` returning test closures
- [x] Refactor all existing runners to return `Result<(), String>` instead of calling `panic!`
- [x] Refactor all existing runners to replace `assert!` with `if !cond { return Err(...) }`
- [x] Refactor duplicate `RUN:` line shell execution into `run_shell_tests`
- [x] Update `test_frontend_fail` to use `run_directory_tests`
- [x] Update `test_backend_fail` to use `run_directory_tests`
- [x] Fix specific compiler warnings in `src/sema/prover.rs`
- [x] Run `cargo test` and `cargo fmt` locally to ensure full suite passes
- [x] Create Walkthrough, Task, and Implementation Plan documents and archive them in `docs/discussions/`
- [x] Commit changes using `git commit`
