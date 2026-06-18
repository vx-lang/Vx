# `src/bin` Refactoring Tasks

- `[x]` **Update Dependencies**

  - `[x]` Add `anyhow = "1.0"` to `Cargo.toml`.

- `[x]` **Refactor `vx-format`**

  - `[x]` Replace manual arg parsing with `clap::Parser`.
  - `[x]` Use `anyhow::Result` within the parallel loop for error reporting.

- `[x]` **Refactor `vx-opt`**

  - `[x]` Use `std::env::args_os()` instead of `std::env::args()`.
  - `[x]` Convert OsStr directly to `CString` using `OsStrExt`.

- `[x]` **Refactor `melior_test2` & `test_loc`**

  - `[x]` `melior_test2.rs`: Wrap in `Result<(), Box<dyn Error>>` and handle `Option`.
  - `[x]` `test_loc.rs`: Selectively register `llvm` and `builtin` dialects instead of `register_all`.
  - `[x]` `test_loc.rs`: Replace `.unwrap()` with `.expect()` for better error messages.

- `[x]` **Refactor `cargo-vx-bench`**

  - `[x]` Switch to `anyhow::Result` for the main loop.
  - `[x]` Implement `rayon::par_iter()` for parallel test execution.
  - `[x]` Inject test harness via AST merging instead of string regex replacement.
  - `[x]` Avoid cloning the entire `ast` node.

- `[x]` **Verification**

  - `[x]` Run `cargo fmt` and `cargo check`.
  - `[x]` Mark review files as done.
