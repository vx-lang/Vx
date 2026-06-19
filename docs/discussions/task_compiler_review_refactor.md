# Driver Refactoring Tasks

- `[x]` Update `src/sema/env.rs` to support `build_from_refs`
  - `[x]` Refactor `GlobalAstEnv` struct to work with borrowed programs effectively or add `build_from_refs` without modifying the struct definitions too much (it currently stores `&'a StructDecl`, etc. so taking `&'a Program` is natural).
- `[x]` Add `tempfile` dependency to `Cargo.toml`
- `[x]` Update `src/driver.rs`
  - `[x]` Clean up `DriverOptions` using `clap` attributes.
  - `[x]` Refactor `execute_vx_pipeline` into smaller stages.
  - `[x]` Update `apply_mlir_opt` to use `tempfile::NamedTempFile`.
  - `[x]` Update `translate_to_llvm_ir` to use `Stdio::piped()`.
- `[x]` Run tests and verify the changes.
- `[x]` Review `Vx_src_parallel_architecture_verifier.md`
- `[x]` Review `Vx_src_main.md`
- `[x]` Review `Vx_src_pipeline.md`
- `[x]` Review `Vx_src_resolver.md`
- `[x]` Review `Vx_src_scratch.md`
- `[x]` Review `Vx_src_session.md`
- `[x]` Verify that all 51 markdown files in `vx-review/review` are marked `done`.
- `[x]` Run formatting and final `cargo test` checks.
