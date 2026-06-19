# Driver Refactoring Plan

This plan addresses the feedback in `Vx_src_driver.md` to improve the readability and efficiency of the `CompilerDriver`.

## Proposed Changes

### `src/driver.rs`

- **Modularize Pipeline**: Split the monolithic `execute_vx_pipeline` into distinct methods:
  - `load_and_expand`
  - `handle_parse_only`
  - `run_semantic_analysis`
  - `execute_vx_pipeline` will act as a concise orchestrator.
- **In-Memory MLIR/LLVM Translation**:
  - Update `translate_to_llvm_ir` to use `Stdio::piped()` instead of temporary files, avoiding I/O overhead.
  - Keep `apply_mlir_opt` using files since it invokes C++ FFI `run_vx_opt` (which reads process stdin if `-` is used), but switch to the `tempfile` crate for secure and collision-free temporary file creation.
- **Better CLI Alias Management**:
  - Refactor `DriverOptions` with `clap`'s `overrides_with` and `conflicts_with` where possible to minimize manual alias processing.
- **Avoid AST Clones**:
  - Pass references `&[&Program]` to `GlobalAstEnv` instead of creating heavy clones (`clone_signature`).

### `src/sema/env.rs`

- Update `GlobalAstEnv` to support building from an array of references: `pub fn build_from_refs(modules: &[&Program]) -> Self`.
- Modify the `GlobalAstEnv::build` method to be a wrapper around `build_from_refs` for backward compatibility, or update all callers.

## Open Questions

- We do not currently have `tempfile` in `Cargo.toml`. Is it acceptable to add the `tempfile` dependency, or would you prefer a custom temporary file generator? I assume we can add `tempfile = "3.10"` to `Cargo.toml`.

## Verification Plan

- Run `cargo test` to ensure that all driver integration tests and MLIR tests pass.
- Run `cargo clippy` and `cargo fmt`.
