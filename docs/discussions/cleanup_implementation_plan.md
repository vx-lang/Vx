# Code Cleanup and Test Consolidation

This plan addresses several outstanding requests related to codebase hygiene, organization, and formatting.

## 1. Refactoring Imports / Scoping

There are over 100 instances of verbose and repetitive scoping in the codebase, primarily `crate::ast::Type::*`, `crate::ast::Topology::*`, `crate::ast::Expr::*`, etc. This makes the code harder to read.

### Proposed Changes

- Create a Python refactoring script to automatically replace inline usages of `crate::ast::Type`, `crate::ast::Expr`, `crate::ast::Topology`, `crate::ast::UnaryOp`, `crate::ast::BinaryOp`, `crate::ast::ElementType`, and other repetitive paths with their direct names (e.g. `Type`, `Expr`).
- Ensure `use crate::ast::*;` or specific `use` statements are added to the top of affected files (primarily `src/codegen/generator.rs` and `src/codegen/lower.rs`).
- The script will be tested against `cargo clippy` and `cargo test` to ensure it doesn't introduce naming collisions.

## 2. Consolidating Test Files

Currently, we have generated many individual files for negative testing (`test_comptime_if_branch_type_mismatch.vx`, `test_comptime_if_condition_type_mismatch.vx`, etc.).

### Proposed Changes

- Group related tests into single, well-structured files to reduce clutter.
- For instance, all `comptime if` tests can be combined into `tests/frontend/fail/comptime_if.vx`, provided the test runner can handle multiple failing constructs in one file, or we can just keep them as separate files inside a `comptime_if/` subdirectory.

## 3. Formatting all files

- After all code changes, run `cargo fmt` for Rust files.
- Run `vx-format` for all `.vx` test files.
- Run `mdformat` for Markdown files.
