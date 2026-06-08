# Implementation Plan: Loop `continue` Support

## Goal

Implement support for the `continue` keyword in `loop` statements within the `Vx` compiler, allowing users to skip to the next iteration of the nearest enclosing loop. This also required verifying complex combinations of `continue`, `break`, and `return` within nested loops.

## Proposed Changes

### Lexer (`src/lexer.rs`)

- [x] Add a `Continue` variant to `TokenType`.
- [x] Register `continue` keyword in the keyword matching logic.

### AST (`src/ast/stmt.rs`)

- [x] Add a `Continue` variant to the `Statement` enum.
- [x] Define a `ContinueStmt` struct containing a span.

### Parser (`src/parser/stmt.rs`)

- [x] Implement parsing for `TokenType::Continue`, returning `Statement::Continue(ContinueStmt)`.

### Sema/Resolution (`src/ast/resolve.rs`, `src/sema/stmt.rs`)

- [x] Add `Continue` matching in `resolve_statement` (no variable resolution needed, just typechecking continuation).
- [x] Implement `typecheck_statement` for `ContinueStmt`. Ensure type validation continues to the next loop scope or ensures basic validity.

### Codegen (`src/codegen/lower.rs`, `src/codegen/generator.rs`, `src/codegen/break_utils.rs`)

- [x] Extend `MeliorGenerator` state to include `continue_flags: Vec<Value<'c, 'c>>` matching the existing `break_flags`.
- [x] Modify `LoopStmt::lower`:
  - Allocate a `continue_ptr` flag via `memref.alloca` alongside the `break_ptr` flag.
  - Push the flag to `continue_flags`.
  - In `before_block`, explicitly store `false` to the `continue_ptr` since `continue` flags only skip the *current* iteration, but shouldn't break the outer loop condition entirely.
  - After generating the condition and forwarding to the `after_block`, pop the flag.
- [x] Implement `ContinueStmt::lower`:
  - Get the latest `continue_ptr` flag from `gen.continue_flags`.
  - Store `true` to this pointer.
- [x] Modify `generate_statements_with_break_guard` in `break_utils.rs` to take the `continue_ptr`. If *either* the `break_ptr` or `continue_ptr` is true, subsequent statements in the block should not execute. We implement this using an `arith.ori` operation to conditionally jump over following statements using an `scf.if` region.

### Tests (`tests/frontend/pass/control_flow_nested.vx`)

- [x] Develop testing infrastructure using nested loops containing combinations of `continue`, `break`, and `return`.

______________________________________________________________________

# Walkthrough

## What Was Accomplished

1. Successfully integrated the `continue` statement into the lexer, AST, parser, semantic analyzer, and MLIR generation pipeline.
1. Handled the complicated state-machine of MLIR strict basic blocks using `scf.while` and `memref.alloca` pointer flags by pushing boolean flags across loops.
1. Created a nested loop rigour test in the tests directory that thoroughly tests the `break`, `continue`, and `return` logic to make sure the loops don't incorrectly jump scopes!

## Verification Plan

1. `cargo test` successfully compiled and passed, specifically with `test_frontend_pass` now compiling nested control flows properly!
1. The `control_flow_nested.vx` program generates MLIR which guarantees control flow guards evaluate properly.
