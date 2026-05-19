# Compile-Time Evaluation (`comptime`) & Assertions

To align with Vx's philosophy of "Ease of Verified Computation," we will implement first-class assertion primitives. Rather than relying on macros, we will introduce a `comptime` block and a built-in `assert` keyword. This enables verifiable mathematical boundaries directly in the AST, supporting both runtime safety and ahead-of-time evaluation.

## User Review Required

Please review the proposed approach for compile-time constant evaluation below. In this initial iteration, `sema.rs` will include a basic interpreter to statically evaluate expressions inside `comptime` blocks. Does this align with your vision for `constexpr`-style evaluation?

## Proposed Changes

### 1. Lexer & AST Updates

- **Tokens**: Add `TokenType::Comptime` and `TokenType::Assert` to `src/lexer.rs`.
- **AST Nodes**:
  - `Statement::Comptime(Vec<Statement>)`: Represents a block of code guaranteed to execute entirely during semantic analysis.
  - `Statement::Assert(Box<Expr>, Option<String>)`: A built-in assertion statement.

### 2. Parser Enhancements (`src/parser.rs`)

- Parse `assert(condition, "optional message");` as a top-level or block-level statement.
- Parse `comptime { ... }` blocks, allowing any valid statements inside.

### 3. Semantic Analysis & Constant Evaluation (`src/sema.rs`)

- Add a static evaluation context (`eval_expr`) inside `Sema` to compute values at compile-time.
- When visiting a `Statement::Comptime`, `sema.rs` will execute the statements. If an `assert` is encountered inside `comptime`, the compiler will statically evaluate the condition. If it is `false`, compilation fails with the custom message.
- For non-comptime `assert`s, `sema.rs` simply verifies that the condition evaluates to the `Bool` (or `i1`) type.

### 4. MLIR Code Generation (`src/codegen.rs`)

- **Runtime Asserts**: Lower `Statement::Assert` into an MLIR `cf.assert` operation (or a conditional branch calling `abort()`).
- **Comptime Blocks**: Strip `Statement::Comptime` blocks during codegen. Since they are guaranteed to have executed during Semantic Analysis, they do not produce runtime MLIR overhead (zero-cost abstraction).

## Verification Plan

1. Add an integration test `tests/backend/pass/assert_runtime.vx` to verify MLIR aborts gracefully.
1. Add a compiler failure test `tests/backend/fail/assert_comptime.vx` to verify that `comptime { assert(false); }` successfully halts the compilation process.
