# Option 1: Full AST Struct Refactor

The goal is to decentralize the MLIR lowering logic (`LowerToMelior`) by delegating it directly to the AST nodes. Since Rust does not allow implementing traits on individual enum variants, we will refactor the `Expr` and `Statement` enums by extracting their variants into distinct structs.

This will provide a highly scalable, robust architecture where each node owns its parsing, type-checking (future), and MLIR lowering logic.

## User Review Required

> [!WARNING]
> This is a massive refactor. It will touch almost every file in the frontend (`ast.rs`, `parser.rs`, `sema.rs`, and `melior_codegen.rs`). The compiler will be temporarily broken while we migrate all instantiations and match expressions.
> Please review the proposed structs and confirm we should proceed with this heavy-duty refactoring.

## Proposed Changes

### `src/ast.rs`

We will introduce a distinct struct for every current variant of `Expr` and `Statement`.

#### [MODIFY] ast.rs

- Create `IdentifierExpr`, `NumberExpr`, `StringLiteralExpr`, `BinaryOpExpr`, `FunctionCallExpr`, `StructInitExpr`, `MemberAccessExpr`, `IfExpr`, `ForLoopStmt`, `LetDeclStmt`, etc.
- Modify the `Expr` enum to wrap these structs:
  ```rust
  pub enum Expr {
      Identifier(IdentifierExpr),
      Number(NumberExpr),
      BinaryOp(BinaryOpExpr),
      // ...
  }
  ```
- Modify the `Statement` enum similarly.
- Update `Expr::span()`, `Expr::substitute()`, and `Statement::substitute()` to delegate to the underlying structs.

### `src/parser.rs`

#### [MODIFY] parser.rs

- Update all parser functions (e.g., `parse_expr`, `parse_statement`, `parse_binary_op`) to instantiate the new structs before wrapping them in the enum variants.
- Example: `Expr::BinaryOp(BinaryOpExpr { lhs, op, rhs, span })`

### `src/sema.rs`

#### [MODIFY] sema.rs

- Update all AST traversals in the semantic analyzer (`infer_expr`, `typecheck_statement`).
- Adjust `match expr` statements to unpack the structs: `Expr::BinaryOp(bin_op) => ... bin_op.lhs ...`.

### `src/melior_codegen.rs`

#### [MODIFY] melior_codegen.rs

- Introduce `pub trait LowerToMelior<'c, Args> { type Output; fn lower(...) -> Self::Output; }`
- Implement `LowerToMelior` on **each specific struct** (e.g., `impl LowerToMelior for BinaryOpExpr`).
- Simplify `generate_expr` to a basic dispatch loop:
  ```rust
  fn generate_expr(&mut self, expr: &Expr, block: &Block) -> (Value, Type) {
      match expr {
          Expr::BinaryOp(e) => e.lower(self, block, ()),
          Expr::Identifier(e) => e.lower(self, block, ()),
          // ...
      }
  }
  ```

## Verification Plan

### Automated Tests

- Run `cargo test -- test_parser` to ensure syntax trees are still built correctly.
- Run `cargo test -- test_sema` to ensure type-checking logic correctly traverses the new structs.
- Run `cargo test -- test_backend` to ensure the MLIR emission behaves identically to the centralized monolithic generator.
