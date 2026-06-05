# Dynamic Cast to Fat Pointer Implementation

- `[x]` **Lexer & AST**:
  - Add `TokenType::As` to `src/lexer.rs`.
  - Add `AsCastExpr` to `src/ast/expr.rs`.
- `[x]` **Parser**:
  - Update `parse_primary_expr` in `src/parser/expr.rs` to parse `as` as a postfix operator.
  - Update `parse_type` in `src/parser/types.rs` to parse `|| -> Type` as `Type::Closure`.
- `[x]` **Semantic Analysis**:
  - Add `check_ascast_expr` to `src/sema/expr.rs` to validate casting `Closure_N` to `Type::Closure`.
  - Add `AsCast` variant to `Statement::substitute` and `Expr::substitute`.
- `[x]` **MLIR Lowering**:
  - Update `generate_expr` in `src/codegen/lower.rs` to lower `Expr::AsCast`.
  - Dynamically construct `!llvm.struct<(ptr, ptr)>` with the function pointer and environment pointer.
- `[x]` **Tests**:
  - Add `tests/frontend/pass/closure_fat_ptr.vx`.
  - Verify that `cargo test` passes.
