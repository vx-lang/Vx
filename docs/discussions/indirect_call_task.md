# Indirect Call Implementation Task

- `[x]` Add `IndirectCallExpr` to `src/ast/expr.rs`.
- `[x]` Add `IndirectCall` variant to `Expr` enum and update `Expr::span` in `src/ast/expr.rs`.
- `[x]` Update `Statement::substitute` and `Expr::substitute` to support `IndirectCallExpr`.
- `[x]` Update `src/parser/expr.rs` to parse postfix `()` as `IndirectCallExpr`.
- `[x]` Implement `check_indirectcall_expr` in `src/sema/expr.rs`.
- `[x]` Route `IndirectCallExpr` through the semantic checking flow.
- `[x]` Add `tests/frontend/pass/indirect_call.vx` and test.
