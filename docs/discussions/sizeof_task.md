# Task List

- `[x]` Define `SizeOfExpr` in `src/ast/expr.rs`.
- `[x]` Add `SizeOf(SizeOfExpr)` to the `Expr` enum and its methods (`span`, `substitute`).
- `[x]` Update parser (`src/parser/expr.rs`) to parse `sizeof<T>()` as `SizeOfExpr`.
- `[x]` Update semantic analyzer (`src/sema/expr.rs`) to handle `SizeOfExpr`, returning `Scalar(I64)`.
- `[x]` Update MLIR codegen (`src/codegen/lower.rs`) to remove string hacks and lower `SizeOfExpr` using the `getelementptr` trick.
- `[x]` Update `stdlib/` and `tests/` to use `i64` instead of `i32` for `sizeof` results.
- `[x]` Run `cargo fmt` and `cargo test`.
