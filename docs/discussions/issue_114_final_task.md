# Option B: Value Type Closures

- `[x]` Update `src/sema/expr.rs` (`check_closure_expr`):
  - `[x]` Generate `Closure_N` name.
  - `[x]` Build `ast::StructDecl` for the environment.
  - `[x]` Build `ast::Function` for the `Closure_N_call` containing capture-bindings.
  - `[x]` Push to `self.monomorphized_functions`.
  - `[x]` Transform `Expr::Closure` AST node into `Expr::StructInit`.
- `[x]` Update `src/sema/expr.rs` (`check_functioncall_expr`):
  - `[x]` Identify calls on `Type::Struct("Closure_...")`.
  - `[x]` Rewrite `FunctionCallExpr` to target the `Closure_N_call` and prepend `&f` argument.
- `[x]` Update `src/codegen/lower.rs`:
  - `[x]` Remove custom `ClosureExpr` lowering block (or leave it unreachable since it's rewritten in sema).
