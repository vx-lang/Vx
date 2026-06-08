# Indirect Call Postfix Operator Walkthrough

The postfix parenthesis operator has been successfully implemented, enabling immediate closure invocation and execution of closures through variables seamlessly!

## What was Changed

1. **AST Improvements**:

   - Introduced `IndirectCallExpr` to the `Expr` AST enum in `src/ast/expr.rs`. This elegantly decouples arbitrary expression calls from statically dispatched `FunctionCallExpr` nodes.
   - Updated structural matching blocks (like `substitute` and `span`) to properly handle this new AST variant.

1. **Parser Upgrades**:

   - Upgraded `parse_primary_expr` in `src/parser/expr.rs` to treat `(` as a postfix operator. This natively supports nested invocations like `(|| || 42)()()`.

1. **Semantic Checking (Sema)**:

   - Added `check_indirectcall_expr` logic in `src/sema/expr.rs`.
   - This phase evaluates the `callee`. If the callee is determined to be an anonymous closure struct, the `IndirectCallExpr` is seamlessly transformed into a standard `FunctionCallExpr` that explicitly targets the automatically-generated `Closure_N_call` function.
   - As part of the transformation, the closure instance itself is safely injected as the first parameter by wrapping the `callee` expression in an `Expr::Borrow`.

1. **Zero Codegen Overhead**:

   - Because the Semantic Check phase perfectly rewrites indirect closure calls into direct method calls (with borrowed contexts), absolutely zero modifications were required in `src/codegen/lower.rs`. The MLIR generator natively leverages its existing standard function call lowering, ensuring efficiency and avoiding `func.call_indirect` wherever possible!

## Validation

A new pass test `tests/frontend/pass/indirect_call.vx` was added containing nested, variable-bound, and immediately-invoked closures. The compiler verified and compiled the test flawlessly.
