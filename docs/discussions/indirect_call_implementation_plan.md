# Postfix Parenthesis Operator (Indirect Calls)

The goal is to support calling closures and arbitrary expressions using the postfix parenthesis operator, e.g., `(|| { 42 })()` or `my_closure()`. Currently, `FunctionCallExpr` only supports string identifiers (`name: String`) as the callee.

## Proposed Changes

1. **AST Updates (`src/ast/expr.rs`)**:

   - Introduce `IndirectCallExpr`:
     ```rust
     #[derive(Debug, PartialEq, Clone)]
     pub struct IndirectCallExpr {
         pub callee: Box<Expr>,
         pub args: Vec<Expr>,
         pub span: Span,
     }
     ```
   - Add `IndirectCall(IndirectCallExpr)` variant to the `Expr` enum.

1. **Parser Updates (`src/parser/expr.rs`)**:

   - In `parse_primary_expr`, extend the postfix operator loop (which currently handles `.member`, `.method()`, and `[index]`) to parse `(` as an indirect call operator.
   - When encountering `(` after an expression, parse the arguments and wrap the `base` expression in an `Expr::IndirectCall`.

1. **Semantic Analysis (`src/sema/expr.rs`)**:

   - Implement `check_indirectcall_expr`.
   - Evaluate the `callee` expression.
   - If the type is `Type::Struct(name)` where `name` starts with `"Closure_"`, rewrite the AST node from `IndirectCallExpr` directly into a `FunctionCallExpr` targeting `Closure_N_call`. We will pass `&callee` as the first argument by wrapping the `callee` in an `Expr::Borrow`.
   - This seamlessly forwards the execution to our existing closure code generation without requiring any MLIR codegen modifications for indirect calls.

## Dynamic Cast to Fat Pointer (Follow-up)

As requested, the second feature (dynamic casting to a fat pointer like `dyn Fn`) will be done in a subsequent PR to keep the changes atomic. It will likely involve:

- Introducing an `as` operator or a builtin `cast` macro.
- Allowing conversions from `Closure_N` to `Type::Closure(args, ret)`.
- Extending `check_indirectcall_expr` and `codegen/lower.rs` to support true indirect MLIR `func.call_indirect` instructions when the callee is a fat pointer.

## Verification

- Add a new pass test case: `tests/frontend/pass/indirect_call.vx` to verify inline closure execution.
- Ensure existing tests continue to pass.
