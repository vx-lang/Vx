# Issue 114: Option B - Value Type Closures (Anonymous Structs)

Refactor Vx closures so that `|| { ... }` evaluates to an anonymous struct value type, storing captured variables directly in its fields, matching Rust's closure semantics.

## User Review Required

> [!IMPORTANT]
> This is a large architectural change that alters how closures are fundamentally represented in Vx's AST and MLIR Backend.
>
> Currently, `ClosureExpr` evaluates to a "fat pointer" tuple `{ fn_ptr, env_ptr }`. After this change, `ClosureExpr` will evaluate to a pure `Struct` containing the captured variables. The function pointer will NOT be stored at runtime, and calls will resolve statically.
>
> **Consequences**:
>
> 1. You cannot place two different closures in the same variable without casting to a trait object (we will keep `Type::Closure` as the "Fat Pointer" type for this purpose eventually, but initially, closures will be strictly strongly-typed anonymous structs).
> 1. Calls to closures will map statically to a generated `call` method, removing the indirect call overhead for inline closures.

## Open Questions

> [!WARNING]
>
> 1. **Parser Limitations**: Currently, Vx parser translates `my_closure(x)` into `FunctionCallExpr { name: "my_closure" }`. It does not support calling an expression like `(get_closure())(x)`. This change won't fix the parser limitation immediately. Is it acceptable to defer parsing improvements (like postfix `()`) to a separate PR?
> 1. **Fat Pointer Casts**: Do we want to implement `as` casting from anonymous struct to `Type::Closure` in this PR? Or just implement the anonymous structs first and tackle `dyn Fn` fat pointers in a follow-up? (Recommendation: tackle the anonymous structs first to keep the PR focused).

## Proposed Changes

### `src/sema/expr.rs`

- **`check_closure_expr`**:

  - Instead of returning `Type::Closure(params, ret)`, generate a unique struct name: `Closure_<ID>`.
  - Record the captured variables as fields in this struct.
  - Insert `StructDecl` for `Closure_<ID>` into `self.env.structs`.
  - Generate an AST `Function` named `Closure_<ID>_call`, which takes `self: &Closure_<ID>` as its first parameter, followed by the closure's original parameters.
  - Insert this function into `self.monomorphized_functions` so the backend compiles it.
  - Return `Type::Struct(Closure_<ID>)`.
  - Morph the AST node from `ClosureExpr` to a `StructInitExpr` that populates the captured fields.

- **`check_functioncall_expr`**:

  - When evaluating `f(x)`, look up the type of `f`.
  - If the type is `Type::Struct` and starts with `Closure_`, rewrite the AST node from `FunctionCallExpr` to a `MethodCallExpr` targeting the `"call"` method, OR rewrite it to `FunctionCallExpr` pointing directly to `Closure_<ID>_call` with `&f` as the first argument.

### `src/codegen/lower.rs`

- **Remove `ClosureExpr` lowering**: Since `ClosureExpr` is lowered to `StructInitExpr` in Semantic Analysis, MLIR codegen will natively handle the struct allocation. No custom closure lowering logic is needed anymore!
- **Function Generation**: The closure body is extracted into `Closure_<ID>_call` during Sema and added to `monomorphized_functions`. The backend will treat it as a standard function and compile it automatically.

### `src/ast/types.rs` & `src/ast/expr.rs`

- We may still keep `Type::Closure` around as a future interface type, but the AST's `ClosureExpr` will now resolve to `Type::Struct`.

## Verification Plan

### Automated Tests

- `cargo test` to ensure existing closure semantics continue to compile and run (with the new underlying struct implementation).
- Verify MLIR output using `cargo test -- --nocapture` to confirm that `Closure_<ID>_call` is a static `func.call` and not an indirect call.
