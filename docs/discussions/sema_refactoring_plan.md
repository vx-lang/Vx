# Goal Description

Refactor the semantic analysis components (`src/sema/expr.rs` and `src/sema/env.rs`) to address technical debt identified during the review phase. This includes fixing a generic unification rollback bug, strengthening `Option<T>` type coercions, and decoupling massive monolithic methods.

## Open Questions

- For the `unify_types` rollback bug in `env.rs`, is cloning the `mapping` `HashMap` on entry the preferred approach, or would you prefer a custom rollback log (saving keys that were inserted and deleting them on failure) to avoid allocation overhead? A rollback log is faster but more complex. (I'll plan for the HashMap clone for simplicity unless you prefer the log).

## Proposed Changes

### Semantic Environment

#### [MODIFY] \[env.rs\](file:///Users/adityak/go/Vx/src/sema/env.rs)

- **`unify_types`**: Wrap the logic such that when checking multi-element types (like `Function`, `GenericInstance`, `Tensor` dimensions), we clone the `mapping` dictionary first. We run the recursive unification on the clone, and if it completely succeeds, we swap/extend the original mapping. If it fails, the original mapping remains unpolluted.

### Semantic Expressions

#### [MODIFY] \[expr.rs\](file:///Users/adityak/go/Vx/src/sema/expr.rs)

- **`is_assignable`**: Replace the weak string prefix checking (`n_source.starts_with(n_target) && n_source.contains('<')`) for `GenericInstance` matching. Instead, resolve the generic base and compare them strictly.
- **`check_functioncall_expr`**: Extract the trait intrinsic resolution logic (e.g., `Tensor::from`, `Math::`) into a separate `resolve_intrinsic_function` method. Extract generic function instantiation into `instantiate_generic_function`.
- **`check_methodcall_expr`**: Extract the array/tensor intrinsics (`map`, `iter`, `transpose`, `reshape`) into a dedicated `resolve_intrinsic_method` method, leaving the core method lookup cleanly separated.

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure that no existing semantic compilation tests break, particularly `test_sema_type_mismatch` and generics tests.
- Re-run the fuzzing tests to guarantee the AST rewrites inside the checker do not panic.
