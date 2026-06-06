# Walkthrough: Comptime Branches & Topology Awareness

I have successfully implemented `if comptime` compile-time branching and the `current_topology` builtin in the Vx compiler.

## Changes Made

### 1. `if comptime` Branching

- Extended `IfExpr` AST node in `src/ast/expr.rs` to include an `is_comptime` flag.
- Updated the parser in `src/parser/expr.rs` to recognize `if comptime <condition> { ... }`.
- Implemented compile-time branch pruning in the Semantic Analyzer (`src/sema/expr.rs`). When the analyzer evaluates an `if comptime` expression:
  - The condition is evaluated at compile time.
  - The un-taken branch is completely removed from the AST (`cleared` / `None`).
  - **Crucially**, the un-taken branch is never type-checked! This allows users to write backend-specific operations (e.g. invalid macros, target-specific builtins) in an inactive branch without throwing compiler semantic errors.

### 2. `current_topology` Builtin Identifier

- Extended parser (`src/parser/types.rs`) to treat `current_topology` seamlessly as `Topology::Current` when utilized in places that require a hard topology constant (e.g., `spawn on (current_topology) { ... }`).
- Extended semantic analysis (`check_identifier_expr` in `src/sema/expr.rs`) to replace any generic occurrence of the `current_topology` identifier with an AST node representing the currently active topological context.
- Ensured backend/codegen (`src/codegen/lower.rs`, `src/codegen/generator.rs`, `src/arch.rs`) handles standard mapping of unreplaced `Topology::Current` defaults (fallback space 0) if it escapes macro contexts.
- Allowed compile-time equality operations for direct evaluation of the `current_topology` identifier against `Topology::...` types.

### 3. Usage & Examples

- Updated the `fill_static` implementation in `stdlib/std/tensor.vx` to showcase branching against different topologies based on the active target:

```rust
if comptime current_topology() == Topology::Host_AVX512 {
    // AVX512 optimized MLIR string
} else {
    // Standard Linalg MLIR string
}
```

- Added a comprehensive frontend test `tests/frontend/pass/comptime_if.vx` that proves invalid semantics in a pruned block do not crash the compiler.
- Added a frontend test `tests/frontend/pass/if_comptime_and_topology.vx` to verify the evaluation of `current_topology` resolving to the active topology defined in `spawn on` blocks.

## Verification

- Verified the build via `cargo check` and ran `cargo test test_frontend_pass` to assert that:
  - Pruned dead code does not error out semantic analysis.
  - Topologies compare correctly against constants.
  - Topologies are correctly parsed.

______________________________________________________________________

# Tasks

- `[x]` Add `is_comptime` to `IfExpr` in `src/ast/expr.rs`
- `[x]` Parse `if comptime` in `src/parser/expr.rs`
- `[x]` Add `Value::Topology` to `src/sema/env.rs`
- `[x]` Handle `if comptime` block pruning and `current_topology()` evaluation in `src/sema/expr.rs`
- `[x]` Update `stdlib/std/tensor.vx` to use `if comptime`
- `[x]` Add frontend test `tests/frontend/pass/comptime_if.vx`
- `[x]` Run test suite to verify
- `[x]` Implement `Topology::Current` variant in `Topology` enum
- `[x]` Support resolving `"current_topology"` identifier to active topology in `eval_expr` for compile-time resolution
- `[x]` Support `current_topology` as a standard identifier evaluating to `Topology::Current` during check phase
- `[x]` Handle defaults and unwrapping of `Topology::Current` in `src/codegen/lower.rs` and `src/arch.rs`
- `[x]` Exhaust match patterns in `src/codegen/generator.rs` to fix `E0004` warnings
- `[x]` Add frontend tests for `current_topology` identifier inside different `spawn on` blocks
