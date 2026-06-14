# Code Review: `src/sema/expr.rs`

This file is the backbone of the Vx compiler's semantic analysis, responsible for type checking expressions, enforcing borrow checker rules, and tracking hardware topology state for distributed execution.

## Core Responsibilities

- **Expression Type Inference**: Recursively derives types for all expressions using `check_expr_type_flag`.
- **Monomorphization**: Instantiates generic functions and methods dynamically upon invocation.
- **Closure Desugaring**: Transforms closure expressions into explicit stateful environment structs (`Closure_<ID>`) and corresponding `_call` functions.
- **Topology Verification**: Enforces boundary checks for variables accessed across different topologies (e.g. CPU vs NPU) inside `spawn on` blocks.
- **Automatic Differentiation Rules**: Type-checks `grad`, `vjp`, and `jvp` expressions, ensuring target functions are continuous.

## Key Mechanisms

### `is_assignable` & Memory Space Rules

The `is_assignable` function handles complex coercions between Types. A critical design decision here is the explicit tracking of `MemorySpace`:

> "We no longer allow implicit unwrapping of Ref<T> or Pinned<T> to T. Users must use transfer(expr, Memory::Space) or .to_host() / .to_device() to move data across memory boundaries."

It also delegates to the 256-bit FastPath Borrow Checker algorithm (`crate::borrow::verify_subtyping_bounds`) when checking assignments between `Type::Borrow`.

### Topology and Hardware Transfers (`check_transfer_expr`)

The checker intercepts `transfer(expr, target_space)` and asks the `TransferCostGraph` for a valid hardware path. If a multi-hop path is required (e.g., length > 2), the AST is **rewritten** to chain multiple `TransferExpr` nodes together so that the lowerer emits the correct sequence of DMA events.

### The `silent` Flag

Most checker methods accept a `silent` flag. This allows speculative type checking (like evaluating method overloads or doing generic inference in `check_functioncall_expr`) without polluting the global error context. If a match isn't found, it fails silently, and the caller can try another path or emit a tailored error.

### Intrinsic Lowering & AST Rewriting

In `check_methodcall_expr`, the semantic analyzer performs significant AST rewriting:

- Method calls are transformed into `FunctionCallExpr` nodes, prepending the `obj` receiver as the first argument.
- `map`, `reshape`, `transpose` are type-checked as compiler intrinsics. For `reshape` and `transpose`, the target dimensions and permutations are statically evaluated at compile time (`eval_expr`), preventing runtime shape mismatch.

### Closure Capture

When a `Closure` is encountered, `check_closure_expr` scans the current depth. Any variables from an outer scope referenced inside the closure are captured. The checker then synthesizes a new `Closure_<ID>` struct to hold the captured state and a `Closure_<ID>_call` monomorphized function, replacing the expression with a `StructInitExpr` for the environment.

## Design Critique & Actionable Items

1. **Monolithic Complexity**: `check_functioncall_expr` and `check_methodcall_expr` are massive. They interleave error reporting, generic unification, trait bound checking, monomorphization, and intrinsic matching.
   - *Action*: Split intrinsic mocking/resolution from generic instantiation and method lookup.
1. **Compile-Time Evaluation**: `eval_expr` is used heavily for `reshape` sizes and `transpose` permutations. It correctly falls back to errors if sizes cannot be evaluated statically.
1. **Implicit Coercion**: `is_assignable` still has some weak checks (`n_source.starts_with(n_target) && n_source.contains('<')` for `Option<T>`). This string-matching based type equivalence is fragile and should be replaced with proper Generic ID matching.
