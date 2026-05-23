# Auto-diff Primitives Implementation Plan

This document outlines the design and implementation strategy for adding IR-level Automatic Differentiation (AD) with language-level intrinsics (`grad`, `vjp`, `jvp`) to the Vx compiler, as specified in Section 9 of the `ROADMAP.md`.

## User Review Required

> [!WARNING]
> **IR-level Autodiff Framework**: Writing a custom MLIR autodiff pass from scratch is a massive undertaking. I propose we use **Enzyme AD**, which operates at the LLVM IR level. Vx will emit standard MLIR, lower it to LLVM IR, and then invoke Enzyme's `__enzyme_autodiff` intrinsic to generate the derivative code. Do you approve of using Enzyme as the backbone?

> [!IMPORTANT]
> **Syntax for Intrinsics**: Since Vx does not currently support first-class functions (closures/function pointers), I propose we introduce new syntax specifically for these intrinsics: `grad(func_name, arg1, arg2)`. Is this syntax acceptable, or would you prefer extending the parser to support first-class function references like `grad(func_name)(arg1, arg2)`?

## Open Questions

1. **Gradient Memory Ownership**: If `x` is `Pinned<Tensor, Topology::NPU[0]>`, should the gradient `dx` returned by `grad` automatically be allocated as `Pinned<Tensor, Topology::NPU[0]>`, or should it default to `HostDRAM` requiring an explicit transfer?
1. **Control Flow**: Do we need to support differentiating through `if` statements and `for` loops in this initial implementation, or should we restrict it to straight-line math functions first?

## Proposed Changes

______________________________________________________________________

### AST & Parser Updates

Introduce new AST nodes to represent the AD intrinsics.

#### [MODIFY] \[src/ast.rs\](file:///Users/adityak/go/Vx/src/ast.rs)

- Add `GradExpr`, `VjpExpr`, and `JvpExpr` structs.
  ```rust
  pub struct GradExpr {
      pub target_fn: String,
      pub args: Vec<Expr>,
      pub span: Span,
  }
  ```
- Add these as variants to the `Expr` enum.

#### [MODIFY] \[src/parser.rs\](file:///Users/adityak/go/Vx/src/parser.rs)

- Register `grad`, `vjp`, and `jvp` as reserved keywords.
- Implement parsing logic to parse `grad(my_func, x, y)` and construct the corresponding AST nodes.

______________________________________________________________________

### Semantic Analysis (Differentiability Proofs)

The semantic analyzer must mathematically guarantee that the target function can be differentiated.

#### [MODIFY] \[src/sema.rs\](file:///Users/adityak/go/Vx/src/sema.rs)

- Implement a `check_differentiability(&Function)` pass.
- **Rules**:
  - The function must only contain differentiable operations (e.g., arithmetic, `Math::` intrinsics).
  - The function must return a continuous type (e.g., `Tensor<f32>`, not `i32` or `bool`).
- **Topology Implications (`spawn on`)**:
  - If the target function contains a `spawn on(Topology::X)` block, the Semantic Analyzer will tag the AST node to ensure that the backward pass (adjoint computation) is also scheduled on `Topology::X`.

______________________________________________________________________

### Backend / MLIR Lowering

Lower the AD intrinsics to MLIR that interfaces with the autodiff backend.

#### [MODIFY] \[src/codegen.rs\](file:///Users/adityak/go/Vx/src/codegen.rs)

- Add `lower_grad`, `lower_vjp`, and `lower_jvp` methods.
- **Implementation Strategy**:
  - The backend will emit an LLVM/MLIR external function declaration for `__enzyme_autodiff`.
  - `codegen.rs` will emit a `func.call` to the Enzyme intrinsic, passing the MLIR symbol reference to the target function, the inputs, and the gradient result buffers.
  - Memory spaces (`NPUHBM` vs `HostDRAM`) will be preserved, ensuring that Enzyme calculates the adjoints in the correct hardware address spaces.

## Verification Plan

### Automated Tests

- Create `tests/backend/pass/autodiff_basic.vx`: Test `grad` on a simple polynomial function.
- Create `tests/backend/pass/autodiff_vjp.vx`: Test Vector-Jacobian Product on matrix multiplication.
- Create `tests/frontend/fail/autodiff_discrete.vx`: Verify the semantic analyzer rejects attempts to differentiate functions returning discrete types (`i32`).

### Manual Verification

- Execute `cargo test` and verify that the generated LLVM IR correctly invokes Enzyme and passes all JIT execution tests.
