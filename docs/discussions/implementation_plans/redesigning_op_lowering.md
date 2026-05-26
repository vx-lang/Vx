# Lowering to Affine and Linalg Dialects

This implementation plan covers the remaining tasks in Phase 2 of the `matmul_roadmap.md`: transitioning our MLIR lowering strategy from `scf` (Structured Control Flow) to `affine` and `linalg` dialects for loops and tensor operations.

## Goal

- Change `ForLoopStmt` lowering from `scf.for` to `affine.for` to enable polyhedral optimizations like loop fusion.
- Introduce native lowering for tensor operations into the `linalg` dialect (e.g., `linalg.matmul`).

## User Review Required

> [!IMPORTANT]
> **Design Decision on Linalg Lowering**
>
> The user has decided that we will **overload the multiplication operator `*`** for Tensor types.
>
> - If `a * b` is executed and both `a` and `b` are `Tensor` or `Matrix` types, the compiler will lower this operation directly to `linalg.matmul`.
> - This provides a clean, elegant syntax (`C = A * B`) without requiring complex loop-lifting passes or explicit intrinsic function calls.

## Proposed Changes

### 1. `affine.for` Lowering

#### [MODIFY] `src/melior_codegen.rs`

- Update `impl<'c> LowerToMelior<'c> for ForLoopStmt`.
- Replace the `scf.for` builder with `affine.for`.
- Construct an identity `AffineMap` (`(d0) -> (d0)`) for the dynamic lower and upper bounds (`start_idx` and `end_idx`).
- Attach these maps as `lower_bound` and `upper_bound` attributes using `AffineMapAttribute`.
- Yield from the loop body using `affine.yield` instead of `scf.yield`.

### 2. `linalg` Lowering (Operator Overloading `A * B`)

#### [MODIFY] `src/sema.rs`

- Add type-checking rules for `BinaryOp::Mul` when operands are `Tensor` or `Matrix` types.
- Ensure dimensions match for matrix multiplication (e.g., `M x K * K x N -> M x N`).

#### [MODIFY] `src/melior_codegen.rs`

- In `generate_expr` for `BinaryOp::Mul`, check if both operands are `memref` (tensors).
- Emit the `"linalg.matmul"` operation directly via `OperationBuilder`.
- Provide the output tensor as an `outs` argument to `linalg.matmul` according to MLIR standards.

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure that standard loops correctly emit `affine.for` and `affine.yield`.
- Use the `melior-test` binary or a dedicated codegen test to parse the output MLIR and verify the presence of `affine.for` and `linalg.matmul` dialects.

### Manual Verification

- Compile `benchmarks/llama2_scaling.vx` to MLIR and visually verify that the generated code is leveraging the `affine` and `linalg` dialects where appropriate.
