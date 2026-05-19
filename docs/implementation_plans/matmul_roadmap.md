# Vx Matmul Execution Roadmap

This implementation plan outlines the architectural phases required to take our current parser and build out the rest of the compiler pipeline to successfully execute a matrix multiplication on Apple Silicon.

## User Review Required

> [!IMPORTANT]
> Please review the chosen technologies (MLIR `melior` crate, LLVM ORC JIT) to ensure they align with your vision for the project.

## Phase 1: Semantic Analysis (Type Checking)

Before generating code, we must validate the AST for correctness.

- **Objective**: Create `src/sema.rs` to traverse the AST and build a symbol table.
- **Key Tasks**:
  - Implement a `TypeEnvironment` to track variable scopes and types.
  - Implement type checking for operations and assignments.
  - Validate memory spaces: ensure operations respect `Memory::Host_DRAM` vs `Memory::NPU_HBM` and enforce `transfer` boundaries.

## Phase 2: MLIR Lowering

Translate the validated AST into MLIR, which provides powerful abstractions for linear algebra.

- **Objective**: Integrate MLIR and lower the AST.
- **Key Tasks**:
  - Add the `melior` crate as a dependency for safe Rust MLIR bindings.
  - Create `src/codegen/mlir.rs`.
  - Lower Vx loops and tensor operations into MLIR's `linalg` and `affine` dialects.
  - Lower host operations (like `LetDecl` and `Return`) into the `func` and `scf` (Structured Control Flow) dialects.

## Phase 3: MLIR Optimization

Optimize the generated MLIR code before converting it to machine code.

- **Objective**: Apply MLIR passes to accelerate the matmul.
- **Key Tasks**:
  - Configure a `PassManager` in `melior`.
  - Apply standard passes: loop unrolling, vectorization, and affine loop fusion.

## Phase 4: LLVM Backend and JIT Execution

Generate native AArch64 machine code and run it directly from the compiler.

- **Objective**: Execute the optimized MLIR code.
- **Key Tasks**:
  - Use MLIR passes to lower the `linalg`/`affine` dialects down to LLVM IR (`llvm` dialect).
  - Integrate an LLVM ORC JIT engine into `vxc`.
  - Execute the function in memory and return the result.

## Phase 5: Runtime & FFI

Provide the data necessary for the matmul to run.

- **Objective**: Bridge Rust and Vx for input/output.
- **Key Tasks**:
  - Define the in-memory representation of an Vx `Tensor` (e.g., a struct with a data pointer and shape).
  - Write a Rust driver in `main.rs` that allocates two matrices, invokes the JIT-compiled Vx matmul via C ABI, and prints the computation time.
