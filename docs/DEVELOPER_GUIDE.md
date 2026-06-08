# Vx Compiler Developer Guide

Welcome to the Vx compiler development guide! This document provides an architectural overview of the `vxc` compiler, explains the core compilation pipeline, and outlines the standard workflow for contributing new features to the language.

## Architecture Overview

The Vx compiler is written entirely in Rust and utilizes a modern, multi-stage compilation pipeline that lowers native source code directly to MLIR (Multi-Level Intermediate Representation) and subsequently to LLVM IR.

### 1. The Frontend (`src/lexer.rs`, `src/parser.rs`, `src/ast.rs`)

- **Lexer**: Tokenizes the `.vx` source text into semantic tokens.
- **Parser**: A recursive descent parser that constructs the Abstract Syntax Tree (AST). It enforces basic syntactical correctness and understands Vx-specific constructs like `Topology::NPU[0]`, `Memory::NPU_HBM`, and `spawn on(...)` blocks.
- **AST**: The data structures representing the parsed program. Modifications to the language syntax usually begin here.

### 2. Semantic Analysis (`src/sema.rs`)

This is the most complex part of the frontend. The `TypeChecker`:

- Enforces strict mathematical and topological boundaries.
- Resolves implicit generics (`Type::GenericInstance`) and performs aggressive monomorphization on-the-fly.
- Statically verifies memory spaces and topological constraints, failing compilation if a variable from Host DRAM is implicitly used inside an NPU context without an explicit `transfer()` mechanism.

### 3. MLIR Code Generation (`src/codegen.rs`)

The `MlirGenerator` takes the fully monomorphized, type-checked AST and translates it into MLIR.

- We utilize standard MLIR dialects like `func`, `arith`, `scf`, `cf`, and `memref`.
- Tensors (`Tensor<f32>`) are lowered directly into MLIR `memref` types.
- The codegen explicitly manages memory casting and runtime FFI hooks.

### 4. JIT Execution Engine (`src/jit.rs`)

For local execution (`cargo run --bin vxc -- --run`), the compiler shells out to standard LLVM tools:

1. `mlir-opt`: Expands and lowers MLIR dialects.
1. `mlir-translate`: Converts MLIR directly into LLVM IR.
1. `lli`: The LLVM execution engine runs the IR natively, dynamically linking our `libvx_rt.dylib` C runtime for FFI operations like high-precision timing.

______________________________________________________________________

## Testing & Benchmarking

### Unit Tests

The compiler's correctness is validated through standard Rust unit tests and snapshot integration tests.

```bash
cargo test
```

Tests in `tests/backend/pass/` are end-to-end tests that parse, compile, and execute Vx code, comparing the final output against expected `// CHECK:` comments.

### Benchmarking

We maintain a custom benchmark harness to measure raw execution performance without the overhead of JIT compilation. The harness dynamically injects `vx_get_time` timing blocks directly into the AST.

```bash
cargo vx-bench
```

Add new, heavy computational tasks to the `benchmarks/` directory to track performance across different topological execution spaces.

______________________________________________________________________

## Adding a New Language Feature

When adding a new feature (e.g., a new control-flow statement or operator), follow this standard workflow:

1. **AST Definition**: Add the new node type to `src/ast.rs` (e.g., inside the `Statement` or `Expr` enum).
1. **Lexing/Parsing**: Update `src/lexer.rs` to recognize any new keywords, and update `src/parser.rs` to build your new AST node.
1. **Type Checking**: Implement the validation logic in `src/sema.rs`. Ensure that topological and memory-space constraints are explicitly verified.
1. **Code Generation**: Add the MLIR lowering logic to `src/codegen.rs`. If you need a new MLIR instruction, ensure it maps correctly to the existing LLVM toolchain.
1. **Testing**: Write a minimal failing test case in `tests/` and iterate until it compiles and runs correctly.

## Debugging

To debug the AST or see the intermediate MLIR output during development:

- Print the generated AST: `cargo run --bin vxc -- --print-ast <file.vx>`
- Print the generated MLIR: `cargo run --bin vxc -- --emit-mlir <file.vx>`
