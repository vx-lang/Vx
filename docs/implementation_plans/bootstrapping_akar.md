# Bootstrapping the Vx Programming Language

This document outlines the high-level roadmap and architectural plan for bootstrapping **Vx**, a natively heterogeneous and topology-aware systems programming language.

## Open Questions & Decisions

1. **Compiler Implementation Language**: We have officially migrated the scaffolding to **Rust** based on feedback and `agents/Coding.md` rules. We will use Cargo to manage dependencies like `proptest` and `rstest` for testing.
2. **Intermediate Representation (IR)**: We will definitely proceed with **MLIR**. This will allow us to define an Vx dialect that natively encodes topologies before lowering to specific accelerator dialects.
3. **Runtime Model**: We have decided on a **Dual-Mode Execution Model**:
   - **Strict Mode (Eager / Thin Runtime)**: When you use `Pinned<T, Topo>`, the code compiles down to direct LLVM/MLIR intrinsics. There is no runtime overhead; it's just raw instructions and DMA transfers. If it can't run, it errors.
   - **Agile Mode (Lazy / Fat Runtime)**: When you use `Verified<T>`, the execution becomes lazy. The runtime builds a Directed Acyclic Graph (DAG) of the computation and dynamically dispatches tasks to available hardware (acting like a smart JIT scheduler or IREE runtime).

## Proposed Execution Phases

We will build the language iteratively, focusing on getting a vertical slice of heterogeneous compilation working as early as possible.

### Phase 1: Formalizing the Syntax and Type System
Before writing a parser, we need a formal specification of the new constructs. We will create a `docs/syntax.md` and `docs/types.md` to define:
- Rust-like basic syntax (functions, variables, control flow).
- **Topology Identifiers:** How to address `NPU_HBM`, `Acc1Core`, etc.
- **Topology-Aware Types:** `Verified<T>`, `Pinned<T, Topology>`.
- **Concurrency & Spatial Primitives:** `spawn on(...)`, `transfer(...)`.
- **Memory Ownership:** How ownership is transferred across address spaces.

### Phase 2: Compiler Scaffolding & Frontend
We will initialize the compiler repository and build the frontend.
- **Lexer & Parser:** Create the AST (Abstract Syntax Tree) to parse standard code plus the new `spawn on` and `transfer` blocks.
- **AST Definition:** Structs to represent the parsed Vx code.

### Phase 3: Semantic Analysis & Topology Checker
This is the "secret sauce" of Vx.
- **Type Checking:** Standard type inference.
- **Spatial Validation:** Ensuring `Ref<Matrix, NPU_HBM>` cannot be added directly to `Ref<Matrix, Host_DRAM>` without a `transfer` primitive.
- **Execution State Resolution:** Resolving the monadic `try_pin` hardware bridges.

### Phase 4: Intermediate Representation (IR) Lowering
- Map the verified AST into an IR (likely MLIR).
- Define the `vx` MLIR dialect.
- Lower standard compute to `llvm` dialect, and heterogeneous compute to specific hardware dialects (e.g., `gpu` or `amdgpu` or custom accelerator dialects).
