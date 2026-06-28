# Epic 164: Formalize the Memory Algebra Graph for Topologies

Currently, `TransferCostGraph` is implemented in `src/arch.rs` and has a cost matrix mapping physical transfers between `CPUDRAM`, `NPUHBM`, `LocalSRAM`, etc. However, the `TypeChecker` currently just throws a hard error (`"Cross-topology access error"`) when a memory boundary is crossed.

Our goal is to **formally integrate the Algebra** so the compiler *automatically* injects DMA transfers when crossing valid topological boundaries, and lowers them to MLIR.

## Proposed Changes

### 1. Auto-Transfer Injection (`src/hir/expr.rs`)
In `check_identifier_expr`, when a topology mismatch occurs:
- Instead of throwing an error, we will query `self.transfer_cost_graph.transfer_path()`.
- If a path exists (e.g., Host to NPU HBM), we will mutate the AST in-place. We will wrap the original `IdentifierExpr` inside a `TransferExpr`, attaching the physical destination `MemorySpace` and the minimum transfer `cost`.
- If no path exists (e.g. GPU attempting to read AccCore LocalSRAM), we keep the hard error.

### 2. Lowering `TransferExpr` to MLIR (`src/codegen/lower/expr.rs`)
`TransferExpr` currently lacks an MLIR lowering implementation.
- I will implement `LowerToMelior<'c> for TransferExpr`.
- Since our custom `vx` dialect (Epic 162) isn't built yet, we will lower this into standard MLIR operations: allocating a new buffer in the target memory space and emitting a `"memref.copy"` operation to move the data across physical domains.

### 3. Verification & Testing (`tests/frontend/pass/memory_algebra.vx`)
I will write integration tests proving that an NPU kernel (`spawn_on NPU`) can transparently read variables declared on the Host (`CPUDRAM`). The AST will implicitly handle the HBM upload.

## User Review Required

> [!IMPORTANT]
> **Implicit vs Explicit Transfers**: This plan proposes *implicit* (automatic) data transfers. The compiler will hide the DMA copy from the developer. Is this the design you want, or do you prefer forcing the developer to write an explicit `transfer(var, NPU)` in their code for performance visibility?
>
> **MLIR Codegen**: Without the `vx` MLIR dialect, using `"memref.copy"` is the cleanest standard way to represent a DMA transfer between two different MLIR memory spaces. Does that sound acceptable for this phase?
