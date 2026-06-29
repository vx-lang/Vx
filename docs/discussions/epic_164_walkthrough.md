# Epic 164 - Walkthrough (Memory Algebra Graph)

I have successfully integrated the `TransferCostGraph` directly into the compiler's semantic type checking pass and MLIR codegen!

## What was Accomplished?

### 1. Automatic Transfer Injection

Instead of throwing a hard error when a variable from one physical topology (e.g., `CPUDRAM`) is accessed from a different physical topology (e.g., `NPUHBM`), the `TypeChecker` now automatically intercepts this.

- It queries the `TransferCostGraph` (defined in `src/arch.rs`) via Dijkstra's shortest path.
- If a valid physical connection exists, it transparently wraps the AST node in a `TransferExpr` targeting the correct memory space.

### 2. MLIR DMA Lowering

I implemented a standard MLIR lowering for these `TransferExpr` nodes inside `src/codegen/lower/expr.rs`.
Currently, it operates by allocating a new `memref` in the target memory space (encoded natively in the MLIR type, e.g., `memref<?xf32, 1>`), and emitting a `"memref.copy"` from the source.

### 3. Verification

I created a test case (`tests/frontend/pass/memory_algebra.vx`) where a host-allocated tensor is accessed from within a `spawn_on NPU(0)` block.
Running `cargo run --bin vxc -- --emit=mlir` yields exactly what we want:

```mlir
// Space 0: Host CPUDRAM 
%0 = "memref.alloc"(%c1024_i64) : (i64) -> memref<?xf32>
// Space 1: NPU HBM
%1 = "memref.alloc"() : () -> memref<?xf32, 1>
// DMA Transfer (Host -> NPU)
"memref.copy"(%0, %1) : (memref<?xf32>, memref<?xf32, 1>) -> ()
```

## Next Steps

This perfectly sets the stage for Epic 165 (NLL Borrow Checking). Now that physical memory spaces and transfers are formalized, we can accurately track if borrowing a `CPUDRAM` reference inside an `NPUHBM` kernel violates borrowing rules or triggers implicit data movement!
