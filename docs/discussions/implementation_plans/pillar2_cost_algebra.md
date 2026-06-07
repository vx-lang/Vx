# Data Movement Cost Algebra & Topology Weights

This plan outlines the final stage of **Pillar 2: Rigorous Topology & Memory Algebra**. While the compiler already parses hardware graphs and correctly verifies memory access using unweighted paths, it currently lacks the mathematical formalization of **data movement costs**. 

## Proposed Changes

### `src/arch.rs` (HardwareGraph Core)
The `HardwareGraph` structure needs to evolve from an unweighted adjacency list to a weighted directed graph, and its BFS pathfinding must be upgraded to Dijkstra's algorithm.

#### [MODIFY] [arch.rs](file:///Users/adityak/go/Vx/src/arch.rs)
- Update `transfer_edges: HashMap<MemorySpace, Vec<MemorySpace>>` to `HashMap<MemorySpace, Vec<(MemorySpace, u32)>>` where `u32` represents the data movement cost (latency/bandwidth penalty).
- Update the default hardware graph instantiation to supply logical costs:
  - `HostDRAM <-> NPUHBM`: Cost 50
  - `NPUHBM <-> LocalSRAM`: Cost 10
- Replace the BFS implementation in `pub fn can_transfer` with `pub fn transfer_cost(&self, source: &MemorySpace, target: &MemorySpace) -> Option<u32>`. This will implement Dijkstra's shortest path algorithm.

### `src/ast/expr.rs` (AST Nodes)
We need to track this calculated cost throughout the compiler pipeline.

#### [MODIFY] [expr.rs](file:///Users/adityak/go/Vx/src/ast/expr.rs)
- Add a `pub cost: Option<u32>` field to `TransferExpr`.
- Update `TransferExpr::new()` to initialize `cost` to `None`.

### `src/sema/expr.rs` (Semantic Analyzer)
The semantic analyzer will execute the pathfinding and embed the cost into the AST.

#### [MODIFY] [expr.rs](file:///Users/adityak/go/Vx/src/sema/expr.rs)
- Update `check_transfer_expr` to call `self.hardware_graph.transfer_cost()`.
- If `None` is returned, throw the existing "no hardware path exists" compile error.
- If `Some(cost)` is returned, assign it to the `expr.cost` field on the AST node.

### `src/codegen/lower.rs` (Optional: MLIR Attributes)
*(If we want MLIR to see this cost, we can append an attribute to the `vx.transfer` operation, though this might be out of scope for the immediate semantics task).*

## Verification Plan

### Automated Tests
- Create `tests/frontend/fail/transfer_no_path.vx` to verify that unconnected memory spaces still correctly error out.
- Create `tests/frontend/pass/transfer_cost_dijkstra.vx` (or similar) ensuring that multi-hop transfers compile without issue.
- Verify `cargo test` passes and Dijkstra's algorithm isn't causing regressions.

## User Review Required

> [!IMPORTANT]
> - Do you want the calculated transfer cost to be emitted directly into the MLIR output as an attribute (e.g., `vx.transfer ... { cost = 50 }`)? If so, I will also update `src/codegen/lower.rs`. 
> - Are you okay with integer costs (`u32`) for edge weights, or would you prefer floating point representation (`f32`)?
