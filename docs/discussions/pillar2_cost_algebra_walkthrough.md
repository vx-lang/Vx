# Pillar 2: Data Movement Cost Algebra Complete

I have successfully finished implementing **Pillar 2: Rigorous Topology & Memory Algebra**. The semantic analyzer now natively computes graph-based physical data movement costs when doing cross-topology data transfers.

## Changes Made

- **Hardware Graph Weights:** Upgraded `HardwareGraph` (`src/arch.rs`) to support weighted, directed edges indicating bandwidth/latency cost:
  - `HostDRAM <-> NPUHBM` penalty: 50
  - `NPUHBM <-> LocalSRAM` penalty: 10
- **Dijkstra's Algorithm:** Implemented `transfer_cost(src, dest)` using a min-heap to find the lowest-cost transfer path across hardware memory spaces.
- **AST Cost Embedding:** Added `cost: Option<u32>` to `TransferExpr`.
- **Semantic Validation:** Modified the semantic analyzer (`src/sema/expr.rs`) to calculate data movement cost and attach it to the `TransferExpr`. If no path exists, it raises the compiler error natively as before.

## Testing & Validation

- **Multi-Hop Tests:** Added `tests/frontend/pass/transfer_cost_dijkstra.vx` which successfully verifies a two-hop transfer (`Host_DRAM -> Local_SRAM`).
- **Integration:** The `cargo test` suite was verified to pass, ensuring that no regressions were caused by updating the standard compiler pipeline.

> [!TIP]
> **What's next for MLIR?**
> The `cost` is now securely stored in the AST. Whenever we move onto **Pillar 1b** (Hardware Kernel Generation) or need advanced backend optimizations, we can easily inject this `cost` as an attribute into the generated `vx.transfer` MLIR dialect nodes.
