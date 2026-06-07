# Pillar 2: Data Movement Cost Algebra

- `[x]` Phase 1: Update TransferCostGraph (`src/arch.rs`)
  - `[x]` Update `transfer_edges` to use `(MemorySpace, u32)`.
  - `[x]` Update `default()` to include realistic edge weights.
  - `[x]` Implement `transfer_cost()` using Dijkstra's algorithm.
- `[x]` Phase 2: Update AST and Semantic Analyzer
  - `[x]` Add `cost: Option<u32>` to `TransferExpr` in `src/ast/expr.rs`.
  - `[x]` Integrate `transfer_cost()` into `src/sema/expr.rs` and attach the cost to the AST node.
- `[x]` Phase 3: Testing
  - `[x]` Create and run `tests/frontend/fail/transfer_no_path.vx`.
  - `[x]` Create and run `tests/frontend/pass/transfer_cost_dijkstra.vx`.
