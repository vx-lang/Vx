# Paper Plan: Data Movement Cost Algebra in Heterogeneous Compilers

This document outlines the strategy for writing the academic/technical paper introducing the **Data Movement Cost Algebra**. Use this as a checklist and structural guide. It links directly to the design artifacts and code implementations that will serve as the primary source material for the paper.

______________________________________________________________________

## 1. Relevant Background & Source Material

Before drafting, review the following documents and code sections:

### Design & Architecture Documents

- **Problem & Justification:** [Pillar 2 Cost Algebra Design Doc](../../discussions/pillar2_cost_algebra_design.md) - Details the *why*: heterogeneous topologies, naive flat-memory limitations, and our formalization.
- **Implementation Steps:** [Pillar 2 Task Checklist](../../discussions/pillar2_cost_algebra_task.md) - Outlines the exact stages we took to integrate this into the Vx compiler.

### Code Implementations (The "How")

- **The Graph & Dijkstra Routing:**
  - File: [src/arch.rs](../../../src/arch.rs#L177)
  - Function: `TransferCostGraph::transfer_cost`
  - *Notes for Paper:* Describe how the boolean adjacency list was converted to a weighted directed graph and how Dijkstra's algorithm determines the optimal multi-hop data movement path.
- **The AST Representation:**
  - File: [src/ast/expr.rs](../../../src/ast/expr.rs#L105)
  - Struct: `TransferExpr`
  - *Notes for Paper:* Explain the `cost: Option<u32>` field, showcasing how the cost becomes a fundamental part of the compiler's Abstract Syntax Tree rather than an afterthought.
- **Semantic Analysis & Cost Injection:**
  - File: [src/sema/expr.rs](../../../src/sema/expr.rs#L825)
  - Function: `TypeChecker::check_transfer_expr`
  - *Notes for Paper:* Describe the frontend pass. When a user writes `transfer(x, MemorySpace)`, the semantic analyzer traverses the topology graph, calculates the shortest path cost, and bakes it into the AST for downstream MLIR optimization.

______________________________________________________________________

## 2. Proposed Paper Structure

### 1. Abstract

- **Hook:** Heterogeneous compute nodes (Host, NPU, GPU) suffer from massive data movement penalties if data routing isn't optimized.
- **Problem:** Standard compilers handle routing via simple static checks or operator fusion, lacking dynamic pathfinding across multi-hop topologies.
- **Solution:** Introduce "Data Movement Cost Algebra", a framework that natively models the hardware topology as a weighted graph inside the compiler frontend.
- **Results:** Briefly mention the efficiency gains or the flexibility achieved by injecting topological transfer costs directly into the AST.

### 2. Introduction

- Discuss the rise of complex ML architectures (e.g., Host DRAM, HBM, SRAM).
- Contrast traditional flat memory models with non-uniform memory access (NUMA) and accelerator interconnects.
- State the core thesis: Data movement must be treated algebraically to enable global, topology-aware compiler optimizations.

### 3. Related Work & Industry Context

- Review how current frameworks (PyTorch `torch.compile`, JAX/XLA, MLIR) handle this.
- Mention their reliance on **Operator Fusion** to avoid transfers, rather than actively solving the multi-hop routing problem.
- Highlight the novelty of applying network routing algorithms (Dijkstra) at the compiler level.

### 4. Methodology: Formalizing the Topology

- Detail the mathematical formalization of the `TransferCostGraph`.
- Explain the edge weights (latency/bandwidth constraints) between different hardware bounds (e.g., `Host_DRAM` to `NPU_HBM`).
- Walk through the Dijkstra shortest-path implementation from [src/arch.rs](../../../src/arch.rs#L177).

### 5. Compiler Integration & The AST

- Detail how the frontend captures the cost.
- Reference the `TransferExpr` changes ([src/ast/expr.rs](../../../src/ast/expr.rs#L105)) and the Semantic Analyzer logic ([src/sema/expr.rs](../../../src/sema/expr.rs#L825)).
- Explain the architectural benefit of doing this in the semantic phase before MLIR lowering, allowing middle-end passes to optimize based on concrete topological costs.

### 6. Evaluation / Case Studies

- Showcase a complex multi-hop transfer scenario (e.g., Host -> NPU -> Local SRAM).
- Compare the routing path chosen by the Cost Algebra vs a naive compiler approach.
- Show performance benchmarks or scheduling improvements.

### 7. Conclusion

- Summarize the impact of introducing explicit cost algebra to compiler frontends.
- Propose future work: ML-driven dynamic edge weighting, extending to multi-node clusters.
