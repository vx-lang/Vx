# Design Document: Data Movement Cost Algebra (Pillar 2)

## 1. Problem Statement

Modern machine learning workloads run on increasingly heterogeneous hardware topologies. A single node may contain a Host CPU, multiple NPUs, GPUs, and distinct memory hierarchies (Host DRAM, NPU HBM, Local SRAM).

Historically, compilers treat all memory as a flat address space or use simple unweighted accessibility checks (e.g., "Can the NPU read this memory? Yes/No"). However, in reality, **data movement is not free**. Moving a large tensor from Host DRAM to NPU HBM has a massive latency and bandwidth penalty compared to moving data from NPU HBM to Local SRAM.

Without understanding the *cost* of data movement, a compiler might make poor optimization decisions, such as:

- Moving data to an NPU for a trivial computation where the transfer cost far outweighs the compute time saved.
- Routing data through an inefficient multi-hop path when a cheaper, direct DMA path exists.

## 2. Why is this needed?

To perform optimal scheduling and kernel placement, the compiler needs a mathematical formalization of the topology. We need to answer not just "Can we transfer data from A to B?", but **"What is the exact penalty of transferring data from A to B?"**

By introducing a **Cost Algebra**, we enable the compiler to:

1. **Global Optimization**: Compare the cost of moving data versus the cost of computing locally.
1. **Pathfinding**: Automatically discover the cheapest multi-hop route for data if a direct connection does not exist (e.g., Host -> NPU 1 -> NPU 2).
1. **Hardware-Aware Lowering**: Pass accurate cost models down to the MLIR back-end, allowing the scheduler to overlap computation with memory transfers optimally.

## 3. The Solution: Weighted Topology Graphs

We solved this by upgrading the `TransferCostGraph` representation within the compiler.

### 3.1. Weighted Directed Graph

Instead of a simple adjacency list representing boolean connectivity, the `TransferCostGraph` is now a **weighted directed graph**. Each edge between two `MemorySpace` nodes carries a `u32` weight representing the relative latency/bandwidth cost of that transfer.

### 3.2. Cost Algebra via Dijkstra's Algorithm

When a `TransferExpr` is encountered in the AST, the compiler's Semantic Analyzer queries the `TransferCostGraph`. We implemented **Dijkstra's Shortest Path Algorithm** to traverse the topology and find the optimal route. The total accumulated weight along the shortest path is the "Data Movement Cost".

### 3.3. AST Integration

This cost is directly injected into the `TransferExpr` AST node during semantic analysis (`expr.cost = Some(calculated_cost)`). Because the cost is computed in the frontend, all downstream middle-end and backend optimization passes (like loop unrolling, vectorization, and MLIR lowering) now have access to realistic hardware metrics to drive their heuristics.

## 4. Comparison with Industry Standards (PyTorch, JAX, MLIR)

This concept of "Transfer Cost Modeling" is highly relevant in the broader ML compilation community, though our specific Dijkstra-based routing represents an advanced topological approach.

- **PyTorch (`torch.compile`) & JAX (XLA):** Both heavily rely on Cost Models to decide when to "fuse" operations (Operator Fusion). The primary goal in XLA is to keep data in fast, local registers to avoid the high transfer cost of writing back to global device memory. However, these frameworks typically use static or ML-driven heuristics to estimate compute vs. transfer times, rather than treating the hardware as a dynamically routable graph.
- **MLIR ecosystem:** MLIR provides dialects (like `memref`) to describe memory hierarchies. Production compilers built on MLIR build device-aware cost models to guide tiling and mapping.
- **Our Dijkstra Approach:** While traditional compilers focus primarily on *fusion* to eliminate transfers, our approach treats the entire heterogeneous system as a network. By using Dijkstra's shortest path, our compiler can actively route data through optimal multi-hop paths (e.g., leveraging a faster interconnect between two accelerators instead of routing through the slower host memory). This brings concepts from network routing into compiler data-flow scheduling, positioning Vx uniquely for complex, multi-accelerator topologies.
