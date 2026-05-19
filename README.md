<div align="center">
  <h1>Vx Language</h1>
  <p><b>One Language, Every Core.</b></p>
  <p>A high-performance systems programming language built from the ground up for heterogeneous computing.</p>
</div>

______________________________________________________________________

## ⚡ What is Vx?

**Vx** (pronounced *"vee-ex"*) is a general-purpose systems programming language designed to unify CPU, GPU, NPU, and accelerator workloads.

Historically, leveraging heterogeneous hardware required disjointed toolchains, painful FFI boundaries, and complex vendor-specific frameworks (like CUDA, Metal, or OpenCL). **Vx treats hardware diversity not as a challenge, but as a first-class citizen.** It bridges execution topologies and memory domains under a single, verifiable syntax.

## Core Philosophy

The language is governed by 7 core tenets:

1. **Heterogeneous Compute**: Address spaces and compute topologies (CPUs, GPUs, NPUs) are first-class primitives. Distributed and parallel computations are expressed natively (e.g., `spawn on(Topology::NPU[0])`).
1. **Ease of Verified Computation**: Hardware-aware type systems, explicit topologies, and `Verified<T>` wrappers allow programmers to verify computation correctness and data locality.
1. **High Performance**: Designed for Ahead-Of-Time (AOT) optimizations. The compiler lowers directly to MLIR and LLVM IR for optimal native machine code.
1. **Deterministic Memory Control**: No mandatory garbage collection. Programmers have control over memory layouts, lifetimes, and pointer arithmetic.
1. **Zero-Cost Abstractions**: High-level constructs compile down to optimal machine code with no runtime overhead.
1. **Direct Hardware Access**: Native support for inline assembly, memory-mapped I/O, and CPU/SIMD intrinsics.
1. **Strong System Interoperability**: Seamless C ABI interoperability and zero-overhead FFI to interact directly with existing OS kernels and C-ecosystem libraries.

## Quick Look

In Vx, you have explicit, type-safe control over where data lives and where code executes:

```rust
// Declare a verified matrix multiplication
fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>, b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {

    // Explicitly dispatch computation to an AI Accelerator
    spawn on(Topology::NPU[0]) {
        let mut result = Tensor([4, 4]).with_memory(Memory::NPU_HBM);

        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0;
                for k in 0..4 {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }

        // Return a verified result
        return Verified(result);
    }
}
```

Here is a trade-off matrix comparing the current heavyweight ML ecosystems against the **Vx** architecture. This table evaluates them across the dimensions of compiler design, hardware targeting, and developer experience.

### The Trade-Off Matrix

| Feature / Dimension | PyTorch | JAX + XLA | Mojo | Vx (Your Design) |
| --- | --- | --- | --- | --- |
| **Execution Model** | Eager by default (Define-by-Run). JIT available via `torch.compile`. | Lazy / JIT Compiled. Purely functional static graphs. | AOT / JIT Compiled. Eager fallback available for Python parity. | Strict **AOT Compiled**. Topologies and regions defined statically. |
| **Hardware Abstraction** | Opaque C++ Run-times (ATen/CUDA). Vendor must write heavy FFI integrations. | High-Level Operations (HLO). XLA compiler handles backend lowering. | MLIR-native. Hardware targeted via specific MLIR dialects and passes. | **Topologies as Types**. Vendor provides MLIR pass plugins; compiler handles the rest. |
| **Memory Control** | Implicit. Python GC and runtime allocator handle VRAM/DRAM. | Implicit. Functional purity means the compiler manages all memory/buffers. | Hybrid. Explicit ownership (borrow checker) available, but implicit allowed. | **Explicit Type-State**. `Pinned<T>` and `Verified<T>` enforce physical affinity at compile-time. |
| **Debuggability (UX)** | **Supreme.** Native Python `print()`, breakpoints, and standard stack traces. | **Difficult.** Tracer errors. Cannot easily inspect intermediate tensors dynamically. | **Good.** Familiar Python syntax, but compiled nature introduces some opacity. | **Shift-Left.** Hardware/memory errors caught at compile-time via type system, preventing runtime segfaults. |
| **Concurrency / Parallelism** | Threading is limited by Python GIL. Distributed requires heavy RPC/NCCL libraries. | SPMD (Single Program, Multiple Data) is first-class via `pmap`/`jit`. | High-performance CPU threading. SIMD and accelerator targeting. | **Lock-Free DOD Compiler.** Language natively supports `spawn on` for distributed asynchronous execution. |
| **Compiler Architecture** | AST to C++ binding. `TorchDynamo` uses bytecode analysis to build graphs. | Python AST to JAXpr to StableHLO to XLA runtime. | Python-superset AST directly lowering to MLIR dialects. | **AST Annihilation.** Parses to flat GID arrays, SIMD patched, directly lowered to MLIR. |

______________________________________________________________________

### Analytical Breakdown: Where Vx Wins and Loses

#### 1. Where Vx Dominates: The Hardware Boundary

- **PyTorch/JAX:** Treat hardware as an *infrastructure problem*. You write code, and a massive runtime environment tries to figure out how to ship it to the GPU/TPU.
- **Mojo:** Treats hardware as a *systems programming problem*. It gives you the pointers and SIMD registers to write fast code.
- **Vx:** Treats hardware as a **Language Semantic**. By elevating `Topology` and `MemorySpace` into the type system (`Pinned<T, NPU_HBM>`), Vx mathematically guarantees that a host CPU cannot accidentally dereference an NPU pointer. You catch cluster-level routing bugs at compile time. No other language does this cleanly.

#### 2. Where Vx Faces Friction: The Eager Penalty

- **PyTorch** won because researchers could treat it like a giant NumPy calculator. They can write chaotic, dynamic `if/else` loops that change on every iteration.
- **Vx** is strictly compiled and data-oriented. If a user wants to read a tensor shape, print it to the console, and dynamically alter the neural network architecture mid-step, Vx will inherently struggle more than PyTorch because Vx wants to build a static MLIR block to hand off to the NPU plugin. You will have to invest heavily in a JIT/REPL environment to win over pure researchers.

#### 3. The MLIR Synergy (Vx vs. Mojo)

Mojo and Vx share the architectural decision: **Lowering directly to MLIR.** However, they take different paths:

- **Mojo** is trying to be a superset of Python. It has to carry the baggage of Python's dynamic semantics, object models, and syntax to win over the existing ecosystem.
- **Vx** (based on your DOD parallel compiler) drops the legacy baggage. By forcing a nominal type system and flat arrays, Vx's compiler frontend will likely be orders of magnitude faster at compiling massive codebases than Mojo's, because Vx doesn't have to negotiate with Python-style dynamic typing heuristics.

Vx is a language that datacenter architects and systems engineers wish they had. PyTorch will always own the "hacky research" phase, but **Vx is positioned perfectly for the "production deployment and hardware scale-out" phase**, where memory determinism, zero-overhead dispatch, and MLIR hardware plugins are the difference between a profitable AI cluster and a bottlenecked one.

## Usage & Tooling

The Vx compiler (`vxc`) is written in Rust and utilizes the LLVM/MLIR infrastructure for lowering and execution.

### Building the Compiler

```bash
cargo build --release
```

### Running Vx Programs

You can run a `.vx` script using the built-in JIT compilation engine:

```bash
cargo run --release --bin vxc -- --run source_file.vx
```

### Benchmarking

Vx includes a custom native benchmarking suite that injects high-resolution timing harnesses directly into the Abstract Syntax Tree (AST) to measure true hardware execution time.

To run the full suite across your heterogeneous topologies:

```bash
cargo vx-bench
```

## Roadmap

We are actively building out the language features, the type-checker, and the MLIR optimization pipeline. Please check out the [ROADMAP.md](./ROADMAP.md) for a detailed list of completed and upcoming milestones!
