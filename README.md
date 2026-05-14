<div align="center">
  <h1>Vx Language</h1>
  <p><b>One Language, Every Core.</b></p>
  <p>A high-performance systems programming language built from the ground up for heterogeneous computing.</p>
</div>

---

## ⚡ What is Vx?

**Vx** (pronounced *"vee-ex"*) is a general-purpose systems programming language designed to unify CPU, GPU, NPU, and accelerator workloads. 

Historically, leveraging heterogeneous hardware required disjointed toolchains, painful FFI boundaries, and complex vendor-specific frameworks (like CUDA, Metal, or OpenCL). **Vx treats hardware diversity not as a challenge, but as a first-class citizen.** It bridges execution topologies and memory domains under a single, verifiable syntax.

## Core Philosophy

The language is governed by 7 core tenets:

1. **Heterogeneous Compute**: Address spaces and compute topologies (CPUs, GPUs, NPUs) are first-class primitives. Distributed and parallel computations are expressed natively (e.g., `spawn on(Topology::NPU[0])`).
2. **Ease of Verified Computation**: Hardware-aware type systems, explicit topologies, and `Verified<T>` wrappers allow programmers to verify computation correctness and data locality.
3. **High Performance**: Designed for Ahead-Of-Time (AOT) optimizations. The compiler lowers directly to MLIR and LLVM IR for optimal native machine code.
4. **Deterministic Memory Control**: No mandatory garbage collection. Programmers have control over memory layouts, lifetimes, and pointer arithmetic.
5. **Zero-Cost Abstractions**: High-level constructs compile down to optimal machine code with no runtime overhead.
6. **Direct Hardware Access**: Native support for inline assembly, memory-mapped I/O, and CPU/SIMD intrinsics.
7. **Strong System Interoperability**: Seamless C ABI interoperability and zero-overhead FFI to interact directly with existing OS kernels and C-ecosystem libraries.

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
