<div align="center">
  <h1>Vx Language</h1>
  <p><b>One Language, Every Core.</b></p>
  <p>A high-performance systems programming language built from the ground up for heterogeneous computing.</p>
</div>

______________________________________________________________________

## ⚡ What is Vx?

**Vx** (pronounced *"vee-ex"*) is a general-purpose systems programming language designed to unify CPU, GPU, NPU, and accelerator workloads.

Historically, leveraging heterogeneous hardware required disjointed toolchains, painful FFI boundaries, and complex vendor-specific frameworks (like CUDA, Metal, or OpenCL). **Vx treats hardware diversity not as a challenge, but as a first-class citizen.** It bridges execution topologies and memory domains under a single, verifiable syntax.

The thesis in one line: **heterogeneity belongs in the type system, not in the runtime.** A host CPU dereferencing an NPU pointer should be a type error, not a segfault.

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

In Vx, developers have explicit, type-safe control over where data lives and where code executes:

```rust
// Declare a verified matrix multiplication
fn custom_matmul(A: Ref<Tensor, Memory::NPU_HBM>,
                 B: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
    // Explicitly dispatch computation to an AI Accelerator
    spawn on(Topology::NPU[0]) {
        // Natively lowers to linalg.matmul
        let C = A * B;
        // Return a verified result
        return Verified(C);
    }
}
```

Crossing a memory domain requires an explicit `transfer()` **even when the hardware boundary is free** (Apple unified memory, for instance), so data locality is always provable from the source text. `spawn on` is an expression yielding `Future<Pinned<T, Topology>>`, so a single host thread can fan out across eight accelerators without blocking.

______________________________________________________________________

## The Machine Is Declared, Not Assumed

Most compilers hard-code a cost model. Vx reads one. A **machine file** describes the memory hierarchy and interconnect of a real part, and the compiler admits or rejects placements against it:

```rust
Memory HBM  { capacity: 80 GiB, bandwidth: 3.35 TB/s, managed: explicit, scope: device }
Memory L2   { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s, managed: cached }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc,
              clock: 1.98 GHz, replicas: 132, granule: 1 KiB, scope: sm }

Topology Device {
    arch: nvptx64,
    memory: Memory::HBM,
    transfer Memory::CPU_DRAM -> Memory::HBM : 63 GB/s,
}
```

From this the compiler derives, before any binary exists:

- **Admission** — whether a tensor's working set fits the space it is placed in, with granule rounding. A rejection is as informative as a cost.
- **Routing** — the cheapest legal path between two spaces, over the declared transfer graph.
- **Transfer cost** — a roofline over the containment tree. A containment hop is charged at *both* endpoints, because data has to leave the parent as well as enter the child.
- **Coherence** — `within:` is acyclic, a child never exceeds its parent's capacity, scope narrows going down, and each edge has exactly one cost source.

Units are exact integer conversions, never floats: SI prefixes are decimal (`GB` = 10⁹) and IEC binary (`GiB` = 2³⁰), so a figure copied from a vendor sheet means what the sheet meant.

`fleet/` ships 13 machine files — H100, H200, B200, A100, MI300X, Apple M4 UMA, multi-GPU nodes and hosts — each citing its sources and marking unverified figures as such. The model is validated against rented hardware rather than asserted; see [`docs/memory_algebra.md`](docs/memory_algebra.md) and `utils/memalg/` for the measurement instruments.

______________________________________________________________________

## What the Compiler Proves

Vx front-loads into type checking a class of bug that normally surfaces as a runtime crash, silent corruption, or an OOM at step 1200:

| Check | What it rules out |
| --- | --- |
| **Address-space typing** | Dereferencing a device pointer from the host; a `Pinned<T, NPU_SRAM>` escaping to a host expression |
| **Borrow checker** | Aliasing and lifetime errors, with variance and region tracking on the flat path |
| **Linear types** | Use-after-move of a consumed buffer |
| **Capacity admission** | A placement whose working set cannot fit the space it targets |
| **Seam contracts** | Reading a buffer whose asynchronous `transfer` has not been made visible — discharged by an SMT prover |
| **Topology reachability** | A transfer between spaces with no declared path |
| **Autodiff** | Differentiating through a region whose adjoint is not defined |

`Verified<T>` marks a value whose computation carried its proof obligations to completion.

______________________________________________________________________

## How It Compiles

### A data-oriented parallel frontend

Every symbol, nominal type and monomorphized variant is a flat **256-bit GID** (`[u64; 4]`: module hash, symbol hash, generic context, flags). A nominal type system plus mandatory boxing for recursive types decouples modules, so the pipeline is parallel across cores with no query engine and no lock contention. Compilation walks flat arrays, not pointer-chased trees.

The output is deterministic: the emitted MLIR is byte-identical regardless of thread count, which is asserted in the test suite rather than hoped for — over 1,000 modules and 16,000 functions, at one thread, at four, and with rayon taken off the path entirely.

### Backends

| Target | Path |
| --- | --- |
| **CPU (x86-64, arm64)** | MLIR → LLVM IR → native, AOT or JIT |
| **NVIDIA GPU** | MLIR → NVVM → PTX → SASS |
| **Apple AMX / ANE** | CoreML primitive dispatch via plugin |
| **Distributed** | Manifest-driven remote regions over a wire protocol |

Vendors extend the compiler through **MLIR pass plugins** rather than by patching it; see `src/plugin/`.

______________________________________________________________________

## Repository Layout

```
src/                  the vxc compiler (Rust)
  lexer, parser/      source → AST
  syntax/             AST, types, topologies, declarations
  hir/                lowering, type checking, borrow checking
    check/            transfer, calls, access, autodiff, region traffic
    memory.rs         the memory algebra: containment, capacity, derived cost
    seam.rs, prover.rs  asynchronous-visibility contracts, SMT discharge
    flatten.rs        the flat (AST-annihilated) path
  codegen/, dialect/  MLIR emission, the Vx dialect and lowering (C++)
  gid.rs, resolver.rs 256-bit global identifiers, parallel symbol resolution
  plugin/             vendor MLIR pass plugins
  jit.rs              JIT execution
fleet/                machine files: real parts, declared and cited
runtime/              dispatch and the distributed fleet runtime (C++)
stdlib/               21 std modules: tensor, simd, io, net, fs, hash_map, …
packages/             vx_nn, vx_linalg, vx_optim, vx_vision, vx_models
examples/llama.vx     a full Llama2 inference port
docs/                 design documents, plans, papers
utils/memalg/         measurement instruments for validating the cost model
vx-analyzer/          language server
vscode-vx/            VS Code extension
tests/                27 integration suites plus ~460 unit tests
```

______________________________________________________________________

## Comparison

| | **PyTorch** | **JAX + XLA** | **Triton** | **Mojo** | **Vx** |
| --- | --- | --- | --- | --- | --- |
| **What it is** | Eager tensor library in Python | Functional array language, traced and JIT'd | Python-embedded DSL for single GPU kernels | Python superset for systems + AI | General-purpose systems language |
| **Scope** | Whole model | Whole program | One kernel — no host program | Whole program | Whole program *and* cluster |
| **Execution** | Define-by-run; `torch.compile` opt-in | Trace → JAXpr → StableHLO → XLA | JIT per kernel → PTX | AOT/JIT, eager fallback | Strict AOT, regions static |
| **Hardware model** | Opaque C++ runtime (ATen/CUDA); vendors write heavy FFI | HLO; XLA owns backend lowering | NVIDIA-first; block-level tiles, threads abstracted away | MLIR dialects and passes | **Topologies as types**; vendors ship MLIR pass plugins |
| **Memory** | Implicit (GC + caching allocator) | Implicit (functional purity, compiler owns buffers) | Explicit *inside* a kernel only | Hybrid: ownership available, implicit allowed | **Explicit type-state**: `Pinned<T, NPU_HBM>` enforced at compile time |
| **Machine model** | None — the runtime discovers the device | None | None | None | **Declared machine files**; capacity and cost checked at compile time |
| **Multi-device** | RPC/NCCL libraries bolted on | SPMD first-class (`pmap`, `shard_map`) | Out of scope | Threading and SIMD; distribution not a language feature | `spawn on` and futures are language primitives |
| **Debuggability** | **Best in class** — `print()`, breakpoints, real stack traces | Hard — tracer errors, opaque intermediates | Hard — kernel-level, limited introspection | Good — Python-familiar, some compile opacity | **Shift-left** — device and memory errors are compile errors |

### Reading the table

**Vx vs PyTorch/JAX** — a different layer entirely. Those treat hardware as *infrastructure*: you write math, and a large runtime figures out how to ship it. Vx treats hardware as *language semantics*. You would not write a research training loop in Vx; you would write the runtime underneath it.

**Vx vs Triton** — Triton is the sharpest tool for one job: making a single fused GPU kernel fast, with thread-level scheduling automated away. It has no host program, no cross-device story, and no type-level memory model. Vx overlaps only at the innermost tile; the rest of Vx is the layer Triton assumes someone else wrote.

**Vx vs Mojo** — the closest comparison. Both lower directly to MLIR and both want one language for CPU plus accelerator. They diverge on legacy: Mojo buys the Python ecosystem and pays for it in dynamic semantics, an object model, and syntax it must honor. Vx drops that entirely — nominal types, flat GID arrays, a lock-free data-oriented frontend. Mojo gives you pointers and SIMD registers to *write* fast code; Vx gives you a type system that makes wrong-device code *unrepresentable*. Mojo has the ecosystem; Vx has the stronger claim on cluster-level correctness.

### Where Vx faces friction: the eager penalty

PyTorch users mutate architecture mid-loop, print a tensor shape, branch on it, and continue. In Vx — AOT, data-oriented, statically regioned — that same dynamism takes real work. **Vx is the right language for the thing that must be correct and fast across ten kinds of silicon. It is not the right language for the thing you are still figuring out.**

______________________________________________________________________

## Usage & Tooling

The Vx compiler (`vxc`) is written in Rust and uses LLVM/MLIR for lowering and execution.

### Building

```bash
# Generate config.local (only needed once)
./setup.sh

# Load environment variables — required before any cargo command
source config.local

cargo build --release
```

### Running

```bash
# JIT a script
cargo run --release --bin vxc -- --run source_file.vx

# AOT compile against a declared machine
cargo run --release --bin vxc -- --machine fleet/h100-sxm.vx program.vx -o program

# Inspect the admission and cost decisions as JSON
cargo run --release --bin vxc -- --machine fleet/h100-sxm.vx program.vx --diagnostics-json out.json
```

### Testing

The suite exercises Apple Silicon AMX (NPU) dispatchers and Enzyme MLIR plugins:

```bash
source config.local
export ENZYME_LIB="$(pwd)/.cargo/enzyme/LLVMEnzyme-22.dylib"

cargo test
```

### Additional tools

| Tool | Purpose |
| --- | --- |
| `vx-format` | Canonical source formatter |
| `vx-opt` | MLIR pass driver for the Vx dialect |
| `cargo vx-bench` | Benchmark suite that injects timing harnesses into the AST to measure true hardware execution time |
| `vx-analyzer` | Language server |
| `vscode-vx` | VS Code extension |

## Roadmap

The project is actively building out the language features, the type-checker, and the MLIR optimization pipeline. See [ROADMAP.md](./ROADMAP.md) for completed and upcoming milestones, and [`docs/`](docs/) for design documents.
