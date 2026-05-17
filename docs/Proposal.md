# Concept Draft: Native Programming Language for Heterogeneous Systems

**Author:** aditya@
**Date:** May 11, 2026

## 1. Abstract & Core Philosophy
Current programming languages model an abstract machine designed for a single address space. To execute programs on modern heterogeneous systems (multi-NPU datacenter topologies, specialized AI accelerators), compilers and runtimes rely on intrinsics and specific dialects to orchestrate compute across different machines.

This concept outlines a new programming language paradigm where **address spaces and heterogeneous compute are modeled as first-class citizens**. Instead of bolting on data movement and remote execution as afterthoughts or library calls (e.g., MPI, `cudaMemcpy`), the language natively understands physical hardware boundaries, spatial execution, and temporal availability.

## 2. Key Architectural Pillars

### 2.1 Topology-Aware Type Systems
Pointers and variables intrinsically encode their physical or logical address space.
* **Example:** `Ref<Matrix, NPU_HBM>` vs. `Ref<Matrix, Host_DRAM>`.
* **Benefit:** The type checker validates that compute operations are localized to the correct memory space, effectively turning data-movement bugs and misaligned execution targets into compile-time errors.

### 2.2 Semantics of Data Movement
Because address spaces are distinct, data movement is elevated to a fundamental language primitive. Moving data from a host to an accelerator is handled as a semantic transfer of ownership and state across physical boundaries, guaranteeing liveness and spatial correctness.

### 2.3 The Ontology of Distributed State
Execution across isolated, heterogeneous memories introduces non-determinism. The language incorporates temporal logic directly into its core semantics to manage synchronization, ensuring valid program states without relying on opaque runtime overhead.

* **Compile-Time Execution Engine (`comptime`)**: An embedded AST interpreter that enforces tensor layouts, hardware assertions, and constraint solving *ahead-of-time*. Shapes and mathematical verification occur as a zero-cost abstraction during Semantic Analysis.

### 2.4 Ecosystem Delegation (The Scale-Out Strategy)
A language without a rich standard library struggles to gain adoption. Rather than rebuilding the world, Vx acts as a pure **topology-aware frontend routing layer**. 
* **Host Topologies:** Standard operations (like networking, file I/O, and data structures like `Option<T>` or `Vec<T>`) are delegated seamlessly to a robust host ecosystem. Instead of transpiling, Vx natively lowers to MLIR and links against pre-compiled standard libraries (e.g., C/C++ or a pre-compiled Rust core exposing a C ABI) via zero-overhead FFI.
* **Accelerator Topologies:** Accelerator execution is delegated to heavily optimized, vendor-specific libraries (e.g., Apple's MLX/NPE, Nvidia's cuDNN, AMD's ROCm) depending on the active `Topology`.
* **Benefit:** By delegating the implementation details to specialized ecosystems, attaching a new hardware topology to Vx becomes trivial. The language focuses entirely on the mathematical correctness of data movement and routing between these ecosystems, which is one of Vx's biggest innovations.

## 3. Hybrid Routing & Control Mechanics
To serve both rapid development and strict performance requirements, the language supports both compiler-inferred and user-defined execution routing.

### 3.1 Implicit Routing (Data Affinity as Destiny)
The compiler acts as a static cost-model evaluator. If variables are typed to reside in specific memory spaces, the compiler infers the compute target based on data affinity.
* *Mechanics:* Writing a standard mathematical operation automatically triggers lowering to the adjacent compute unit (e.g., an NPU or Tensor Core) because moving the data back to the Host CPU would violate the cost model.

### 3.2 Explicit Control (Spatial and Temporal Scoping)
For deterministic performance and strict locality, programmers can use **Execution Scopes** to override the compiler's cost model.

```rust
spawn on(Topology::Acc1Core[0]) {
    let local_data = transfer(input_tensor, Memory::LocalSRAM);
    let optimized_result = custom_matmul(local_data);
}

```

## 4. Hardware-Aware Typestates: Execution Modes as Types

To manage the unpredictability of hardware availability, the language encodes the *execution mode* directly into the type system, treating the datacenter topology as a state machine.

### 4.1 The Agile Default: `Verified<T>`

Standard operations yield a `Verified` type. The compiler guarantees execution correctness but retains the freedom to route the computation across the heterogeneous mesh based on real-time availability.

```rust
let async_result: Verified<Tensor> = matmul(A, B);

```

### 4.2 The Strict Mode: `Pinned<T, Topology>`

For systems requiring deterministic latency, computation is explicitly bound to a physical target. If the compiler cannot guarantee this execution path, it throws a compile-time error.

```rust
let strict_result: Pinned<Tensor, Topology::NPU[0]> = matmul_optimized(A, B);

```

### 4.3 The Monadic Hardware Bridge

Developers can smoothly transition between agile and strict modes using monadic constructs, handling *absent hardware* much like handling absent data.

```rust
let target_compute = async_result.try_pin(Topology::Acc1Core);

match target_compute {
    HardwareState::Available(pinned_tensor) => {
        pinned_tensor.execute();
    },
    HardwareState::Saturated(fallback_tensor) => {
        fallback_tensor.execute_anywhere(); // Reverts to Verified<T> behavior
    }
}

```

## 5. Next Steps for Compiler Lowering

The next phase of research involves defining how these frontend syntactical types map down through the compiler stack—specifically, whether modeling temporal hardware states requires a novel IR dialect or can be represented through existing abstraction layers.

