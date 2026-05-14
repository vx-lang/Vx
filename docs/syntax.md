# Akar Language Syntax

This document outlines the core syntax of the **Akar** programming language. Akar uses a C-family, Rust-like syntax but introduces novel constructs for spatial and temporal logic, distributed state, and hardware topologies.

## 1. Basic Structure

Akar programs are structured into modules, functions, and scopes.

```rust
// A basic function
fn compute_metrics(data: Ref<Tensor, Host_DRAM>) -> Verified<Tensor> {
    // Variable declaration
    let intermediate = data.map(|x| x * 2.0);
    return intermediate;
}
```

## 2. Topologies & Memory Spaces

Akar introduces hardware-aware keywords: `Topology` and `Memory`.

### Topologies
Topologies represent physical compute resources (e.g., CPU, GPU, NPU, TPU).

```rust
// Examples of topology identifiers:
Topology::Host
Topology::Acc1Core[0]    // 0th core of Accelerator 1
Topology::NPU[0..4]      // A slice of 4 NPUs
```

### Memory Spaces
Memory spaces represent physical memory boundaries.

```rust
// Examples of memory spaces:
Memory::Host_DRAM
Memory::NPU_HBM
Memory::LocalSRAM
```

## 3. Spatial Execution Scopes: `spawn on`

To explicitly route computation to a specific topology, Akar uses the `spawn on` block. This overrides the compiler's default cost-model inferred routing.

```rust
fn distributed_matmul(a: Tensor, b: Tensor) {
    // Spawn computation on a specific NPU core
    spawn on(Topology::Acc1Core[0]) {
        let result = custom_matmul(a, b);
        // Compute happens entirely on Acc1Core[0]
    }
}
```

## 4. Semantics of Data Movement: `transfer`

Data cannot be implicitly moved across address spaces. Moving data requires the `transfer` primitive, which explicitly tracks ownership and liveness across boundaries.

```rust
fn heterogeneous_pipeline(host_input: Ref<Tensor, Memory::Host_DRAM>) {
    spawn on(Topology::NPU[0]) {
        // Explicitly transfer data from Host DRAM to NPU HBM
        let local_data = transfer(host_input, Memory::NPU_HBM);
        
        // Execute computation on the local data
        let result = process(local_data);
        
        // Transfer result back to Host DRAM
        let host_result = transfer(result, Memory::Host_DRAM);
    }
}
```

## 5. Hardware-Aware Typestates

Akar encodes execution modes as types to handle non-deterministic hardware availability.

### 5.1 The Agile Default: `Verified<T>`
Yielded by standard operations. The compiler guarantees execution but routes it dynamically based on the best available hardware.

```rust
let async_result: Verified<Tensor> = matmul(A, B);
```

### 5.2 The Strict Mode: `Pinned<T, Topology>`
Computation is strictly bound to a physical target.

```rust
let strict_result: Pinned<Tensor, Topology::NPU[0]> = matmul_optimized(A, B);
```

### 5.3 Monadic Hardware Bridge: `try_pin`
Transitions an agile computation into a strict computation, handling hardware saturation gracefully.

```rust
let target_compute = async_result.try_pin(Topology::Acc1Core);

match target_compute {
    HardwareState::Available(pinned) => {
        pinned.execute(); 
    },
    HardwareState::Saturated(fallback) => {
        // Hardware is busy/unavailable, fallback to agile execution
        fallback.execute_anywhere(); 
    }
}
```

## 6. Control Flow

Standard Rust-like control flow is supported: `if`, `else`, `match`, `for`, `while`.
Loops can be annotated for spatial unrolling.

```rust
// Unroll this loop across 4 NPUs
unroll across(Topology::NPU[0..4]) { |npu_id|
    spawn on(npu_id) {
        let chunk = transfer(data[npu_id], Memory::NPU_HBM);
        process(chunk);
    }
}
```
