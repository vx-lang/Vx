# Akar Type System

This document outlines the topology-aware type system of the **Akar** programming language. Akar's type system is designed to catch data-movement bugs, misaligned execution targets, and hardware saturation at compile time.

## 1. Core Philosophy: Address Spaces as First-Class Types

In standard languages, a pointer `*mut T` or reference `&T` only encodes the data type, assuming a uniform, flat memory address space.
In Akar, a reference intrinsically encodes both the data type and its physical or logical address space.

### The `Ref<T, Memory>` Type

The fundamental data reference type is `Ref<T, Memory>`.

```rust
// A reference to a Matrix located in the Host's DRAM
let host_matrix: Ref<Matrix, Memory::Host_DRAM> = ...;

// A reference to a Matrix located in an NPU's High Bandwidth Memory
let npu_matrix: Ref<Matrix, Memory::NPU_HBM> = ...;
```

**Type Checking Rule 1 (Spatial Isolation):**
The compiler strictly prohibits direct operations between types in different memory spaces without explicit data movement.

```rust
// COMPILE ERROR: Cannot add variables in different address spaces
let c = host_matrix + npu_matrix; 

// CORRECT: Must explicitly transfer ownership
let npu_matrix_b = transfer(host_matrix, Memory::NPU_HBM);
let c = npu_matrix_b + npu_matrix; 
```

## 2. Hardware-Aware Typestates

Akar models the non-deterministic nature of distributed, heterogeneous execution using typestates. Computations and tasks are typed based on their topological binding and availability.

### 2.1 `Verified<T>`: The Agile Compute Type

When you define a computation without specifying an exact physical target, the compiler returns a `Verified<T>`. 

```rust
let task: Verified<Tensor> = matmul(A, B);
```

- **Semantics:** The compiler has statically verified that the computation is spatially correct and memory-safe.
- **Routing:** The compiler retains the right to dynamically route this computation to any available hardware that satisfies the cost model (e.g., an available NPU, GPU, or falling back to CPU).

### 2.2 `Pinned<T, Topology>`: The Strict Compute Type

When you require deterministic latency or specific hardware features, you bind a computation to a strict topology.

```rust
let strict_task: Pinned<Tensor, Topology::NPU[0]> = matmul(A, B);
```

- **Semantics:** The computation *must* execute on the specified topology.
- **Routing:** If the target topology is unavailable or saturated, the program cannot proceed unless explicitly handled.

## 3. The Hardware State Monad

Because physical hardware may be saturated, failed, or unavailable, bridging `Verified<T>` to `Pinned<T, Topology>` is an inherently fallible operation. Akar represents this via the `HardwareState` enum, which acts like a monad.

```rust
enum HardwareState<T, Topo> {
    // The hardware is available and the computation is successfully pinned.
    Available(Pinned<T, Topo>),
    
    // The hardware is saturated or unavailable. Returns the original unpinned computation.
    Saturated(Verified<T>),
}
```

**Usage via `try_pin`:**

```rust
let compute_task: Verified<Tensor> = matmul(A, B);
let target: HardwareState<Tensor, Topology::Acc1Core> = compute_task.try_pin(Topology::Acc1Core);

match target {
    HardwareState::Available(pinned_task) => {
        // Safe to execute strictly on Acc1Core
        pinned_task.execute();
    },
    HardwareState::Saturated(agile_task) => {
        // Hardware busy, let the runtime route it anywhere
        agile_task.execute_anywhere();
    }
}
```

## 4. Effect Tracking for Synchronization

Akar tracks synchronization and side-effects across address spaces as part of the function signature.

```rust
// This function signature indicates that it performs asynchronous data 
// movement to the NPU and has temporal side-effects.
fn pipeline() -> Verified<()> 
    effects(DataMovement(Memory::Host_DRAM -> Memory::NPU_HBM)) 
{
    // ...
}
```

## 5. Topology-Aware Allocation

When defining arrays, you can use Topology literals to specify the physical location of the memory as well as minimum alignment requirements (if applicable). If no alignment is specified, the compiler will use the default alignment for the type. For example:

```rust
// Allocate memory on the NPU
let x: Ref<Tensor, Topology::NPU_Core> = Tensor::new([8, 8]) with Topology::NPU_Core;

// Allocate memory on the Host
let y: Ref<Tensor, Topology::Host_Core> = Tensor::new([8, 8]) with Topology::Host_Core;

// Align to 128 bytes (A16)
let z: Ref<Tensor, Topology::NPU_Core(128)> = Tensor::new([8, 8]) with Topology::NPU_Core(128);
```