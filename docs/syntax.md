# Vx Language Syntax

This document outlines the core syntax of the **Vx** programming language. Vx uses a C-family, Rust-like syntax but introduces novel constructs for spatial and temporal logic, distributed state, and hardware topologies.

## 1. Basic Structure

Vx programs are structured into modules, functions, and scopes.

```rust
// A basic function with statically verified layouts
fn compute_metrics(data: Ref<Tensor<f32, [128, 256]>, Host_DRAM>) -> Verified<Tensor<f32, [128, 256]>> {
    // Variable declaration
    let intermediate = data.map(|x| x * 2.0);
    return intermediate;
}
```

## 2. Topologies & Memory Spaces

Vx introduces hardware-aware keywords: `Topology` and `Memory`.

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

To explicitly route computation to a specific topology, Vx uses the `spawn on` block. This overrides the compiler's default cost-model inferred routing.

```rust
fn distributed_matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    comptime {
        assert(a.shape[1] == b.shape[0], "Inner dimensions must match for matmul!");
    }
    // Spawn computation on a specific NPU core
    spawn on(Topology::Acc1Core[0]) {
        let result = custom_matmul(a, b);
        // Compute happens entirely on Acc1Core[0]
    }
}
```

## 4. Logical and Relational Operators

- Compound assignment: `+=`, `*=`
- Relational Operators: `==`, `!=`, `<`, `>`, `<=`, `>=` (Returns a Boolean evaluation)
- Logical Operators: `&&`, `||`, `!` (Requires Boolean operands)

## 5. Semantics of Data Movement: `transfer`

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

Vx encodes execution modes as types to handle non-deterministic hardware availability.

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

## 7. Foreign Function Interface (FFI) & Safety

Vx supports calling external C functions via the `extern` block. By default, all external functions are considered `unsafe` because the compiler cannot statically verify their memory safety across the language boundary. Calling an `unsafe` function requires an `unsafe { ... }` block.

However, many C functions (like simple math functions, standard library I/O, or thoroughly tested user kernels) are inherently safe or have been manually verified by the programmer. Vx allows you to claim responsibility for this safety by annotating the FFI declaration with the `safe` keyword:

```rust
extern {
    // Unsafe by default. Requires `unsafe { ... }` at call sites.
    fn vx_malloc_f32(num_elements: i32) -> *mut f32;
    
    // Explicitly marked as safe. Can be called freely in pure Vx code!
    safe fn vx_decode_token(tokenizer_ptr: *mut i8, prev_token: i32, token: i32) -> *mut i8;
}
```

**Motivation**: The `safe` keyword delegates the safety assertion to the interface boundary. This prevents the codebase from being littered with repetitive `unsafe` blocks for functions that are already trusted, keeping your application logic clean and robust while maintaining strict boundaries for actual unsafe operations (like pointer arithmetic or arbitrary memory mapping).
