# Vx Language Syntax

This document outlines the core syntax of the **Vx** programming language. Vx uses a C-family, Rust-like syntax but introduces novel constructs for spatial execution and distributed state.

## 1. Basic Structure

Vx programs are structured into modules, functions, and scopes.

```rust
// A basic function
fn compute_metrics(data: Tensor<f32>) -> Tensor<f32> {
    // Variable declaration
    let intermediate = data.map(|x| x * 2.0);
    intermediate
}
```

> [!NOTE]
> **Implicit Returns**: Vx supports Rust-style implicit returns. The final expression in a block (such as a function body, `comptime` block, or `unsafe` block) can omit the trailing semicolon, causing the block to evaluate to the value of that expression. This avoids boilerplate `return` statements.

## 2. Spatial Execution Scopes: `spawn on`

To explicitly route computation to a specific physical topology, Vx uses the `spawn on` block. This overrides the compiler's default cost-model inferred routing.

```rust
fn distributed_matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    comptime {
        assert(a.shape[1] == b.shape[0], "Inner dimensions must match for matmul!");
    }
    // Spawn computation on a specific NPU core
    spawn on(Topology::Acc1Core[0]) {
        let result = custom_matmul(a, b);
        // Compute happens entirely on Acc1Core[0]
        result
    }
}
```

## 3. Logical and Relational Operators

- Compound assignment: `+=`, `*=`
- Relational Operators: `==`, `!=`, `<`, `>`, `<=`, `>=` (Returns a Boolean evaluation)
- Logical Operators: `&&`, `||`, `!` (Requires Boolean operands)

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

## 5. Control Flow

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

## 6. Foreign Function Interface (FFI) & Safety

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
