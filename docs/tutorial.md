# Vx Programming Language Tutorial

Welcome to Vx! Vx is a systems programming language and deep learning compiler that seamlessly bridges the gap between high-level hardware-agnostic coding and low-level hardware-specific optimizations.

This tutorial will guide you through the basics of Vx, from syntax to advanced hardware targeting.

## 1. Basic Syntax

Vx uses a syntax familiar to Rust and Swift developers.

```vx
fn main() -> i32 {
    let x: i32 = 42;
    let mut y = 10;
    y += x;
    return 0;
}
```

### Variables and Types

Variables are declared with `let`. Use `mut` to make them mutable.
Supported types include standard primitives (`i32`, `f32`, `f64`, `bool`), custom structs, and n-dimensional tensors.

## 2. Advanced Semantics

### Tensors and Math

Vx has native support for tensors, designed for machine learning workflows.

```vx
fn compute(a: Tensor<f32, 100x100>, b: Tensor<f32, 100x100>) -> Tensor<f32, 100x100> {
    return a @ b; // Native matrix multiplication
}
```

### Automatic Differentiation

Vx incorporates built-in primitives for automatic differentiation (Autograd).
You can calculate gradients natively using `grad`, `vjp`, and `jvp` expressions:

```vx
fn my_loss(x: f32) -> f32 {
    return x * x;
}

fn compute_gradient() {
    let gradient_fn = grad(my_loss);
    // ...
}
```

## 3. Hardware Targeting and Topologies

Vx enables you to deploy functions specifically to targeted hardware like the Apple Neural Engine (ANE), GPUs, and custom accelerator cores using **Topologies**.

### The `spawn` Expression

You can execute tasks on specific hardware using the `spawn` block:

```vx
fn offloaded_task() -> f32 {
    return 3.14;
}

fn main() -> i32 {
    // Spawns the task onto the Apple Neural Engine (ANE)
    let result = spawn on(Topology::ANE) { offloaded_task() };
    return 0;
}
```

### Supported Topologies

- `@Host`: The default CPU host.
- `@GPU`: Standard GPU offloading.
- `@ANE`: Apple Neural Engine.
- `@AMX`: Apple Matrix Co-processor.
- `@NPU(n)`: Specific Neural Processing Units.
- `@AccCore(n)`: Custom Accelerator Cores.

### Data Transfers

Memory spaces are explicit, allowing predictable performance. Use `transfer` to move data between memory spaces (e.g., from Host DRAM to NPU HBM).

```vx
let npu_tensor = transfer(my_host_tensor, Memory::NPU_HBM);
```
