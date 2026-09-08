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
fn compute(a: Tensor<f32, [100, 100]>, b: Tensor<f32, [100, 100]>) -> Tensor<f32, [100, 100]> {
    return a @ b; // Native matrix multiplication
}
```

A tensor's shape is a bracketed list and is part of its type. Use `?` for an extent that is only
known at run time — `Tensor<f32, [?, ?]>` — and read it back with `.extent(0)`.

### Automatic Differentiation

Vx has built-in primitives for automatic differentiation: `grad`, `vjp` and `jvp`. Each takes the
function **and the point to differentiate at**, and evaluates to a value — they do not return a
function.

```vx
fn my_loss(x: f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let x : f32 = 3.0;
    let dx : f32 = grad(my_loss, x);   // 6.0
    print(dx);
    return 0;
}
```

Autodiff lowers through the Enzyme MLIR plugin, which is an optional component: build it with
`./scripts/provision/install_enzyme.sh` and point `ENZYME_LIB` at the result. Without it the rest
of the compiler works and only `grad`/`jvp`/`vjp` are unavailable.

## 3. Hardware Targeting and Topologies

Vx enables you to deploy functions specifically to targeted hardware like the Apple Neural Engine (ANE), GPUs, and custom accelerator cores using **Topologies**.

### The `spawn on` block

`spawn on` runs a block of code on a named topology. The work goes **inside** the block, and every
value it touches has to already live in a memory that topology can address — which is what the
`transfer` below is for.

```vx
fn main() -> i32 {
    let mut host : Tensor<f32, [4, 4]> = Tensor<f32, [4, 4]>::uninit();
    for i in 0..4 {
        for j in 0..4 {
            host[i][j] = 1.0;
        }
    }

    let mut device = transfer(host, Memory::NPU_HBM);

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                device[i][j] += 1.0;
            }
        }
    }

    return 0;
}
```

Note that the block calls no helper function. A function declared without a topology belongs to
the host, and calling it from inside a device region is a compile error
(`E6001: Function 'f' requires topology 'CPU', but is called from 'ANE'`) — the address-space rule
doing its job. Code meant to run on a device is written in the region, or in a function declared
for that topology.

> **Not implemented yet.** `spawn on` is a statement today. The design intends it to be an
> *expression* yielding a `Future`, so a host thread could fan work out across several
> accelerators and join them later — but there is no `await` in the language at present, and no
> future type. Treat any document describing that shape as design intent rather than as
> something you can call.

### Built-in topologies

Written `Topology::NAME`, as in the example above. The `@` spelling in earlier drafts of this
document was never implemented — `@` is the matrix-multiplication operator.

| Topology | Meaning |
| --- | --- |
| `Topology::CPU` | The host CPU. |
| `Topology::CPU_AVX512`, `Topology::CPU_Neon` | The host CPU, restricted to an instruction set. |
| `Topology::GPU`, `Topology::GPU[n]` | A GPU. Bare `GPU` means device 0. |
| `Topology::NPU[n]` | A neural processing unit. The index is required. |
| `Topology::ANE` | Apple Neural Engine. |
| `Topology::AMX` | Apple matrix co-processor. |
| `Topology::AccCore[n]` | A custom accelerator core. The index is required. |
| `Topology::Current` | Whichever topology the enclosing region compiles for. |

Any other name is treated as a **user-defined topology**, resolved from the `Topology`
declarations in a [machine file](lang/hosts_and_machines.md). That is how a new accelerator is
added without changing the compiler — but it also means a typo in a built-in name is silently read
as a custom topology rather than reported.

### Data Transfers

Memory spaces are explicit, allowing predictable performance. Use `transfer` to move data between memory spaces (e.g., from Host DRAM to NPU HBM).

```vx
let npu_tensor = transfer(my_host_tensor, Memory::NPU_HBM);
```
