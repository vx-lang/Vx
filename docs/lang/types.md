# Vx Type System

This document outlines the topology-aware type system of the **Vx** programming language. Vx's type system is designed to catch data-movement bugs, misaligned execution targets, and hardware saturation at compile time.

## 1. Core Philosophy: Address Spaces as First-Class Types

In standard languages, a pointer `*mut T` or reference `&T` only encodes the data type, assuming a uniform, flat memory address space.
In Vx, a reference intrinsically encodes both the data type and its physical or logical address space.

### The `Ref<T, Memory>` Type

The fundamental data reference type is `Ref<T, Memory>`.

```rust
// A reference to a generic Matrix located in the Host's DRAM
let host_matrix: Ref<Matrix, Memory::CPU_DRAM> = ...;

// A reference to a Tensor with statically known layouts [128, 256] in NPU's High Bandwidth Memory
let npu_tensor: Ref<Tensor<f32, [128, 256]>, Memory::NPU_HBM> = ...;
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

## 2. Compile-Time Layouts & Shapes

Vx brings tensor dimensions and bounds checking entirely into the type system via `comptime` const-generics. Instead of relying on runtime panics for dimension mismatches, the `Sema` pass executes layout verification ahead-of-time.

```rust
// Tensor shapes are encoded directly in the type.
fn batch_norm(input: Tensor<f32, [N, C, H, W]>) -> Tensor<f32, [N, C, H, W]> { ... }

// Dimension assertions are evaluated at compile time
fn matmul(A: Tensor<f32, [M, K]>, B: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    comptime {
        assert(A.shape[1] == B.shape[0], "Inner dimensions must match!");
    }
    // ...
}
```

If dimensions are statically evaluated to mismatch, the compiler will refuse to compile, effectively removing zero-day out-of-bounds runtime errors for standard tensor operations.

## 3. Hardware-Aware Typestates

Vx models the non-deterministic nature of distributed, heterogeneous execution using typestates. Computations and tasks are typed based on their topological binding and availability.

### 3.1 `Verified<T>`: The Agile Compute Type

When you define a computation without specifying an exact physical target, the compiler returns a `Verified<T>`.

```rust
let task: Verified<Tensor> = matmul(A, B);
```

- **Semantics:** The compiler has statically verified that the computation is spatially correct and memory-safe.
- **Routing:** The compiler retains the right to dynamically route this computation to any available hardware that satisfies the cost model (e.g., an available NPU, GPU, or falling back to CPU).

### 3.2 `Pinned<T, Topology>`: The Strict Compute Type

When you require deterministic latency or specific hardware features, you bind a computation to a strict topology.

```rust
let strict_task: Pinned<Tensor, Topology::NPU[0]> = matmul(A, B);
```

- **Semantics:** The computation *must* execute on the specified topology.
- **Routing:** If the target topology is unavailable or saturated, the program cannot proceed unless explicitly handled.

### 3.3 Pinned Cross-Topology Access Rules

A `Pinned<T, TopologyA>` value can only be accessed from topologies that have visibility to `TopologyA`'s default memory space. The compiler enforces this statically from the topology registry (`arch::builtin_descriptors`); the built-in visibility sets are:

| Active Topology | Default space | Directly visible memory spaces |
|---|---|---|
| CPU | CPU_DRAM | CPU_DRAM, NPU_HBM |
| GPU | GPU_HBM | GPU_HBM, CPU_DRAM |
| NPU[i] | NPU_HBM | NPU_HBM |
| ANE | NPU_HBM | NPU_HBM, CPU_DRAM |
| AMX | CPU_DRAM | CPU_DRAM |
| AccCore[i] | Local_SRAM | Local_SRAM |
| CpuAvx512 | CPU_DRAM | CPU_DRAM |
| CpuNeon | CPU_DRAM | CPU_DRAM |

> [!NOTE]
> The table lists *direct* visibility (the `Visible` verdict). Where a space is not directly
> visible but a **transfer path exists**, access is still possible via an explicit
> `transfer(...)` (the `NeedsSeam` verdict); only a space with no path at all is truly
> `Unreachable`. User-defined `Topology` declarations (see [syntax.md](./syntax.md#31-user-defined-topologies))
> contribute their own visibility sets and transfer edges to this matrix.

**Example — rejected at compile time:**

```rust
fn invalid_cross_access() -> i32 {
    let t : Tensor<f32> = 1.0;
    let t_npu = transfer(t, Memory::NPU_HBM);

    spawn on(Topology::GPU) {
        // COMPILE ERROR: GPU cannot access NPU_HBM.
        // Pinned<Tensor, NPU> lives in NPU_HBM, which is
        // not in GPU's visibility set.
        let invalid = t_npu;
    }
    return 0;
}
```

> [!WARNING]
> This check applies to both explicit `Pinned<T, Topology>` types and any variable that was transferred to a topology-specific memory space (e.g., `transfer(t, Memory::NPU_HBM)` pins the data to NPU).

### 3.4 Topology-Polymorphic Types

`Pinned<T, Topology>` can be abstracted over the topology: a function generic with a
`D: Topology` bound takes and returns `Pinned<T, Topology::D>`, and the compiler monomorphizes
it per concrete target at each call site. Moving a value between two topology variables
requires a `where Transfer<S, D>` constraint (discharged against the transfer cost graph), and
the same relation is available as a compile-time `Transfer<A, B>` predicate for `if comptime`
branch pruning. See [syntax.md §3.2](./syntax.md#32-topology-polymorphic-functions) for the
surface syntax and [`hardware_monad.md`](./hardware_monad.md) for the categorical model.

## 4. The Hardware State Monad

> [!WARNING]
> **Experimental / Unimplemented Feature**
> The `HardwareState` tracking and `try_pin` features are currently planned but not yet implemented.

Because physical hardware may be saturated, failed, or unavailable, bridging `Verified<T>` to `Pinned<T, Topology>` is an inherently fallible operation. Vx represents this via the `HardwareState` enum, which acts like a monad.

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
let target: HardwareState<Tensor, Topology::AccCore[0]> = compute_task.try_pin(Topology::AccCore[0]);

match target {
    HardwareState::Available(pinned_task) => {
        // Safe to execute strictly on AccCore[0]
        pinned_task.execute();
    },
    HardwareState::Saturated(agile_task) => {
        // Hardware busy, let the runtime route it anywhere
        agile_task.execute_anywhere();
    }
}
```

## 5. Effect Tracking (Upcoming Feature)

> [!WARNING]
> **Experimental / Unimplemented Feature**
> The `effects(...)` syntax is currently planned but not yet implemented in the parser.

Vx extends the type system with *effect tracking* to trace non-local behavior such as cross-topology data movement, implicit synchronization points, and potentially mutating global hardware states. Effects are checked at compile time to ensure functions do not silently introduce performance bottlenecks.

### The `effects(...)` annotation

Functions that incur significant effects must explicitly document them in their signature using the `effects` keyword. If a function calls another function with an effect, the caller must either propagate the effect in its own signature or handle it explicitly (if possible).

```rust
// This function signature indicates that it performs asynchronous data
// movement to the NPU and has temporal side-effects.
fn pipeline() -> Verified<()>
    effects(DataMovement(Memory::CPU_DRAM -> Memory::NPU_HBM))
{
    // ...
}
```

## 6. Topology-Aware Allocation

> [!WARNING]
> **Experimental / Unimplemented Feature**
> The `with Topology::...` allocation syntax is currently planned but not yet implemented in the parser.

When defining arrays, you can use Topology literals to specify the physical location of the memory as well as minimum alignment requirements (if applicable). If no alignment is specified, the compiler will use the default alignment for the type. For example:

```rust
// Allocate memory on the NPU
let x: Ref<Tensor, Topology::NPU_Core> = Tensor::new([8, 8]) with Topology::NPU_Core;

// Allocate memory on the Host
let y: Ref<Tensor, Topology::Host_Core> = Tensor::new([8, 8]) with Topology::Host_Core;

// Align to 128 bytes (A16)
let z: Ref<Tensor, Topology::NPU_Core(128)> = Tensor::new([8, 8]) with Topology::NPU_Core(128);
```

## 7. Type Coercion and Assignability

Vx evaluates type compatibility through a formal `is_assignable` constraint check during the Semantic Analysis phase. It is important to distinguish this from the ownership, move, or copy semantics found in languages like Rust.

In Vx, `is_assignable` is purely a **Type Compatibility Checker**. It determines if a value of a `Source` type can be legally bound to a slot expecting a `Target` type. It does not enforce linear typing or borrow checking (e.g., whether a value is bitwise copied or ownership is moved).

### Assignability Rules

When verifying `let target: TargetType = source_expression;`, the compiler permits the following structural coercions:

1. **Strict Equality:** If the resolved `Target` and `Source` types are identical, the assignment is valid.
1. **Implicit Unwrapping:** A hardware-specific wrapper type can implicitly decay to its base type. For example, `Ref<T>` or `Pinned<T>` can be safely assigned to a variable explicitly requesting a raw `T`.
1. **Literal Broadcasting (Scalar to Tensor):** Scalar numerical literals (e.g., `1.0` or `42`) can be implicitly coerced and broadcasted into `Tensor<T>` configurations, provided they are not boolean mismatches.
1. **Numeric Scalar Coercions:** Standard numeric types are permitted to automatically coerce across differing precisions (e.g., `f64` to `f32`) to accommodate literals during compilation, ensuring mathematical continuity without verbose casting.
1. **Pointer Decay:** Safe borrows (`&mut T`) implicitly decay into raw unsafe pointers (`*mut T`) when crossing FFI or unsafe boundaries.
1. **Safety Coercions:** A `Ref<T, HostDRAM>` can be coerced into a `Verified<T>` boundary type, signaling that host memory access requires no further spatial validation.

If the types pass the `is_assignable` constraint matrix, the Semantic Analyzer accepts the program. Advanced lifecycle validation (like borrow constraints) operates entirely independently of this type-compatibility pass.

## 8. Primitive Types and Arrays

Vx provides a comprehensive set of primitive types:

- **Signed Integers**: `i4`, `i8`, `i16`, `i32`, `i64`, `i128`
- **Unsigned Integers**: `u4`, `u8`, `u16`, `u32`, `u64`, `u128`
- **Floating Point**: `f16`, `bf16`, `f32`, `f64`
- **Boolean**: `bool`

### Arrays, Tensors, and Matrices

- **Arrays**: Fixed-size arrays are supported using the `[T; N]` syntax.
- **Tensors and Matrices**: Built-in `Tensor<T, Shape>` and `Matrix` types are first-class constructs natively understood by the compiler for high-performance algebraic operations.
- **SIMD Vectors**: Explicit SIMD types are available (e.g., `<4 x f32>`) for low-level vectorization control.

## 9. Linear (Affine) Types vs. Copyable Types

Vx uses a **linear type discipline** for resource-owning types. A linear value must be used exactly once — using it consumes it, and any subsequent use is a compile-time error. This statically prevents double-free, use-after-free, and resource leaks.

### Linear Types (consumed on use)

| Type | Example |
|------|---------|
| `Tensor<T, Shape>` | `let a : Tensor<f32> = 1.0;` |
| `Matrix` | `let m : Matrix = ...;` |
| `Ref<T, Memory>` | `let r : Ref<Tensor, Memory::CPU_DRAM> = ...;` |
| `Verified<T>` | `let v : Verified<Tensor> = ...;` |
| `Pinned<T, Topology>` | `let p : Pinned<Tensor, NPU[0]> = ...;` |
| `struct` instances | `let cfg = Config { value: 1.0 };` |
| `enum` instances | `let opt = Option::Some(42);` |

```rust
let a : Tensor<f32> = 1.0;
let b = a;  // a is consumed here
let c = a;  // COMPILE ERROR: Use of moved or consumed linear variable: a
```

### Copyable Types (reusable freely)

| Type | Example |
|------|---------|
| `i4`, `i8`, `i16`, `i32`, `i64`, `i128` | `let x = 42;` |
| `u4`, `u8`, `u16`, `u32`, `u64`, `u128` | `let y : u32 = 10;` |
| `f16`, `bf16`, `f32`, `f64` | `let pi = 3.14;` |
| `bool` | `let flag = true;` |

```rust
let x = 42;
let y = x + 1;  // OK: scalars are NOT linear
let z = x + 2;  // OK: x can be used multiple times
```

> [!NOTE]
> The `transfer()` primitive uses **non-destructive copy semantics** at the sema level. It creates a DMA copy in the destination memory space without consuming the source variable. The source remains accessible in its original memory space.
