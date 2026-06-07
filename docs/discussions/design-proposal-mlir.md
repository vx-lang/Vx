# Design Document: Inline MLIR Assembly Macros (`mlir!`)

## 1. Executive Summary & Rationale

In the era of heterogeneous compute, hardware-specific assembly (x86/ARM) is insufficient. MLIR (Multi-Level Intermediate Representation) acts as the universal "assembly language" for diverse accelerators like NPUs and TPUs.

By introducing an `mlir!` macro, Vx elevates inline assembly. Programmers can write explicit MLIR code directly within their Vx programs, effectively treating MLIR dialects as first-class namespaces. This allows library developers to write highly optimized, accelerator-specific kernels directly in user-space without modifying the Vx compiler.

## 2. Core Architectural Constraints

To prevent the `mlir!` block from becoming an opaque black box that breaks the Vx compiler's liveness analysis and LLVM lowering, the macro is strictly governed by three structural constraints:

### A. The Isolation Principle (`IsolatedFromAbove`)

The `mlir!` macro fundamentally acts as an `IsolatedFromAbove` MLIR region.

- **No Implicit Captures:** Implicitly capturing Vx environment variables or closures is strictly forbidden.
- **Explicit Binding:** Any data entering the MLIR region must be explicitly passed as an input parameter and bound to an MLIR block argument. This ensures the MLIR pass manager can operate on the region without locking the surrounding SSA dominance tree.

### B. Control Flow Integrity (SESE)

The macro must form a Single-Entry, Single-Exit (SESE) region within the Vx Control Flow Graph.

- **No External Branching:** An `mlir!` block cannot contain terminator instructions (like `cf.br`) that attempt to branch to a Vx label outside the macro.
- **Predictable Yields:** The block must terminate with a `macro.yield` operation, returning control (and optionally a value) safely back to the Vx AST.

### C. Liveness and Side-Effect Transparency (Clobbers)

Because the internal MLIR operations are opaque to the Vx semantic analyzer, the programmer must explicitly declare memory side-effects.

- **Clobber Lists:** If an `mlir!` block mutates a reference, buffer, or tensor, it must be listed in the `clobbers` array. Failure to do so may result in the Vx compiler aggressively reordering or eliminating subsequent reads.

## 3. Preconditions & Postconditions

**Preconditions (Compiler Guarantees before entry):**

1. **Type Mapping:** The Vx semantic analyzer guarantees that the Vx types provided in the `inputs` mapping are legally convertible to the specified MLIR types (e.g., Vx `f32` to MLIR `f32`, Vx `&mut Tensor` to MLIR `memref`).
1. **Initialization:** All input variables are guaranteed to be initialized and live at the point of the macro invocation.
1. **Syntax Validation & Wrapper:** The Vx compiler will silently wrap the macro's internal MLIR string inside a dummy operation (e.g., `"vx.macro_wrapper"() ({ ^bb0... }) : () -> ()`). It then parses this wrapper using MLIR's C++ parser (Melior). This ensures we can parse isolated blocks correctly.
1. **Dialect Loading:** The compiler will load the dialects specified in the `dialects` array before parsing.

**Postconditions (Compiler Guarantees after exit):**

1. **State Mutation:** The Vx semantic analyzer marks all variables listed in the `clobbers` array as having been mutated. Any previous statically known state for those variables is invalidated.
1. **Type Safety:** The value yielded by `vx.yield` is guaranteed to be bitcast back into the Vx type specified in the `returns` field.

## 4. Syntax & Examples

### Example 1: The Pure Function (Scalar Arithmetic)

This example demonstrates explicit positional input binding and returning a value.

```rust
let x: f32 = 10.0;
let y: f32 = 20.0;

// The macro acts as an Isolated boundary.
let result = mlir!(
    // Inputs are mapped explicitly to MLIR block arguments
    inputs: (%arg0 = x: f32, %arg1 = y: f32),
    returns: f32,
    dialects: ["arith", "vx"]
) {
^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    vx.yield %0 : f32
};
```

### Example 2: Side Effects and Clobbers (Tensor Mutation)

When an operation mutates state, the `clobbers` array must explicitly list the mutated variables.

```rust
let mut t = Tensor::<f32>::new([10, 10]);

mlir!(
    inputs: (%arg0 = t: memref<?x?xf32>),
    clobbers: [t], // Let the compiler know 't' is mutated!
    returns: void,
    dialects: ["linalg", "arith", "vx"]
) {
^bb0(%arg0: memref<?x?xf32>):
    %cst = arith.constant 0.0 : f32
    linalg.fill ins(%cst : f32) outs(%arg0 : memref<?x?xf32>)
    vx.yield
};
```

### The Heterogeneous Kernel (Memory Mutation)

This example demonstrates offloading a task to an NPU and using MLIR's affine dialect to mutate memory. Notice the explicit clobbers declaration, which prevents the Vx optimizer from assuming the tensor is empty after the block executes.

```rust
let mut w = Tensor::<f32>::new([128, 128]);
// In a real scenario, this buffer might be allocated on an NPU
let device_buffer = align_to_npu_granularity(w);

mlir!(
    inputs: (%arg0 = device_buffer: memref<?x?xf32, #npu_memory_space>),
    clobbers: [w], // "w" is mutated
    returns: void,
    dialects: ["npu", "affine"]
) {
^bb0(%arg0: memref<?x?xf32, #npu_memory_space>):
    %c1 = affine.constant 1 : f32
    // 1. Use NPU-specific attributes
    %2 = npu.load_constant %c1 : f32

    // 2. Use affine dialect for memory access
    affine.for %i in (0 .. 128) {
        affine.for %j in (0 .. 128) {
            %idx = dim %arg0, #1
            affine.store %2, %arg0[%i, %j] : memref<?x?xf32, #npu_memory_space>
        }
    }
    vx.yield
};
```

Spawning a mlir code block on NPU

```rust
// Allocate a tensor on the NPU topology
let mut tensor = Tensor::<f32, [1024]>::alloc(Topology::NPU[0]);

spawn on(Topology::NPU[0]) {
    mlir!(
        inputs: (t = &mut tensor: memref<1024xf32>),
        clobbers: [t], // Critical: Tells Vx this memory is mutated
        dialect: ["affine", "arith", "memref"]
    ) {
    ^bb0(%arg0: memref<1024xf32>):
        // MLIR affine loop to zero out the tensor
        %cst = arith.constant 0.0 : f32
        affine.for %i = 0 to 1024 {
            affine.store %cst, %arg0[%i] : memref<1024xf32>
        }
        macro.yield
    };
}
```

### 5. Implementation Requirements

- **`vx` Dialect:** We must register a minimal custom MLIR dialect `vx` within the compiler backend. This dialect will define `vx.macro_wrapper` (for parsing isolated blocks) and `vx.yield` (to terminate the SESE region).
- **Region Lowering:** During codegen, the compiler strips the `vx.macro_wrapper`, extracts the block, replaces `vx.yield` with the appropriate parent block terminator, and splices the MLIR directly into the generated function.
- **Higher-Order Functions:** Passing Vx closures (e.g. `|x| x * 2`) to be used as MLIR regions (like for `linalg.generic`) is deferred to a future extension of the `mlir!` macro, requiring specialized syntax for region arguments.
