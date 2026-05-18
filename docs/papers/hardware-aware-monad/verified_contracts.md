NOTE: This discussion mentions contract for Deferred<T> but later we realized these contracts
are better suited for Verified<T> and then we can say that Deferred<T> is anything that is a Verified<T>:

If `spawn_on(Topology)` or `topology.dispatch()` accepts an arbitrary computation, the compiler must guarantee that the computation can physically and legally execute on the target hardware.

In C++ (CUDA/SYCL), this is achieved with hacky macro annotations like `__device__`. In Vx, because hardware is a first-class citizen, we can enforce these rules purely through the type system and the semantic analyzer (Phase 1.3 of your compiler architecture).

To make a function or callable offloadable (capable of returning a `Deferred<T>`), it must satisfy **The Offload Contract**. Here are the four strict compile-time constraints Vx must enforce.

### 1. The `DeviceSafe` Purity Constraint (No Host Side-Effects)

An offloaded computation runs in an isolated hardware context. The compiler must prove that the function does not attempt to perform actions that only the Host OS can handle.

**Compile-Time Checks:**

* **No Host I/O:** The function cannot call `print()`, read files, or open network sockets.
* **No Host Allocations:** It cannot dynamically allocate unbounded memory on the Host DRAM (e.g., standard `malloc` or dynamically sizing a `Vec` without a pre-allocated device arena).
* **No System Calls:** Any instruction that lowers to an OS interrupt (`syscall`) immediately fails compilation.

```rust
fn bad_kernel(a: Pinned<Tensor, NPU>) -> Deferred<Tensor> {
    print("Computing..."); // COMPILE ERROR: Host I/O in DeviceSafe context
    return a * 2.0;
}

```

### 2. The Strict Memory Affinity Constraint

The inputs to the callable must be physically guaranteed to exist on the target topology *before* execution begins.

**Compile-Time Checks:**

* If a `Deferred<T>` is constructed, the compiler inspects the `MemorySpace` of all arguments.
* When `.spawn_on(Target)` is invoked, the semantic analyzer asserts that every captured argument is exactly `Pinned<Type, Target>`.

```rust
let host_tensor = Tensor::ones([100]);
let npu_tensor = host_tensor.to_device(Topology::NPU[0]);

// COMPILE ERROR: Cannot pass Host memory to an NPU-bound callable
let fut = tensor::matmul(host_tensor, npu_tensor).spawn_on(Topology::NPU[0]);

```

### 3. Zero Implicit Capture (The Pointer Boundary)

If Vx allows closures or inline blocks to be offloaded, the compiler must aggressively restrict the environment capture rules.

In standard programming, a closure implicitly captures a pointer to the surrounding stack frame. If you send that stack pointer to an NPU, dereferencing it will cause a hardware fault because the NPU cannot read Host CPU L1 cache.

**Compile-Time Checks:**

* Offloadable callables must have **Zero Implicit Captures**.
* Any data required by the function must be passed explicitly as an argument.
* If a closure is used, it must only capture data that is `Copy` (like a primitive `f32` scalar) or already `Pinned` to the target device.

```rust
let scaling_factor: f32 = 0.5; // Trivially copyable, allowed to cross boundary
let host_config: HostString = "v1_mode"; // Complex host reference

let fut = NPU[0].dispatch(|t: Pinned<Tensor, NPU[0]>| {
    let scaled = t * scaling_factor; // OK: Scalar is copied by value into MLIR region
    let mode = host_config;          // COMPILE ERROR: Implicit capture of Host Reference
    return scaled;
});

```

### 4. The MLIR Dialect Bound (Hardware Lowerability)

Just because a function is memory-safe doesn't mean the hardware has the silicon to execute it. The operations inside the callable must be translatable to the specific MLIR dialects supported by the target's plugin.

**Compile-Time Checks:**

* The Vx standard library defines a core set of operations guaranteed to lower to MLIR's `linalg` or `tensor` dialects (which all hardware vendors agree to support in their pass pipelines).
* If a user writes a custom loop, the Vx compiler must be able to lower it into an `scf.for` (Structured Control Flow) block.
* If the NPU plugin's `is_op_supported` (from our plugin contract) returns false for any MLIR operation in the block, and Vx cannot decompose it into simpler math, the compiler halts.

### Implementing the Contract in Vx

To implement this elegantly in your compiler, you don't need to burden the user with writing `#[device_safe]` everywhere. You can use your Data-Oriented compiler architecture to infer it.

When your semantic analyzer (Phase 1) builds the `LOCAL_HIR_STREAM`, it flags every function. If a function contains an I/O call, it sets a bit `HAS_HOST_SIDE_EFFECTS = 1`.

When the user calls `.spawn_on()` or `.dispatch()`, the compiler simply checks:

1. Are all arguments `Pinned` to the target?
2. Is `HAS_HOST_SIDE_EFFECTS == 0`?
3. Are all variables explicitly passed or trivially copied?

If yes, it wraps the MLIR in an `async.execute` region. If no, it emits a beautiful, highly specific error message explaining exactly which physical constraint was violated.
