# Mapping Vx Linear Types to Rust's Borrow Checker

This design document outlines the strategy for integrating the `Vx` compiler's frontend linear typing and `Ref<T, MemorySpace>` constraints with the pre-compiled `libvx_std.a` Rust core's strict borrow checker.

## 1. Problem Statement

`Vx` treats data distributed across heterogeneous memory spaces (Host DRAM, NPU HBM) as first-class, affine/linear types. When a variable of type `Ref<T, Memory::Host_DRAM>` is moved via a `transfer(a, Memory::NPU_HBM)`, the original `a` is consumed and can no longer be accessed.

However, the actual execution happens by calling into Rust FFI functions (like `vx_transfer_host_to_npu`). Rust's borrow checker cannot enforce lifetimes across an opaque C FFI boundary dynamically. If `Vx` passes a raw pointer `*mut T`, Rust assumes no lifecycle invariants.

## 2. The Semantic Bridge

To safely bridge these two environments, we enforce a strict **Monomorphized Opaque Pointer Lifecycle**.

### 2.1 The FFI Boundary

All generic `Vx` types (`Tensor<T>`, `Matrix`) cross the FFI boundary as opaque `*mut i8` (or equivalently `*mut c_void`).
The Rust backend defines strongly-typed wrappers that consume these pointers and instantly convert them into `Box<T>` or `&mut T` based on the operation.

```rust
// In Rust (libvx_std.a)
#[no_mangle]
pub unsafe extern "C" fn vx_tensor_f32_add(
    a_ptr: *mut c_void,
    b_ptr: *mut c_void
) -> *mut c_void {
    // Re-assert ownership and lifecycle
    let a = Box::from_raw(a_ptr as *mut Tensor<f32>);
    let b = Box::from_raw(b_ptr as *mut Tensor<f32>);

    let result = a.add(&*b);

    // b is dropped automatically here
    // a is dropped automatically here
    Box::into_raw(Box::new(result)) as *mut c_void
}
```

### 2.2 Linear Consumption in Vx

In the `Vx` semantic analyzer (`sema.rs`), calling an FFI function that maps to a consuming operation *must* drop the variable from the active `SymbolMap`.

```vx
let a: Ref<Tensor, Host_DRAM> = Tensor::new([128]);
let b: Ref<Tensor, Host_DRAM> = Tensor::new([128]);

// The `add` operation consumes `a` and `b`.
let c = a + b;

// COMPILE ERROR: `a` was moved in the previous operation.
let d = a + c;
```

## 3. Topologies and Memory Spaces

Memory spaces act as distinct typestates. A Rust backend function is exposed for each legal memory transition.

### 3.1 The `transfer` Primitive

The `transfer` keyword in `Vx` maps to memory-space specific FFI endpoints.

```vx
let npu_data = transfer(host_data, Memory::NPU_HBM);
```

Maps to:

```rust
#[no_mangle]
pub unsafe extern "C" fn vx_transfer_host_to_npu_f32(
    host_ptr: *mut c_void
) -> *mut c_void {
    // 1. Claim ownership of host data
    let host_tensor = Box::from_raw(host_ptr as *mut Tensor<f32>);

    // 2. Perform DMA transfer to NPU
    let npu_tensor = pcie_dma_transfer(&*host_tensor);

    // 3. Drop host_tensor (Rust handles host memory cleanup)
    // 4. Return new NPU pointer
    Box::into_raw(Box::new(npu_tensor)) as *mut c_void
}
```

By ensuring that every `transfer` or consuming operator maps to a `Box::from_raw` on the Rust side, we guarantee that the Rust borrow checker accurately frees memory at the exact moment the `Vx` frontend declares the linear type "consumed".

## 4. Borrowing (Non-Consuming)

For non-consuming operations (like read-only `.shape` or `.len()` access), the Rust FFI uses `&T`.

```rust
#[no_mangle]
pub unsafe extern "C" fn vx_tensor_f32_get_len(
    a_ptr: *const c_void
) -> i64 {
    // Safe borrow, no ownership claimed. Memory is not freed.
    let a = &*(a_ptr as *const Tensor<f32>);
    a.len() as i64
}
```

In `Vx`, this maps to a method call that does *not* remove the variable from the `SymbolMap`.
