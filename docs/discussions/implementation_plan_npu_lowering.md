# Implement full LLVM lowering for NPU dispatch operations

The goal is to map the `vx.dispatch` operation to the actual NPU hardware ABI (`vx_plugin_dispatch_async`) during the `vx-to-llvm` lowering phase.

## Current State
Currently, `DispatchOpLowering` in `src/dialect/VxLowering.cpp` converts `vx.dispatch @kernel (args...)` to a standard MLIR `func::CallOp`. This works for testing with a CPU-based sequential fallback, but it does not invoke the hardware NPU plugin architecture via `npu_dispatch.h`.

## Open Questions

> [!WARNING]
> **LLVM Type Converter Dependency**
> The standard MLIR way to pack `memref` arguments into a `void**` array (which `vx_plugin_dispatch_async` expects) requires `LLVMTypeConverter`. Should we refactor `ConvertVxToLLVMPass` to use `DialectConversion` and `LLVMTypeConverter` (similar to MLIR's `gpu-to-llvm` pass), or should we introduce a separate `convert-vx-to-llvm-ir` pass that specifically runs *after* the standard `func-to-llvm` conversion?

> [!IMPORTANT]
> **Runtime API selection**
> The `npu_dispatch.h` runtime defines two async APIs:
> 1. Generic: `uint64_t vx_plugin_dispatch_async(const void* binary_payload, size_t payload_size, void** device_args);`
> 2. Flat (matmul specific): `uint64_t vx_plugin_dispatch_async_flat(float* xout, float* x, float* w, int n, int d);`
> 
> The generic approach is required for arbitrary `vx.dispatch` operations, but requires pointer packing. I plan to implement the generic pointer packing approach. Do you agree with this design direction?

## Proposed Changes

### `src/dialect/VxLowering.cpp`

We will update the `ConvertVxToLLVMPass` (or a newly created `ConvertVxToLLVMDialectPass`) to:

1. **Declare the External Runtime API:**
   Insert an `llvm.func` declaration for `vx_plugin_dispatch_async` at the top of the `ModuleOp` if it doesn't already exist.

2. **Lower `vx.dispatch`:**
   Modify `DispatchOpLowering` to:
   - Create an `llvm.mlir.global` string containing the kernel name.
   - Use `llvm.addressof` to get a pointer to the kernel name string (`binary_payload`).
   - Emit `llvm.alloca` to allocate an array of `!llvm.ptr` equal to the number of operands.
   - For each operand, if it's a `memref`, obtain its `MemRefDescriptor` (via `LLVMTypeConverter`), emit an `llvm.alloca` for the descriptor, store it, and place its pointer into the argument array. If it's a scalar, similarly alloca and store.
   - Emit an `llvm.call` to `vx_plugin_dispatch_async` with the packed arguments.
   - If the operation expects a returned future ID, return it.

#### [MODIFY] [VxLowering.cpp](file:///Users/adityak/go/Vx/src/dialect/VxLowering.cpp)
Refactor pass execution to use `DialectConversion` framework and `LLVMTypeConverter`. Add logic to pack LLVM types into `void**` arrays.

### `tests/optimizations/pass/npu_lowering.vx`

#### [MODIFY] [npu_lowering.vx](file:///Users/adityak/go/Vx/tests/optimizations/pass/npu_lowering.vx)
Update the `FileCheck` assertions to look for `llvm.call @vx_plugin_dispatch_async` instead of a direct function call to the outlined kernel.

## Verification Plan

### Automated Tests
- Run `cargo test` and ensure all frontend tests pass.
- Run `vx-opt` and `vxc --emit-llvm` manually on `npu_lowering.vx` to verify that the generated LLVM IR correctly references `vx_plugin_dispatch_async` and correctly builds the argument array.
- Verify that `utils/update_mlir_test_checks.rs` works cleanly with the new lowering.
