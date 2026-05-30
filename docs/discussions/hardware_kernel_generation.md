# Real Hardware Kernel Generation & Dispatch

## Walkthrough

### Overview
This walkthrough summarizes the completion of the "Real Hardware Kernel Generation & Dispatch" objective for the Vx language compiler, specifically targeting the NPU backend lowering pass from the `vx` dialect to `LLVM IR`.

### Changes Made
1. **Dialect Enhancements**: 
   - Introduced `vx.return` to properly terminate hardware kernels instead of using `func.return` which violates structure invariants.
   - Refined the signature of `vx.launch` (replacing `vx.dispatch`) and `vx.kernel` to accurately model grid and topology dimensions needed by the backend dispatcher.

2. **Backend Lowering Patterns (`VxLowering.cpp`)**:
   - Implemented `KernelOpLowering` and `ReturnOpLowering` to directly outline `vx.kernel` blocks into LLVM-compatible top-level functions (`func.func`).
   - Implemented `LaunchOpLowering` which dynamically translates `vx.launch` nodes into native runtime calls to `vx_plugin_dispatch_async`.
   - Elegantly resolved `llvm.store` verifier crashes by explicitly bridging untyped operands (`memref` and `index`) to their underlying target representations via transient `UnrealizedConversionCastOp` casts, allowing the MLIR type conversion machinery to naturally resolve them in subsequent conversion passes.

3. **Compiler Pass Configurations**:
   - Expanded dialect legalization definitions in `ConvertVxToLLVMPass` to mark `UnrealizedConversionCastOp` as explicitly valid. 
   - Avoided SFINAE C++ compiler crashes by extending directly from `OpRewritePattern` rather than the finicky `ConvertOpToLLVMPattern`.

4. **Testing Infrastructure**:
   - Validated backend lowering operations by updating FileCheck assertions within `tests/optimizations/pass/npu_lowering.vx`.
   - Verified functionality via successful end-to-end NPU passes (`test_backend` and `test_optimizations` inside `compile_test.rs`).

### Verification
- All automated backend test cases for the `VxLowering.cpp` compilation and the NPU hardware lowering sequence are successfully passing (`cargo test compile_test`).

---

## Tasks Completed
- `[x]` 1. Dialect Definition (`VxDialect.td`)
  - `[x]` Add `Vx_KernelOp`
  - `[x]` Add `Vx_LaunchOp`
  - `[x]` Add `Vx_ReturnOp`
- `[x]` 2. Lowering Pipeline (`VxLowering.cpp`)
  - `[x]` Update `SpawnOpLowering` to extract into `vx.kernel`
  - `[x]` Replace original `vx.spawn` with `vx.launch`
- `[x]` 3. Hardware Translation (Bypass GPU Dialect)
  - `[x]` Directly lower `vx.launch` to ANE/CPU shims (`vx_plugin_dispatch_async`)
  - `[x]` Directly lower `vx.kernel` to outlined `func.func`
- `[x]` 4. Verification
  - `[x]` Run MLIR tests to verify new dialect ops
  - `[x]` Validate end-to-end compilation without GPU dialect errors

---

## Original Implementation Plan

We previously planned to lower `vx.spawn` bodies to the MLIR `gpu` dialect to achieve real hardware kernel generation. Since the MLIR `gpu` dialect can be brittle and overly restrictive for our specific heterogeneous topology needs, we modeled kernel generation natively within our custom `vx` dialect.

### Proposed Changes

We introduced hardware execution primitives directly into the `vx` dialect, allowing us to maintain full control over the lowering pipeline.

### Dialect Definition
We expanded the `vx` dialect to include primitives for kernel definition and execution.

- **`vx.kernel`**: A specialized function operation to represent the outlined hardware kernel (replacing our current hack of using `func.func` with a `vx.kernel` attribute).
- **`vx.launch`**: An operation to invoke a `vx.kernel` from the host, explicitly taking grid and block dimensions as operands.
- **`vx.return`**: Operation to terminate kernel execution blocks cleanly.

### Lowering Pipeline
We updated the compiler to outline `vx.spawn` blocks into `vx.kernel` operations instead of standard functions.

- Update `SpawnOpLowering` to extract the body of a `vx.spawn` (for topology > 0) into a newly defined `vx.kernel` within the module scope.
- Replace the original `vx.spawn` with a `vx.launch` operation instead of a `vx.dispatch` call. 
- Map any parallel loops (`linalg.generic` or affine loops) inside the spawn region directly to coordinate computations.

### Hardware Translation (Bypassing `gpu` dialect)
We created a new set of lowering patterns that bridge the gap between our `vx` dialect and the final hardware binary representation.

- Implement a direct lowering pattern from `vx.kernel` and `vx.launch` to LLVM calls mapping directly to `vx_plugin_dispatch_async`.
- This gives us a mathematically rigorous and direct path to hardware binaries without relying on the MLIR `gpu` dialect's rigid abstractions.
