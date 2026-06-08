# Implement NPU Hardware Dispatch Lowering (`vx.spawn` for topology 100)

This document outlines the plan for the second pillar of the NPU execution model: correctly lowering `vx.spawn on(Topology::NPU[0])` blocks down to MLIR and interfacing with our `Objective-C++` / Metal NPU backend dispatcher (`npu_dispatch.mm`).

## Background

Currently, `melior_codegen.rs` successfully emits a `vx.spawn` operation with a nested region, but the MLIR lowering pass in `VxLowering.cpp` merely inlines the NPU blocks sequentially to avoid crashes. We need to implement true hardware dispatch by outlining the `vx.spawn` block into a standalone kernel function and replacing the `vx.spawn` operation with a call to the external NPU dispatcher (`vx_dispatch_npu` or `vx_dispatch_ane`).

## User Review Required

> [!WARNING]
> **Outlining Design Decision:** In MLIR, passing function pointers to C ABIs can be tricky depending on the LLVM translation pipeline. Instead of passing an MLIR function pointer dynamically, I propose that the `VxLowering` pass will outline the `vx.spawn` body into a new top-level `func.func private @__vx_npu_kernel_0(...)`, and we will insert a call to an external `vx_dispatch_npu_kernel(...)` that passes the extracted arguments (tensors) dynamically to the Metal runtime.
> Let me know if you prefer we use standard GPU dialects (`gpu.launch`) instead, or stick to our custom dispatch approach!

## Proposed Changes

### 1. MLIR C++ Lowering Pass

#### [MODIFY] `src/dialect/VxLowering.cpp`

- **Region Outlining:** Update `SpawnOpLowering` for `topology == 100` (NPU).
- Create a new `func.func` operation at the `ModuleOp` level.
- Extract the variables captured by the `vx.spawn` block and define them as parameters for the outlined function.
- Move the `vx.spawn` region body into the outlined function.
- Replace the `vx.spawn` operation with a `func.call` to the external runtime hardware dispatcher, passing the operands and potentially an ID mapping to the outlined kernel.
- **Resource Cleanup:** Ensure that `memref.dealloc` is still respected inside the outlined NPU kernel.

### 2. Runtime C++ Implementation

#### [MODIFY] `runtime/npu_dispatch.mm`

- **Dispatcher Interface:** Implement the required external C-ABI function (e.g. `extern "C" void vx_dispatch_npu(...)`) that takes the arguments necessary to initialize the Metal Command Queue and execute the kernel operations.

## Verification Plan

### Automated Tests

- Create a test `tests/backend/pass/npu_dispatch_spawn.vx` that uses `spawn on(Topology::NPU[0])`.
- We will run the MLIR compilation step and use `FileCheck` to ensure that `func.func private @__vx_npu_kernel` is generated and that the original `vx.spawn` is replaced with `func.call @vx_dispatch_npu`.

### Manual Verification

- Output the generated `.vxm` file to inspect the generated MLIR outliner logic.
