# Task: Implement Custom `vx-kernel-outlining` Pass

- `[x]` Create a custom C++ MLIR pass `ConvertVxToStandardPass` modification or a new `vx-kernel-outlining` pass to outline `vx.spawn` for NPU.
- `[x]` Update `VxLowering.cpp` to outline `vx.spawn` directly into a module-level `func.func` with a specific attribute (e.g. `npu.kernel`).
- `[x]` Replace `vx.spawn` with a `vx.dispatch` operation or standard `func.call` + hardware annotations.
- `[x]` Add `vx.dispatch` to the `vx` dialect in TableGen (`VxOps.td`).
- `[x]` Update `vx_hardware_runtime.h` to define the runtime calls for `vx.dispatch` lowering. (N/A for MLIR level)
- `[x]` Test with `tests/optimizations/pass/gpu_lowering.vx` (which we should rename to `npu_lowering.vx`).
