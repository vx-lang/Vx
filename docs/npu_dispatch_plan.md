# Hardware Dispatcher Implementation Plan: MPS (GPU) and CoreML (ANE)

> **Status, 2026-08-13.** Both phases have landed — `runtime/npu_dispatch.mm` implements
> `vx_dispatch_gpu` via `MPSMatrixMultiplication` and `vx_dispatch_ane` via CoreML with
> `MLComputeUnitsAll`. Two things below no longer describe the code, and are corrected in place:
> the buffers are **copied, not zero-copy** (see Phase 1 §3), and `build_ane_matmul.py` now lives
> in `scripts/legacy/`.

The goal of this plan is to upgrade Vx's `spawn on(Topology::...)` hardware offloading capabilities to natively target Apple Silicon's GPU and Apple Neural Engine. Per your request, we will implement these in two distinct phases, committing the first before moving to the second.

I propose we implement **Phase 1: GPU (MPS)** first, as it is highly generic and doesn't require compiling external `.mlmodelc` binaries.

## Phase 1: Metal Performance Shaders (GPU) Dispatcher

### 1. Compiler Frontend & Codegen Updates

- **Lexer & Parser:** Add `"GPU"` as a valid `Topology` variant.
- **AST & Sema:** Add `Topology::GPU` to the AST and `MemorySpace::HostDRAM` (for unified memory access).
- **Codegen:** Update the MLIR lowering for `spawn on(Topology::GPU)` to intercept nested dense matrix multiplications and rewrite them as a function call to the external C-ABI function `vx_dispatch_gpu`.

### 2. Build System Integration

- Modify `build.rs` to dynamically link `-framework Metal` and `-framework MetalPerformanceShaders` during the JIT initialization.

### 3. Objective-C++ Runtime (`npu_dispatch.mm`)

- Implement `extern "C" int vx_dispatch_gpu(float* xout, float* x, float* w, int n, int d)`.

- Acquire the GPU using `MTLCreateSystemDefaultDevice()`.

- Create a command queue and command buffer.

- ~~Wrap the raw CPU pointers `xout`, `x`, and `w` into `MTLBuffer` objects using `newBufferWithBytesNoCopy` to leverage Apple Silicon's Unified Memory Architecture (zero-copy transfers).~~
  **Not what shipped.** `runtime/npu_dispatch.mm` uses `newBufferWithBytes:` with
  `MTLResourceStorageModeShared` — which *copies*. The reason is in the code comment: mmap'd weight
  pointers are not guaranteed 4 KiB aligned, and `newBufferWithBytesNoCopy` requires a page-aligned
  address and a page-multiple length, returning nil otherwise.

  This is a defensible trade, but it is not free, and we now know the price on the development M4
  (`utils/memalg/results/m4-*`, Vx `73f7fdb3`): a kernel reading a buffer the CPU just wrote runs at
  **106.0 GB/s** against **106.3 GB/s** for one the GPU already had — a ratio of 0.997, so the
  zero-copy path genuinely costs nothing. An explicit copy of the same data runs at **48.9 GB/s**.
  So every dispatched matmul pays roughly `2 x bytes / 48.9 GB/s` that the unified-memory path would
  not.

  Recovering it needs the tensor allocator to hand out page-aligned, page-multiple buffers. Worth
  doing when dispatch volume justifies it; recorded here so the copy stays a decision rather than an
  accident.

- Wrap the `MTLBuffer` objects into `MPSMatrix` descriptors.

- Encode an `MPSMatrixMultiplication` kernel into the command buffer.

- Execute and synchronously wait `[commandBuffer waitUntilCompleted]`.

### 4. Verification & Commit

- Write an Vx script `tests/backend/pass/mps_test.vx` wrapping a `matmul` in `spawn on(Topology::GPU)`.
- Run the test and observe logs indicating MPS execution.
- Commit Phase 1.

______________________________________________________________________

## Phase 2: Core ML (ANE) Dispatcher

### 1. Model Generation

Unlike MPS, the ANE is accessed via Core ML which expects pre-compiled model graphs.

- Create a Python script (`scripts/legacy/build_ane_matmul.py`) using `coremltools` to generate a generic `matmul.mlpackage` with flexible shapes using `RangeDim`.
- Compile it via Xcode's `coremlcompiler` to yield `matmul.mlmodelc` in the workspace.

### 2. Objective-C++ Runtime (`npu_dispatch.mm`)

- Implement `extern "C" int vx_dispatch_ane(float* xout, float* x, float* w, int n, int d)`.
- Load `matmul.mlmodelc` via `[MLModel modelWithContentsOfURL:]`.
- Configure the model with `MLModelConfiguration.computeUnits = MLComputeUnitsAll` (which forces ANE prioritization).
- Wrap `x`, `w`, and `xout` inside `MLMultiArray` instances using `initWithDataPointer:shape:dataType:strides:deallocator:error:`.
- Execute the graph via `predictionFromFeatures:error:` and copy the output to `xout`.

### 3. Verification & Commit

- Modify `llama2.vx`'s Feed-Forward Network to use `spawn on(Topology::ANE)` to dispatch `w1`, `w2`, and `w3` matrix multiplications directly to the Neural Engine.
- Run the LLaMA2 text generation test to confirm valid logits and output text.
- Commit Phase 2.

______________________________________________________________________

## User Review Required — resolved

Both questions below were answered by shipping: Phase 1 went first, and the `(1 x n)` / `(n x d)`
descriptor encoding is what `npu_dispatch.mm` uses. Kept for the record.

> Does starting with Phase 1 (GPU/MPS) sound good? Also, since MPS and CoreML both expect multidimensional shapes and we only receive flattened `1D` pointers from Vx's current backend (`d` and `n`), I will encode the matrices as `(1 x n)` and `(n x d)` vectors. Do you approve this matrix descriptor alignment?
