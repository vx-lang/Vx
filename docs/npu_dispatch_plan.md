# Hardware Dispatcher Implementation Plan: MPS (GPU) and CoreML (ANE)

The goal of this plan is to upgrade Vx's `spawn on(Topology::...)` hardware offloading capabilities to natively target Apple Silicon's GPU and Apple Neural Engine. Per your request, we will implement these in two distinct phases, committing the first before moving to the second.

I propose we implement **Phase 1: GPU (MPS)** first, as it is highly generic and doesn't require compiling external `.mlmodelc` binaries.

## Phase 1: Metal Performance Shaders (GPU) Dispatcher

### 1. Compiler Frontend & Codegen Updates
*   **Lexer & Parser:** Add `"GPU"` as a valid `Topology` variant.
*   **AST & Sema:** Add `Topology::GPU` to the AST and `MemorySpace::HostDRAM` (for unified memory access).
*   **Codegen:** Update the MLIR lowering for `spawn on(Topology::GPU)` to intercept nested dense matrix multiplications and rewrite them as a function call to the external C-ABI function `vx_dispatch_gpu`.

### 2. Build System Integration
*   Modify `build.rs` to dynamically link `-framework Metal` and `-framework MetalPerformanceShaders` during the JIT initialization.

### 3. Objective-C++ Runtime (`npu_dispatch.mm`)
*   Implement `extern "C" int vx_dispatch_gpu(float* xout, float* x, float* w, int n, int d)`.
*   Acquire the GPU using `MTLCreateSystemDefaultDevice()`.
*   Create a command queue and command buffer.
*   Wrap the raw CPU pointers `xout`, `x`, and `w` into `MTLBuffer` objects using `newBufferWithBytesNoCopy` to leverage Apple Silicon's Unified Memory Architecture (zero-copy transfers).
*   Wrap the `MTLBuffer` objects into `MPSMatrix` descriptors.
*   Encode an `MPSMatrixMultiplication` kernel into the command buffer.
*   Execute and synchronously wait `[commandBuffer waitUntilCompleted]`.

### 4. Verification & Commit
*   Write an Vx script `tests/backend/pass/mps_test.vx` wrapping a `matmul` in `spawn on(Topology::GPU)`.
*   Run the test and observe logs indicating MPS execution.
*   Commit Phase 1.

---

## Phase 2: Core ML (ANE) Dispatcher

### 1. Model Generation
Unlike MPS, the ANE is accessed via Core ML which expects pre-compiled model graphs.
*   Create a Python script (`build_ane_matmul.py`) using `coremltools` to generate a generic `matmul.mlpackage` with flexible shapes using `RangeDim`.
*   Compile it via Xcode's `coremlcompiler` to yield `matmul.mlmodelc` in the workspace.

### 2. Objective-C++ Runtime (`npu_dispatch.mm`)
*   Implement `extern "C" int vx_dispatch_ane(float* xout, float* x, float* w, int n, int d)`.
*   Load `matmul.mlmodelc` via `[MLModel modelWithContentsOfURL:]`.
*   Configure the model with `MLModelConfiguration.computeUnits = MLComputeUnitsAll` (which forces ANE prioritization).
*   Wrap `x`, `w`, and `xout` inside `MLMultiArray` instances using `initWithDataPointer:shape:dataType:strides:deallocator:error:`.
*   Execute the graph via `predictionFromFeatures:error:` and copy the output to `xout`.

### 3. Verification & Commit
*   Modify `llama2.vx`'s Feed-Forward Network to use `spawn on(Topology::ANE)` to dispatch `w1`, `w2`, and `w3` matrix multiplications directly to the Neural Engine.
*   Run the LLaMA2 text generation test to confirm valid logits and output text.
*   Commit Phase 2.

---

## User Review Required
Does starting with Phase 1 (GPU/MPS) sound good? Also, since MPS and CoreML both expect multidimensional shapes and we only receive flattened `1D` pointers from Vx's current backend (`d` and `n`), I will encode the matrices as `(1 x n)` and `(n x d)` vectors. Do you approve this matrix descriptor alignment?
