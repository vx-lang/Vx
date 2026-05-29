# Replacing the MLIR GPU Dialect for NPU Offloading

The MLIR `gpu` dialect provides an excellent out-of-the-box infrastructure for host/device outlining, but it is fundamentally designed around the execution model of GPUs (CUDA/ROCm) and requires a compiled PTX binary to lower to LLVM IR. Since we are targeting a custom NPU and not a GPU, relying on the `gpu` dialect adds brittle assumptions to our compiler pipeline that we constantly have to work around.

Here are the primary architectural alternatives to replace the `gpu` dialect for `spawn on (Topology::NPU)`:

## Alternative 1: Custom `vx-kernel-outlining` Pass (Recommended)

Instead of lowering `vx.spawn` to `gpu.launch`, we keep the `vx.spawn` operation as-is during the initial lowering. Then, we write a custom C++ MLIR pass (`vx-kernel-outlining`) that performs the host/device separation ourselves. 

**How it works:**
1. The pass finds `vx.spawn` operations targeted at the NPU.
2. It extracts the region inside `vx.spawn` into a standard `func.func` (or a custom `vx.func`) placed at the module level.
3. It replaces the `vx.spawn` with a `vx.dispatch` call that references the newly created function.
4. A final `vx-to-llvm` pass simply lowers `vx.dispatch` into standard LLVM IR calls to our custom NPU runtime API (`npu_launch_kernel`).

**Pros:** Total control over the outlining process; zero dependency on PTX/CUDA assumptions; perfectly aligns with our custom hardware dispatcher.
**Cons:** We have to implement the outlining logic ourselves (variable capturing, argument passing) rather than getting it for free from `gpu-kernel-outlining`.

## Alternative 2: The `async` Dialect with Target Attributes

Currently, we lower `vx.spawn` on the `Host` to `async.execute`. We could do the same for the NPU, but annotate the `async.execute` operation with an attribute like `target = "NPU"`.

**How it works:**
1. Lower `vx.spawn on (NPU)` to `async.execute { target = "NPU" }`.
2. Use MLIR's built-in `async-to-async-runtime` outlining pass, which automatically outlines `async.execute` regions into coroutine-like functions.
3. Write a custom pass to intercept the execution of these specific async functions and redirect them to the NPU driver.

**Pros:** Reuses MLIR's robust `async` outlining infrastructure; no need to manually capture variables.
**Cons:** The `async` dialect is primarily designed for CPU thread pools, not hardware accelerators. Routing it to an NPU would require hacking the async runtime interface.

## Alternative 3: The OpenMP Dialect (`omp.target`)

MLIR has a mature OpenMP dialect used by Flang/Clang for offloading computation to accelerators. 

**How it works:**
1. Lower `vx.spawn` to `omp.target`.
2. Rely on the OpenMP dialect's device outlining passes.

**Pros:** Extremely robust, industry-standard offloading model.
**Cons:** OpenMP brings a massive amount of overhead and complexity (its own massive runtime, data mapping clauses, etc.) that is overkill for our streamlined NPU execution model.

---

## Open Questions
> [!IMPORTANT]
> **Recommendation:** I strongly recommend **Alternative 1 (Custom Outlining)**. Since we already have the C++ MLIR pass infrastructure in `src/dialect/VxLowering.cpp`, writing a custom outlining pass gives us the cleanest, most direct path from Vx code to our NPU runtime without fighting MLIR's built-in assumptions.
> 
> How would you like to proceed? Should we move forward with Alternative 1 and start implementing `vx-kernel-outlining`?
