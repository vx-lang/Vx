# Vx v3.0 Release Proposal: From Shim to Production Compiler

Based on our recent architectural milestones—specifically establishing the "Melior all the way" MLIR pipeline, native object emission, and Enzyme Auto-Diff integration—Vx is perfectly positioned to graduate from a visionary prototype to a mathematically sound, production-grade heterogeneous compiler.

Here is a proposed roadmap for **Vx v3.0**, addressing the major technical gaps identified in our architectural teardown.

## 1. Custom `vx` MLIR Dialect (Completed)

**The Gap:** The compiler relied on custom AST walking and C-FFI shims instead of an industry-standard intermediate representation.
**The Fix:**

- [x] Implement a custom `vx` MLIR dialect (e.g., `vx.spawn`, `vx.transfer`, `vx.tensor`).
- [x] Lower tensor math into MLIR's `linalg` and `affine` dialects rather than emitting function calls.
- [x] Introduce hardware-specific optimization passes (loop unrolling, tiling, and vectorization) using native MLIR passes before dropping down to the LLVM dialect.

## 1b. Real Hardware Kernel Generation & Dispatch

**The Gap:** While `vx.spawn` successfully outlines code for NPU topologies, the resulting function is still compiled for the CPU. It doesn't actually execute on an accelerator.
**The Fix:**

- Expand the custom `vx` MLIR dialect (e.g., `vx.kernel_launch`, `vx.module`) to directly represent hardware-agnostic accelerator kernels. We intentionally avoid MLIR's built-in `gpu` dialect, which is too rigidly coupled to traditional GPU architectures and poorly suited for heterogeneous edge devices (NPUs/TPUs).
- Lower the `vx` kernel representations into hardware-specific target binaries (e.g., SPIR-V for Vulkan/Metal, PTX for NVIDIA, or CoreML representations).
- Inject host-side runtime API calls (e.g., `vulkanLaunchKernel` or a custom Metal C-API) to dispatch the compiled kernel blob to the actual hardware accelerator at runtime.

## 2. Rigorous Topology & Memory Algebra

**The Gap:** The topology type-checker uses hardcoded `if/else` enums.
**The Fix:**

- Introduce a generic mathematical algebra for memory. The compiler will construct a graph of connected hardware topologies (e.g., `Host_DRAM` \<-> `NPU_HBM`).
- `transfer()` calls will be structurally verified at compile time against this graph to ensure physical legality and calculate data movement costs.

## 3. Formal Verification (`Verified<T>`) (Completed)

**The Gap:** The `Verified<T>` label is hollow and has no SMT solver backing.
**The Fix:**

- [x] Implement basic dependent types to verify matrix shapes and dimensions at compile time.
- [x] Add pre-condition and post-condition tracking to the semantic analyzer, ensuring bounds checks and shape transformations are mathematically proven before execution (integrated Z3 solver via `prover.rs`).

## 4. Complete Borrow Checker (NLL & Aliasing)

**The Gap:** The borrow checker only tracks basic linear/affine consumption.
**The Fix:**

- Implement Non-Lexical Lifetimes (NLL) and strict aliasing rules for mutable (`&mut`) and immutable (`&`) borrows.
- This is critical given how heavily our benchmarks rely on raw `*mut f32` pointers. We need to prevent data races during parallel execution across different topologies.

## 5. Standard Library Ecosystem (`std`)

**The Gap:** Users are forced to bypass safety features with raw C-FFI calls.
**The Fix:**

- [x] Bootstrap a native Vx standard library.
- [x] Split the ecosystem into `stdlib/` (compiler intrinsic types, IO) and `packages/` (third-party style libraries like `vx_nn` and `vx_linalg`).
- [x] Include File I/O, native Strings, core mathematical functions, and topologically-aware Tensors (`Tensor<f32, ANE_SRAM>`) backed directly by MLIR `memref`.

## 6. Const Generics Support (Completed)

**The Gap:** The language currently only supports type generics (e.g., `T`, `U`) with optional trait bounds, and relies on built-in compiler intrinsics to handle static, multidimensional tensor shapes.
**The Fix:**

- [x] Expand the AST, parser, and semantic analyzer to support constant values in generic parameter lists (e.g., `struct Array<T, const N: usize>`).
- [x] Implement compile-time constant evaluation within the semantic analyzer to ensure static shape compatibility across tensor bounds.
- [x] Empower user-defined libraries to represent compile-time properties (like kernel sizes or channel counts) securely without hardcoded compiler magic.

## 7. Native `SizeOf` Expressions and Scalar Casts (Completed)

**The Gap:** The compiler relied on string-based type name matching to determine memory allocation sizes, which was fragile and didn't support target-dependent type sizes. Furthermore, explicit type casting between basic scalar types was missing.
**The Fix:**

- [x] Introduced a dedicated `SizeOfExpr` AST node natively evaluated during LLVM lowering using `llvm.getelementptr` and `llvm.mlir.zero`.
- [x] Implemented safe scalar-to-scalar casting in the semantic analyzer and MLIR lowering pipeline.
- [x] Replaced hardcoded type size strings in `malloc` and tensor allocations with native `sizeof()` calls.

## 8. Unified `FileCheck` Test Infrastructure (Completed)

**The Gap:** Tests were verified through a brittle combination of Python generation scripts (`gen_tests.py`) and ad-hoc string matching rather than an industry-standard testing framework.
**The Fix:**

- [x] Migrated all test files (`.vx`, `.mlir`, `.mlr`) to use LLVM's `FileCheck` verification pipeline (`// RUN: ... | FileCheck %s`).
- [x] Consolidated hundreds of auto-generated tests into maintainable, aggregated test files, entirely removing the need for `gen_tests.py`.

______________________________________________________________________

> [!NOTE]
> **Which pillar should we tackle first?**
> The most natural continuation of our current trajectory is **Pillar 1b: Real Hardware Kernel Generation**. Since we just laid down the Melior backend infrastructure, defining our own dialect is the key that unlocks true kernel generation and auto-vectorization.

Let me know if you agree with this assessment or if you'd like to prioritize the Borrow Checker or Topology Algebra first!
