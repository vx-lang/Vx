# Pillar 1: Custom `vx` MLIR Dialect & Real Kernel Generation

The goal of this phase is to move away from treating `vx.spawn` and `vx.transfer` as opaque black boxes or relying on C-FFI shims. Instead, we will implement a dedicated MLIR conversion pass in C++ that organically lowers `vx` dialect operations into standard LLVM/MLIR constructs (like `async` blocks or AMX hardware dispatcher calls) so that they can be optimized by native compiler passes.

## Proposed Changes

### 1. Extend TableGen (`include/VxDialect.td`)
- Refine the `vx.spawn` and `vx.transfer` operations.
- Add `IsolatedFromAbove` or `AutomaticAllocationScope` traits to the `vx.spawn` region to ensure it behaves correctly during MLIR loop unrolling and optimization passes.

### 2. Implement C++ Lowering Pass (`src/dialect/VxLowering.cpp`)
#### [NEW] [src/dialect/VxLowering.cpp](file:///Users/adityak/go/Vx/src/dialect/VxLowering.cpp)
- Create an MLIR `ConversionTarget` and `OpRewritePattern` for `vx::SpawnOp`.
- **CPU Topology (`HostDRAM`)**: Lower `vx.spawn` into an `async.execute` block. This allows MLIR to natively handle multi-threading without requiring us to write a manual pthread C-FFI shim.
- **Hardware Topology (`NPUHBM`)**: Lower `vx.spawn` into an explicit MLIR `func.call` directed to the `npu_dispatch` AMX runtime.

### 3. Expose Pass via C-API (`src/dialect/VxDialect.cpp` / `VxDialect.h`)
#### [MODIFY] [src/dialect/VxDialect.cpp](file:///Users/adityak/go/Vx/src/dialect/VxDialect.cpp)
- Create a `createConvertVxToStandardPass()` function.
- Expose a C-API function (`registerVxLoweringPass`) so that Rust can inject this custom pass into the `melior` pipeline.

### 4. Rust Build Integration (`build.rs` & `melior_codegen.rs`)
#### [MODIFY] [build.rs](file:///Users/adityak/go/Vx/build.rs)
- Update the build script to compile `VxLowering.cpp` alongside `VxDialect.cpp` and bundle it into `libvx_dialect.a`.
#### [MODIFY] [src/melior_codegen.rs](file:///Users/adityak/go/Vx/src/melior_codegen.rs)
- Declare the `extern "C" { fn registerVxLoweringPass(...) }`.
- Insert the Vx lowering pass into the `melior::pass::PassManager` pipeline immediately before `lower-affine` and `convert-scf-to-cf`.

## Open Questions

> [!IMPORTANT]
> 1. **Lowering Target for `vx.spawn`:** Should we use the MLIR `async` dialect for CPU threading, or should we lower it to `scf.parallel`? The `async` dialect is generally more flexible for task-based spawning, but `scf.parallel` is better for strict data-parallel loops.
> 2. **Reviewing C++ Code:** Since MLIR C++ can be quite verbose, do you want me to write the full Tablegen and C++ rewrite patterns first, or would you prefer to mock the pass structure and test the Rust FFI connection first?
