# NPU Hardware Dispatch (`vx.spawn` & `vx.transfer`)

## Overview

We've successfully established a hardware isolation layer using the new operations `vx.spawn` and `vx.transfer`. The Vx MLIR backend is now capable of capturing operands destined for an accelerator (`Topology::NPU[0]`), offloading computations by outlining regions into separated functions, and maintaining memory safety using Zero-Leak RAII patterns in LLVM dialect generation!

## `vx.transfer`

The compiler now recognizes explicit transfer of a `memref` to a target memory space (`Memory::NPU_HBM` or `Topology::NPU[0]`).

- Converts `vx.transfer` to explicit `memref.alloc` bound to the appropriate physical memory layer.
- Immediately emits `memref.copy` to move data from host memory space to accelerator space.
- Implements a "Zero Leak RAII" guard where a `memref.dealloc` is guaranteed to be injected at the terminator of the current lexical block so that memory transitions remain strictly memory-safe without requiring garbage collection or explicit `free()`.

## `vx.spawn` and Dynamic Region Outlining

We use MLIR C++ passes to orchestrate custom execution layers:

- The MLIR compiler identifies all `vx.spawn` blocks tied to NPU architecture topologies.
- It scans the nested scope using a localized block traversal to identify values used inside the block but defined outside of it (captured operands).
- It generates a custom wrapper `func.func` kernel named `__vx_npu_kernel_{N}`.
- It creates a direct `IRMapping` from the captured host values to the arguments of the new kernel function.
- It clones the original computations cleanly bypassing Greedy Pattern Rewrite "Rollback" restrictions.
- Replaces the inline block with a `func.call` invocation to pass execution control cleanly!

## Validation pipeline

We tested JIT execution using standard MLIR tools without triggering Dialect Conversion mismatches:

- Replaced the `melior_codegen.rs` tensor mock implementation with proper dynamic `memref.alloc` accompanied by explicit `operandSegmentSizes` to enable end-to-end `finalize-memref-to-llvm` pass compilation!
- Discovered and resolved a C ABI pointer conversion issue in MLIR where `llvm.emit_c_interface` misidentified address space mapping, permitting transparent UMA execution.

## Testing and Verifying

```bash
cargo run --bin vxc -- tests/backend/pass/npu_dispatch_spawn.vx
cargo run --bin vxc -- --emit-mlir tests/backend/pass/npu_dispatch_spawn.vx
```

The test passes completely via our custom orchestration JIT engine using `mlir-opt`, `mlir-translate`, and `lli`!
