# NPU Hardware Dispatch (`vx.spawn` & `vx.transfer`)

- [x] Implement explicit `vx.transfer` lowering to `memref.alloc` and `memref.copy`.
- [x] Enforce `memref.dealloc` zero-leak RAII semantics at block exit.
- [x] Modify semantic analyzer to enforce explicit `transfer(...)` for hardware execution.
- [x] Update frontend tests to use explicit transfers.
- [x] Implement `vx.spawn` region outlining in `VxLowering.cpp` for `Topology::NPU[0]`.
- [x] Capture operands and inject block arguments dynamically via `IRMapping` instead of mutating the spawn region (to bypass greedy pattern rewrite limitations).
- [x] Remove `llvm.emit_c_interface` to maintain consistent `memref` pointer address space conversions inside JIT execution context.
- [x] Update `melior_codegen.rs` to register and run `VxLowering` correctly inside its own `PassManager` to avoid pipeline overwrite conflicts.
- [x] Fix `Tensor([...])` initialization in `melior_codegen.rs` to emit `memref.alloc` instead of `builtin.unrealized_conversion_cast` with `operandSegmentSizes` support to satisfy standard MLIR LLVM translation.
- [x] Validate JIT execution pipeline using `cargo run --bin vxc tests/backend/pass/npu_dispatch_spawn.vx`.
