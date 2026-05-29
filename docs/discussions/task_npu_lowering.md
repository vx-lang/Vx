# Task: NPU Dispatch Lowering

- [x] Create a new MLIR pass `ConvertVxToLLVMIRPass` or update `ConvertVxToLLVMPass` to use `LLVMTypeConverter` to lower `vx.dispatch` to `llvm.call` with packed pointers.
- [x] Add the required LLVM IR types and attributes in C++.
- [x] Implement the argument packing logic (`llvm.alloca` for the argument pointers, and saving `memref` pointers).
- [x] Update `src/dialect/VxLowering.cpp` and `src/dialect/CMakeLists.txt` / `src/CMakeLists.txt` if needed.
- [x] Update `tests/optimizations/pass/npu_lowering.vx` to test the new LLVM IR generation.
- [x] Run `cargo check` and `cargo test`.
- [x] Walkthrough and commit.
