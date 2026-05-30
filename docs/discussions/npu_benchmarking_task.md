## Core Testing Tasks
- [x] 1. `npu_vector_transfer.vx` - Transfer vector CPU <-> ANE, verify.
- [x] 2. `npu_vector_add.vx` - Transfer vector CPU <-> ANE, add, read back, verify.
- [x] 3. `npu_matrix_transpose.vx` - Transfer matrix CPU <-> ANE, transpose, read back, verify.
- [x] 4. `npu_matmul.vx` - Transfer matrices CPU <-> ANE, matmul, read back, verify.
- [x] 5. `npu_tiled_matmul.vx` - Transfer large matrices CPU <-> ANE, tile and matmul, read back, verify.
- [x] **Note**: All the above 5 tests should cover `bf16` and `f32`.

## Tooling and Benchmarking Tasks
- [x] 6. For all 10 tests, add `// RUN` lines emitting and comparing `mlir`, `vx-mlir`, `llvm-ir` (FileCheck rules).
- [x] 7. `cpu_fusion_overhead.vx` - Simple CPU example `matmul -> add bias -> ReLU`. Measure time w/ and w/o fusion.
- [x] 8. `npu_fusion_overhead.vx` - Simple ANE example `matmul -> add bias -> ReLU`. Measure time w/ and w/o fusion.
