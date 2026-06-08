# Tasks

- [x] Extract dimensions and resolve shape tracking issues in `src/sema/expr.rs`
- [x] Relax type checking for `BinaryOp::MatMul` in `src/sema/expr.rs`
- [x] Update `linalg.fill` and `linalg.matmul` emission in `src/codegen/lower.rs` to attach empty regions.
- [x] Expand `mod.rs` LLVM pipeline to include `convert-linalg-to-loops`.
- [x] Add detailed debug MLIR dumping to `src/driver.rs` when MLIR verification fails.
- [x] Update file path strings in `tests/backend/pass/llama2_v2.vx` to point to the correct `.bin` paths to avoid JIT segfaults.
- [x] Run `cargo run --offline --bin vxc tests/backend/pass/llama2_v2.vx --run` to verify full functionality.
