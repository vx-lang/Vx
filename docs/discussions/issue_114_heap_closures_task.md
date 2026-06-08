# Heap Allocate Closure Environments Tasks

- `[x]` 1. Emit module-level `malloc` declaration in `src/codegen/generator.rs`
- `[x]` 2. Refactor `ClosureExpr` lowering in `src/codegen/lower.rs` to compute environment size using MLIR
- `[x]` 3. Refactor `ClosureExpr` lowering in `src/codegen/lower.rs` to call `malloc` instead of `llvm.alloca`
- `[x]` 4. Run tests and verify the changes
- `[x]` 5. Commit changes
