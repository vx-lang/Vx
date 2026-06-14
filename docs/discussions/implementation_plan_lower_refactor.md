# Goal: Refactor `src/codegen/lower.rs`

The `lower.rs` file is the "engine room" of the backend and has grown to over 4800 lines. The primary goals are to split it into smaller, more manageable submodules and to improve efficiency by reducing string allocations during type lowering.

## User Review Required

> [!WARNING]
> Splitting a ~5000 line file is a massive refactoring step. I propose writing a Python script to reliably migrate the `impl LowerToMelior for ...` blocks into their respective new files, then committing this purely structural change before applying any functional optimizations. This makes it easier to review and bisect if necessary. Do you approve of this approach?

## Open Questions

- Should tensor-specific and autodiff AST nodes (e.g. `GradExpr`, `TransferExpr`, `VjpExpr`) go into `tensors.rs` or `ad.rs`? I plan to put them in `tensors.rs` for now.

## Proposed Changes

We will create a new directory `src/codegen/lower/` and split the AST lowering logic.

### Structural Changes (Submodules)

#### [NEW] src/codegen/lower/mod.rs

- Will contain the `LowerToMelior` trait definition.
- Helper methods like `build_constant`, `is_memref`, and type utilities to replace raw `to_string()` checks.

#### [NEW] src/codegen/lower/expr.rs

- Lowering logic for basic expressions: `IdentifierExpr`, `NumberExpr`, `StringLiteralExpr`, `BinaryOpExpr`, `RelationalOpExpr`, `LogicalOpExpr`, `UnaryOpExpr`, etc.

#### [NEW] src/codegen/lower/stmt.rs

- Lowering logic for statements: `LetDeclStmt`, `AssignStmt`, `CompoundAssignStmt`, `ExprStmtStmt`, `ReturnStmt`.

#### [NEW] src/codegen/lower/control_flow.rs

- Lowering logic for control flow: `IfExpr`, `ForLoopStmt`, `LoopStmt`, `MatchExpr`, `BreakStmt`, `ContinueStmt`.

#### [NEW] src/codegen/lower/tensors.rs

- Lowering logic for AD and specific ML constructs: `GradExpr`, `VjpExpr`, `JvpExpr`, `TransferExpr`, `SpawnOnExpr`.

#### [MODIFY] src/codegen/lower.rs -> [DELETE] src/codegen/lower.rs

- The current 4850-line file will be deleted and replaced by the module structure above.

### Performance Optimizations

#### [MODIFY] src/codegen/lower/mod.rs (and others)

- Replace `Type::parse(gen.context, "index").unwrap()` and other hardcoded types with the newly cached `gen.index_ty`, `gen.i32_ty`, etc., from `MeliorGenerator`.
- Abstract string-based type checks (`ty.to_string().starts_with("memref<")`) into `gen.is_memref(&ty)` and `gen.is_llvm_ptr(&ty)` helpers. This centralizes string allocations and sets the stage for future O(1) C-API checks.

## Verification Plan

### Automated Tests

- Run `cargo fmt` and `cargo clippy`.
- Run `cargo test` to ensure that breaking down the file into submodules and optimizing type checks doesn't break the lowering of any existing constructs.

### Manual Verification

- N/A - The tests are comprehensive enough.
