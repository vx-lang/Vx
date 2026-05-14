# Generic Tensor Types & Execution Tests

To write execution tests for `f32`, `bf16`, and `int64`, Akar needs to support parametrizing `Tensor` with different hardware element types. Currently, everything is hardcoded to `f32` (`memref<?x?xf32>`).

## User Review Required

> [!IMPORTANT]
> Since Akar is a systems language, I propose adding Rust-like generic type parsing for Tensors. For example: `Tensor<bf16>`, `Tensor<i64>`.
> Does this syntax look good to you? If so, I will implement the parser for `<T>` and lower these directly into MLIR `bf16` and `i64` dialects.

## Proposed Changes

### 1. AST & Type System (`src/ast.rs`)
- Add `ElementType` enum: `F32`, `F64`, `BF16`, `I64`.
- Modify `Type::Tensor` to hold an `ElementType` (e.g., `Type::Tensor(ElementType)`).

### 2. Lexer & Parser (`src/parser.rs`)
- Update `parse_type()` to parse `<>` after `Tensor`. For example: `Tensor<i64>`.
- Default to `<f32>` if omitted to maintain backwards compatibility with existing tests.
- Update `parse_primary()` to support generic initialization: `Tensor<i64>([4, 4])`.

### 3. Semantic Analysis & Codegen (`src/sema.rs`, `src/codegen.rs`)
- Update type coercion and checking for the new `Tensor(ElementType)`.
- Update `lower_type` in Codegen to map to `memref<?x?xbf16>`, `memref<?x?xi64>`, etc.
- Update floating-point formatting to support `f64` and integer formats for `i64`.
- Dynamically invoke `@printMemrefBF16` or `@printMemrefI64` based on the tensor's element type.

### 4. Tests (`tests/backend/`)
- `matmul_bf16.ak`: Testing `custom_matmul` with `Tensor<bf16>`.
- `matmul_i64.ak`: Testing `custom_matmul` with `Tensor<i64>`.
- Both will include `EXPECT` statements to verify the correct math via JIT.

## Verification Plan
1. `cargo test --lib` to ensure the parser properly extracts `<bf16>` and `<i64>`.
2. `cargo test --test compile_test` to verify JIT execution seamlessly lowers MLIR types and prints the correct arrays.
