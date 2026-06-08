# Fix `sizeof` Implementation

Currently, `sizeof<T>()` is parsed as a regular function call to a string-mangled name `sizeof<T>`. In the codegen phase, the compiler performs string-matching on the function name, extracts `T` as a string, and computes its size by hardcoding sizes of primitives and naively summing up struct fields, completely ignoring target data layout, alignment, and padding.

## Proposed Changes

### `src/ast/expr.rs`

- Add `SizeOfExpr` to explicitly represent the `sizeof` operator.
  ```rust
  #[derive(Debug, PartialEq, Clone)]
  pub struct SizeOfExpr {
      pub ty: Type,
      pub span: Span,
  }
  ```
- Add `SizeOf(SizeOfExpr)` to the `Expr` enum.

### `src/parser/expr.rs`

- Update expression parsing to recognize `sizeof` as a keyword.
- When `sizeof` is encountered, expect `<` followed by a type, `>`, `(`, and `)`.
- Return `Expr::SizeOf(SizeOfExpr)`.

### `src/sema/expr.rs`

- Add type-checking for `Expr::SizeOf`.
- `sizeof` will unconditionally return a `Scalar(I64)` (since sizes can be large, and LLVM `ptrtoint` naturally produces an `i64` on 64-bit systems).
- Resolve the inner type of `SizeOfExpr` using the current generic mappings.

### `src/codegen/lower.rs`

- Remove the string matching hack (`if name.starts_with("sizeof<")`) from function call lowering.
- Implement the `LowerToMelior` trait for `Expr::SizeOf`.
- Use the standard LLVM `getelementptr` trick to dynamically evaluate the type's size according to the exact LLVM data layout:
  ```mlir
  %null = llvm.mlir.zero : !llvm.ptr
  %gep = llvm.getelementptr %null[1] : (!llvm.ptr) -> !llvm.ptr, <target_type>
  %size = llvm.ptrtoint %gep : !llvm.ptr to i64
  ```

## User Review Required

> [!IMPORTANT]
> The current size calculation returns `i32` but standard `sizeof` in LLVM usually maps to `size_t` (`i64` on 64-bit architectures). Changing this to `i64` will require updating any tests that currently expect `sizeof` to return an `i32` (e.g. `tests/backend/test_generic_vec.vx`, `stdlib/std/vec.vx`, `stdlib/std/box.vx`). Is this acceptable?

## Open Questions

> [!WARNING]
> You asked if we're doing stringly-typed computations elsewhere. The answer is yes, unfortunately. `src/codegen/lower.rs` and `generator.rs` heavily rely on `ty.to_string().starts_with(...)` for types like `memref<...>`, `!llvm.ptr`, `Option<T>`, and `Tensor_T`. While we can fix `sizeof` cleanly now, refactoring the entire type-lowering system to avoid string matching is a massive structural change. I recommend we focus purely on `sizeof` for this task.
