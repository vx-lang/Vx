# Implement Native Iterator System

Currently, the `for` loop syntax (`for i in start..end`) in Vx is hardcoded to emit a C-style loop over a bounded range of integers. This plan outlines the necessary steps to generalize `for` loops using a native Iterator trait system, similar to Rust.

GitHub Issue: [#110](https://github.com/hiraditya/Vx/issues/110)

## Open Questions

> [!IMPORTANT]
>
> - Do we currently have support for methods taking `self`? If not, we will need to implement `fn next(&mut self)` method resolution.
> - Do we want to implement `IntoIterator` as well so we can seamlessly use `for item in array`, or should we stick to explicitly implementing `Iterator` for things like `Range` first?
> - Does MLIR lowering currently support `enum Option<T>` and branching on its variants? We need to be able to desugar the loop condition effectively.

## Proposed Changes

### Parser and AST

#### [MODIFY] src/ast/stmt.rs

- Update `ForLoopStmt` to replace `start: Box<Expr>` and `end: Box<Expr>` with a single `iterable: Box<Expr>`.

#### [MODIFY] src/parser/stmt.rs

- Update `parse_for_loop` to parse `for identifier in expression { block }`.

#### [MODIFY] src/parser/decl.rs

- Extend `parse_trait_decl` to allow parsing generic type parameters (`trait Iterator<T> { ... }`).
- Extend `parse_impl_block` to allow parsing generic type parameters for the implementation block (`impl<T> Iterator<T> for Range<T> { ... }`).

### Standard Library

#### [NEW] tests/modules/iter.vx

- Define `enum Option<T> { Some(T), None }`.
- Define `trait Iterator<T> { fn next(self: &mut Self) -> Option<T>; }`.
- Define `struct Range { start: i32, end: i32 }`.
- Implement `Iterator<i32>` for `Range`.

### Semantic Analysis

#### [MODIFY] src/sema/stmt.rs

- When analyzing `ForLoopStmt`, perform trait resolution to verify that `iterable` evaluates to a type that implements the `Iterator` trait.
- Extract the loop variable's type based on the `<T>` generic parameter of the matched `Iterator` trait.

### Code Generation (MLIR)

#### [MODIFY] src/codegen/lower.rs

- Lower `ForLoopStmt` into an MLIR construct equivalent to:
  ```rust
  let mut iter = iterable;
  loop {
      let opt = iter.next();
      match opt {
          Some(val) => { /* body */ },
          None => break,
      }
  }
  ```

## Verification Plan

### Automated Tests

- Create a test file `tests/backend/pass/iterator_for_loop.vx` executing a loop over our new custom `Range` iterator structure and asserting the mathematical result.
- Extend `cargo test` to verify no regressions in existing range loops once they are updated to the new syntax parsing.
