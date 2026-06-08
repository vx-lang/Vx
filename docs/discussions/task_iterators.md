# Iterator System Implementation Tasks

- `[x]` 1. **Parser and AST Updates**
  - `[x]` Update `ForLoopStmt` AST definition in `src/ast/stmt.rs`
  - `[x]` Update `parse_for_loop` in `src/parser/stmt.rs`
  - `[x]` Update `parse_trait_decl` in `src/parser/decl.rs` to support generic parameters
  - `[x]` Update `parse_impl_block` in `src/parser/decl.rs` to support generic parameters
- `[x]` 2. **Semantic Analysis Updates**
  - `[x]` Implement semantic checking for the generalized `ForLoopStmt` in `src/sema/stmt.rs`
  - `[x]` Integrate `Option<T>` parsing/type inference in semantic logic if necessary
- `[x]` 3. **Code Generation (MLIR) Updates**
  - `[x]` Update `ForLoopStmt` lowering in `src/codegen/lower.rs` to handle iterables.
  - `[x]` Map `Range` expressions directly to MLIR `scf.for` start/end bounds for now to maintain existing behavior while the new Iterator API is bootstrapped.
- `[x]` 4. **Standard Library Additions**
  - `[x]` Create `tests/modules/iter.vx` with `Option<T>`, `Iterator<T>`, `Range`
- `[x]` 5. **Testing & Verification**
  - `[x]` Write tests in `tests/backend/pass/iterator_for_loop.vx`
  - `[x]` Verify new iterator API doesn't break `tests/modules/llama.vx` loop constructs or implementations
