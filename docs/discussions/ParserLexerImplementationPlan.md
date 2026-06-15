# Lexer & Parser Refactoring Plan

This plan addresses all the actionable items from the `Vx_src_lexer.md` and `Vx_src_parser_*.md` review documents. The goal is to improve the efficiency, memory footprint, and maintainability of the frontend parsing pipeline.

## Proposed Changes

______________________________________________________________________

### Lexer

#### [MODIFY] src/lexer.rs

- **Keyword Lookup**: Replace the giant `match` block with a hash map lookup using `once_cell::sync::Lazy` and `rustc-hash::FxHashMap`.
- **Token Unification**: Consolidate `OwnedToken` and `Token` by introducing a `TokenBase<Storage>` pattern to reduce duplication.
- **String Lexing**: Refactor `string_literal` so it defaults to returning `Cow::Borrowed` and only allocates a new `String` when an escape character like `\n` is actually encountered.
- **Iterator Peeking**: Use string slices and indices directly in `skip_whitespace` instead of `.clone()` on the iterator.

______________________________________________________________________

### Parser Core

#### [MODIFY] src/parser/mod.rs

- **Zero-Copy Errors**: Update `ParserError::UnexpectedToken` to hold a reference `Token<'a>` instead of cloning the entire `OwnedToken`.
- **Consume Logic**: Update `consume` to avoid cloning the token eagerly, propagating borrowed lifetimes.
- **Peek optimization**: Update `peek_n` to use explicit bound checks.
- **Helper encapsulation**: Migrate `From<&str> for Function` to a dedicated static helper `parse_fn_from_str`.

______________________________________________________________________

### Declarations & Expressions

#### [MODIFY] src/parser/decl.rs

- **Loop Optimization**: Reviewed `Vec::new()` + `push()` loop constructs.

#### [MODIFY] src/parser/expr.rs

- **Control Flow**: Refactor complex expression AST construction where applicable.

______________________________________________________________________

## Verification Plan

### Automated Tests

- `cargo test --all` confirmed green across all suites.
