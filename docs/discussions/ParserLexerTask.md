# Parser & Lexer Refactoring

- **Lexer (`src/lexer.rs`)**
  - `[x]` Use `once_cell` + `FxHashMap` for keyword lookup
  - `[x]` Consolidate `OwnedToken` and `Token` if possible
  - `[x]` Optimize string lexing (avoid eager allocation)
  - `[x]` Optimize iterator peeking (use `peek()` instead of iterator clones)
- **Parser Core (`mod.rs`)**
  - `[x]` Use zero-copy reference for `UnexpectedToken` error
  - `[x]` Avoid cloning token in `consume` on success path
  - `[x]` Explicit bound checks in `peek_n`
  - `[x]` Replace `impl From<&str> for Function` with `parse_fn_from_str`
- **Parser Declarations (`decl.rs`)**
  - `[x]` Replace `Vec::new()` + `push()` loops with more functional iterators if clear.
  - `[x]` Avoid repetitive clone/allocation during error paths.
- **Parser Expressions (`expr.rs`)**
  - `[x]` Flatten nested `if`s (use `match` or extract methods)
  - `[x]` Extract parsing of complex expressions (e.g., `parse_binary_expr`)
