# AST Refactoring Tasks

- `[/]` **Refactor `src/ast/expr.rs`**

  - `[x]` Skip AST substitution entirely when `mapping` is empty to eliminate clone overhead.
  - `[x]` Optimize `InlineMlirExpr::substitute` to avoid regex compilation if `block_str` contains no mapping keys.

- `[x]` **Refactor `src/ast/macro_expand.rs`**

  - `[x]` Eliminate `String` round-tripping in macro token parsing by feeding `OwnedToken` slice directly to `Parser`.
  - `[x]` Update `expand_expr` to mutate `&mut expr::Expr` in-place, preventing deep AST cloning.
  - `[x]` Convert block manipulation loops (`remove(i)` / `insert`) to use efficient `drain` / `extend` in `expand_function` and control flow statements.

- `[x]` **Refactor `src/ast_printer.rs`**

  - `[x]` Create `Indent` struct to manage prefix formatting without generating new `String` objects on every node.
  - `[x]` Convert all `AstPrinter` methods to accept `&mut impl std::io::Write` instead of hardcoded `println!`.
  - `[x]` Update `AstPrinter` call sites in the codebase to pass `&mut std::io::stdout()`.

- `[ ]` **Verification**

  - `[ ]` Run `cargo build` and `cargo test` to ensure changes didn't break functionality.
  - `[ ]` Mark remaining review files as `done`.
