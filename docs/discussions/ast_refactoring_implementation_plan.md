# AST Refactoring Implementation Plan

This plan addresses the pending actionable improvements identified in the three AST-related review files: `Vx_src_ast_expr.md`, `Vx_src_ast_macro_expand.md`, and `Vx_src_ast_printer.md`.

## User Review Required

Please review the proposed changes below. Some of the actions from the review have already been organically completed in earlier refactors (e.g., using `Symbol` instead of `String` for identifiers). I want to ensure we align on the approach before modifying core AST code.

## Open Questions

- In `ast_printer.rs`, changing `println!` to an `std::io::Write` trait will change the signature of `AstPrinter::print_program`. Any consumers of this API (like `main.rs` or tests) will need to be updated to pass `&mut std::io::stdout()`. Is this acceptable?

## Proposed Changes

### `src/ast/expr.rs`

#### [MODIFY] \[expr.rs\](file:///Users/adityak/go/Vx/src/ast/expr.rs)

- **Regex Optimization**: In `InlineMlirExpr::substitute`, the dynamic compilation of a `Regex` on every call is highly inefficient. We will optimize this by checking if the block string actually contains any of the substitution keys before attempting to regex-replace. (Note: the `Symbol` migration for identifiers is already complete!).
- **Cloning Overhead**: We will investigate if `Expr` substitution can be skipped entirely when `mapping` is empty, avoiding a massive clone tree.

### `src/ast/macro_expand.rs`

#### [MODIFY] \[macro_expand.rs\](file:///Users/adityak/go/Vx/src/ast/macro_expand.rs)

- **Zero-Copy Parsing**: Rewrite `parse_expanded_expr` and `parse_expanded_exprs`. Instead of turning `OwnedToken` back into strings to be re-lexed (an O(N) string allocation + re-parse bottleneck), we'll implement logic to convert `OwnedToken` directly into the `Token` references expected by the `Parser`.
- **In-place Node Expansion**: Modify `expand_expr` to take `&mut expr::Expr` instead of taking ownership by value. This allows us to modify expressions in place, eliminating the need to do `*b.lhs = self.expand_expr(*b.lhs.clone())?` which currently triggers a deep clone of the AST on every traversal step.
- **Efficient Block Traversal**: In functions like `expand_function`, `expand_stmt_children`, and control flow blocks (`if`, `loop`, `unsafe`), we'll replace the O(N^2) `.remove(i)` / `.insert(i)` logic with an efficient `drain`/collect approach.

### `src/ast_printer.rs`

#### [MODIFY] \[ast_printer.rs\](file:///Users/adityak/go/Vx/src/ast_printer.rs)

- **Generic Writer API**: Change all printing methods to accept `&mut impl std::io::Write` instead of hardcoding `println!`. This allows printing to a `String` buffer or streaming efficiently.
- **Indentation Object**: Introduce an `Indent` tracking struct to avoid allocating new `String` objects with `format!("{}...", indent)` on every recursive AST node. This will drastically reduce memory overhead during large AST dumps.

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure that macro expansions still produce valid code and AST parsing logic hasn't broken.
- Ensure that the `ast_printer` API updates are propagated successfully to their call sites in `main.rs` and other compilation stages.
