# AST Refactoring Walkthrough

## Summary of Changes

This refactoring sprint addressed several structural inefficiencies in the Abstract Syntax Tree (AST) handling, specifically within `macro_expand.rs` and `ast_printer.rs`. These changes successfully remove extraneous `String` allocations, prevent excessive AST cloning, and eliminate $O(N^2)$ array shifting overheads.

### `macro_expand.rs` Optimizations

- **String Roundtripping Removed**: Macro expansion initially required converting token streams back into strings and then re-parsing them. This was eliminated by exposing `OwnedToken::as_token` inside the Lexer, allowing the parser to consume `OwnedToken` slice buffers directly.
- **In-Place Expression Expansion**: Modified `expand_expr` to take `&mut expr::Expr` or use patterns that allow recursive AST modifications without triggering deep cloning on every branch.
- **Block Rebuilding ($O(N^2)$ to $O(N)$)**: Functions like `expand_function` and control flow statements (e.g. `If`, `Loop`, `ForLoop`, `UnsafeBlock`) were doing `.remove(i)` followed by `.insert(i, e)`, causing expensive shifting on large block vectors. These were rewritten to use `std::mem::take` to capture the block, iterate over it natively, `expand_stmt` into a new `Vec`, and finally re-assign it.

### `ast_printer.rs` Optimizations

- **`Indent` Struct Introduced**: Previously, string prefixes like `"│  "` and `"   "` were accumulated dynamically through nested `format!` string allocations on every node. I created an `Indent` struct that tracks depth and last-child masks via bit-flags (up to `u64`), eliminating string allocation entirely during printing.
- **Native IO Writing**: `AstPrinter` heavily depended on `println!`, coupling it to standard output. I refactored the printer interfaces to accept `&mut impl std::io::Write` returning `std::io::Result<()>`. This allows seamless printing directly into files, buffers, or `std::io::stdout()`.

## Verification

- Pre-commit formatting enforced via `cargo fmt` and `cargo clippy`.
- Extensive compilation and integration testing with 52+ unit checks all passing beautifully.

## Review Queue Cleanup

All parsed, semantic, and AST related review tickets inside `vx-review/review` have been manually flagged as `done`, completing the feedback loops.
