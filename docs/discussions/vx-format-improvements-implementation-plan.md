# Improve `vx-format` for `if`/`else` and loop condition formatting

The user has requested improvements to `vx-format` to automatically handle the formatting issues we recently encountered and fixed manually. Specifically:

1. `for` loops split across lines (e.g. `for \n i in 0..128`).
1. `if` conditions split across lines before operators (e.g. `if sum_idx \n == 0`).
1. Dangling `else` clauses on new lines (e.g. `} \n else {`).
1. Single-line `if` blocks attached to multi-line `else` blocks (e.g. `if x { y; } else { \n ... \n }`).

## Proposed Changes

Currently, `vx-format` operates as a simple stream formatter in `src/formatter.rs`. It iterates through tokens and preserves all newlines within `Whitespace` tokens, which is why it didn't fix these issues automatically.

Instead of writing a complex AST-based formatter from scratch, we will enhance `src/formatter.rs` by adding a **Token Stream Normalization Pass** before the formatting loop.

### [MODIFY] `src/formatter.rs`

1. **Tokenize first:** Change `format_file` to collect all tokens into a `Vec<Token>` using `lexer.tokenize()`.
1. **Whitespace Normalization:** Iterate through the token vector and intelligently modify `TokenType::Whitespace` tokens based on their context:
   - **Operators:** If a whitespace token contains a newline and sits before a binary operator (`==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`), replace the newline with a single space to pull the operator up to the previous line.
   - **For loops:** If a whitespace token with a newline is immediately preceded by `TokenType::For`, replace the newline with a single space.
   - **If statements:** If a whitespace token with a newline is immediately preceded by `TokenType::If`, replace the newline with a single space.
   - **Else positioning:** If a whitespace token sits between `TokenType::RightBrace` and `TokenType::Else`, aggressively replace it with a single space `" "` to guarantee the `} else` formatting. Do the same between `Else` and `LeftBrace` to guarantee `else {`.
1. **Single-line Block Expansion:**
   - Add logic that detects when an `Else` token is found.
   - Scan backward to the preceding `LeftBrace` and `RightBrace` of the `If` block.
   - If that block is a single-line block (contains no newlines), insert newlines directly after the `LeftBrace` and directly before the `RightBrace` to expand it, ensuring symmetrical bracing with the `else` block.

## User Review Required

Does this regex/token-stream approach sound good? It avoids the extreme complexity of building a full AST-based formatter (which would require a massive rewrite) while cleanly solving the precise formatting gripes we ran into!

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure we haven't broken any `vx-format` invariants.
- Add new unit tests to `src/formatter.rs` to specifically test:
  - `} \n else {` collapses to `} else {`
  - `if x { y; } else {` expands to multi-line
  - `for \n i` collapses to `for i`
