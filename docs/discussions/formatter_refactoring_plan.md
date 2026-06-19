# Refactor Formatter Logic

The current formatter logic in `src/formatter.rs` suffers from readability issues and performance bottlenecks (O(N^2) brace matching, O(N) vector insertions per edit, monolithic `format_file` function). This plan addresses the feedback in `Vx_src_formatter.md`.

## Proposed Changes

### [MODIFY] \[formatter.rs\](file:///Users/adityak/go/Vx/src/formatter.rs)

1. **Stack-based Brace Matching**:
   Implement a single-pass O(N) brace matcher that uses a stack to pre-compute the indices of matching braces (`LeftBrace` and `RightBrace`).
1. **Modular Passes**:
   Break `format_file` into the following smaller functions:
   - `tokenize(content: &str) -> Vec<Token>`
   - `normalize_and_expand_blocks(tokens: Vec<Token>) -> Vec<Token>` (handles `{` spacing, `unsafe` collapsing, and block expansion).
   - `adjust_spacing(tokens: Vec<Token>) -> Vec<Token>` (handles operators and `} else {` spacing).
   - `emit_formatted_string(tokens: Vec<Token>, indent_spaces: usize) -> String`
1. **Builder Pattern for Tokens**:
   Replace all `tokens.insert` calls with pushing to a new `Vec::with_capacity(tokens.len())` stream to eliminate O(N) shifting.
1. **Optimized Indentation**:
   Use `out.push_str(&" ".repeat(level * spaces))` or a similar pre-calculated indentation approach in the final emission loop instead of nested `for` loops.

## Verification Plan

### Automated Tests

Run the existing formatting tests (which are quite robust for `format_off_top_level`, `format_off_block`, `format_newline_stripping`, `format_else_single_line_expansion`, and `format_dangling_else`):

```bash
source config.local && cargo test --test compile_test --test formatter_test
source config.local && cargo test formatter::
```

### Manual Verification

Ensure that running `vx-format` on the codebase itself doesn't crash or hang, demonstrating that the O(N) rewrite correctly replicates the previous behavior.
