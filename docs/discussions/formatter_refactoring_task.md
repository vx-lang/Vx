# Formatter Refactoring Tasks

- `[x]` Replace `format_file` with discrete passes
  - `[x]` Implement `normalize_and_expand_blocks` (O(N) builder pattern)
  - `[x]` Implement `adjust_spacing` (O(N) builder pattern)
  - `[x]` Update `format_file` to orchestrate these passes
- `[x]` Verify formatter behaves identically using `cargo test`
