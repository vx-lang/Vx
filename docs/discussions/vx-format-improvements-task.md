# `vx-format` Improvements

- [x] Modify `src/formatter.rs` to tokenize the entire file into a `Vec<Token>`.
- [x] Implement Token Stream Normalization Pass:
  - [x] Strip newlines before binary operators.
  - [x] Strip newlines after `for` and `if`.
  - [x] Normalize `} \n else {`.
  - [x] Expand single-line `if` blocks attached to `else`.
- [x] Update the formatting loop to iterate over the normalized `Vec<Token>`.
- [x] Add unit tests in `src/formatter.rs`.
- [x] Run `cargo test` to verify changes.
