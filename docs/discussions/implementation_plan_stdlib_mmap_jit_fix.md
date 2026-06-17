# Phase 3: V3 Roadmap (stdlib, mmap, llama.c, vscode-vx)

This implementation plan outlines the steps necessary to complete the Vx compiler V3 roadmap, focusing on standard library features, external bindings, language utility routines, and tooling updates.

## Proposed Changes

### Part 1: Standard Library Bindings (`libc` and `mmap`)

To support memory-mapped file loading (required for the `llama.c` weight loading), we need to implement standard C-library bindings natively within Vx.

#### [NEW] `stdlib/libc.vx`

- Expose essential file I/O and POSIX operations via `extern` declarations:
  - `extern fn open(path: *const u8, oflag: i32) -> i32;`
  - `extern fn close(fd: i32) -> i32;`
  - `extern fn lseek(fd: i32, offset: i64, whence: i32) -> i64;`

#### [NEW] `stdlib/mmap.vx`

- Provide high-level, safe wrapper implementations for memory mapping using `libc`:
  - `extern fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;`
  - `extern fn munmap(addr: *mut u8, length: usize) -> i32;`
  - Encapsulate memory flags (`PROT_READ`, `MAP_SHARED`, etc.) within `enum`s or constant variables to provide a safe idiomatic Vx interface.

### Part 2: String Routines

Basic string routines are needed to parse strings effectively within `llama.c` config loading contexts.

#### [MODIFY] `stdlib/string.vx`

- Implement foundational string slice routines:
  - `fn string_length(s: *const u8) -> usize`
  - `fn string_compare(s1: *const u8, s2: *const u8) -> i32`
  - `fn parse_int(s: *const u8) -> i32`

### Part 3: `llama.c` Config & Weights Loader Rewrite

Currently, the LLM loader heavily relies on file streams. We need to refactor it to leverage `mmap` for fast, zero-copy weight loading.

#### [MODIFY] `examples/llama.vx`

- Replace file stream I/O loading logic with the new `mmap` implementations.
- Refactor model parameter parsing to utilize the new string routines, reading metadata and quantization settings directly from the mapped memory region.

### Part 4: Tooling Updates (`vscode-vx`)

#### [MODIFY] `vscode-vx/syntaxes/vx.tmLanguage.json`

- Update the TextMate syntax highlighting grammar to match and appropriately colorize macros (`print!`, `println!`, etc.).
- Ensure the regex correctly captures `[a-zA-Z_][a-zA-Z0-9_]*!` identifiers and tags them with the standard `entity.name.function.macro.vx` classification.

#### [MODIFY] `vscode-vx/package.json`

- Bump the version for the new release of the VS Code extension.

## Open Questions

> [!WARNING]
> Do we want to construct the memory map bindings for both Unix (`mmap`) and Windows (`MapViewOfFile`) or strictly constrain it to POSIX Unix bindings for now? I assume Unix-only `mmap` based on the task description.

> [!WARNING]
> Are there specific `vscode-vx` macros you want explicitly handled, or should all postfix bang operators (`identifier!`) be styled as macros universally?

## Verification Plan

### Automated Tests

- `cargo test --test compile_test`
- Compile and run the updated `examples/llama.vx` model with dummy weights to verify the `mmap` loading successfully populates tensors without SEGFAULTS.

### Manual Verification

- Install the packaged `vscode-vx` `.vsix` file locally into VS Code to verify that the syntax highlighting changes correctly colorize macro invocations like `print!("Hello")`.
