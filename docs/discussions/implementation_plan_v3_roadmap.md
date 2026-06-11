# Rewrite LLaMa Runtime in Pure Vx

The current LLaMa implementation relies heavily on `llama_rt.vx` and `llama.rs` to provide C FFI bindings for file I/O, memory mapping, and string manipulation. This hides a lot of the logic in low-level Rust C-stubs. The goal of this plan is to write these routines natively in Vx.

## Proposed Changes

We will remove the Rust FFI functions from `stdlib/rust_core/src/ffi/llama.rs` and reimplement their logic natively inside the Vx codebase.

### `stdlib/std/fs.vx` & `stdlib/std/libc.vx`

We will introduce bindings to standard `libc` functions to allow pure Vx to handle I/O without custom Rust bridges.

- Add `libc` bindings for `fopen`, `fread`, `fclose`, `fseek`, `ftell`.
- Add bindings for `mmap`, `munmap`, `open`, `close`, `fstat` to allow zero-copy weight loading.
- Add higher-level wrapper methods to `std::fs::File`.

### `stdlib/std/string.vx`

The tokenizer requires string comparisons and manipulation. We will augment the `String` type in Vx with:

- `fn eq(self: &String, other: &String) -> bool`
- `fn starts_with(self: &String, prefix: &String) -> bool`
- `fn char_at(self: &String, index: i32) -> i8`
- `fn push_char(self: &mut String, c: i8)`
- `fn from_bytes(bytes: *const u8, len: i32) -> String`

### `tests/modules/llama_rt.vx` (or merged into `llama2_v2.vx`)

We will replace the FFI declarations with native Vx implementations for:

1. **`vx_load_config`**: Open the model file with `fopen`, `fread` 28 bytes directly into a `Config` struct or array.
1. **`vx_load_weights`**: Open the model file via `open()`, get size with `fstat`, and map it into memory using `mmap()`. Return the memory mapped pointer.
1. **`vx_build_tokenizer`**: Read `tokenizer.bin` using `fread` to parse the `vocab_size`, token scores, lengths, and string bytes into native `Vec<f32>` and `Vec<String>` structs.
1. **`vx_decode_token`**: Reimplement the decoding logic using the native `Vec<String>` vocab array and string comparison.
1. **`vx_encode_prompt`**: Reimplement prompt tokenization using simple string matching against the loaded vocab array.

## Pillar 4: General `unsafe` Code Audit & Refactor

**Goal:** Continue removing remaining `unsafe` blocks across other test files as part of the overall memory safety initiative, transitioning FFI logic to higher-level wrappers and abstracting pointer manipulations.

- [x] **Audit:** Search for remaining `unsafe` occurrences in core model files (e.g., `llama2_v2.vx`).
- [x] **Refactoring `TransformerWeights`:** Migrate `vx_load_weights` from returning pointers to a `Tensor` (`all_weights`).
- [x] **Pointer Math Elimination:** Replace `TransformerWeights` struct members with `TransformerWeightOffsets` and compute tensor slices dynamically in `transformer()`.
- [x] **FFI Wrappers:** Wrap FFI calls like `vx_get_rope_freq` and `vx_advance_ptr` in safe internal helper functions.
- [x] **Final Cleanup:** Ensured `tests/backend/pass/llama2_v2.vx` has almost zero raw pointers in the model definition loop.

## Open Questions

> [!WARNING]
> **UTF-8 Encoding/Decoding**
> The Rust tokenizer in `llama.rs` uses full UTF-8 char parsing and hex fallbacks (`<0xXX>`). Doing full UTF-8 handling in native Vx will require adding UTF-8 decoding to `std::string`. Do you want a complete UTF-8 implementation in Vx, or is a simpler byte-level fallback acceptable for now?

> [!IMPORTANT]
> **Codegen / MLIR**
> You mentioned "let our codegen write a proper mlir that handles this." Do you mean we should ensure the AST-to-MLIR generation fully supports compiling these new `File` and `String` methods down to `libc` calls, or is there a specific MLIR dialect (like `llvm.intrinsics` or `pdl`) you had in mind for file I/O?

## Verification Plan

We will build the `llama2_v2.vx` test and ensure it still passes and produces the correct story generation output without relying on `llama.rs` FFI bindings.
