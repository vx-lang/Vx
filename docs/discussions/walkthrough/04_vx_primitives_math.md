# Vx Module System & Imports

## Changes Made

- Deprecated the `Expr::Import` node and replaced it with a top-level `ImportDecl` in the AST, allowing `import path::to::module;` syntax.
- Updated the Lexer and Parser to support the new `ImportDecl` syntax at the global file scope, aligning with the Rust-like `use` or `import` conventions requested.
- Created `ModuleLoader` (`src/module_loader.rs`) to recursively discover, parse, and load dependencies based on the `VX_STD_PATH` (defaulting to `stdlib/std`).
- Wired `ModuleLoader` into the main compilation pipeline (`src/main.rs`) and test runners (`tests/compile_test.rs`).
- Corrected the `MlirGenerator` to process `externs` from all imported modules so that FFI calls like `vx_vec_new_i32()` are emitted properly in the MLIR output.
- Migrated the standard library `extern` blocks for Vectors, Options, Results, and IO from individual test files to the new `stdlib/std/*.vx` modules.

## What Was Tested

- We ran `cargo test test_backend` to ensure the MLIR JIT compiler could properly link and execute code depending on the new standard library wrapper modules.
- We confirmed that `tests/compile_test.rs` correctly resolves dependencies across the `frontend` and `middle-end` test phases, ensuring all legacy test infrastructure is compatible with the new multi-file AST environment.

## Validation Results

All tests are currently passing (`test test_backend_fail ... ok` and `test test_backend ... ok`). The Vx compiler can now dynamically load `stdlib/std/*.vx` FFI definitions dynamically without explicitly defining `extern` blocks in user-written code!

## Extending the Standard Library with Memory-Safe Collections

To provide robust and memory-safe data structures, we integrated core Rust collections (`HashMap`, `HashSet`, `String`) into the Vx standard library using a cross-boundary FFI approach.

1. **FFI Macros Generation (`stdlib/rust_core/src/ffi/macros.rs`)**:
   - Created FFI macros `instantiate_hash_map_ffi!`, `instantiate_hash_set_ffi!`, and `instantiate_string_ffi!` to expose Rust collections to the C-ABI.
   - Specialized these macros in `stdlib/rust_core/src/collections/` for common primitive types, avoiding C-ABI limitations regarding generics.
1. **Vx Language Expositions**:
   - Created corresponding standard library declarations: `stdlib/std/hash_map.vx`, `stdlib/std/hash_set.vx`, and `stdlib/std/string.vx`.
   - Solved `void` function syntax requirements and properly mapped boolean types to `Bool`.
1. **Backend Integration and Testing**:
   - Added complete lifecycle tests for each collection in `tests/backend/pass/`.
   - Verified successful lowering to MLIR, translation to LLVM IR, and JIT execution via `lli`, ensuring symbols dynamically link with `libvx_std_core.dylib`.

## Primitives and Math Traits

To expand numerical computing capabilities (Issue #12), we enabled full support for advanced numerical types and standard math traits.

1. **Numeric Primitives**:
   - Updated `src/parser.rs` and `src/codegen.rs` to parse and lower advanced datatypes into MLIR: `I4` through `I128`, unsigned counterparts `U4` through `U128`, and `F16`, `BF16`, `F32`, `F64`.
1. **Standard Math & SIMD (mapped to libcore)**:
   - Implemented standard math traits (`sin`, `cos`, `tan`, `abs`, `sqrt`, `exp`, `log`, etc.) mapped to Rust's FFI in `stdlib/rust_core/src/math.rs`.
   - Implemented SIMD vector operations (`add_f32x4`, `sub_f32x4`, `mul_f32x4`, `fma_f32x4`) inside `stdlib/rust_core/src/simd.rs` using raw contiguous pointers.
   - Exposed these interfaces in the `Vx` standard library through `stdlib/std/math.vx` and `stdlib/std/simd.vx` wrapper modules.
   - Backend validated via `ffi_math.vx` and `ffi_simd.vx` integration tests.
