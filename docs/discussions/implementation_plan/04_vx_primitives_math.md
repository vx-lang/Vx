# Implement Vx StdLib: Primitives & Math

This plan addresses Issue #12: Provide standard numerical primitives, math traits, and SIMD intrinsics mapped to `libcore`.

## User Review Required

> [!IMPORTANT]
> The SIMD intrinsics mapping will use Rust's `core::simd` (portable SIMD) if possible, or target-specific intrinsics via `core::arch::x86_64` (or standard MLIR `vector` dialect mappings if preferred). Currently, the FFI boundary relies on pointers and primitive scalars. For SIMD, we will map them to scalar arrays (`*mut f32` or `*mut i32` for 128/256-bit vectors) across the FFI, then utilize `libcore` SIMD instructions inside the Rust runtime. Please confirm if this approach fits the architecture!

## Open Questions

> [!WARNING]
>
> 1. **Math Implementation:** Currently, `Math::sin`, `Math::cos`, etc., are hardcoded in `codegen.rs` (MLIR). Since the issue states "mapped to `libcore`", should I remove the hardcoded MLIR math ops and implement them entirely as `extern` FFI functions (`stdlib/std/math.vx` mapped to Rust `libm` or `std`), or should I keep MLIR lowering and ONLY add FFI fallbacks for missing operations?
> 1. **SIMD Intrinsics:** Does Vx have a native SIMD/Vector type in the AST (like `f32x4`), or should SIMD intrinsics take pointers/arrays of primitives as arguments?

## Proposed Changes

### AST & Parser

- Ensure `f16` is fully parsed (`"f16" => Ok(Type::Scalar(ElementType::F16))`) and supported in `src/parser.rs`.

______________________________________________________________________

### CodeGen (MLIR Lowering)

- Update `lower_type` in `src/codegen.rs` to properly map all newly requested primitive sizes to MLIR types:
  - `I4, I8, I16, I32, I64, I128` -> `"i4", "i8", "i16", "i32", "i64", "i128"`
  - `U4, U8, U16, U32, U64, U128` -> `"i4", "i8", "i16", "i32", "i64", "i128"` (signless)
  - `F16, BF16, F32, F64` -> `"f16", "bf16", "f32", "f64"`

______________________________________________________________________

### Rust Core (`stdlib/rust_core/`)

#### [NEW] `stdlib/rust_core/src/math.rs`

- Implement FFI wrappers mapping to `libcore` (or `libm`/`std`) for standard Math traits:
  - `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
  - `abs`, `sqrt`, `exp`, `log`, `log2`, `log10`
  - Implement specialized versions for `f32` and `f64`.

#### [NEW] `stdlib/rust_core/src/simd.rs`

- Implement SIMD intrinsics mappings exposing core architecture instructions (e.g., vectorized add, sub, mul, fma).
  - Define `vx_simd_add_f32x4(a: *const f32, b: *const f32, out: *mut f32)`.

#### [MODIFY] `stdlib/rust_core/src/lib.rs` and `mod.rs`

- Export `math` and `simd` modules.

______________________________________________________________________

### Vx StdLib (`stdlib/std/`)

#### [NEW] `stdlib/std/math.vx`

- Expose Math traits via `extern` block and safe wrappers for primitives.

#### [NEW] `stdlib/std/simd.vx`

- Expose SIMD intrinsics via `extern` block.

## Verification Plan

### Automated Tests

- Create `tests/backend/pass/ffi_math.vx` to test Math trait functions (sin, cos, abs).
- Create `tests/backend/pass/ffi_simd.vx` to test SIMD vectorized operations.
- Run `cargo test test_backend` to ensure the MLIR lowering and execution succeeds without missing symbol errors.
