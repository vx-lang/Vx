# Review Audit: Implementation Plan for Remaining Items

Based on the audit, the following high-impact items will be addressed in priority order.

## Phase 1: Pipeline Structured Diagnostics

### [MODIFY] pipeline.rs

- Add a `PipelineError` enum with `IO`, `Parse`, `Semantic`, `Metadata` variants.
- Change `compile_pipeline` return type from `Result<(), String>` to `Result<(), PipelineError>`.
- Implement `Display` for `PipelineError`.

## Phase 2: Codegen — Cache Primitive MLIR Types

### [MODIFY] codegen/generator.rs

- Create an `MlirTypes<'c>` struct that caches `i1`, `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`, `f64`, `index`, `ptr`, `none` types.
- Initialize the cache in `MeliorGenerator::new`.
- Replace ad-hoc `Type::parse(context, "i32")` calls in `lower_type` with cached lookups.

## Phase 3: Codegen — Decompose `lower_type`

### [MODIFY] codegen/generator.rs

- Extract `lower_tensor_type`, `lower_struct_type`, `lower_enum_type` helper methods.
- Keep the main `lower_type` as a dispatcher.

## Phase 4: Parser — Factor out `parse_identifier_expr`

### [MODIFY] parser/expr.rs

- Extract `parse_struct_literal`, `parse_call_args`, `parse_enum_variant_construction` from the monolithic `parse_identifier_expr`.

## Verification Plan

- `source config.local && cargo test` after each phase.
- `cargo fmt` before commits.
