# Review Audit — Remaining Implementation Tasks

## Phase 1: Pipeline Structured Diagnostics

- `[x]` Add `PipelineError` enum to `pipeline.rs` (already existed)
- `[x]` Update `compile_pipeline` return type (already done)
- `[x]` Run tests

## Phase 2: Codegen — Cache Primitive MLIR Types

- `[x]` Create `MlirTypes<'c>` struct in `codegen/generator.rs` (already existed as fields)
- `[x]` Initialize cache in `MeliorGenerator::new` (already done)
- `[x]` Replace ad-hoc `Type::parse` calls with cached lookups
- `[x]` Run tests

## Phase 3: Codegen — Decompose `lower_type`

- `[x]` Extract `lower_tensor_type` helper
- `[ ]` Extract `lower_struct_type` helper (deferred — tightly coupled to enum logic)
- `[ ]` Extract `lower_enum_type` helper (deferred — shares state with struct branch)
- `[x]` Run tests

## Phase 4: Parser — Factor out `parse_identifier_expr`

- `[x]` Extract `apply_type_args` (removed 6x duplicated 6-line block)
- `[x]` Extract `parse_enum_variant_expr` (removed 2x duplicated 30-line block)
- `[x]` Run tests
