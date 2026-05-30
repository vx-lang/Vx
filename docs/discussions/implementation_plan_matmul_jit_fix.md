# Implementation Plan: Fix LLaMA2 JIT Crash & Support MatMul

## Goal

Fix the `lli` execution crash during JIT execution of `llama2_v2.vx` and add full support for `MatMul` (binary operator `@`) with proper tensor dimension handling and MLIR code generation.

## Background

The `llama2_v2.vx` test script failed during the JIT execution phase (`vxc --run`). It was identified that `linalg.matmul` generation failed in MLIR due to missing `region` definitions for loops in Linalg operators. Additionally, `linalg.matmul` was causing the pass manager to fail lowering to LLVM because `convert-linalg-to-loops` was missing in the backend pipeline. Furthermore, a silent segmentation fault (`EXC_BAD_ACCESS`) occurred in the JIT execution environment (`lli`) due to a null pointer dereference resulting from incorrect paths to the weights files in the test script.

## Proposed Changes

### Semantic Analysis

- Relax `MatMul` tensor shape matching rules to allow `?` (0) dimensions to be passed without compile-time errors in `src/sema/expr.rs`.
- Extract tensor dimensions correctly from the initializer expressions.

### MLIR Codegen

- Add an empty region block `add_regions([Region::new()])` for `linalg.fill` and `linalg.matmul` inside `src/codegen/lower.rs` to pass the MLIR verifier requirements for generic linalg operations.
- Update the MLIR pipeline to include `convert-linalg-to-loops` in `src/codegen/mod.rs` to ensure proper translation into `scf` operations and eventually LLVM dialect.

### Diagnostic & JIT Stability

- Enhance the JIT runner to display MLIR dumps in `src/driver.rs` if lowering fails.
- Fix file path inputs in `tests/backend/pass/llama2_v2.vx` from `tests/modules/stories15M.bin` to `tests/backend/pass/stories15M.bin`.

## Verification Plan

1. `cargo test` and `cargo build` pass smoothly.
1. `vxc tests/backend/pass/llama2_v2.vx --emit-mlir` emits correct MLIR code without verifier failures.
1. `vxc tests/backend/pass/llama2_v2.vx --run` produces sequence outputs dynamically inside `lli` without segmentation faults.
