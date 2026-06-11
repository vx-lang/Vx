# Walkthrough: Stdlib FFI Refactoring

## Changes Made

Based on `SKILLS-stdlib.md`, we identified that `malloc`, `realloc`, and `free` FFI bindings were duplicated across `box.vx`, `option.vx`, and `vec.vx`.
We also noticed that `benchmark.vx` and `tracing.vx` had bare FFI definitions.

Following your guidance, we centralized the memory bindings to `stdlib/std/alloc.vx` but kept the `unsafe` FFI bindings restricted internally to avoid exporting public safe wrappers for inherently unsafe operations. We then updated downstream library files to use the centralized allocations.

1. **Centralized Allocation**:
   - [NEW] `stdlib/std/alloc.vx` was created containing the `extern "C"` declarations for memory operations.
1. **Refactored Data Structures**:
   - [MODIFY] `stdlib/std/box.vx`, `stdlib/std/option.vx`, and `stdlib/std/vec.vx` were refactored to `import std::alloc` and remove their local `extern "C"` copies.
1. **Safe Wrappers for Benchmarks**:
   - [MODIFY] `stdlib/benchmarks/benchmark.vx` and `stdlib/benchmarks/tracing.vx` were updated with safe internal native wrapper functions.

## Validation Results

We ran the `Vx` compiler tests and the result was perfectly clean. The centralized module system allows `malloc` calls to correctly cross-resolve in MLIR after the `import std::alloc;` declaration is included in downstream stdlib modules.

```
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.57s
...
test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
```
