# Goal Description

Scale the Vx compiler test suite from ~250 tests to 1000+ tests to guarantee maturity, robustness, and semantic correctness. This will be achieved by moving from coarse-grained tests to highly specific, granular test cases, generating exhaustive behavioral test suites, and leveraging randomized fuzzing.

## User Review Required

Please review the proposed four-phase plan. Once approved, I will use the `gh` CLI to create a GitHub issue with this plan to track our future progress, and then immediately start executing **Phase 1: Grammar Tests**.

## Proposed Changes

The scaling effort will be broken down into four distinct phases, executed sequentially:

### Phase 1: Granular Grammar & Feature Tests (The "Spec" Suite)

Currently, many language features are tested in single monolithic files (e.g., `closures.vx`). We will break these down and expand them into hundreds of specific tests.

- **Frontend Fails:** Create tiny `.vx` files explicitly designed to fail the parser or type-checker for specific reasons (e.g., mismatched types, invalid generic arguments, missing fields). Use `FileCheck` to assert the exact diagnostic output.
- **Borrow Checker Rules:** Exhaustive tests for lifetimes, aliasing rules, and linear types.
- **Syntax Edge Cases:** Operator precedence, empty blocks, complex nested macros, and dangling commas.

### Phase 2: Standard Library Conformance Tests

The standard library must be bulletproof. We will create exhaustive test suites corresponding to each file in `stdlib/std/`:

- `math.vx`: Tests for `NaN`, `Infinity`, overflow/underflow, and edge cases in transcendental functions.
- `tensor` / `simd`: Extensive tests for broadcasting rules, memory aliasing, and multidimensional indexing bounds.
- `hash_map.vx` & `vec.vx`: Capacity boundaries, rehashing, reallocation, and iterator exhaustion.

### Phase 3: Autograd & Mathematics Verification Suite

To ensure Vx is robust for AI workloads, its autodiff must be numerically flawless.

- **Combinatorial Primitives:** Generate specific tests verifying `jvp` and `vjp` rules for all primitive operations (Add, Mul, Exp, Log, Sin, etc.).
- **Complex Graphs:** Tests verifying gradient flows through nested loops, closures, and conditional branches.
- **Formal Verification Assertions:** Adding specific tests verifying that the MLIR autodiff lowering matches theoretical ground truth.

### Phase 4: Fuzzing & Property-Based Testing

To find the bugs we cannot think of:

- **Parser Fuzzing:** Generate deeply nested, chaotic, or heavily randomized AST strings to ensure the parser handles them gracefully without panicking.
- **Semantic Fuzzing:** Generate well-formed but semantically nonsensical programs to stress-test memory consumption and type-checker resilience.

## Verification Plan

For every phase, verification will involve running `cargo test` and asserting that the newly added test count reflects the additions, with `0 failed`. We will verify the `gh issue` is created and visible in the repository.
