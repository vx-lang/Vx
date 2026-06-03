# Tracking Issue: Scale Vx Test Suite to 1000+ Tests

## Goal Description

Scale the Vx compiler test suite from ~250 tests to 1000+ tests to guarantee maturity, robustness, and semantic correctness. This will be achieved by moving from coarse-grained tests to highly specific, granular test cases, generating exhaustive behavioral test suites, and leveraging randomized fuzzing.

## Proposed Changes

The scaling effort will be broken down into four distinct phases, executed sequentially:

### Phase 1: Granular Grammar & Feature Tests (The "Spec" Suite)

- **Frontend Fails:** Create tiny `.vx` files explicitly designed to fail the parser or type-checker for specific reasons.
- **Borrow Checker Rules:** Exhaustive tests for lifetimes, aliasing rules, and linear types.
- **Syntax Edge Cases:** Operator precedence, empty blocks, complex nested macros, and dangling commas.

### Phase 2: Standard Library Conformance Tests

- `math.vx`: Tests for `NaN`, `Infinity`, overflow/underflow, and edge cases in transcendental functions.
- `tensor` / `simd`: Extensive tests for broadcasting rules, memory aliasing, and multidimensional indexing bounds.
- `hash_map.vx` & `vec.vx`: Capacity boundaries, rehashing, reallocation, and iterator exhaustion.

### Phase 3: Autograd & Mathematics Verification Suite

- **Combinatorial Primitives:** Generate specific tests verifying `jvp` and `vjp` rules for all primitive operations.
- **Complex Graphs:** Tests verifying gradient flows through nested loops, closures, and conditional branches.

### Phase 4: Fuzzing & Property-Based Testing

- **Parser Fuzzing:** Generate deeply nested, chaotic, or heavily randomized AST strings.
- **Semantic Fuzzing:** Generate well-formed but semantically nonsensical programs to stress-test memory consumption and type-checker resilience.
