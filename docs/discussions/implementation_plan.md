# Strategy to Scale Compiler Testing

Scaling from ~87 to ~300 tests efficiently requires a shift from manually writing individual `.vx` files to automated, systematic, and programmatic test generation. 

Here is my proposed plan to rapidly expand the testing suite and improve confidence in the compiler.

## User Review Required

> [!IMPORTANT]
> Please review the three primary approaches below. Which direction do you prefer we tackle first?
> 
> 1. **Generative Testing Scripts** (Fastest way to get 100-200 tests)
> 2. **Fuzz Testing** (Best for finding edge cases and compiler panics)
> 3. **Coverage-Guided Expansion** (Best for tracking un-tested branches in the codebase)

## Proposed Changes

### Phase 1: Generative Test Scripting
Writing tests manually for every combination of data types, operators, and functions is tedious. Instead, we can write a generator script.
- **Action**: Create `scripts/generate_tests.py`.
- **Details**: The script will dynamically generate `.vx` test files covering combinatorial edge cases. For instance:
    - **Binary Operators**: Cross-product of `(+, -, *, /, &&, ||, <, >)` with `(i32, i64, f32, f64, bool)`.
    - **Tensor Dimensions**: Generate tests validating type-checking rules for matrix multiplication across various static shape combinations (`[N, M] * [M, P] = [N, P]`).
    - **Formal Verification**: Procedurally generate 50 deep compound math expressions that pass, and 50 that fail.
- **Output**: This will easily generate 150+ deterministic `.vx` files covering semantic checking bounds.

### Phase 2: Table-Driven / Inline Testing
Adding 200 separate `.vx` files can clutter the file system and slow down the compiler test runner (`compile_test.rs`).
- **Action**: Update `tests/compile_test.rs`.
- **Details**: Allow a single `.vx` file to contain multiple isolated snippets. We can use a custom macro or annotation format (e.g., `// TEST-CASE: name`) inside a single file, and `compile_test.rs` will parse and run them individually.
- **Output**: This keeps the repository clean while vastly scaling the number of test assertions.

### Phase 3: Fuzz Testing (`cargo fuzz`)
To ensure industrial-grade robustness, we should use property-based testing and fuzzing to catch compiler panics.
- **Action**: Introduce `cargo fuzz` (via `libfuzzer-sys`).
- **Details**: Create a fuzzer target that continuously feeds randomized AST structures and raw strings into the `parser` and `sema` (Type Checker). 
- **Output**: While this doesn't create static `.vx` files, it provides the coverage equivalent of thousands of tests by finding infinite loops or panics.

### Phase 4: Coverage-Guided Manual Tests
- **Action**: Run `cargo tarpaulin` or `cargo llvm-cov` to generate an HTML coverage report of the compiler.
- **Details**: Identify exactly which `match` arms in `parser.rs` and `sema.rs` have 0% coverage.
- **Output**: Manually write the remaining ~50 tests to hit those specific unreachable edge cases (e.g., specific `ffi` boundaries, layout mismatches, specific error propagations).

## Open Questions

> [!QUESTION]
> 1. Are you okay with introducing a python script to autogenerate test files in `tests/frontend/` and `tests/backend/`?
> 2. Do you want me to start with the **Generative Script** to immediately crank out 100+ tests for type coercion, formal verification, and autodiff, or do you want to start by gathering **Code Coverage** data to see what we are currently missing?
