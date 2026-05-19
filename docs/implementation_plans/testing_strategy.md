# Testing Strategy Architecture

We need to transition from the single `test.vx` root file to a scalable, layered testing architecture suitable for a modern compiler. This plan proposes a strategy that isolates tests for the frontend, middle-end, and backend, using standard compiler testing practices (like FileCheck-style assertions).

## User Review Required

> [!IMPORTANT]
> Please review the proposed directory structure and the choice of testing tools. Specifically, do you approve of writing a lightweight `FileCheck`-style parser in Rust to verify MLIR codegen output, or would you prefer we require installing LLVM's actual `FileCheck` binary on the host system?

## Proposed Testing Architecture

We will organize the `tests/` directory into three specialized domains, creating a clear boundary between compilation phases:

### 1. Frontend Tests (`tests/frontend/`)

**Goal:** Verify Lexing, Parsing, and Semantic Analysis (Type Checking).

- **Structure:**
  - `tests/frontend/pass/`: `.vx` files that must compile without semantic errors.
  - `tests/frontend/fail/`: `.vx` files that must trigger specific compiler errors.
- **Mechanism:** We will build a Rust integration test runner (`tests/frontend_runner.rs`) that iterates over these directories, compiles the files up to the semantic analysis phase, and asserts the success/failure state. `test.vx` will be moved into `tests/frontend/pass/`.

### 2. Middle-End Tests (`tests/middle_end/`)

**Goal:** Verify MLIR Code Generation and Optimization passes.

- **Structure:** `tests/middle_end/`
- **Mechanism (FileCheck-style):** We want to verify that specific Vx syntax produces specific MLIR dialects. Vx test files will include special comments:
  ```rust
  // CHECK: scf.for %i = %v1 to %v7 step %v8
  // CHECK: memref.store %v13, %result[%i, %j]
  fn loop() { ... }
  ```
  We will implement a lightweight test runner in Rust (`tests/middle_end_runner.rs`) that compiles the code with `--emit-mlir`, parses the `// CHECK:` lines, and verifies they appear in order within the MLIR output.

### 3. Backend Tests (`tests/backend/`)

**Goal:** Verify End-to-End Execution and Runtime correctness.

- **Structure:** `tests/backend/`
- **Mechanism:** These tests will compile Vx code to LLVM IR, link it against `vx_rt.c`, execute it using `lli` via `jit.rs`, and capture the standard output.
  - Vx files will print values using `vx_print()`.
  - The test runner (`tests/backend_runner.rs`) will assert that the standard output matches the expected output defined in the test file (e.g., `// EXPECT: 42`).

## Execution Steps

1. Create the `tests/frontend/{pass,fail}`, `tests/middle_end`, and `tests/backend` directories.
1. Move `test.vx` into `tests/frontend/pass/custom_matmul.vx`.
1. Create a unified `tests/compile_test.rs` driver using `std::fs` to iterate over directories and execute the respective tests.
1. Implement the basic `FileCheck` line-matching utility for the middle-end tests.
1. Create initial test cases for each pipeline phase to prove the infrastructure works.

## Verification Plan

We will verify the framework by running `cargo test`. If successful, the runner will automatically locate `custom_matmul.vx`, pass it through the frontend, verify its MLIR generation via comments, and report success without leaving stray `test.vx` or `temp.mlir` files in the root directory.
