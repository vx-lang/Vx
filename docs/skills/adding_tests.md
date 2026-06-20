# Vx Compiler Testing Guidelines & Skills

This document serves as a "skills file" or reference guide for developers (and AI agents) adding tests to the Vx compiler. The testing infrastructure is multi-layered, covering everything from parser unit tests to end-to-end MLIR FileCheck integration tests.

## 1. Test Directory Structure

```text
├── src/                  # Inline Rust unit tests (e.g., `#[test]`)
├── tests/
│   ├── frontend/         # Integration `.vx` tests for lexer/parser/sema
│   │   ├── pass/         # Valid programs
│   │   └── fail/         # Invalid programs (expected compile errors)
│   ├── middle_end/       # Tests focusing on borrow checking, lowering, MLIR
│   ├── backend/          # LLVM IR and execution tests
│   ├── borrow_test.rs    # FastPath borrow checker unit tests
│   ├── compile_test.rs   # Test runner for `.vx` and `.mlr` integration tests
│   └── registry_test.rs  # Module registry dependency tests
```

______________________________________________________________________

## 2. Integration Tests (`.vx` files)

Integration tests are written in the Vx language (`.vx`) and executed by the test runner in `tests/compile_test.rs`.

### 2.1 FileCheck "Pass" Tests

For valid programs, place the file in a `pass/` directory (e.g., `tests/frontend/pass/`).

**Key Requirements:**

1. You MUST include a `// RUN:` line at the top of the file.
1. Use `FileCheck` to verify the generated MLIR or AST.
1. Use `// CHECK:` or `// CHECK-NOT:` annotations.

**Template:**

```vx
// RUN: vxc %s --action emit-mlir 2>&1 | FileCheck %s

fn test_valid_operation() -> i32 {
  let t : Tensor<f32> = 1.0;
  // CHECK-NOT: error
  return 0;
}

// CHECK: module
```

### 2.2 FileCheck "Fail" Tests

For invalid programs that *must* produce a compilation error, place the file in a `fail/` directory (e.g., `tests/frontend/fail/`).

**Key Requirements:**

1. The `// RUN:` line must use `not vxc` to invert the exit code (if `vxc` fails, `not` succeeds).
1. You can use `FileCheck` to ensure the exact error message is emitted.

**Template:**

```vx
// RUN: not vxc %s --action emit-mlir 2>&1 | FileCheck %s

fn test_invalid_access() -> i32 {
  let t : Tensor<f32> = 1.0;
  let t_npu = transfer(t, Memory::NPU_HBM);

  spawn on(Topology::GPU) {
    // CHECK: error: GPU cannot access NPU_HBM
    let invalid = t_npu;
  }
  return 0;
}
```

______________________________________________________________________

## 3. Rust Unit Tests (`src/` and `tests/*.rs`)

For internal compiler logic (e.g., parser precedence, type substitution, borrow checker bitwise logic, or graph routing), write standard Rust `#[test]` functions.

### 3.1 Inline Module Tests

Place these at the bottom of the relevant file inside a `tests` module.
**Example (`src/arch.rs`):**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_cost() {
        let graph = TransferCostGraph::new();
        // Assert expectations
        assert_eq!(graph.get_cost(&Topology::CPU, &Topology::NPU(Box::new(Expr::...))), 50);
    }
}
```

### 3.2 Running the Tests

**CRITICAL RULE:** Always source `config.local` before running tests to ensure the correct Rust toolchain and LLVM environment variables are set.

```bash
# Run all inline library unit tests
source config.local && cargo test --lib

# Run a specific integration test runner
source config.local && cargo test --test compile_test
source config.local && cargo test --test borrow_test

# Run tests with output enabled (useful for debugging)
source config.local && cargo test -- --nocapture
```

______________________________________________________________________

## 4. MLIR FileCheck Tests (`.mlr`)

For tests that operate strictly on the MLIR level (Middle-End/Backend), `.mlr` files can be added and checked. `tests/compile_test.rs` contains explicit test functions like `test_melior_matmul` that parse a `.mlr` file and look for `// CHECK:` lines directly within the MLIR output.

______________________________________________________________________

## 5. Quick Reference Checklist for Adding a Feature

1. **Unit tests**: Add inline `#[test]` functions for the core data structures/algorithms.
1. **Pass tests (`tests/frontend/pass/`)**: Add a `.vx` file testing the "happy path" with a `// RUN:` line and `FileCheck`.
1. **Fail tests (`tests/frontend/fail/`)**: Add `.vx` files for all invalid/error cases to verify diagnostics. Use `not vxc` in the `// RUN:` line.
1. **Formatting**: Run `source config.local && cargo run --bin vx-format -- <file>` on any new `.vx` files.
1. **Validation**: Run `source config.local && cargo test` to verify everything passes.
