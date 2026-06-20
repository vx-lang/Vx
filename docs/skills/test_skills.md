# Vx Compiler Testing: Strategy & Reference

This document serves as a complete guide for developers (and AI agents) adding
tests to the Vx compiler. It covers both the mechanics (templates, commands) and
the strategy (gap-finding, coverage thinking). It distills lessons from the test
expansion work done since `e7a0b8a` (June 2026).

______________________________________________________________________

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

### Test Placement Guide

| What you're testing | Where to put it |
|---------------------|-----------------|
| Parser syntax (lexer/parser only) | `tests/frontend/pass/` or `fail/` |
| Sema type checking, borrow rules | `tests/middle_end/pass/` or `fail/` |
| Topology + memory spaces | `tests/frontend/pass/` (sema-level) |
| MLIR codegen output | `tests/backend/pass/` |
| Internal algorithm (e.g., FastPath) | `tests/borrow_test.rs` or `src/**/mod.rs` |
| Parser error messages | `tests/frontend/fail/parser/` |
| Formal verification | `tests/frontend/pass/formal_verification/` |

**Rule of thumb**: Use `frontend/` when the test primarily validates
parsing + sema. Use `middle_end/` when the test exercises borrow checking,
lowering, or type system features beyond basic parsing. Use `backend/` when
verifying MLIR output or LLVM lowering.

______________________________________________________________________

## 2. The Feature Coverage Triangle

Every compiler feature needs tests at three levels:

```
        ┌─────────────┐
        │  .vx fail   │   ← "Does it reject bad input?"
        │  (diagnostic │
        │   accuracy)  │
        ├─────────────┤
        │  .vx pass   │   ← "Does it accept good input?"
        │  (end-to-end │
        │   pipeline)  │
        ├─────────────┤
        │  Rust #[test]│   ← "Does the algorithm work?"
        │  (unit logic)│
        └─────────────┘
```

**Rule**: A feature is not considered tested until all three levels exist.

### Example: Borrow Checking

| Level | Test | What it proves |
|-------|------|---------------|
| Unit | `test_sema_borrow_blocks_access` | Mutable borrow blocks identifier access |
| Pass | `borrow_lexical.vx` | Scope-based borrow release compiles |
| Fail | `borrow_mut_twice.vx` | Double mutable borrow is rejected with correct error message |

______________________________________________________________________

## 3. Integration Tests (`.vx` files)

Integration tests are written in the Vx language (`.vx`) and executed by the
test runner in `tests/compile_test.rs`.

### 3.1 FileCheck "Pass" Tests

For valid programs, place the file in a `pass/` directory (e.g.,
`tests/frontend/pass/`).

**Key Requirements:**

1. You MUST include a `// RUN:` line at the very first line of the file.
1. Use `FileCheck` to verify the generated MLIR or AST.
1. Use `// CHECK:` or `// CHECK-NOT:` annotations.

**Template:**

```vx
// RUN: vxc %s --action emit-mlir 2>&1 | FileCheck %s
// CHECK: module

fn test_valid_operation() -> i32 {
  let t : Tensor<f32> = 1.0;
  return 0;
}
```

For pass tests, the simplest check is `// CHECK: module` which verifies that
MLIR codegen succeeded. Add more specific checks for critical output:

```vx
// CHECK: func.func @my_kernel
// CHECK: memref.alloc
// CHECK-NOT: error
```

### 3.2 FileCheck "Fail" Tests

For invalid programs that *must* produce a compilation error, place the file in
a `fail/` directory (e.g., `tests/frontend/fail/`).

**Key Requirements:**

1. The `// RUN:` line must use `not vxc` to invert the exit code (if `vxc`
   fails, `not` succeeds).
1. Use `FileCheck` to check the **exact error message** the compiler emits.

**Template:**

```vx
// RUN: not vxc %s 2>&1 | FileCheck %s

fn main() -> i32 {
  let t : Tensor<f32> = 1.0;
  let t_npu = transfer(t, Memory::NPU_HBM);

  spawn on(Topology::GPU) {
    let invalid = t_npu;
  }
  return 0;
}

// CHECK: Cross-topology access error
```

Use `CHECK-DAG` when multiple errors may appear in any order:

```vx
// CHECK-DAG: Use of moved or consumed linear variable: cap
// CHECK-DAG: Closure 'f' expects 1 arguments, got 2
```

### 3.3 Common FileCheck Pitfalls

- **Don't `CHECK` for debug output**: The `[CODEGEN]` log lines are printed to
  stdout. If your error goes to stderr and you're piping `2>&1`, the CHECK will
  match against codegen logs instead of the error. Make sure your CHECK string
  is unique to the error.
- **Use `not` correctly**: `not vxc %s 2>&1 | FileCheck %s` — the `not` inverts
  `vxc`'s exit code but doesn't affect `FileCheck`. If `vxc` succeeds when it
  shouldn't, `not` makes the pipeline fail.

______________________________________________________________________

## 4. Rust Unit Tests (`src/` and `tests/*.rs`)

For internal compiler logic (e.g., parser precedence, type substitution, borrow
checker bitwise logic, or graph routing), write standard Rust `#[test]`
functions.

### 4.1 Inline Module Tests

Place these at the bottom of the relevant file inside a `tests` module.

**Example (`src/arch.rs`):**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_cost() {
        let graph = TransferCostGraph::new();
        assert_eq!(graph.get_cost(&Topology::CPU, &Topology::NPU(...)), 50);
    }
}
```

### 4.2 MLIR FileCheck Tests (`.mlr`)

For tests that operate strictly on the MLIR level (Middle-End/Backend), `.mlr`
files can be added and checked. `tests/compile_test.rs` contains explicit test
functions like `test_melior_matmul` that parse a `.mlr` file and look for
`// CHECK:` lines directly within the MLIR output.

### 4.3 Running the Tests

**CRITICAL RULE:** Always source `config.local` before running tests to ensure
the correct Rust toolchain and LLVM environment variables are set.

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

## 5. How to Find Missing Tests

### 5.1 Error Message Audit

Grep the compiler source for error messages, then check if each has a
corresponding `.vx` fail test:

```bash
grep -rn 'self.errors.push' src/sema/ | grep -oP '"[^"]*"' | sort -u
```

Every error string should appear in at least one `// CHECK:` line in a fail
test. If it doesn't, there's a coverage gap.

### 5.2 Feature Enumeration

List all AST variants, type constructors, or language keywords, then verify each
has pass + fail coverage:

```bash
# All topology variants in the parser
grep -n 'Topology::' src/parser/types.rs

# All memory spaces
grep -n 'MemorySpace::' src/arch.rs
```

For each variant, ask: "Is there a test that uses this? Is there a test that
misuses this?"

### 5.3 Boundary Analysis

For any feature with ranges or limits, test:

- The **happy path** (normal case)
- The **boundary** (max valid value)
- The **overflow** (one past the boundary)

Example: `NPU[0..4]` slice syntax needs tests for:

- `NPU[0..4]` (valid range) → pass
- `NPU[0..0]` (empty range) → what happens?
- `NPU[5..3]` (inverted range) → should fail

### 5.4 Cross-Feature Interaction

The most subtle bugs hide at feature intersections. Ask:

- What happens when feature A meets feature B?
- Closures + linear types → does the closure consume the variable?
- Topology + formal verification → are `verified` constraints checked inside
  `spawn on`?
- Generics + borrows → does a generic function correctly propagate lifetime
  constraints?

______________________________________________________________________

## 6. Naming Conventions

### 6.1 File Names

Use the pattern: `<feature>_<aspect>.vx`

- **Feature**: The primary language feature being tested (e.g., `topology`,
  `borrow`, `linear`, `closure`, `macro`)
- **Aspect**: What about it (e.g., `spawn`, `transfer`, `double_use`,
  `split_fields`)

Good names:

- `linear_tensor_double_use.vx` — linear types, double-use case
- `borrow_split_fields.vx` — borrow checker, split struct borrows
- `topology_spawn.vx` — topology, spawn blocks

Bad names:

- `test1.vx` — meaningless
- `bug_fix_234.vx` — couple to issue tracker, not feature

### 6.2 Unifying vs. Splitting

**Unify** tests into one file when they test closely related aspects of the same
feature (e.g., all topology spawn variants in `topology_spawn.vx`). This reduces
file count and makes it easy to see all variants together.

**Split** tests into separate files when:

- They test different features (don't mix borrow + topology)
- A fail test needs its own file (each fail test is a separate compilation)
- The file would exceed ~100 lines

______________________________________________________________________

## 7. Discovering Implementation vs. Specification Gaps

During test writing, you will often discover that the compiler's behavior
diverges from what the documentation says, or vice versa. Document these
findings immediately:

### Pattern: "The test proves the docs wrong"

Example: We discovered that `transfer()` does **not** consume the source
variable (contrary to what `borrow_checker_mapping.md` implied). The test
`linear_transfer_reuse.vx` proves that `transfer` uses copy semantics.

**Action**: Update the docs AND add the test.

### Pattern: "The algorithm exists but is unreachable"

Example: The FastPath borrow checker supports Invariant (0x0) and Contravariant
(0x2) variance modes, but `lower_to_type_id()` always emits Covariant (0x1).
The algorithm tests pass, but no real compilation flow exercises those paths.

**Action**: Write the algorithm test anyway (proves correctness), file an issue
for the integration gap, add `TODO` or `assert` to prevent silent misuse.

### Pattern: "The code does NLL but it doesn't work end-to-end"

Example: NLL dead-borrow cleanup runs at borrow-creation time, but the
identifier-access check fires earlier. The feature is partially implemented.

**Action**: Don't write a pass test that assumes the feature works. Instead,
document the limitation in the docs with a workaround, and write the test that
proves the *workaround* works (`borrow_lexical.vx`).

______________________________________________________________________

## 8. Quick Reference Checklist

When adding a new feature:

1. **Unit tests**: Add inline `#[test]` functions for the core algorithms.
1. **Pass tests (`tests/frontend/pass/`)**: Add a `.vx` file testing the "happy
   path" with a `// RUN:` line and `FileCheck`.
1. **Fail tests (`tests/frontend/fail/`)**: Add `.vx` files for all
   invalid/error cases to verify diagnostics. Use `not vxc` in the `// RUN:`
   line.
1. **Formatting**: Run `source config.local && cargo run --bin vx-format -- <file>`
   on any new `.vx` files (never clang-format).
1. **Validation**: Run `source config.local && cargo test` to verify everything
   passes.

When modifying existing tests:

- [ ] Ensure `// RUN:` line is the very first line of the file
- [ ] Ensure `// CHECK:` lines match the current compiler output exactly
- [ ] Run `cargo run --bin vx-format -- <file>` on `.vx` files
- [ ] Run `source config.local && cargo test` to validate everything passes
- [ ] If a test was renamed/moved, check that `compile_test.rs` discovers it
  (it uses directory globbing, so placement matters)
