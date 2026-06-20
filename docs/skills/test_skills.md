# Test Strategy & Skills for the Vx Compiler

This document captures the thinking process and strategies for writing effective
tests in the Vx compiler. It distills lessons from the test expansion work done
since `e7a0b8a` (June 2026).

For the mechanical reference (templates, commands, directory structure), see
[adding_tests.md](adding_tests.md).

______________________________________________________________________

## 1. The Feature Coverage Triangle

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

## 2. How to Find Missing Tests

### 2.1 Error Message Audit

Grep the compiler source for error messages, then check if each has a
corresponding `.vx` fail test:

```bash
grep -rn 'self.errors.push' src/sema/ | grep -oP '"[^"]*"' | sort -u
```

Every error string should appear in at least one `// CHECK:` line in a fail test.
If it doesn't, there's a coverage gap.

### 2.2 Feature Enumeration

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

### 2.3 Boundary Analysis

For any feature with ranges or limits, test:

- The **happy path** (normal case)
- The **boundary** (max valid value)
- The **overflow** (one past the boundary)

Example: `NPU[0..4]` slice syntax needs tests for:

- `NPU[0..4]` (valid range) → pass
- `NPU[0..0]` (empty range) → what happens?
- `NPU[5..3]` (inverted range) → should fail

### 2.4 Cross-Feature Interaction

The most subtle bugs hide at feature intersections. Ask:

- What happens when feature A meets feature B?
- Closures + linear types → does the closure consume the variable?
- Topology + formal verification → are `verified` constraints checked inside `spawn on`?
- Generics + borrows → does a generic function correctly propagate lifetime constraints?

______________________________________________________________________

## 3. Naming Conventions

### 3.1 File Names

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

### 3.2 Unifying vs. Splitting

**Unify** tests into one file when they test closely related aspects of the same
feature (e.g., all topology spawn variants in `topology_spawn.vx`). This reduces
file count and makes it easy to see all variants together.

**Split** tests into separate files when:

- They test different features (don't mix borrow + topology)
- A fail test needs its own file (each fail test is a separate compilation)
- The file would exceed ~100 lines

______________________________________________________________________

## 4. Writing Effective `// CHECK:` Lines

### 4.1 Pass Tests

For pass tests, the simplest check is `// CHECK: module` which verifies that
MLIR codegen succeeded. Add more specific checks for critical output:

```vx
// CHECK: func.func @my_kernel
// CHECK: memref.alloc
// CHECK-NOT: error
```

### 4.2 Fail Tests

For fail tests, check the **exact error message** the compiler produces:

```vx
// CHECK: Use of moved or consumed linear variable: a
```

Use `CHECK-DAG` when multiple errors may appear in any order:

```vx
// CHECK-DAG: Use of moved or consumed linear variable: cap
// CHECK-DAG: Closure 'f' expects 1 arguments, got 2
```

### 4.3 Common Pitfalls

- **Don't `CHECK` for debug output**: The `[CODEGEN]` log lines are printed to
  stdout. If your error goes to stderr and you're piping `2>&1`, the CHECK will
  match against codegen logs instead of the error. Make sure your CHECK string
  is unique to the error.
- **Use `not` correctly**: `not vxc %s 2>&1 | FileCheck %s` — the `not` inverts
  `vxc`'s exit code but doesn't affect `FileCheck`. If `vxc` succeeds when it
  shouldn't, `not` makes the pipeline fail.

______________________________________________________________________

## 5. Test Placement Guide

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

## 6. Discovering Implementation vs. Specification Gaps

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

## 7. Test Maintenance Checklist

When modifying existing tests:

- [ ] Run `cargo run --bin vx-format -- <file>` on `.vx` files (never clang-format)
- [ ] Ensure `// RUN:` line is the very first line of the file
- [ ] Ensure `// CHECK:` lines match the current compiler output exactly
- [ ] Run `source config.local && cargo test` to validate everything passes
- [ ] If a test was renamed/moved, check that `compile_test.rs` discovers it
  (it uses directory globbing, so placement matters)
