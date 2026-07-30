# Borrow Checker Test Expansion & Documentation

## Date: 2026-06-20

## Summary

Comprehensive review of the Vx borrow checker, test expansion for linear types
and FastPath variance checking, and documentation updates for `docs/lang/`.
Uncovered an NLL implementation limitation during testing.

______________________________________________________________________

## Architecture Overview

The Vx borrow checker operates at two levels:

### Level 1: Sema (Linear Types + Lexical Borrows)

- **Linear consumption**: `is_linear()` types (Tensor, Matrix, Ref, Verified,
  Pinned, Struct, Enum) are consumed on use via `consume()`. Using a consumed
  variable produces `"Use of moved or consumed linear variable"`.
- **Active borrows**: `BorrowRecord { is_mut, scope_depth, borrower_name, path }`
  tracks which variables are borrowed.
- **Lexical lifetime cleanup**: `pop_scope()` removes borrows originating in
  that scope, enabling re-borrowing after scope exit.
- **NLL (partial)**: Dead borrowers (via `is_variable_used_after()`) are cleaned
  up when creating new borrows inside `check_borrow_expr`.
- **Split borrows**: Struct fields can be borrowed independently via path-based
  overlap checking.
- **Enforcement**: Mutable borrow blocks all access, double-mutable and
  mutable+immutable borrows are rejected.

### Level 2: FastPath Borrow Checker (`borrow.rs`)

- 256-bit TypeId with Word 2 encoding 4 × 16-bit lifetime parameters.
- Each 16-bit slot: `[Variance Flags (4 bits) | Region ID (12 bits)]`.
- Variance modes: Invariant (0x0, exact match), Covariant (0x1, `<=`),
  Contravariant (0x2, `>=`).
- `verify_subtyping_bounds()`: bitwise register math for O(1) lifetime checking.
- Slow path fallback for complex signatures (> 4 lifetime parameters).

______________________________________________________________________

## Tests Added

### Unit Tests (src/sema/mod.rs) — +3 tests

| Test | What it validates |
|------|-------------------|
| `test_sema_linear_move_consumed` | Tensor double-use rejected |
| `test_sema_scalar_not_consumed` | Scalars (i32) freely reusable |
| `test_sema_borrow_blocks_access` | Mutable borrow blocks variable access |

### FastPath Tests (tests/borrow_test.rs) — +3 tests

| Test | What it validates |
|------|-------------------|
| `test_fast_path_contravariant` | `region_a >= region_b` for contravariant |
| `test_fast_path_mismatched_variance` | Mismatched variance flags rejected |
| `test_fast_path_multi_slot` | Multi-slot (slots 0-3) checked simultaneously |

### Integration Tests (.vx) — +5 tests

| File | Type | What it validates |
|------|------|-------------------|
| `linear_scalar_reuse.vx` | pass | i32/bool/f32 survive multiple uses |
| `linear_tensor_double_use.vx` | fail | Tensor consumed on first use |
| `linear_transfer_reuse.vx` | pass | transfer() uses copy semantics |
| `borrow_split_fields.vx` | pass | Disjoint struct fields → split borrows |
| `borrow_use_after_mut.vx` | fail | Access during live mutable borrow rejected |

______________________________________________________________________

## Findings

### Discovery: Transfer is Non-Destructive

`transfer(t, Memory::NPU_HBM)` does NOT consume `t`. The sema calls
`check_expr_type_flag(&mut t.expr, false, silent)` with `consume=false`.
This means transfer uses copy semantics at the language level — the source
remains accessible in its original memory space.

### Discovery: NLL Implementation Limitation

NLL dead-borrow cleanup runs inside `check_borrow_expr` (at borrow-creation
time), but `check_identifier_expr` performs a separate "mutably borrowed"
access check that fires BEFORE the NLL cleanup can run. This means NLL
reborrowing only works via **lexical scope exit** (`pop_scope`), not via
usage-based liveness within the same scope.

**Impact**: The following pattern is valid in Rust but fails in Vx:

```rust
let y = &mut x;
// y is never used again
let z = &mut x;  // ERROR in Vx (should be OK with NLL)
```

The workaround is to use an explicit scope:

```rust
{ let y = &mut x; }  // y goes out of scope
let z = &mut x;       // OK: borrow released by scope exit
```

______________________________________________________________________

## Documentation Updates

### `docs/lang/borrow_checker_mapping.md` *(retired)*

- **§5 Split Borrows**: Path-based overlap analysis, disjoint fields example.
- **§6 Non-Lexical Lifetimes**: Partial NLL via liveness analysis, with
  `[!IMPORTANT]` note on the identifier-access limitation.

> [!NOTE]
> That file has since been retired. §5 and §6 duplicated
> [`borrow_checker_architecture.md`](../borrow_checker_architecture.md), which now owns both and
> supersedes §6 — loans release at *last use*, not only at lexical scope exit, so the
> "use explicit scopes" workaround the old §6 recommended is no longer needed. Its one durable
> part, the Rust-FFI opaque-pointer ownership contract, moved to
> [`docs/lang/abi.md` §2.1](../../lang/abi.md).

### `docs/lang/types.md`

- **§9 Linear (Affine) Types vs. Copyable Types**: Complete classification
  table of which types are linear (consumed on use) vs. copyable (scalars).
- `transfer()` copy semantics note.

______________________________________________________________________

## Verification

- `cargo test --lib` — 159 unit tests pass
- `cargo test --test borrow_test` — 4 FastPath tests pass
- `cargo test --test compile_test` — 13 integration tests pass
