# Close Memory Safety Gaps

## Phase 1: Implement Integer to Pointer Casting

- `[x]` Revert `nullptr` keyword additions (Done)
- `[x]` Modify `check_ascast_expr` in `src/sema/expr.rs` to allow casting from integers to pointers
- `[x]` Ensure MLIR backend handles integer-to-pointer casts (`llvm.inttoptr`)

## Phase 2: Standard Library Bounds Checking & Null Safety

- `[x]` Run all existing unit and integration tests (`cargo test`) to ensure regressions haven't been introduced.
- `[x]` Create a compiler test case in `tests/` that checks `0 as *mut T` is disallowed outside `unsafe {}`
- `[x]` Create a test case showing successful bounds-checking in `Vec::get`.

## Phase 3: Verification

- `[x]` Run `cargo test` on compiler core (All tests pass)
- `[x]` Run `vxc` on standard library (Asserts and codegen casts working perfectly)
