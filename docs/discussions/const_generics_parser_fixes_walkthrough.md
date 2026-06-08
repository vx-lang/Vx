# Walkthrough: Const Generics Parser/Sema Fixes & Test Suite Recovery

## Changes Made

- **Parser & Semantic Analysis**:
  - Modified `Type::substitute` to recursively substitute expressions within `Type::Const`, ensuring proper generic substitution during monomorphization.
  - Updated `parse_ty_str` in `sema/env.rs` to try parsing expressions via `parse_expr_str` when type parsing fails, correctly enabling constants as type parameters in generics.
- **Test File Corrections**:
  - `tests/frontend/fail/const_generic_errors.vx`: Updated `CHECK` to match the new parser error where the token after a valid type argument is now parsed successfully.
  - `tests/frontend/pass/control_flow_rigorous.vx`: Fixed invalid `=>` token that should have had a space.
  - `tests/frontend/pass/transfer_cost_dijkstra.vx`: Replaced the undefined `malloc` call with `Tensor::zeros` to pass semantic checks.
- **Optimization Test Runner Fix**:
  - Modified `tests/optimizations/pass/*.vx` and `*.mlr` files to remove dangling `-X` arguments and merged multi-line `RUN` commands. This fixed regressions caused when `vx-format` previously wrapped these commands, causing the runner to fail due to broken command-line arguments.
- **Repository Maintenance**:
  - Re-ran `vx-format` across `.vx` files, excluding optimization tests to avoid breaking `RUN` commands again.
  - Fixed clippy warning in `src/arch.rs` related to unnecessary `map_or`.

## Verification

- Ran the full `cargo test` suite (`compile_test`, `frontend_pass`, `middle_end`, `optimizations`, etc.).
- All tests pass cleanly, confirming `vxc` now successfully handles constant generic values and that the test suite harness functions appropriately.
