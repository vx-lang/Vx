# Close Memory Safety Gaps

## 1. Implementation Plan

To align properly with Rust's philosophy, we avoided adding a `nullptr` language keyword and instead allowed **casting integers to pointers** inside the type checker (`sema`), which enabled us to write `0 as *mut T` in `Vec::free` just like we would in Rust. We also enforced that casting integers to pointers must occur inside an `unsafe {}` block, or else a compiler error is thrown.

## 2. Tasks Completed

- Reverted `nullptr` keyword additions in frontend.
- Modified `check_ascast_expr` in `src/sema/expr.rs` to allow casting from integers to pointers while enforcing `unsafe {}`.
- Lowered integer-to-pointer casts (`llvm.inttoptr`) in MLIR backend (`src/codegen/lower.rs`).
- Added safe `get(index)` and `set(index, val)` methods to `Vec<T>` with bounds checking asserts.
- Updated `Vec::free` to assign `self.data = 0 as *mut T`.
- Fixed `Option::unwrap` to correctly assert dynamically without failing at comptime.
- Verified changes with `cargo test` and `vxc`.

## 3. Walkthrough

- **Removed `malloc(0)` Workarounds**: Instead of using an expensive `malloc(0)` to initialize empty pointers in `Option` and `Vec`, we now use proper techniques. `Vec::new` uses `Vec::with_capacity(2)`, and `Vec::free` now uses the `0 as *mut T` pattern to represent a null pointer.
- **Integer to Pointer Casts**: We modified Semantic Analysis to natively support casting integers to pointers (`0 as *mut T`).
- **Safety Assertions**: The integer-to-pointer cast checks if you are inside an `unsafe {}` block, throwing: `"Casting an integer to a raw pointer requires an unsafe block"`.
- **Bounds Checking for `Vec`**: Safe `get` and `set` methods automatically `assert` that the requested `index` is within valid bounds.
- **No `nullptr` Keyword**: We avoided adding a `nullptr` language keyword in favor of integer casting, perfectly aligning with Rust's design.
