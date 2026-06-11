# Memory Safety Improvements Walkthrough

## What Changed?

1. **Removed `malloc(0)` Workarounds**: Instead of using an expensive `malloc(0)` to initialize empty pointers in `Option` and `Vec`, we now use proper techniques. `Vec::new` uses `Vec::with_capacity(2)`, and `Vec::free` now uses the `0 as *mut T` pattern to represent a null pointer.
1. **Integer to Pointer Casts**: We modified the type checker (Semantic Analysis) to natively support casting integers to pointers (`0 as *mut T`).
1. **Safety Assertions**: The integer-to-pointer cast now checks if you are inside an `unsafe {}` block, otherwise it will throw an error: `"Casting an integer to a raw pointer requires an unsafe block"`.
1. **Bounds Checking for `Vec`**: Added safe `get` and `set` methods to `Vec<T>`. These methods will automatically `assert` that the requested `index` is within the valid `[0, len)` bounds of the vector.
1. **No `nullptr` Keyword**: After confirming your feedback, we avoided adding a `nullptr` language keyword in favor of the integer casting approach to perfectly align with Rust's philosophy!

## Verification

- We verified that the Semantic Analysis cleanly accepts `0 as *mut T` while correctly enforcing `unsafe {}`.
- We ran `cargo test` to ensure that standard tests pass without regressions.
- The standard library `Vec` and `Option` implementations compile correctly with `vxc`.
