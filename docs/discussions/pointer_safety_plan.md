# Close Memory Safety Gaps (Revised)

You are absolutely right. Rust doesn't have a `nullptr` keyword—it handles null pointers through `std::ptr::null()` (which under the hood is an integer-to-pointer cast) and uses `Option<&T>` in safe code instead of null references.

To align properly with Rust's philosophy, we should revert the `nullptr` language keyword attempt and instead allow **casting integers to pointers** inside the type checker (`sema`), which will enable us to write `0 as *mut T` in `Vec::free` just like we would in Rust.

## Open Questions

- Does this new approach of implementing integer-to-pointer casts (`0 as *mut T`) and rolling back the `nullptr` keyword sound better to you?

## Proposed Changes

### 1. Revert `nullptr` Keyword Additions

We will remove the `nullptr` changes I just started adding to:

- `src/lexer.rs`
- `src/ast/expr.rs`
- `src/parser/expr.rs`

### 2. Implement Integer-to-Pointer Casting

We will update Semantic Analysis to support integer to pointer casts.

#### [MODIFY] src/sema/expr.rs

- In `check_ascast_expr`, add support for casting `Type::Scalar(I32 | I64 | U64)` to `Type::Pointer`. This allows `0 as *mut T`.

#### [MODIFY] src/codegen/expr.rs (or relevant MLIR lowering)

- If necessary, update the backend to correctly emit a bitcast/inttoptr for the MLIR representation of integer to pointer casts.

### 3. Standard Library Bounds Checking & Null Safety

We will use the new casting mechanism in `Vec` and add array bounds checking.

#### [MODIFY] stdlib/std/vec.vx

- Update `Vec::free` to assign `self.data = 0 as *mut T;`
- Add a safe method `pub fn get(self: &Vec<T>, index: i32) -> T` which does:
  ```vx
  assert(index >= 0 && index < self.len, "Index out of bounds");
  return unsafe { self.data[index] };
  ```
- Add a safe method `pub fn set(self: &mut Vec<T>, index: i32, val: T)` with the same bounds assertion.

## Verification Plan

### Automated Tests

- Run `cargo test` to ensure the frontend compiles perfectly.
- Create a simple test file using `0 as *mut i32` and compile it with `vxc`.
- Run standard library tests to verify `Vec` bounds checking.
