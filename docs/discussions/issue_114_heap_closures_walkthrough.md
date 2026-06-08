# Walkthrough: Heap Allocated Closure Environments

## Overview

We have successfully refactored `Vx`'s codegen to heap-allocate closure environments, resolving issue-114. Previously, the environments for closures (the structures holding the variables captured from the outer scope) were allocated on the stack via `llvm.alloca`. This made it unsafe to return closures or pass them across boundaries where the stack frame might be popped.

By heap-allocating these environments using `malloc`, closures can now safely escape their defining scopes, turning them into "full-fledged" closures.

## Changes Made

1. **Module-level `malloc` Declaration:**

   - In `src/codegen/lower.rs` inside `ClosureExpr::lower`, we added logic to ensure that an external `malloc` function `(i64) -> !llvm.ptr` is declared exactly once per module if it does not already exist.

1. **Environment Size Computation via MLIR GEP:**

   - Instead of static sizes or guesswork, we instruct MLIR to dynamically compute the size of the closure environment struct `env_struct_ty`.
   - We generate an `llvm.mlir.null` pointer, then an `llvm.getelementptr` on it with a `rawConstantIndices` offset of `1`, and an `llvm.ptrtoint` to convert the pointer representation into a raw `i64` byte-size.

1. **Replacing `llvm.alloca` with `malloc`:**

   - We completely removed the `llvm.alloca` instruction and replaced it with a `func.call @malloc(%size)` returning a heap-allocated pointer. This memory is now used as the fat-pointer's environment struct context.

## Validation

- Ran the full rust test suite (`cargo test`) to ensure no codegen invariants were broken.
- `test_frontend_pass` and `test_backend` complete perfectly, which includes `tests/frontend/pass/closures.vx`.
- The new `malloc` calls accurately link with the libc `malloc` during the Clang binary linkage phase seamlessly without any changes needed to the C-linker configuration!
