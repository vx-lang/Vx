# Heap Allocate Closure Environment

The issue requests to heap-allocate the closure environment (the struct containing captured variables) so that `Vx` supports full-fledged closures (closures that can safely escape the scope they are defined in, e.g. when returned from a function).

## Current Implementation

In `src/codegen/lower.rs`, the `ClosureExpr` lowering currently allocates the environment structure on the stack using `llvm.alloca`:

```rust
let alloca_op = block.append_operation(
    melior::ir::operation::OperationBuilder::new(
        "llvm.alloca",
        Location::unknown(gen.context),
    )
    // ...
);
```

When a closure escapes, the stack memory is popped, leading to dangling pointers and memory corruption.

## Proposed Changes

We will modify the MLIR generation for `ClosureExpr` to allocate the environment on the heap using `malloc`.

### 1. Module-level `malloc` Declaration

In `src/codegen/generator.rs`, we will ensure that an external `malloc` function is declared:

```mlir
func.func private @malloc(i64) -> !llvm.ptr
```

This maps to the standard C library `malloc`.

### 2. Computing the Environment Size

To safely determine the size of the dynamically generated environment struct in MLIR, we will generate the standard "getelementptr on null" trick:

```mlir
%null = llvm.mlir.null : !llvm.ptr
%gep = llvm.getelementptr %null[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<...>
%size = llvm.ptrtoint %gep : !llvm.ptr to i64
```

### 3. Replace `llvm.alloca` with `func.call @malloc`

We will replace the `llvm.alloca` operation in `ClosureExpr::lower` with a call to `@malloc`:

```mlir
%env_ptr = func.call @malloc(%size) : (i64) -> !llvm.ptr
```

This will change the environment from a stack allocation to a heap allocation.

## Open Questions

- Is there any specific memory management mechanism (like ARC or a Garbage Collector) that we should hook into for `free`ing this memory later, or is `malloc` sufficient for now to fulfill the "heap allocate" requirement?
- We are currently assuming `malloc` will be provided by the system linker (`libc`). Is this correct for the Vx toolchain?

## Verification Plan

### Automated Tests

- Run the full `cargo test` suite to ensure closures still work for all basic test cases (`tests/frontend/pass/closures.vx`).
- The `test_no_panics_or_asserts_in_tests` lint test will continue to pass since we are modifying codegen, not tests.
