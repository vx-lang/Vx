# Standard Library & MLIR Tensor Refactoring

We have successfully implemented four major standard library and compiler backend pillars for the Vx v3.0 release. These additions transform the language from utilizing raw C-FFI hacks into providing an ergonomic, native standard library and topologically-aware MLIR constructs.

## Changes Made

### Phase 1: File I/O (`fs.vx` & `io.vx`)
- Retained the `File` struct wrapping low-level `vx_file_*` extern functions.
- Introduced `print_i32` into `io.vx` to provide standard formatting capabilities, mirroring the standard `stdin_read`, `stdout_write`, and `stderr_write`.

### Phase 2: Native Strings (`string.vx`)
- Defined a formal `struct String` wrapper over `*mut i8` to abstract raw C string operations.
- Implemented core memory and string-manipulation methods: `new()`, `from_c_str()`, `push_c_str()`, `len()`, `as_c_str()`, and `drop()`.
- This lays the groundwork for standard string iteration and manipulation.

### Phase 3: Core Mathematical Functions (`math.vx`)
- Refactored `trait Math` to accept `self`, making it function as an object-oriented interface.
- Implemented `Math` for both `f32` and `f64`.
- The trait implementation safely bridges to the highly optimized `libvx_std_core.dylib` LLVM intrinsics.

### Phase 4: Topologically-Aware Tensors (`melior_codegen.rs`)
- In Vx, the AST supports a Topology identifier inside the type system: `Type::Tensor(ElementType, Vec<Expr>, Option<Topology>)`.
- Previously, this topology was discarded, lowering to raw `memref<?x?xf32>`.
- Modified `melior_codegen.rs` to intercept the tensor's `Topology` and map it dynamically to an MLIR Memory Space integer (e.g. `1` for `NPUHBM`, `2` for `AccCore`, `0` for `HostDRAM`).
- Tensors correctly emit topology-aware MLIR variables like `memref<?x?xf32, 1>`.

## Validation
- All backend generation tests pass! The compiler successfully emits the memory space integers for tensor `memref` constructs.
- Core standard library parses successfully and acts as a solid baseline for `stdlib/std`.

> [!NOTE]
> These changes were grouped under Issue ID `[#43]`.
