# Monomorphized Generics & Traits

This plan outlines the architecture and implementation steps to introduce zero-cost generic programming and trait-based constraints into the Vx compiler. Because Vx emphasizes deterministic memory and high-performance lowering to MLIR, all generics must be resolved statically at compile time (monomorphization) rather than relying on dynamic dispatch (vtables).

## Background
Currently, Vx's type system only supports concrete types (like `Tensor<f32>`, `Config`). To build reusable systems abstractions (like `Ref<T, MemSpace>` or generic custom math kernels), we need to allow `T` placeholders. Trait bounds ensure that generic type parameters satisfy required behaviors (e.g., `<T: Add>`).

> [!WARNING]
> **User Review Required**: Since Vx lowers directly to MLIR, monomorphization will generate a distinct `func.func` for every unique type signature combination (e.g., `process_Tensor_f32`, `process_Tensor_i32`). This can increase compile time and binary size (code bloat), similar to Rust and C++. Please confirm this trade-off is acceptable over dynamic dispatch.

> [!IMPORTANT]
> **Open Question**: For this initial implementation, should we support implicit type deduction on function calls (e.g., `process(my_tensor)`) or should we mandate explicit turbofish syntax (`process::<Tensor<f32>>(my_tensor)`)? Type deduction is more complex to implement but far more ergonomic.

## Proposed Changes

### Lexer & AST `src/lexer.rs` & `src/ast.rs`
- Add `trait` and `impl` keywords to the lexer.
- Introduce new AST nodes:
  - `TraitDecl`: Defines a trait name and a list of abstract function/method signatures.
  - `ImplBlock`: Associates a specific implementation of a trait with a target `Struct` or `Type`.
- Update `StructDecl` and `Function` AST nodes to include an optional list of generic parameters (and bounds): `generics: Vec<(String, Option<String>)>` (e.g., `[("T", Some("Printable"))]`).
- Add `Type::Generic(String)` to represent abstract types like `T`.

### Parser `src/parser.rs`
- Parse angle brackets `<T>` in struct and function declarations.
- Parse `trait { ... }` blocks containing abstract method definitions.
- Parse `impl Trait for Type { ... }` blocks containing concrete function bodies.
- Update `parse_type` to handle parsing generic identifiers as `Type::Generic(name)`.

### Semantic Analyzer & Monomorphizer `src/sema.rs`
This requires a structural refactor of how type checking is executed:
1. **Definition Collection Phase**: Gather all generic structs, functions, traits, and impl blocks into symbol tables *without* checking their bodies.
2. **Entry Point Execution**: Begin type checking from the `main` function (or a specified root).
3. **Lazy Instantiation (Monomorphization)**:
   - When a `FunctionCall` or `MethodCall` is encountered, resolve the target function.
   - If the target is generic, deduce `T` by mapping the argument types to the parameter types.
   - Clone the generic function's AST body, recursively replace `Type::Generic("T")` with the concrete type, and enforce that the concrete type satisfies any specified Trait bounds by looking up the `ImplBlock` registry.
   - Add the new instantiated function (e.g., `func_name_mangled`) to a "pending instantiation" queue.
   - Type-check the instantiated function body.
4. **Output Generation**: Produce a pruned AST containing *only* fully concrete, monomorphized structs and functions.

### Codegen `src/codegen.rs`
- No major conceptual changes required! Because `sema.rs` will hand `codegen.rs` a strictly monomorphized AST with zero generics, the MLIR generator simply processes concrete `Tensor<f32>` or `Config_f32` types identically to how it does today.
- Method calls tied to traits will be statically lowered to direct MLIR function calls (e.g., `func.call @impl_Printable_for_TensorF32_print(...)`).

## Verification Plan

### Automated Tests
- **Frontend / Unit**: Ensure `parse_trait`, `parse_impl`, and `parse_generic_function` correctly build the AST.
- **Sema**: Write tests that verify a type error is thrown if a generic constraint isn't met (e.g., passing a struct that doesn't `impl Add` to a function requiring `<T: Add>`).
- **Middle-end**: Verify `FileCheck` on an `.vx` script that instantiates a generic function twice (e.g., with `f32` and `i32`) and generates exactly two distinct `func.func` MLIR outputs.
