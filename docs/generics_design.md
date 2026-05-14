# Akar Generics: Design Decisions

Akar introduces zero-cost monomorphized generics to provide highly reusable, statically typed abstractions. When extending the compiler to support generics, several critical design decisions were made to align with our core philosophies of predictable execution and deterministic memory control.

## 1. Static Monomorphization over Dynamic Dispatch
**Decision:** We opted for static monomorphization (like C++ templates or Rust generics) rather than dynamic dispatch (like Java's type erasure or Swift's witness tables).
**Rationale:**
- **Zero-Cost Abstractions:** Akar's primary domain is high-performance systems and heterogeneous compute. Dynamic dispatch introduces runtime overhead via vtable lookups and prevents aggressive compiler optimizations (like inlining or MLIR loop fusion).
- **MLIR Compatibility:** Akar lowers directly to MLIR. Emitting distinct, concrete functions (`func.func @process_TensorF32`) maps perfectly to MLIR's strongly-typed static `memref` architecture.
- **Trade-off:** This decision increases compilation times and binary sizes (code bloat) since a new function is generated for every unique type signature combination. We explicitly accepted this trade-off in favor of maximum runtime performance.

## 2. Implicit Type Deduction
**Decision:** We chose to implement implicit type deduction during generic function calls rather than requiring explicit turbofish syntax (e.g., `process::<f32>(val)`).
**Rationale:**
- **Ergonomics:** Systems code can become heavily littered with angle brackets when manipulating complex tensor memory spaces or topologies. By unifying the formal parameter types against the evaluated argument types at the call site, the compiler handles the heavy lifting, keeping the code clean.
- **Implementation:** The Semantic Analyzer (`sema.rs`) uses a 2-pass lazy instantiation engine. It first builds a mapping (`HashMap<String, Type>`) by verifying arguments against generic constraints, then clones the AST node and substitutes the deduced types.

## 3. AST Rewriting at Semantic Analysis
**Decision:** The type checker recursively rewrites the AST (`&mut AST`) in-place during semantic analysis to permanently replace generic `FunctionCall` nodes with concrete, mangled function names.
**Rationale:**
- To keep the backend `codegen.rs` as clean and linear as possible, we decided that the codegen pass should *never* encounter a generic type or a generic function call.
- By having Sema rewrite `process(t1)` into `process_TensorF32(t1)`, the backend operates purely on a fully concretized AST, emitting standard static MLIR calls without knowing that the abstraction ever existed.

## 4. Trait Constraints (Upcoming)
**Decision:** Traits are parsed as standard AST declarations but act purely as *static constraints* evaluated during monomorphization.
**Rationale:** Similar to C++ Concepts or Rust Traits, traits will not produce runtime fat pointers (unless we explicitly design a `dyn Trait` equivalent later). They exist solely to allow the semantic analyzer to reject invalid generic substitutions *before* instantiating a potentially massive function body.
