# Design Document: Inline MLIR Assembly Macros (`mlir!`)

## 1. Executive Summary & Rationale

Traditionally, systems programming languages like C and Rust provide an `asm!` macro to allow programmers to write inline hardware assembly (x86, ARM) for maximum performance and hardware-specific optimizations.

In the era of heterogeneous compute (CPUs, GPUs, NPUs, TPUs), hardware-specific assembly is no longer sufficient. Instead, **MLIR (Multi-Level Intermediate Representation)** acts as the universal "assembly language" for these diverse accelerators. By introducing an `mlir!` macro to Vx, we are elevating the concept of inline assembly. Programmers can write explicit MLIR code directly within their Vx programs. This effectively turns Vx into a first-class MLIR metaprogramming language, allowing developers to write highly optimized, accelerator-specific kernels directly in user-space without modifying the compiler.

## 2. Underlying Principles

To make this feature feel natural and powerful, the design is guided by the following principles:

### A. Dialects as First-Class Namespaces

MLIR is heavily modularized into "dialects" (e.g., `linalg`, `affine`, `scf`, `npucore`). Writing raw strings like `"linalg.fill"` is brittle and error-prone. Instead, Vx will treat dialects as namespaces. This allows the compiler to reason structurally about the dialect being targeted and provides a natural syntax that feels native to the language.

### B. Explicit Type Control for Heterogeneous Lowering

Data types in Vx (like `Tensor<f32>`) have default MLIR lowerings (like `memref<?xf32>`). However, when targeting custom hardware, a programmer might want to lower that exact same `Tensor` into a custom dialect type (e.g., `npu_buffer<?xf32>`). The `mlir!` macro must support explicit type overrides, allowing the programmer to strictly dictate the MLIR type signatures.

### C. Seamless Variable Binding

Just like Rust's `asm!`, the macro must safely bridge Vx variables into the MLIR block. The programmer should be able to pass Vx expressions directly as operands, and the compiler will handle the data flow and SSA value mapping automatically.

## 3. Proposed Syntax

The syntax bridges the gap between Vx's Rust-like aesthetics and MLIR's structural requirements.

### Basic Operations

```rust
impl<T> Tensor<T> {
    fn fill(&mut self, val: T) {
        // Dialects are namespaces. Vx variables are passed as operands.
        mlir! {
            linalg::fill(val, self) {
                // Attributes block
                operandSegmentSizes = array<i32: 1, 1>
            } : (T, Tensor<T>) -> ()
        }
    }
}
```

### Explicit Type Overrides

If a programmer wants to target an NPU-specific dialect, they can explicitly override the type signatures:

```rust
fn custom_npu_kernel(t: Tensor<f32>) {
    mlir! {
        // Here we explicitly tell the macro to lower `t` into an `npu_buffer`
        // instead of the default `memref`.
        npu::compute(t) : (npu_buffer<?xf32, NPU_HBM>) -> ()
    }
}
```

### Regions and Blocks (Advanced)

Many MLIR operations (like `linalg.generic` or `scf.for`) require nested regions and basic blocks. The macro syntax will use standard braces to delineate regions, and block arguments will look like closure parameters:

```rust
fn map<U>(&self, closure: |T| -> U) -> Tensor<U> {
    mlir! {
        linalg::generic(self, closure) {
            indexing_maps = [ ... ],
            iterator_types = ["parallel"]
        }
        // Region 1
        ({
            // Block 1 with arguments
            ^bb0(|arg0: f32|):
                let res = math::exp(arg0) : (f32) -> f32;
                linalg::yield(res) : (f32) -> ();
        }) : (Tensor<T>, |T| -> U) -> Tensor<U>
    }
}
```

*(Note: For complex closures passed into regions, we may be able to auto-lower Vx closures into MLIR regions directly to save boilerplate!)*

## 4. Compiler Architecture Integration

To support this, the compiler pipeline will be updated as follows:

### 1. AST (`src/ast/expr.rs`)

Introduce `InlineMlirExpr` and `InlineMlirRegion` to represent the structured inline assembly.

```rust
pub struct InlineMlirExpr {
    pub dialect: String,
    pub op_name: String,
    pub operands: Vec<Expr>,
    pub attributes: Vec<(String, AttributeExpr)>,
    pub results: Vec<Type>, // Supports explicit MLIR types
    pub regions: Vec<InlineMlirRegion>,
}
```

### 2. Macro Expander (`src/ast/macro_expand.rs`)

The `MacroExpander` will intercept the `mlir!` token tree. Instead of using standard declarative macro rules, it will use a custom recursive descent parser to parse the namespace syntax, operands, attributes, and regions, constructing the `InlineMlirExpr` node.

### 3. Semantic Analyzer (`src/sema/expr.rs`)

The typechecker will seamlessly type-check the Vx expressions passed as operands. If explicit MLIR types are provided in the macro signature, the typechecker will verify that the Vx types can legally lower into the requested MLIR types.

### 4. Code Generation (`src/codegen/lower.rs`)

The `InlineMlirExpr` lowers directly to `melior::ir::operation::OperationBuilder`:

1. Evaluate Vx AST operand expressions to `melior::ir::Value`s.
1. Parse and attach the requested attributes using Melior's attribute APIs.
1. Recursively lower any inner regions/blocks.
1. Emit the operation directly into the current MLIR basic block.

## 5. Next Steps / Open Questions

> [!IMPORTANT]
> **Open Questions for Design Review**
>
> 1. **Attributes**: MLIR has highly complex attributes (affine maps, dictionary attributes). Should we parse a simplified, JSON-like attribute syntax in the macro, or try to match MLIR's textual attribute syntax exactly?
> 1. **Type Safety**: MLIR operations normally rely on their respective Dialect Validators to catch invalid types/operands. In Vx, should the Semantic Analyzer attempt to validate `mlir!` operations, or do we blindly trust the programmer and let the MLIR backend crash if the assembly is invalid (similar to C++ `asm!`)?
>
> Let me know your thoughts on this expanded design! Once we finalize the design, we can write this into `docs/discussions/implementation_plans/mlir_macro_design.md` for permanent documentation.
