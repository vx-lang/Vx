# Akar Abstract Syntax Tree (AST) Reference

This document outlines the Abstract Syntax Tree (AST) definitions used by the Akar compiler. This reference serves as a structural map for components like `akar-format`, `akarc` semantic analysis, and third-party tooling integrating with Akar's core.

The AST is located in `src/ast.rs`.

---

## 1. Top-Level Program Structure
The root of any parsed Akar file is the `Program` struct, which contains all definitions in a flat list (declarations are globally scoped in a single file module).

```rust
pub struct Program {
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
}
```

### 1.1 Declarations
- **`ExternDecl`**: C FFI function declarations (`extern "C" { fn foo(); }`).
  - `name`: String
  - `params`: `Vec<(String, Type)>`
  - `return_type`: `Type`
- **`StructDecl`**: User-defined data types (`struct Foo<T> { x: T }`).
  - `name`: String
  - `generics`: `Vec<(String, Option<String>)>` (Name, Optional Trait Bound)
  - `fields`: `Vec<(String, Type)>`
- **`TraitDecl`**: Interface definitions (`trait Foo { fn bar(); }`).
  - `name`: String
  - `methods`: `Vec<(method_name, params, return_type)>`
- **`ImplBlock`**: Trait implementations or inherent methods (`impl Foo for Bar { ... }`).
  - `trait_name`: `Option<String>` (None for inherent impls)
  - `target_type`: `Type`
  - `methods`: `Vec<Function>`
- **`Function`**: Executable blocks (`fn main() { ... }`).
  - `name`: String
  - `generics`: `Vec<(String, Option<String>)>`
  - `params`: `Vec<(String, Type)>`
  - `return_type`: `Type`
  - `body`: `Vec<Statement>`

---

## 2. Hardware Topologies & Memory Contexts
Akar natively encodes heterogeneous hardware contexts into the AST.

### 2.1 `Topology`
Represents the execution environment or domain.
- `Host`: The CPU or primary orchestration host.
- `NPU(Box<Expr>)`: A specific Neural Processing Unit (e.g. `NPU[0]`).
- `AccCore(Box<Expr>)`: An Accelerator Core.
- `Slice(Box<Topology>, Start, End)`: A range of hardware units (e.g., `NPU[0..4]`).

### 2.2 `MemorySpace`
Represents distinct physical memory regions.
- `HostDRAM`: System main memory.
- `NPUHBM`: High Bandwidth Memory attached to an NPU.
- `LocalSRAM`: Fast local accelerator memory.

---

## 3. Type System (`Type`)
The type system incorporates safety primitives, memory contexts, and basic scalar types.

```rust
pub enum Type {
    Tensor(ElementType),                           // e.g. Tensor<f32>
    Matrix,                                      // High-level matrix type
    Ref(Box<Type>, MemorySpace),                 // e.g. Ref<T, Memory::HostDRAM>
    Borrow(Box<Type>, Option<MemorySpace>, bool),// e.g. &mut T or &T
    Pointer(Box<Type>, Option<MemorySpace>, bool),// e.g. *mut T or *const T
    Struct(String),                              // Custom Struct type
    Verified(Box<Type>),                         // Verified<T> bounds
    Pinned(Box<Type>, Topology),                 // Pinned<T, NPU[0]>
    Generic(String),                             // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),       // e.g. Config<f32>
}
```

### `ElementType`
Scalars supported by the compiler: `Bool`, `BF16`, `F16`, `F32`, `F64`, `I4` through `I128`, and `U4` through `U128`.

---

## 4. Statements (`Statement`)
Execution flow is defined by the following statements:

```rust
pub enum Statement {
    LetDecl(String, bool, Option<Type>, Expr),             // let mut x: T = expr;
    Return(Expr),                                          // return expr;
    SpawnOn(Topology, Vec<Statement>),                     // spawn on(NPU[0]) { ... }
    ExprStmt(Expr),                                        // func_call();
    ForLoop(String, Box<Expr>, Box<Expr>, Vec<Statement>), // for i in 0..10 { ... }
    Assign(Expr, Expr),                                    // a = b;
    CompoundAssign(Expr, BinaryOp, Expr),                  // a += b;
}
```

---

## 5. Expressions (`Expr`)
Mathematical, logical, and memory-based computations.

```rust
pub enum Expr {
    Identifier(String),                       // x
    Number(f64),                              // 3.14 (Floats map dynamically)
    Transfer(Box<Expr>, MemorySpace),         // Implicitly derived from .to_device()
    FunctionCall(String, Vec<Expr>),          // foo(a, b)
    Array(Vec<Expr>),                         // [1, 2, 3]
    MemberAccess(Box<Expr>, String),          // obj.field
    IndexAccess(Box<Expr>, Box<Expr>),        // arr[i]
    MethodCall(Box<Expr>, String, Vec<Expr>), // obj.method(a)
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>), // a + b
    UnaryOp(UnaryOp, Box<Expr>),              // !a
    Borrow(Box<Expr>, bool),                  // &mut a / &a
    Dereference(Box<Expr>),                   // *a
    UnsafeBlock(Vec<Statement>, Option<Box<Expr>>), // unsafe { ... }
    StructInit(String, Vec<(String, Expr)>),  // Config { x: 1 }
    MemorySpace(MemorySpace),                 // Memory::HostDRAM literal
    Topology(Topology),                       // Topology::NPU[0] literal
}
```

### `BinaryOp`
`Add`, `Sub`, `Mul`, `Div`, `Eq`, `NotEq`, `Lt`, `Gt`, `Le`, `Ge`, `And`, `Or`.

### `UnaryOp`
`Not`.

---

## 6. Helper Utilities
The AST structures include `substitute` methods allowing dynamic type substitutions during the Monomorphization pass of the Semantic Analyzer.
