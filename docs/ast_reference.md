# Akar AST Reference Guide

This document formally details the Abstract Syntax Tree (AST) definitions for the Akar programming language. As a language designed for heterogeneous hardware (CPUs, NPUs, Accelerators), Akar's AST includes first-class constructs for topology, memory spaces, and explicit data transfers.

The AST is primarily defined in `src/ast.rs` and forms the foundation for semantic analysis (`sema.rs`) and MLIR code generation (`codegen.rs`).

## High-Level Program Structure

An Akar `Program` is the root node of the AST. It encapsulates all top-level declarations within a single compilation unit.

```rust
pub struct Program {
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
}
```

## Topologies & Memory Spaces

Akar explicitly models hardware locality to prevent unexpected, implicit cross-device memory operations.

### Topology
Topologies define *where* computation executes. They can represent the Host CPU, an NPU cluster, or specific slices of a distributed topology.

```rust
pub enum Topology {
    Host,
    NPU(Box<Expr>),                             // e.g., NPU[0]
    AccCore(Box<Expr>),                         // e.g., AccCore[1]
    Slice(Box<Topology>, Box<Expr>, Box<Expr>), // e.g., NPU[0..4]
}
```

### Memory Space
Memory Spaces define *where* data resides physically.

```rust
pub enum MemorySpace {
    HostDRAM,   // System CPU memory
    NPUHBM,     // High Bandwidth Memory attached to NPU
    LocalSRAM,  // Fast, small scratchpad memory on Accelerator Cores
}
```

## Type System

Akar features a strictly verified, generic-capable type system.

```rust
pub enum Type {
    Tensor(ElementType),
    Matrix,
    Ref(Box<Type>, MemorySpace),                  // Reference bound to a specific memory space
    Borrow(Box<Type>, Option<MemorySpace>, bool), // &T or &mut T
    Pointer(Box<Type>, Option<MemorySpace>, bool),// *const T or *mut T
    Struct(String),                               // User-defined structs
    Verified(Box<Type>),                          // Formally verified types
    Pinned(Box<Type>, Topology),                  // Memory physically pinned to a topology
    Generic(String),                              // Generic type parameter 'T'
    GenericInstance(Box<Type>, Vec<Type>),        // E.g., Config<f32>
}
```

### Element Types
Native scalar primitives supported in `Tensor`s:
* **Floating Point**: `F16`, `BF16`, `F32`, `F64`
* **Signed Integer**: `I4`, `I8`, `I16`, `I32`, `I64`, `I128`
* **Unsigned Integer**: `U4`, `U8`, `U16`, `U32`, `U64`, `U128`
* **Boolean**: `Bool`

## Expressions (`Expr`)

Expressions represent operations that evaluate to a value.

```rust
pub enum Expr {
    Identifier(String),
    Number(f64),
    Transfer(Box<Expr>, MemorySpace),               // Explicit data movement (to_device / to_host)
    FunctionCall(String, Vec<Expr>),
    Array(Vec<Expr>),
    MemberAccess(Box<Expr>, String),                // obj.field
    IndexAccess(Box<Expr>, Box<Expr>),              // arr[idx]
    MethodCall(Box<Expr>, String, Vec<Expr>),       // obj.method(args)
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>),       // lhs + rhs
    UnaryOp(UnaryOp, Box<Expr>),                    // !expr
    Borrow(Box<Expr>, bool),                        // &expr or &mut expr
    Dereference(Box<Expr>),                         // *expr
    UnsafeBlock(Vec<Statement>, Option<Box<Expr>>), // unsafe { ... }
    StructInit(String, Vec<(String, Expr)>),        // Point { x: 1.0, y: 2.0 }
    MemorySpace(MemorySpace),
    Topology(Topology),
}
```

## Statements (`Statement`)

Statements represent actions or control flow operations that do not yield a usable value directly.

```rust
pub enum Statement {
    LetDecl(String, bool, Option<Type>, Expr),             // let [mut] name: Type = expr;
    Return(Expr),                                          // return expr;
    SpawnOn(Topology, Vec<Statement>),                     // spawn on(Topology::NPU[0]) { ... }
    ExprStmt(Expr),                                        // func();
    ForLoop(String, Box<Expr>, Box<Expr>, Vec<Statement>), // for i in 0..10 { ... }
    Assign(Expr, Expr),                                    // lhs = rhs;
    CompoundAssign(Expr, BinaryOp, Expr),                  // lhs += rhs;
}
```

## Declarations

### Functions
```rust
pub struct Function {
    pub name: String,
    pub generics: Vec<(String, Option<String>)>, // e.g., <T: Trait>
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Vec<Statement>,
}
```

### Structs
```rust
pub struct StructDecl {
    pub name: String,
    pub generics: Vec<(String, Option<String>)>,
    pub fields: Vec<(String, Type)>,
}
```

### Traits & Impls
```rust
pub struct TraitDecl {
    pub name: String,
    pub methods: Vec<(String, Vec<(String, Type)>, Type)>,
}

pub struct ImplBlock {
    pub trait_name: Option<String>,
    pub target_type: Type,
    pub methods: Vec<Function>,
}
```

### Extern (FFI)
```rust
pub struct ExternDecl {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}
```

> [!NOTE]
> The AST fully supports generic substitution via the `substitute()` method implemented on `Type`, `Expr`, and `Statement`. This powers the monomorphization engine in `sema.rs` to generate statically verified, hardware-specific machine code paths without runtime overhead.
