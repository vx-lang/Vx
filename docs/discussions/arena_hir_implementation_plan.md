# Epic 163: Arena-based Flat HIR Foundation

This is the first step toward achieving "AST Annihilation" and strict Data-Oriented Design (DOD) for the Vx compiler. Currently, the AST heavily relies on `Box<Expr>` and `Vec<Statement>`, leading to pointer-chasing, cache misses, and deep cloning.

Instead of rewriting the entire compiler in one massive breaking change, we will take an incremental, additive approach. We will lay the foundational data structures for the new HIR (High-Level Intermediate Representation) and build a conversion pass.

## Proposed Changes

### 1. New Dependency

We will add `id-arena` to `Cargo.toml` (or implement a simple custom index-based `Vec` arena if preferred) to manage our flat data structures.

### 2. Define the Flat HIR (`src/hir/arena.rs` or `src/hir/ast.rs`)

We will create the new DOD-compliant structs where recursive pointers (`Box`) are replaced with lightweight integer indices:

```rust
pub type ExprId = id_arena::Id<HirExpr>;
pub type StmtId = id_arena::Id<HirStmt>;

pub struct HirArena {
    pub exprs: id_arena::Arena<HirExpr>,
    pub stmts: id_arena::Arena<HirStmt>,
}
```

### 3. Define the HIR Nodes

We will mirror the existing `syntax::Expr` and `syntax::Statement` enums into `HirExpr` and `HirStmt`, but replace all `Box<Expr>` with `ExprId` and `Vec<Statement>` with `Vec<StmtId>`.

### 4. Create the Conversion Pass (`src/hir/lower_ast.rs`)

We will implement a lowering pass that takes the tree-based `syntax::Expr` / `syntax::Statement` emitted by the parser and flattens them into the `HirArena`.

```rust
impl HirArena {
    pub fn lower_expr(&mut self, expr: &syntax::Expr) -> ExprId { ... }
    pub fn lower_stmt(&mut self, stmt: &syntax::Statement) -> StmtId { ... }
}
```

## User Review Required

> [!IMPORTANT]
> **Incremental Rollout**: This plan is purely *additive*. We will build the new `HirArena` and the conversion pass, but we will **not** immediately rewrite the Semantic Analyzer (`TypeChecker`) or MLIR Codegen to use it. This ensures we don't break the compiler while laying the foundation. Does this incremental approach sound good to you?

> [!NOTE]
> **Dependency Choice**: Do you want me to add the `id-arena` crate to `Cargo.toml`, or would you prefer I write a lightweight custom `Vec`-based arena from scratch to keep dependencies minimal?
