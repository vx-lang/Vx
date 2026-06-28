use crate::symbol::Symbol;
use crate::symbol::Symbol;
//===- arena.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Flat, arena-allocated High-Level Intermediate Representation (HIR)
//
//===----------------------------------------------------------------------===//

use crate::syntax::*;
use id_arena::{Arena, Id};

pub type ExprId = Id<HirExpr>;
pub type StmtId = Id<HirStmt>;

#[derive(Debug, Clone)]
pub struct HirArena {
    pub exprs: Arena<HirExpr>,
    pub stmts: Arena<HirStmt>,
}

impl HirArena {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            exprs: Arena::new(),
            stmts: Arena::new(),
        }
    }

    pub fn alloc_expr(&mut self, expr: HirExpr) -> ExprId {
        self.exprs.alloc(expr)
    }

    pub fn alloc_stmt(&mut self, stmt: HirStmt) -> StmtId {
        self.stmts.alloc(stmt)
    }
}

// -----------------------------------------------------------------------------
// HIR AST Nodes
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct HirIdentifierExpr {
    pub name: Symbol,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirEnumVariantExpr {
    pub enum_name: Symbol,
    pub variant_name: Symbol,
    pub payload: Option<Vec<ExprId>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirNumberExpr {
    pub value: Symbol,
    pub ty: Option<ElementType>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirStringLiteralExpr {
    pub value: Symbol,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirSpawnOnExpr {
    pub top: Topology,
    pub stmts: Vec<StmtId>,
    pub ret: Option<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirTransferExpr {
    pub expr: ExprId,
    pub space: MemorySpace,
    pub cost: Option<u32>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirFunctionCallExpr {
    pub name: Symbol,
    pub type_args: Option<Vec<Type>>,
    pub args: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirAsCastExpr {
    pub expr: ExprId,
    pub target_ty: Type,
    pub source_ty: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirIndirectCallExpr {
    pub callee: ExprId,
    pub args: Vec<ExprId>,
    pub target_func_ty: Option<syntax::Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirArrayExpr {
    pub elements: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMemberAccessExpr {
    pub base: ExprId,
    pub member: Symbol,
    pub struct_name: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirIndexAccessExpr {
    pub base: ExprId,
    pub index: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMethodCallExpr {
    pub base: ExprId,
    pub method_name: Symbol,
    pub type_args: Option<Vec<Type>>,
    pub args: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirBinaryOpExpr {
    pub lhs: ExprId,
    pub op: BinaryOp,
    pub rhs: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirRelationalOpExpr {
    pub lhs: ExprId,
    pub op: RelationalOp,
    pub rhs: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirLogicalOpExpr {
    pub lhs: ExprId,
    pub op: LogicalOp,
    pub rhs: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirUnaryOpExpr {
    pub op: UnaryOp,
    pub expr: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirBorrowExpr {
    pub expr: ExprId,
    pub is_mut: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirDereferenceExpr {
    pub expr: ExprId,
    pub ty: Option<syntax::Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirUnsafeBlockExpr {
    pub stmts: Vec<StmtId>,
    pub ret: Option<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirComptimeBlockExpr {
    pub stmts: Vec<StmtId>,
    pub ret: Option<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirStructInitExpr {
    pub name: Symbol,
    pub fields: Vec<(Symbol, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMemorySpaceExpr {
    pub space: MemorySpace,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirTopologyExpr {
    pub top: Topology,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirGradExpr {
    pub target_fn: Symbol,
    pub args: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirVjpExpr {
    pub target_fn: Symbol,
    pub args: Vec<ExprId>,
    pub cotangent: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirJvpExpr {
    pub target_fn: Symbol,
    pub args: Vec<ExprId>,
    pub tangent: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirIfExpr {
    pub is_comptime: bool,
    pub cond: ExprId,
    pub then_block: Vec<StmtId>,
    pub else_block: Option<Vec<StmtId>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirRangeExpr {
    pub start: ExprId,
    pub end: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMatchExpr {
    pub expr: ExprId,
    pub arms: Vec<MatchArm>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirVecMacroExpr {
    pub elements: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirClosureExpr {
    pub params: Vec<(Symbol, Type)>,
    pub body: ExprId,
    pub captures: Vec<(Symbol, Type)>,
    pub ret_ty: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMacroCallExpr {
    pub name: Symbol,
    pub token_tree: TokenTree,
    pub block_tree: Option<TokenTree>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirPrintExpr {
    pub args: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirPrintlnExpr {
    pub args: Vec<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirSizeOfExpr {
    pub target_ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirInlineMlirExpr {
    pub inputs: Vec<(Symbol, Expr, String)>, // "%argN", expr,
    pub clobbers: Vec<ExprId>,
    pub returns: Option<Type>,
    pub dialects: Vec<String>,
    pub block_str: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirLetDeclStmt {
    pub name: Symbol,
    pub is_mut: bool,
    pub ty_ann: Option<Type>,
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirReturnStmt {
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirExprStmtStmt {
    pub expr: Expr,
    pub has_semi: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirForLoopStmt {
    pub iter: String,
    pub iterable: ExprId,
    pub invariants: Vec<ExprId>,
    pub body: Vec<StmtId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirAssignStmt {
    pub lhs: Expr,
    pub rhs: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirCompoundAssignStmt {
    pub lhs: Expr,
    pub op: BinaryOp,
    pub rhs: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirAssertStmt {
    pub expr: ExprId,
    pub msg: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirLoopStmt {
    pub invariants: Vec<ExprId>,
    pub body: Vec<StmtId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirBreakStmt {
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirContinueStmt {
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirMacroCallStmt {
    pub name: Symbol,
    pub token_tree: TokenTree,
    pub block_tree: Option<TokenTree>,
    pub has_semi: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Identifier(HirIdentifierExpr),
    EnumVariant(HirEnumVariantExpr),
    Number(HirNumberExpr),
    StringLiteral(HirStringLiteralExpr),
    SpawnOn(HirSpawnOnExpr),
    Transfer(HirTransferExpr),
    FunctionCall(HirFunctionCallExpr),
    AsCast(HirAsCastExpr),
    IndirectCall(HirIndirectCallExpr),
    Array(HirArrayExpr),
    MemberAccess(HirMemberAccessExpr),
    IndexAccess(HirIndexAccessExpr),
    MethodCall(HirMethodCallExpr),
    BinaryOp(HirBinaryOpExpr),
    RelationalOp(HirRelationalOpExpr),
    LogicalOp(HirLogicalOpExpr),
    UnaryOp(HirUnaryOpExpr),
    Borrow(HirBorrowExpr),
    Dereference(HirDereferenceExpr),
    UnsafeBlock(HirUnsafeBlockExpr),
    ComptimeBlock(HirComptimeBlockExpr),
    StructInit(HirStructInitExpr),
    MemorySpace(HirMemorySpaceExpr),
    Topology(HirTopologyExpr),
    Grad(HirGradExpr),
    Vjp(HirVjpExpr),
    Jvp(HirJvpExpr),
    If(HirIfExpr),
    Range(HirRangeExpr),
    Match(HirMatchExpr),
    VecMacro(HirVecMacroExpr),
    Closure(HirClosureExpr),
    MacroCall(HirMacroCallExpr),
    Print(HirPrintExpr),
    Println(HirPrintlnExpr),
    SizeOf(HirSizeOfExpr),
    InlineMlir(HirInlineMlirExpr),
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    LetDecl(HirLetDeclStmt),
    Return(HirReturnStmt),
    Expr(HirExprStmtStmt),
    ForLoop(HirForLoopStmt),
    Assign(HirAssignStmt),
    CompoundAssign(HirCompoundAssignStmt),
    Assert(HirAssertStmt),
    Loop(HirLoopStmt),
    Break(HirBreakStmt),
    Continue(HirContinueStmt),
    MacroCall(HirMacroCallStmt),
    Error(Span),
}
