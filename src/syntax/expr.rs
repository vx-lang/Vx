//===- expr.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for Vx expressions, including mathematical operators and function calls.
//
//===----------------------------------------------------------------------===//

use super::*;

use crate::symbol::Symbol;
use crate::syntax;
#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    MatMul,
    Div,
}

#[derive(Debug, PartialEq, Clone)]
pub enum RelationalOp {
    Eq,
    NotEq,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, PartialEq, Clone)]
pub struct IdentifierExpr {
    pub name: Symbol,
    pub span: Span,
}
impl IdentifierExpr {
    pub fn new(name: Symbol, span: Span) -> Self {
        Self { name, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumVariantExpr {
    pub enum_name: Symbol,
    pub variant_name: Symbol,
    pub payload: Option<Vec<Expr>>,
    pub span: Span,
}
impl EnumVariantExpr {
    pub fn new(
        enum_name: Symbol,
        variant_name: Symbol,
        payload: Option<Vec<Expr>>,
        span: Span,
    ) -> Self {
        Self {
            enum_name,
            variant_name,
            payload,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct NumberExpr {
    pub value: Symbol,
    pub ty: Option<ElementType>,
    pub span: Span,
}
impl NumberExpr {
    pub fn new(value: String, ty: Option<ElementType>, span: Span) -> Self {
        Self {
            value: value.into(),
            ty,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StringLiteralExpr {
    pub value: Symbol,
    pub span: Span,
}
impl StringLiteralExpr {
    pub fn new(value: String, span: Span) -> Self {
        Self {
            value: value.into(),
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct SpawnOnExpr {
    pub top: Topology,
    pub stmts: Vec<Statement>,
    pub ret: Option<Box<Expr>>,
    pub span: Span,
}
impl SpawnOnExpr {
    pub fn new(top: Topology, stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self {
            top,
            stmts,
            ret,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TransferExpr {
    pub expr: Box<Expr>,
    pub space: MemorySpace,
    /// The bandwidth-derived roofline cost, filled in by sema. Cycles for a `B/cyc` hop,
    /// picoseconds for a `B/s` one — `u64` because picoseconds overflow `u32` at 4.3 ms.
    pub cost: Option<u64>,
    /// The `impl Transfer` lowering sema matched for this site, when one exists and is
    /// emittable: codegen inlines that lowering's body in place of the builtin copy.
    /// Recorded as the (from, to, topology) key -- the lowering itself is looked up again
    /// at emission, where the bodies live.
    ///
    /// The topology is part of the key because the edge alone no longer identifies a
    /// lowering: two machines may implement `Memory::L2 -> Memory::SMEM` differently, and
    /// the one that runs here is the one sema picked, not whichever the lookup finds first.
    pub lowering: Option<(MemorySpace, MemorySpace, String)>,
    pub span: Span,
}

/// `Reachable<A, B>` used as a value: a compile-time boolean, true iff a transfer path exists
/// from `from` to `to` in the cost graph. Evaluated at comptime (e.g. inside
/// `comptime { if Reachable<A, B> { … } }`); topology variables are substituted during
/// monomorphization first.
///
/// Its arguments are TOPOLOGIES, and it names no particular lowering -- which is why it is no
/// longer spelled `Transfer<A, B>`. That name now belongs to the edge lowering
/// (`impl Transfer<Memory::A, Memory::B> for Topology::X`), whose arguments are memory spaces.
#[derive(Debug, PartialEq, Clone)]
pub struct TransferPredicateExpr {
    pub from: Topology,
    pub to: Topology,
    pub span: Span,
}
impl TransferExpr {
    pub fn new(expr: Box<Expr>, space: MemorySpace, span: Span) -> Self {
        Self {
            expr,
            space,
            cost: None,
            lowering: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionCallExpr {
    pub name: Symbol,
    pub type_args: Option<Vec<Type>>,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl FunctionCallExpr {
    pub fn new(name: Symbol, type_args: Option<Vec<Type>>, args: Vec<Expr>, span: Span) -> Self {
        Self {
            name,
            type_args,
            args,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AsCastExpr {
    pub expr: Box<Expr>,
    pub target_ty: Type,
    pub source_ty: Option<Type>, // Added by sema
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct IndirectCallExpr {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
    pub target_func_ty: Option<syntax::Type>,
    pub span: Span,
}
impl IndirectCallExpr {
    pub fn new(callee: Box<Expr>, args: Vec<Expr>, span: Span) -> Self {
        Self {
            callee,
            args,
            target_func_ty: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArrayExpr {
    pub elements: Vec<Expr>,
    pub span: Span,
}
impl ArrayExpr {
    pub fn new(elements: Vec<Expr>, span: Span) -> Self {
        Self { elements, span }
    }

    /// If this is a nested initializer list (its first element is itself an array), return the
    /// inferred shape: this level's length followed by the sub-shape. A *flat* array returns
    /// `None` -- the `Tensor` constructor reads a flat array as an explicit dims list (a shape),
    /// so `Tensor<f32>([2, 4])` stays a 2x4 shape while `Tensor<f32>([[..],[..]])` is data.
    pub fn initializer_shape(&self) -> Option<Vec<usize>> {
        match self.elements.first() {
            Some(Expr::Array(inner)) => {
                let mut shape = vec![self.elements.len()];
                match inner.initializer_shape() {
                    Some(sub) => shape.extend(sub),
                    None => shape.push(inner.elements.len()),
                }
                Some(shape)
            }
            _ => None,
        }
    }

    /// Row-major flatten of a nested initializer list into its scalar element expressions,
    /// e.g. `[[a, b], [c, d]]` -> `[a, b, c, d]`.
    pub fn initializer_values(&self) -> Vec<&Expr> {
        let mut out = Vec::new();
        for el in &self.elements {
            match el {
                Expr::Array(inner) => out.extend(inner.initializer_values()),
                other => out.push(other),
            }
        }
        out
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MemberAccessExpr {
    pub base: Box<Expr>,
    pub member: Symbol,
    pub struct_name: Option<Symbol>,
    pub span: Span,
}
impl MemberAccessExpr {
    pub fn new(base: Box<Expr>, member: Symbol, span: Span) -> Self {
        Self {
            base,
            member,
            struct_name: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct IndexAccessExpr {
    pub base: Box<Expr>,
    pub index: Box<Expr>,
    pub span: Span,
}
impl IndexAccessExpr {
    pub fn new(base: Box<Expr>, index: Box<Expr>, span: Span) -> Self {
        Self { base, index, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MethodCallExpr {
    pub base: Box<Expr>,
    pub method_name: Symbol,
    pub type_args: Option<Vec<Type>>,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl MethodCallExpr {
    pub fn new(
        base: Box<Expr>,
        method_name: Symbol,
        type_args: Option<Vec<Type>>,
        args: Vec<Expr>,
        span: Span,
    ) -> Self {
        Self {
            base,
            method_name,
            type_args,
            args,
            span,
        }
    }
}

pub trait BinaryOperator {
    fn lhs(&self) -> &Expr;
    fn rhs(&self) -> &Expr;
    fn span(&self) -> &Span;
}

#[derive(Debug, PartialEq, Clone)]
pub struct BinaryOpExpr {
    pub lhs: Box<Expr>,
    pub op: BinaryOp,
    pub rhs: Box<Expr>,
    pub span: Span,
}
impl BinaryOpExpr {
    pub fn new(lhs: Box<Expr>, op: BinaryOp, rhs: Box<Expr>, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

impl BinaryOperator for BinaryOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct RelationalOpExpr {
    pub lhs: Box<Expr>,
    pub op: RelationalOp,
    pub rhs: Box<Expr>,
    pub span: Span,
}
impl RelationalOpExpr {
    pub fn new(lhs: Box<Expr>, op: RelationalOp, rhs: Box<Expr>, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

impl BinaryOperator for RelationalOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LogicalOpExpr {
    pub lhs: Box<Expr>,
    pub op: LogicalOp,
    pub rhs: Box<Expr>,
    pub span: Span,
}
impl LogicalOpExpr {
    pub fn new(lhs: Box<Expr>, op: LogicalOp, rhs: Box<Expr>, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

impl BinaryOperator for LogicalOpExpr {
    fn lhs(&self) -> &Expr {
        &self.lhs
    }
    fn rhs(&self) -> &Expr {
        &self.rhs
    }
    fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnaryOpExpr {
    pub op: UnaryOp,
    pub expr: Box<Expr>,
    pub span: Span,
}
impl UnaryOpExpr {
    pub fn new(op: UnaryOp, expr: Box<Expr>, span: Span) -> Self {
        Self { op, expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BorrowExpr {
    pub expr: Box<Expr>,
    pub is_mut: bool,
    pub span: Span,
}
impl BorrowExpr {
    pub fn new(expr: Box<Expr>, is_mut: bool, span: Span) -> Self {
        Self { expr, is_mut, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct DereferenceExpr {
    pub expr: Box<Expr>,
    pub ty: Option<syntax::Type>,
    pub span: Span,
}
impl DereferenceExpr {
    pub fn new(expr: Box<Expr>, span: Span) -> Self {
        Self {
            expr,
            ty: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnsafeBlockExpr {
    pub stmts: Vec<Statement>,
    pub ret: Option<Box<Expr>>,
    pub span: Span,
}
impl UnsafeBlockExpr {
    pub fn new(stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self { stmts, ret, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ComptimeBlockExpr {
    pub stmts: Vec<Statement>,
    pub ret: Option<Box<Expr>>,
    pub span: Span,
}
impl ComptimeBlockExpr {
    pub fn new(stmts: Vec<Statement>, ret: Option<Box<Expr>>, span: Span) -> Self {
        Self { stmts, ret, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructInitExpr {
    pub name: Symbol,
    pub fields: Vec<(Symbol, Expr)>,
    /// The resolved GID of the struct being initialized, attached by the type checker
    /// (`check_structinit_expr`). `None` until type-checked, or for a monomorphized/generated struct
    /// whose identity isn't a plain module symbol. Lets consumers (e.g. the flat-HIR lowerer) map the
    /// construction to its registry layout without re-resolving the name (#199).
    pub type_id: Option<crate::gid::TypeId>,
    pub span: Span,
}
impl StructInitExpr {
    pub fn new(name: Symbol, fields: Vec<(Symbol, Expr)>, span: Span) -> Self {
        Self {
            name,
            fields,
            type_id: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MemorySpaceExpr {
    pub space: MemorySpace,
    pub span: Span,
}
impl MemorySpaceExpr {
    pub fn new(space: MemorySpace, span: Span) -> Self {
        Self { space, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TopologyExpr {
    pub top: Topology,
    pub span: Span,
}
impl TopologyExpr {
    pub fn new(top: Topology, span: Span) -> Self {
        Self { top, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct GradExpr {
    pub target_fn: Symbol,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl GradExpr {
    pub fn new(target_fn: Symbol, args: Vec<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct VjpExpr {
    pub target_fn: Symbol,
    pub args: Vec<Expr>,
    pub cotangent: Box<Expr>,
    pub span: Span,
}
impl VjpExpr {
    pub fn new(target_fn: Symbol, args: Vec<Expr>, cotangent: Box<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            cotangent,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct JvpExpr {
    pub target_fn: Symbol,
    pub args: Vec<Expr>,
    pub tangent: Box<Expr>,
    pub span: Span,
}
impl JvpExpr {
    pub fn new(target_fn: Symbol, args: Vec<Expr>, tangent: Box<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            tangent,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct IfExpr {
    pub is_comptime: bool,
    pub cond: Box<Expr>,
    pub then_block: Vec<Statement>,
    pub else_block: Option<Vec<Statement>>,
    pub span: Span,
}
impl IfExpr {
    pub fn new(
        is_comptime: bool,
        cond: Box<Expr>,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
        span: Span,
    ) -> Self {
        Self {
            is_comptime,
            cond,
            then_block,
            else_block,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct RangeExpr {
    pub start: Box<Expr>,
    pub end: Box<Expr>,
    pub span: Span,
}
impl RangeExpr {
    pub fn new(start: Box<Expr>, end: Box<Expr>, span: Span) -> Self {
        Self { start, end, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Expr), // e.g. Number, StringLiteral
    Identifier(Symbol),
    EnumVariant(Symbol, Symbol, Option<Vec<Pattern>>), // enum_name, variant_name, payload patterns
}

#[derive(Debug, PartialEq, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MatchExpr {
    pub expr: Box<Expr>,
    pub arms: Vec<MatchArm>,
    pub span: Span,
}
impl MatchExpr {
    pub fn new(expr: Box<Expr>, arms: Vec<MatchArm>, span: Span) -> Self {
        Self { expr, arms, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct VecMacroExpr {
    pub elements: Vec<Expr>,
    pub span: Span,
}
impl VecMacroExpr {
    pub fn new(elements: Vec<Expr>, span: Span) -> Self {
        Self { elements, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ClosureExpr {
    pub params: Vec<(Symbol, Type)>,
    pub body: Box<Expr>,
    pub captures: Vec<(Symbol, Type)>,
    pub ret_ty: Option<Type>,
    pub span: Span,
}
impl ClosureExpr {
    pub fn new(params: Vec<(Symbol, Type)>, body: Box<Expr>, span: Span) -> Self {
        Self {
            params,
            body,
            captures: Vec::new(),
            ret_ty: None,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroCallExpr {
    pub name: Symbol,
    pub token_tree: TokenTree,
    pub block_tree: Option<TokenTree>,
    pub span: Span,
}
impl MacroCallExpr {
    pub fn new(
        name: String,
        token_tree: TokenTree,
        block_tree: Option<TokenTree>,
        span: Span,
    ) -> Self {
        Self {
            name: name.into(),
            token_tree,
            block_tree,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PrintExpr {
    pub args: Vec<Expr>,
    pub span: Span,
}
impl PrintExpr {
    pub fn new(args: Vec<Expr>, span: Span) -> Self {
        Self { args, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PrintlnExpr {
    pub args: Vec<Expr>,
    pub span: Span,
}
impl PrintlnExpr {
    pub fn new(args: Vec<Expr>, span: Span) -> Self {
        Self { args, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct SizeOfExpr {
    pub target_ty: Type,
    pub span: Span,
}
impl SizeOfExpr {
    pub fn new(target_ty: Type, span: Span) -> Self {
        Self { target_ty, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct InlineMlirExpr {
    pub inputs: Vec<(Symbol, Expr, String)>, // "%argN", expr, mlir_type_str
    pub clobbers: Vec<Expr>,
    pub returns: Option<Type>, // None for void
    pub dialects: Vec<String>,
    pub block_str: String,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Identifier(IdentifierExpr),
    EnumVariant(EnumVariantExpr),
    Number(NumberExpr),
    StringLiteral(StringLiteralExpr),
    Transfer(TransferExpr),
    TransferPredicate(TransferPredicateExpr),
    FunctionCall(FunctionCallExpr),
    IndirectCall(IndirectCallExpr),
    Array(ArrayExpr),
    MemberAccess(MemberAccessExpr),
    IndexAccess(IndexAccessExpr),
    MethodCall(MethodCallExpr),
    BinaryOp(BinaryOpExpr),
    RelationalOp(RelationalOpExpr),
    LogicalOp(LogicalOpExpr),
    UnaryOp(UnaryOpExpr),
    Borrow(BorrowExpr),
    Dereference(DereferenceExpr),
    UnsafeBlock(UnsafeBlockExpr),
    ComptimeBlock(ComptimeBlockExpr),
    StructInit(StructInitExpr),
    MemorySpace(MemorySpaceExpr),
    Topology(TopologyExpr),
    If(IfExpr),
    Range(RangeExpr),
    Match(MatchExpr),
    Grad(GradExpr),
    Vjp(VjpExpr),
    Jvp(JvpExpr),
    SpawnOn(SpawnOnExpr),
    VecMacro(VecMacroExpr),
    Closure(ClosureExpr),
    MacroCall(MacroCallExpr),
    AsCast(AsCastExpr),
    Print(PrintExpr),
    Println(PrintlnExpr),
    SizeOf(SizeOfExpr),
    InlineMlir(InlineMlirExpr),
}

macro_rules! delegate_expr {
    ($self:ident, $method:ident) => {
        match $self {
            Expr::Identifier(e) => e.$method.clone(),
            Expr::EnumVariant(e) => e.$method.clone(),
            Expr::Number(e) => e.$method.clone(),
            Expr::StringLiteral(e) => e.$method.clone(),
            Expr::Transfer(e) => e.$method.clone(),
            Expr::TransferPredicate(e) => e.$method.clone(),
            Expr::FunctionCall(e) => e.$method.clone(),
            Expr::IndirectCall(e) => e.$method.clone(),
            Expr::Array(e) => e.$method.clone(),
            Expr::MemberAccess(e) => e.$method.clone(),
            Expr::IndexAccess(e) => e.$method.clone(),
            Expr::MethodCall(e) => e.$method.clone(),
            Expr::BinaryOp(e) => e.$method.clone(),
            Expr::RelationalOp(e) => e.$method.clone(),
            Expr::LogicalOp(e) => e.$method.clone(),
            Expr::UnaryOp(e) => e.$method.clone(),
            Expr::Borrow(e) => e.$method.clone(),
            Expr::Dereference(e) => e.$method.clone(),
            Expr::UnsafeBlock(e) => e.$method.clone(),
            Expr::ComptimeBlock(e) => e.$method.clone(),
            Expr::StructInit(e) => e.$method.clone(),
            Expr::MemorySpace(e) => e.$method.clone(),
            Expr::Topology(e) => e.$method.clone(),
            Expr::If(e) => e.$method.clone(),
            Expr::Range(e) => e.$method.clone(),
            Expr::Match(e) => e.$method.clone(),
            Expr::Grad(e) => e.$method.clone(),
            Expr::Vjp(e) => e.$method.clone(),
            Expr::Jvp(e) => e.$method.clone(),
            Expr::SpawnOn(e) => e.$method.clone(),
            Expr::VecMacro(e) => e.$method.clone(),
            Expr::Closure(e) => e.$method.clone(),
            Expr::MacroCall(e) => e.$method.clone(),
            Expr::AsCast(e) => e.$method.clone(),
            Expr::Print(e) => e.$method.clone(),
            Expr::Println(e) => e.$method.clone(),
            Expr::SizeOf(e) => e.$method.clone(),
            Expr::InlineMlir(e) => e.$method.clone(),
        }
    };
}

/// Replace whole identifiers in a fragment of MLIR text with their bound types. Only complete
/// words are replaced: the `N` in `memref<N x M x f32>` becomes the bound extent, the one
/// inside `NxMxf32` does not.
fn substitute_words(text: &str, mapping: &std::collections::HashMap<Symbol, Type>) -> String {
    if mapping.is_empty() || !mapping.keys().any(|k| text.contains(&**k)) {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut word = String::new();
    let flush = |word: &mut String, result: &mut String| {
        if word.is_empty() {
            return;
        }
        match mapping.get(word.as_str()) {
            Some(ty) => result.push_str(&ty.to_string()),
            None => result.push_str(word),
        }
        word.clear();
    };
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut result);
            result.push(c);
        }
    }
    flush(&mut word, &mut result);
    result
}

/// The local an `@` operand reads, when the operand is a bare name or a borrow of one.
///
/// `None` for anything else, which is the answer a caller wants: `dst = a @ b` fills `dst` in
/// place, and that is only correct while `dst` is disjoint from both operands. A name, and the
/// declared type behind it, is as far as the caller can settle that without alias analysis.
pub fn matmul_operand_root(e: &Expr) -> Option<&Symbol> {
    let e = match e {
        Expr::Borrow(b) => b.expr.as_ref(),
        other => other,
    };
    match e {
        Expr::Identifier(id) => Some(&id.name),
        _ => None,
    }
}

impl Expr {
    pub fn span(&self) -> Span {
        delegate_expr!(self, span)
    }

    pub fn is_binary_operator(&self) -> bool {
        matches!(
            self,
            Expr::BinaryOp(_) | Expr::RelationalOp(_) | Expr::LogicalOp(_)
        )
    }

    pub fn get_binary_operands(&self) -> Option<(&Expr, &Expr)> {
        match self {
            Expr::BinaryOp(e) => Some((&e.lhs, &e.rhs)),
            Expr::RelationalOp(e) => Some((&e.lhs, &e.rhs)),
            Expr::LogicalOp(e) => Some((&e.lhs, &e.rhs)),
            _ => None,
        }
    }

    pub fn substitute(&self, mapping: &std::collections::HashMap<Symbol, Type>) -> Expr {
        if mapping.is_empty() {
            return self.clone();
        }
        match self {
            Expr::Transfer(e) => Expr::Transfer(TransferExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                space: e.space.clone(),
                cost: e.cost,
                lowering: e.lowering.clone(),
                span: e.span,
            }),
            // Type substitution does not touch topologies; topology substitution is done
            // separately during monomorphization.
            Expr::TransferPredicate(e) => Expr::TransferPredicate(e.clone()),
            Expr::ComptimeBlock(e) => Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span,
            }),
            Expr::FunctionCall(e) => {
                let mut new_name = e.name.clone();
                if new_name.starts_with("Tensor_") {
                    let t_name = new_name.strip_prefix("Tensor_").unwrap();
                    if let Some(concrete_el) = mapping.get(t_name) {
                        new_name = format!("Tensor_{}", concrete_el).into();
                    }
                } else if let Some(idx) = new_name.find('<') {
                    if let Some(end_idx) = new_name.find('>') {
                        let base = &new_name[..idx];
                        let ty_arg = &new_name[idx + 1..end_idx];
                        let remainder = &new_name[end_idx + 1..];
                        let mut substituted_args = Vec::new();
                        for t in ty_arg.split(',') {
                            let t = t.trim();
                            if let Some(mapped_ty) = mapping.get(t) {
                                substituted_args.push(mapped_ty.to_string());
                            } else {
                                substituted_args.push(t.to_string());
                            }
                        }
                        new_name =
                            format!("{}<{}>{}", base, substituted_args.join(", "), remainder)
                                .into();
                    }
                }
                let substituted_type_args = e
                    .type_args
                    .as_ref()
                    .map(|tys| tys.iter().map(|ty| ty.substitute(mapping)).collect());
                Expr::FunctionCall(FunctionCallExpr {
                    name: new_name,
                    type_args: substituted_type_args,
                    args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                    span: e.span,
                })
            }
            Expr::IndirectCall(e) => Expr::IndirectCall(IndirectCallExpr {
                callee: Box::new(e.callee.substitute(mapping)),
                args: e.args.iter().map(|arg| arg.substitute(mapping)).collect(),
                target_func_ty: e.target_func_ty.clone(),
                span: e.span,
            }),
            Expr::Array(e) => Expr::Array(ArrayExpr {
                elements: e.elements.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span,
            }),
            Expr::MemberAccess(e) => Expr::MemberAccess(MemberAccessExpr {
                base: Box::new(e.base.substitute(mapping)),
                member: e.member.clone(),
                struct_name: e.struct_name.clone(),
                span: e.span,
            }),
            Expr::IndexAccess(e) => Expr::IndexAccess(IndexAccessExpr {
                base: Box::new(e.base.substitute(mapping)),
                index: Box::new(e.index.substitute(mapping)),
                span: e.span,
            }),
            Expr::MethodCall(e) => {
                let substituted_type_args = e
                    .type_args
                    .as_ref()
                    .map(|tys| tys.iter().map(|ty| ty.substitute(mapping)).collect());
                Expr::MethodCall(MethodCallExpr {
                    base: Box::new(e.base.substitute(mapping)),
                    method_name: e.method_name.clone(),
                    type_args: substituted_type_args,
                    args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                    span: e.span,
                })
            }
            Expr::BinaryOp(e) => Expr::BinaryOp(BinaryOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span,
            }),
            Expr::RelationalOp(e) => Expr::RelationalOp(RelationalOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span,
            }),
            Expr::LogicalOp(e) => Expr::LogicalOp(LogicalOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span,
            }),
            Expr::UnaryOp(e) => Expr::UnaryOp(UnaryOpExpr {
                op: e.op.clone(),
                expr: Box::new(e.expr.substitute(mapping)),
                span: e.span,
            }),
            Expr::Borrow(e) => Expr::Borrow(BorrowExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                is_mut: e.is_mut,
                span: e.span,
            }),
            Expr::Dereference(e) => Expr::Dereference(DereferenceExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                ty: e.ty.as_ref().map(|t| t.substitute(mapping)),
                span: e.span,
            }),
            Expr::UnsafeBlock(e) => Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span,
            }),
            Expr::StructInit(e) => {
                let mut new_name = e.name.clone();
                if let Some(idx) = new_name.find('<') {
                    let base = &new_name[..idx];
                    let ty_args_str = &new_name[idx + 1..new_name.len() - 1];
                    let mut substituted_args = Vec::new();
                    for ty_arg in ty_args_str.split(',') {
                        let ty_arg = ty_arg.trim();
                        if let Some(mapped_ty) = mapping.get(ty_arg) {
                            substituted_args.push(mapped_ty.to_string());
                        } else {
                            substituted_args.push(ty_arg.to_string());
                        }
                    }
                    new_name = format!("{}<{}>", base, substituted_args.join(", ")).into();
                }
                Expr::StructInit(StructInitExpr {
                    name: new_name,
                    fields: e
                        .fields
                        .iter()
                        .map(|(n, ex)| (n.clone(), ex.substitute(mapping)))
                        .collect(),
                    type_id: e.type_id,
                    span: e.span,
                })
            }
            Expr::AsCast(e) => Expr::AsCast(AsCastExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                target_ty: e.target_ty.substitute(mapping),
                source_ty: e.source_ty.as_ref().map(|t| t.substitute(mapping)),
                span: e.span,
            }),
            Expr::If(e) => Expr::If(IfExpr {
                is_comptime: e.is_comptime,
                cond: Box::new(e.cond.substitute(mapping)),
                then_block: e.then_block.iter().map(|s| s.substitute(mapping)).collect(),
                else_block: e
                    .else_block
                    .as_ref()
                    .map(|b| b.iter().map(|s| s.substitute(mapping)).collect()),
                span: e.span,
            }),
            Expr::Range(e) => Expr::Range(RangeExpr {
                start: Box::new(e.start.substitute(mapping)),
                end: Box::new(e.end.substitute(mapping)),
                span: e.span,
            }),
            Expr::Match(e) => Expr::Match(MatchExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                arms: e
                    .arms
                    .iter()
                    .map(|arm| MatchArm {
                        pattern: arm.pattern.clone(), // Patterns don't currently have types to substitute
                        body: arm.body.iter().map(|s| s.substitute(mapping)).collect(),
                    })
                    .collect(),
                span: e.span,
            }),
            Expr::Grad(e) => Expr::Grad(GradExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span,
            }),
            Expr::Vjp(e) => Expr::Vjp(VjpExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                cotangent: Box::new(e.cotangent.substitute(mapping)),
                span: e.span,
            }),
            Expr::Jvp(e) => Expr::Jvp(JvpExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                tangent: Box::new(e.tangent.substitute(mapping)),
                span: e.span,
            }),
            Expr::SpawnOn(e) => Expr::SpawnOn(SpawnOnExpr {
                // The device index is substituted too: `spawn on(NPU[R])` in a generic body must
                // become `NPU[5]` at `f<5>`, or dispatch silently falls back to device 0 (#284).
                top: e.top.substitute(mapping),
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span,
            }),
            Expr::Identifier(id) => {
                if let Some(mapped_ty) = mapping.get(&*id.name) {
                    if let Type::Generic(val_str, _) = mapped_ty {
                        if val_str.parse::<f64>().is_ok() {
                            return Expr::Number(syntax::NumberExpr {
                                value: val_str.to_string().into(),
                                ty: None,
                                span: id.span,
                            });
                        } else {
                            return Expr::Identifier(syntax::IdentifierExpr {
                                name: val_str.clone(),
                                span: id.span,
                            });
                        }
                    } else if let Type::Const(expr) = mapped_ty {
                        return *expr.clone();
                    }
                }
                self.clone()
            }
            Expr::EnumVariant(e) => {
                let mut new_name = e.enum_name.clone();
                if let Some(idx) = new_name.find('<') {
                    let base = &new_name[..idx];
                    let ty_args_str = &new_name[idx + 1..new_name.len() - 1];
                    let mut substituted_args = Vec::new();
                    for ty_arg in ty_args_str.split(',') {
                        let ty_arg = ty_arg.trim();
                        if let Some(mapped_ty) = mapping.get(ty_arg) {
                            substituted_args.push(mapped_ty.to_string());
                        } else {
                            substituted_args.push(ty_arg.to_string());
                        }
                    }
                    new_name = format!("{}<{}>", base, substituted_args.join(", ")).into();
                }
                Expr::EnumVariant(EnumVariantExpr {
                    enum_name: new_name,
                    variant_name: e.variant_name.clone(),
                    payload: e
                        .payload
                        .as_ref()
                        .map(|p| p.iter().map(|ex| ex.substitute(mapping)).collect()),
                    span: e.span,
                })
            }
            Expr::VecMacro(e) => Expr::VecMacro(VecMacroExpr {
                elements: e.elements.iter().map(|ex| ex.substitute(mapping)).collect(),
                span: e.span,
            }),
            Expr::Closure(e) => Expr::Closure(ClosureExpr {
                params: e
                    .params
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute(mapping)))
                    .collect(),
                body: Box::new(e.body.substitute(mapping)),
                captures: e
                    .captures
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute(mapping)))
                    .collect(),
                ret_ty: e.ret_ty.as_ref().map(|t| t.substitute(mapping)),
                span: e.span,
            }),
            Expr::MacroCall(e) => Expr::MacroCall(MacroCallExpr {
                name: e.name.clone(),
                token_tree: e.token_tree.clone(),
                block_tree: e.block_tree.clone(),
                span: e.span,
            }),
            Expr::Print(e) => Expr::Print(PrintExpr {
                args: e.args.iter().map(|ex| ex.substitute(mapping)).collect(),
                span: e.span,
            }),
            Expr::Println(e) => Expr::Println(PrintlnExpr {
                args: e.args.iter().map(|ex| ex.substitute(mapping)).collect(),
                span: e.span,
            }),
            Expr::SizeOf(e) => Expr::SizeOf(SizeOfExpr {
                target_ty: e.target_ty.substitute(mapping),
                span: e.span,
            }),
            // An `mlir!` block is text, so its type parameters are substituted word by word.
            // The declared input types are the same text and take the same pass: leaving them
            // alone gave the generated wrapper a `memref<NxMxf32>` signature over a body that
            // had already become `2x2` (Vx#414).
            Expr::InlineMlir(e) => Expr::InlineMlir(InlineMlirExpr {
                inputs: e
                    .inputs
                    .iter()
                    .map(|(n, ex, t)| {
                        (
                            n.clone(),
                            ex.substitute(mapping),
                            substitute_words(t, mapping),
                        )
                    })
                    .collect(),
                clobbers: e.clobbers.iter().map(|ex| ex.substitute(mapping)).collect(),
                returns: e.returns.as_ref().map(|t| t.substitute(mapping)),
                dialects: e.dialects.clone(),
                block_str: substitute_words(&e.block_str, mapping),
                span: e.span,
            }),
            Expr::Number(_) | Expr::StringLiteral(_) | Expr::MemorySpace(_) | Expr::Topology(_) => {
                self.clone()
            }
        }
    }
}
