use super::*;

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
    pub name: String,
    pub span: Span,
}
impl IdentifierExpr {
    pub fn new(name: String, span: Span) -> Self {
        Self { name, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumVariantExpr {
    pub enum_name: String,
    pub variant_name: String,
    pub payload: Option<Vec<Expr>>,
    pub span: Span,
}
impl EnumVariantExpr {
    pub fn new(
        enum_name: String,
        variant_name: String,
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
    pub value: String,
    pub ty: Option<ElementType>,
    pub span: Span,
}
impl NumberExpr {
    pub fn new(value: String, ty: Option<ElementType>, span: Span) -> Self {
        Self { value, ty, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StringLiteralExpr {
    pub value: String,
    pub span: Span,
}
impl StringLiteralExpr {
    pub fn new(value: String, span: Span) -> Self {
        Self { value, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TransferExpr {
    pub expr: Box<Expr>,
    pub space: MemorySpace,
    pub span: Span,
}
impl TransferExpr {
    pub fn new(expr: Box<Expr>, space: MemorySpace, span: Span) -> Self {
        Self { expr, space, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionCallExpr {
    pub name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl FunctionCallExpr {
    pub fn new(name: String, args: Vec<Expr>, span: Span) -> Self {
        Self { name, args, span }
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
    pub span: Span,
}
impl IndirectCallExpr {
    pub fn new(callee: Box<Expr>, args: Vec<Expr>, span: Span) -> Self {
        Self { callee, args, span }
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
}

#[derive(Debug, PartialEq, Clone)]
pub struct MemberAccessExpr {
    pub base: Box<Expr>,
    pub member: String,
    pub struct_name: Option<String>,
    pub span: Span,
}
impl MemberAccessExpr {
    pub fn new(base: Box<Expr>, member: String, span: Span) -> Self {
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
    pub method_name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl MethodCallExpr {
    pub fn new(base: Box<Expr>, method_name: String, args: Vec<Expr>, span: Span) -> Self {
        Self {
            base,
            method_name,
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
    pub ty: Option<crate::ast::Type>,
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
    pub name: String,
    pub fields: Vec<(String, Expr)>,
    pub span: Span,
}
impl StructInitExpr {
    pub fn new(name: String, fields: Vec<(String, Expr)>, span: Span) -> Self {
        Self { name, fields, span }
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
    pub target_fn: String,
    pub args: Vec<Expr>,
    pub span: Span,
}
impl GradExpr {
    pub fn new(target_fn: String, args: Vec<Expr>, span: Span) -> Self {
        Self {
            target_fn,
            args,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct VjpExpr {
    pub target_fn: String,
    pub args: Vec<Expr>,
    pub cotangent: Box<Expr>,
    pub span: Span,
}
impl VjpExpr {
    pub fn new(target_fn: String, args: Vec<Expr>, cotangent: Box<Expr>, span: Span) -> Self {
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
    pub target_fn: String,
    pub args: Vec<Expr>,
    pub tangent: Box<Expr>,
    pub span: Span,
}
impl JvpExpr {
    pub fn new(target_fn: String, args: Vec<Expr>, tangent: Box<Expr>, span: Span) -> Self {
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
    pub cond: Box<Expr>,
    pub then_block: Vec<Statement>,
    pub else_block: Option<Vec<Statement>>,
    pub span: Span,
}
impl IfExpr {
    pub fn new(
        cond: Box<Expr>,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
        span: Span,
    ) -> Self {
        Self {
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
    Identifier(String),
    EnumVariant(String, String, Option<Vec<Pattern>>), // enum_name, variant_name, payload patterns
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
    pub params: Vec<(String, Type)>,
    pub body: Box<Expr>,
    pub captures: Vec<(String, Type)>,
    pub ret_ty: Option<Type>,
    pub span: Span,
}
impl ClosureExpr {
    pub fn new(params: Vec<(String, Type)>, body: Box<Expr>, span: Span) -> Self {
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
    pub name: String,
    pub token_tree: TokenTree,
    pub span: Span,
}
impl MacroCallExpr {
    pub fn new(name: String, token_tree: TokenTree, span: Span) -> Self {
        Self {
            name,
            token_tree,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Identifier(IdentifierExpr),
    EnumVariant(EnumVariantExpr),
    Number(NumberExpr),
    StringLiteral(StringLiteralExpr),
    Transfer(TransferExpr),
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
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Identifier(e) => e.span.clone(),
            Expr::EnumVariant(e) => e.span.clone(),
            Expr::Number(e) => e.span.clone(),
            Expr::StringLiteral(e) => e.span.clone(),
            Expr::Transfer(e) => e.span.clone(),
            Expr::FunctionCall(e) => e.span.clone(),
            Expr::IndirectCall(e) => e.span.clone(),
            Expr::Array(e) => e.span.clone(),
            Expr::MemberAccess(e) => e.span.clone(),
            Expr::IndexAccess(e) => e.span.clone(),
            Expr::MethodCall(e) => e.span.clone(),
            Expr::BinaryOp(e) => e.span.clone(),
            Expr::RelationalOp(e) => e.span.clone(),
            Expr::LogicalOp(e) => e.span.clone(),
            Expr::UnaryOp(e) => e.span.clone(),
            Expr::Borrow(e) => e.span.clone(),
            Expr::Dereference(e) => e.span.clone(),
            Expr::UnsafeBlock(e) => e.span.clone(),
            Expr::ComptimeBlock(e) => e.span.clone(),
            Expr::StructInit(e) => e.span.clone(),
            Expr::MemorySpace(e) => e.span.clone(),
            Expr::Topology(e) => e.span.clone(),
            Expr::If(e) => e.span.clone(),
            Expr::Range(e) => e.span.clone(),
            Expr::Match(e) => e.span.clone(),
            Expr::Grad(e) => e.span.clone(),
            Expr::Vjp(e) => e.span.clone(),
            Expr::Jvp(e) => e.span.clone(),
            Expr::SpawnOn(e) => e.span.clone(),
            Expr::VecMacro(e) => e.span.clone(),
            Expr::Closure(e) => e.span.clone(),
            Expr::MacroCall(e) => e.span.clone(),
            Expr::AsCast(e) => e.span.clone(),
        }
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

    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Expr {
        match self {
            Expr::Transfer(e) => Expr::Transfer(TransferExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                space: e.space.clone(),
                span: e.span.clone(),
            }),
            Expr::ComptimeBlock(e) => Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span.clone(),
            }),
            Expr::FunctionCall(e) => {
                let mut new_name = e.name.clone();
                if new_name.starts_with("Tensor_") {
                    let t_name = new_name.strip_prefix("Tensor_").unwrap();
                    if let Some(concrete_el) = mapping.get(t_name) {
                        new_name = format!("Tensor_{}", concrete_el);
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
                            format!("{}<{}>{}", base, substituted_args.join(", "), remainder);
                    }
                }
                Expr::FunctionCall(FunctionCallExpr {
                    name: new_name,
                    args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                    span: e.span.clone(),
                })
            }
            Expr::IndirectCall(e) => Expr::IndirectCall(IndirectCallExpr {
                callee: Box::new(e.callee.substitute(mapping)),
                args: e.args.iter().map(|arg| arg.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::Array(e) => Expr::Array(ArrayExpr {
                elements: e.elements.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::MemberAccess(e) => Expr::MemberAccess(MemberAccessExpr {
                base: Box::new(e.base.substitute(mapping)),
                member: e.member.clone(),
                struct_name: e.struct_name.clone(),
                span: e.span.clone(),
            }),
            Expr::IndexAccess(e) => Expr::IndexAccess(IndexAccessExpr {
                base: Box::new(e.base.substitute(mapping)),
                index: Box::new(e.index.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::MethodCall(e) => Expr::MethodCall(MethodCallExpr {
                base: Box::new(e.base.substitute(mapping)),
                method_name: e.method_name.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::BinaryOp(e) => Expr::BinaryOp(BinaryOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::RelationalOp(e) => Expr::RelationalOp(RelationalOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::LogicalOp(e) => Expr::LogicalOp(LogicalOpExpr {
                lhs: Box::new(e.lhs.substitute(mapping)),
                op: e.op.clone(),
                rhs: Box::new(e.rhs.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::UnaryOp(e) => Expr::UnaryOp(UnaryOpExpr {
                op: e.op.clone(),
                expr: Box::new(e.expr.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::Borrow(e) => Expr::Borrow(BorrowExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                is_mut: e.is_mut,
                span: e.span.clone(),
            }),
            Expr::Dereference(e) => Expr::Dereference(DereferenceExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                ty: e.ty.as_ref().map(|t| t.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::UnsafeBlock(e) => Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span.clone(),
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
                    new_name = format!("{}<{}>", base, substituted_args.join(", "));
                }
                Expr::StructInit(StructInitExpr {
                    name: new_name,
                    fields: e
                        .fields
                        .iter()
                        .map(|(n, ex)| (n.clone(), ex.substitute(mapping)))
                        .collect(),
                    span: e.span.clone(),
                })
            }
            Expr::AsCast(e) => Expr::AsCast(AsCastExpr {
                expr: Box::new(e.expr.substitute(mapping)),
                target_ty: e.target_ty.substitute(mapping),
                source_ty: e.source_ty.as_ref().map(|t| t.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::If(e) => Expr::If(IfExpr {
                cond: Box::new(e.cond.substitute(mapping)),
                then_block: e.then_block.iter().map(|s| s.substitute(mapping)).collect(),
                else_block: e
                    .else_block
                    .as_ref()
                    .map(|b| b.iter().map(|s| s.substitute(mapping)).collect()),
                span: e.span.clone(),
            }),
            Expr::Range(e) => Expr::Range(RangeExpr {
                start: Box::new(e.start.substitute(mapping)),
                end: Box::new(e.end.substitute(mapping)),
                span: e.span.clone(),
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
                span: e.span.clone(),
            }),
            Expr::Grad(e) => Expr::Grad(GradExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::Vjp(e) => Expr::Vjp(VjpExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                cotangent: Box::new(e.cotangent.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::Jvp(e) => Expr::Jvp(JvpExpr {
                target_fn: e.target_fn.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                tangent: Box::new(e.tangent.substitute(mapping)),
                span: e.span.clone(),
            }),
            Expr::SpawnOn(e) => Expr::SpawnOn(SpawnOnExpr {
                top: e.top.clone(),
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span.clone(),
            }),
            Expr::Identifier(id) => {
                if let Some(Type::Generic(val_str, _)) = mapping.get(&id.name) {
                    if val_str.parse::<f64>().is_ok() {
                        return Expr::Number(crate::ast::NumberExpr {
                            value: val_str.clone(),
                            ty: None,
                            span: id.span.clone(),
                        });
                    } else {
                        return Expr::Identifier(crate::ast::IdentifierExpr {
                            name: val_str.clone(),
                            span: id.span.clone(),
                        });
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
                    new_name = format!("{}<{}>", base, substituted_args.join(", "));
                }
                Expr::EnumVariant(EnumVariantExpr {
                    enum_name: new_name,
                    variant_name: e.variant_name.clone(),
                    payload: e
                        .payload
                        .as_ref()
                        .map(|p| p.iter().map(|ex| ex.substitute(mapping)).collect()),
                    span: e.span.clone(),
                })
            }
            Expr::VecMacro(e) => Expr::VecMacro(VecMacroExpr {
                elements: e.elements.iter().map(|ex| ex.substitute(mapping)).collect(),
                span: e.span.clone(),
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
                span: e.span.clone(),
            }),
            Expr::MacroCall(e) => Expr::MacroCall(MacroCallExpr {
                name: e.name.clone(),
                token_tree: e.token_tree.clone(),
                span: e.span.clone(),
            }),
            Expr::Number(_) | Expr::StringLiteral(_) | Expr::MemorySpace(_) | Expr::Topology(_) => {
                self.clone()
            }
        }
    }
}
