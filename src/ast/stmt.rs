use super::*;

#[derive(Debug, PartialEq, Clone)]
pub struct LetDeclStmt {
    pub name: String,
    pub is_mut: bool,
    pub ty_ann: Option<Type>,
    pub expr: Expr,
    pub span: Span,
}
impl LetDeclStmt {
    pub fn new(name: String, is_mut: bool, ty_ann: Option<Type>, expr: Expr, span: Span) -> Self {
        Self {
            name,
            is_mut,
            ty_ann,
            expr,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ReturnStmt {
    pub expr: Expr,
    pub span: Span,
}
impl ReturnStmt {
    pub fn new(expr: Expr, span: Span) -> Self {
        Self { expr, span }
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
pub struct ExprStmtStmt {
    pub expr: Expr,
    pub has_semi: bool,
    pub span: Span,
}
impl ExprStmtStmt {
    pub fn new(expr: Expr, has_semi: bool, span: Span) -> Self {
        Self {
            expr,
            has_semi,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ForLoopStmt {
    pub iter: String,
    pub iterable: Box<Expr>,
    pub body: Vec<Statement>,
    pub span: Span,
}
impl ForLoopStmt {
    pub fn new(iter: String, iterable: Box<Expr>, body: Vec<Statement>, span: Span) -> Self {
        Self {
            iter,
            iterable,
            body,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AssignStmt {
    pub lhs: Expr,
    pub rhs: Expr,
    pub span: Span,
}
impl AssignStmt {
    pub fn new(lhs: Expr, rhs: Expr, span: Span) -> Self {
        Self { lhs, rhs, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct CompoundAssignStmt {
    pub lhs: Expr,
    pub op: BinaryOp,
    pub rhs: Expr,
    pub span: Span,
}
impl CompoundAssignStmt {
    pub fn new(lhs: Expr, op: BinaryOp, rhs: Expr, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AssertStmt {
    pub expr: Box<Expr>,
    pub msg: Option<String>,
    pub span: Span,
}
impl AssertStmt {
    pub fn new(expr: Box<Expr>, msg: Option<String>, span: Span) -> Self {
        Self { expr, msg, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LoopStmt {
    pub body: Vec<Statement>,
    pub span: Span,
}
impl LoopStmt {
    pub fn new(body: Vec<Statement>, span: Span) -> Self {
        Self { body, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BreakStmt {
    pub span: Span,
}
impl BreakStmt {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ContinueStmt {
    pub span: Span,
}
impl ContinueStmt {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetDecl(LetDeclStmt),
    Return(ReturnStmt),
    ExprStmt(ExprStmtStmt),
    ForLoop(ForLoopStmt),
    Assign(AssignStmt),
    CompoundAssign(CompoundAssignStmt),
    Assert(AssertStmt),
    Loop(LoopStmt),
    Break(BreakStmt),
    Continue(ContinueStmt),
}

impl Statement {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Statement {
        match self {
            Statement::LetDecl(e) => Statement::LetDecl(LetDeclStmt {
                name: e.name.clone(),
                is_mut: e.is_mut,
                ty_ann: e.ty_ann.as_ref().map(|t| t.substitute(mapping)),
                expr: e.expr.substitute(mapping),
                span: e.span.clone(),
            }),
            Statement::Return(e) => Statement::Return(ReturnStmt {
                expr: e.expr.substitute(mapping),
                span: e.span.clone(),
            }),
            Statement::ExprStmt(e) => Statement::ExprStmt(ExprStmtStmt {
                expr: e.expr.substitute(mapping),
                has_semi: e.has_semi,
                span: e.span.clone(),
            }),
            Statement::ForLoop(e) => Statement::ForLoop(ForLoopStmt {
                iter: e.iter.clone(),
                iterable: Box::new(e.iterable.substitute(mapping)),
                body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Statement::Assign(e) => Statement::Assign(AssignStmt {
                lhs: e.lhs.substitute(mapping),
                rhs: e.rhs.substitute(mapping),
                span: e.span.clone(),
            }),
            Statement::CompoundAssign(e) => Statement::CompoundAssign(CompoundAssignStmt {
                lhs: e.lhs.substitute(mapping),
                op: e.op.clone(),
                rhs: e.rhs.substitute(mapping),
                span: e.span.clone(),
            }),
            Statement::Assert(e) => Statement::Assert(AssertStmt {
                expr: Box::new(e.expr.substitute(mapping)),
                msg: e.msg.clone(),
                span: e.span.clone(),
            }),
            Statement::Loop(e) => Statement::Loop(LoopStmt {
                body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Statement::Break(e) => Statement::Break(e.clone()),
            Statement::Continue(e) => Statement::Continue(e.clone()),
        }
    }
}
