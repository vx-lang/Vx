#[derive(Debug, PartialEq, Clone, Default)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Topology {
    Host,
    NPU(Box<Expr>),
    AccCore(Box<Expr>),
    AMX,
    ANE,
    GPU,
    Slice(Box<Topology>, Box<Expr>, Box<Expr>), // For NPU[0..4] etc.
}

#[derive(Debug, PartialEq, Clone)]
pub enum MemorySpace {
    HostDRAM,
    NPUHBM,
    LocalSRAM,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ElementType {
    Bool,
    BF16,
    F16,
    F32,
    F64,
    I4,
    I8,
    I16,
    I32,
    I64,
    I128,
    U4,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Tensor(ElementType, Vec<Expr>, Option<Topology>),
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Borrow(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Pointer(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Scalar(ElementType),
    Struct(String),
    Enum(String),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(String),                                         // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),                   // e.g. Config<f32>
    Module(String, std::collections::HashMap<String, Type>), // (path, exported_symbols)
}

impl Type {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Type {
        match self {
            Type::Generic(name) => {
                if let Some(concrete) = mapping.get(name) {
                    concrete.clone()
                } else {
                    self.clone()
                }
            }
            Type::GenericInstance(base, args) => {
                let new_base = base.substitute(mapping);
                let new_args = args.iter().map(|a| a.substitute(mapping)).collect();
                Type::GenericInstance(Box::new(new_base), new_args)
            }
            Type::Borrow(inner, mem, is_mut) => {
                Type::Borrow(Box::new(inner.substitute(mapping)), mem.clone(), *is_mut)
            }
            Type::Pointer(inner, mem, is_mut) => {
                Type::Pointer(Box::new(inner.substitute(mapping)), mem.clone(), *is_mut)
            }
            Type::Tensor(el_ty, dims, top) => {
                let new_dims = dims.iter().map(|d| d.substitute(mapping)).collect();
                Type::Tensor(*el_ty, new_dims, top.clone())
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(inner.substitute(mapping)), mem.clone()),
            Type::Verified(inner) => Type::Verified(Box::new(inner.substitute(mapping))),
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(inner.substitute(mapping)), top.clone())
            }
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    // Relational
    Eq,
    NotEq,
    Lt,
    Gt,
    Le,
    Ge,
    // Logical
    And,
    Or,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    Not,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Identifier(String, Span),
    EnumVariant(String, String, Span),
    Number(f64, Span),
    StringLiteral(String, Span),
    Transfer(Box<Expr>, MemorySpace, Span),
    FunctionCall(String, Vec<Expr>, Span),
    Array(Vec<Expr>, Span),
    MemberAccess(Box<Expr>, String, Span),
    IndexAccess(Box<Expr>, Box<Expr>, Span),
    MethodCall(Box<Expr>, String, Vec<Expr>, Span),
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>, Span),
    UnaryOp(UnaryOp, Box<Expr>, Span),
    Borrow(Box<Expr>, bool, Span), // &expr or &mut expr
    Dereference(Box<Expr>, Span),  // *expr
    UnsafeBlock(Vec<Statement>, Option<Box<Expr>>, Span),
    ComptimeBlock(Vec<Statement>, Option<Box<Expr>>, Span),
    StructInit(String, Vec<(String, Expr)>, Span),
    MemorySpace(MemorySpace, Span),
    Topology(Topology, Span),
    Import(String, Span),
}

impl Expr {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Expr {
        match self {
            Expr::Transfer(expr, mem, _) => Expr::Transfer(
                Box::new(expr.substitute(mapping)),
                mem.clone(),
                Span::default(),
            ),
            Expr::Import(path, span) => Expr::Import(path.clone(), span.clone()),
            Expr::ComptimeBlock(stmts, ret, span) => Expr::ComptimeBlock(
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span.clone(),
            ),
            Expr::FunctionCall(name, args, span) => Expr::FunctionCall(
                name.clone(),
                args.iter().map(|a| a.substitute(mapping)).collect(),
                span.clone(),
            ),
            Expr::Array(items, _) => Expr::Array(
                items.iter().map(|a| a.substitute(mapping)).collect(),
                Span::default(),
            ),
            Expr::MemberAccess(expr, member, _) => Expr::MemberAccess(
                Box::new(expr.substitute(mapping)),
                member.clone(),
                Span::default(),
            ),
            Expr::IndexAccess(expr, idx, span) => Expr::IndexAccess(
                Box::new(expr.substitute(mapping)),
                Box::new(idx.substitute(mapping)),
                span.clone(),
            ),
            Expr::MethodCall(expr, method, args, span) => Expr::MethodCall(
                Box::new(expr.substitute(mapping)),
                method.clone(),
                args.iter().map(|a| a.substitute(mapping)).collect(),
                span.clone(),
            ),
            Expr::BinaryOp(lhs, op, rhs, span) => Expr::BinaryOp(
                Box::new(lhs.substitute(mapping)),
                op.clone(),
                Box::new(rhs.substitute(mapping)),
                span.clone(),
            ),
            Expr::UnaryOp(op, expr, _) => Expr::UnaryOp(
                op.clone(),
                Box::new(expr.substitute(mapping)),
                Span::default(),
            ),
            Expr::Borrow(expr, is_mut, _) => {
                Expr::Borrow(Box::new(expr.substitute(mapping)), *is_mut, Span::default())
            }
            Expr::Dereference(expr, _) => {
                Expr::Dereference(Box::new(expr.substitute(mapping)), Span::default())
            }
            Expr::UnsafeBlock(stmts, ret, span) => Expr::UnsafeBlock(
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span.clone(),
            ),
            Expr::StructInit(name, fields, span) => Expr::StructInit(
                name.clone(),
                fields
                    .iter()
                    .map(|(n, e)| (n.clone(), e.substitute(mapping)))
                    .collect(),
                span.clone(),
            ),
            Expr::Identifier(..)
            | Expr::EnumVariant(..)
            | Expr::Number(..)
            | Expr::StringLiteral(..)
            | Expr::MemorySpace(..)
            | Expr::Topology(..) => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetDecl(String, bool, Option<Type>, Expr, Span), // (name, is_mut, type_annotation, expr)
    Return(Expr, Span),
    SpawnOn(Topology, Vec<Statement>, Span),
    ExprStmt(Expr, Span),
    ForLoop(String, Box<Expr>, Box<Expr>, Vec<Statement>, Span), // (iterator, start, end, body)
    If(Box<Expr>, Vec<Statement>, Option<Vec<Statement>>, Span), // (condition, then_block, else_block)
    Assign(Expr, Expr, Span),                                    // lhs = rhs
    CompoundAssign(Expr, BinaryOp, Expr, Span),                  // lhs += rhs
    Comptime(Vec<Statement>, Span),                              // comptime { ... }
    Assert(Box<Expr>, Option<String>, Span),                     // assert(expr, "message")
}

impl Statement {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Statement {
        match self {
            Statement::LetDecl(name, is_mut, ty_ann, expr, span) => Statement::LetDecl(
                name.clone(),
                *is_mut,
                ty_ann.as_ref().map(|t| t.substitute(mapping)),
                expr.substitute(mapping),
                span.clone(),
            ),
            Statement::Return(expr, _) => {
                Statement::Return(expr.substitute(mapping), Span::default())
            }
            Statement::SpawnOn(top, stmts, span) => Statement::SpawnOn(
                top.clone(),
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
                span.clone(),
            ),
            Statement::ExprStmt(expr, _) => {
                Statement::ExprStmt(expr.substitute(mapping), Span::default())
            }
            Statement::ForLoop(iter, start, end, body, span) => Statement::ForLoop(
                iter.clone(),
                Box::new(start.substitute(mapping)),
                Box::new(end.substitute(mapping)),
                body.iter().map(|s| s.substitute(mapping)).collect(),
                span.clone(),
            ),
            Statement::If(cond, then_block, else_block, span) => Statement::If(
                Box::new(cond.substitute(mapping)),
                then_block.iter().map(|s| s.substitute(mapping)).collect(),
                else_block
                    .as_ref()
                    .map(|b| b.iter().map(|s| s.substitute(mapping)).collect()),
                span.clone(),
            ),
            Statement::Assign(lhs, rhs, _) => Statement::Assign(
                lhs.substitute(mapping),
                rhs.substitute(mapping),
                Span::default(),
            ),
            Statement::CompoundAssign(lhs, op, rhs, span) => Statement::CompoundAssign(
                lhs.substitute(mapping),
                op.clone(),
                rhs.substitute(mapping),
                span.clone(),
            ),
            Statement::Comptime(stmts, _) => Statement::Comptime(
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
                Span::default(),
            ),
            Statement::Assert(expr, msg, _) => Statement::Assert(
                Box::new(expr.substitute(mapping)),
                msg.clone(),
                Span::default(),
            ),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub generics: Vec<(String, Option<String>)>, // (TypeParamName, OptionalTraitBound)
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    pub name: String,
    pub generics: Vec<(String, Option<String>)>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExternDecl {
    pub name: String,
    pub is_safe: bool,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TraitDecl {
    pub name: String,
    // (method_name, params, return_type)
    #[allow(clippy::type_complexity)]
    pub methods: Vec<(String, Vec<(String, Type)>, Type)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImplBlock {
    pub trait_name: Option<String>,
    pub target_type: Type,
    pub methods: Vec<Function>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub enums: Vec<EnumDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
}

pub type AkModule = Program;
