#[derive(Debug, PartialEq, Clone)]
pub enum Topology {
    Host,
    NPU(Box<Expr>),
    AccCore(Box<Expr>),
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
    Tensor(ElementType),
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Borrow(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Pointer(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Struct(String),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(String),                       // e.g. T
    GenericInstance(Box<Type>, Vec<Type>), // e.g. Config<f32>
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
    Identifier(String),
    Number(f64),
    Transfer(Box<Expr>, MemorySpace),
    FunctionCall(String, Vec<Expr>),
    Array(Vec<Expr>),
    MemberAccess(Box<Expr>, String),
    IndexAccess(Box<Expr>, Box<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
    Borrow(Box<Expr>, bool), // &expr or &mut expr
    Dereference(Box<Expr>),  // *expr
    UnsafeBlock(Vec<Statement>, Option<Box<Expr>>),
    StructInit(String, Vec<(String, Expr)>),
    MemorySpace(MemorySpace),
    Topology(Topology),
}

impl Expr {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Expr {
        match self {
            Expr::Transfer(expr, mem) => {
                Expr::Transfer(Box::new(expr.substitute(mapping)), mem.clone())
            }
            Expr::FunctionCall(name, args) => Expr::FunctionCall(
                name.clone(),
                args.iter().map(|a| a.substitute(mapping)).collect(),
            ),
            Expr::Array(items) => {
                Expr::Array(items.iter().map(|a| a.substitute(mapping)).collect())
            }
            Expr::MemberAccess(expr, member) => {
                Expr::MemberAccess(Box::new(expr.substitute(mapping)), member.clone())
            }
            Expr::IndexAccess(expr, idx) => Expr::IndexAccess(
                Box::new(expr.substitute(mapping)),
                Box::new(idx.substitute(mapping)),
            ),
            Expr::MethodCall(expr, method, args) => Expr::MethodCall(
                Box::new(expr.substitute(mapping)),
                method.clone(),
                args.iter().map(|a| a.substitute(mapping)).collect(),
            ),
            Expr::BinaryOp(lhs, op, rhs) => Expr::BinaryOp(
                Box::new(lhs.substitute(mapping)),
                op.clone(),
                Box::new(rhs.substitute(mapping)),
            ),
            Expr::UnaryOp(op, expr) => {
                Expr::UnaryOp(op.clone(), Box::new(expr.substitute(mapping)))
            }
            Expr::Borrow(expr, is_mut) => Expr::Borrow(Box::new(expr.substitute(mapping)), *is_mut),
            Expr::Dereference(expr) => Expr::Dereference(Box::new(expr.substitute(mapping))),
            Expr::UnsafeBlock(stmts, ret) => Expr::UnsafeBlock(
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
            ),
            Expr::StructInit(name, fields) => Expr::StructInit(
                name.clone(),
                fields
                    .iter()
                    .map(|(n, e)| (n.clone(), e.substitute(mapping)))
                    .collect(),
            ),
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetDecl(String, bool, Option<Type>, Expr), // (name, is_mut, type_annotation, expr)
    Return(Expr),
    SpawnOn(Topology, Vec<Statement>),
    ExprStmt(Expr),
    ForLoop(String, Box<Expr>, Box<Expr>, Vec<Statement>), // (iterator, start, end, body)
    Assign(Expr, Expr),                                    // lhs = rhs
    CompoundAssign(Expr, BinaryOp, Expr),                  // lhs += rhs
}

impl Statement {
    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Statement {
        match self {
            Statement::LetDecl(name, is_mut, ty_ann, expr) => Statement::LetDecl(
                name.clone(),
                *is_mut,
                ty_ann.as_ref().map(|t| t.substitute(mapping)),
                expr.substitute(mapping),
            ),
            Statement::Return(expr) => Statement::Return(expr.substitute(mapping)),
            Statement::SpawnOn(top, stmts) => Statement::SpawnOn(
                top.clone(),
                stmts.iter().map(|s| s.substitute(mapping)).collect(),
            ),
            Statement::ExprStmt(expr) => Statement::ExprStmt(expr.substitute(mapping)),
            Statement::ForLoop(iter, start, end, body) => Statement::ForLoop(
                iter.clone(),
                Box::new(start.substitute(mapping)),
                Box::new(end.substitute(mapping)),
                body.iter().map(|s| s.substitute(mapping)).collect(),
            ),
            Statement::Assign(lhs, rhs) => {
                Statement::Assign(lhs.substitute(mapping), rhs.substitute(mapping))
            }
            Statement::CompoundAssign(lhs, op, rhs) => Statement::CompoundAssign(
                lhs.substitute(mapping),
                op.clone(),
                rhs.substitute(mapping),
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
pub struct ExternDecl {
    pub name: String,
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
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
}
