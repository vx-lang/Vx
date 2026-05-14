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

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExternDecl {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub functions: Vec<Function>,
}
