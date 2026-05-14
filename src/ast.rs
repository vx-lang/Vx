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

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Tensor,
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Identifier(String),
    Number(f64),
    Transfer(Box<Expr>, MemorySpace),
    FunctionCall(String, Vec<Expr>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetDecl(String, Expr),
    Return(Expr),
    SpawnOn(Topology, Vec<Statement>),
    ExprStmt(Expr),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}
