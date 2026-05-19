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
    Struct(String, Option<crate::gid::TypeId>),
    Enum(String, Option<crate::gid::TypeId>),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(String, Option<crate::gid::TypeId>), // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),       // e.g. Config<f32>
    Module(String, std::collections::HashMap<String, Type>), // (path, exported_symbols)
    Simd(ElementType, usize),                    // e.g. <4 x f32>
}

impl Type {
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            Type::Ref(_, _)
                | Type::Tensor(_, _, _)
                | Type::Matrix
                | Type::Verified(_)
                | Type::Pinned(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _)
        )
    }

    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Type {
        match self {
            Type::Generic(name, _) => {
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
            Type::Simd(el_ty, n) => Type::Simd(*el_ty, *n),
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
    pub span: Span,
}
impl EnumVariantExpr {
    pub fn new(enum_name: String, variant_name: String, span: Span) -> Self {
        Self {
            enum_name,
            variant_name,
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
    pub span: Span,
}
impl MemberAccessExpr {
    pub fn new(base: Box<Expr>, member: String, span: Span) -> Self {
        Self { base, member, span }
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
    pub span: Span,
}
impl DereferenceExpr {
    pub fn new(expr: Box<Expr>, span: Span) -> Self {
        Self { expr, span }
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
pub enum Expr {
    Identifier(IdentifierExpr),
    EnumVariant(EnumVariantExpr),
    Number(NumberExpr),
    StringLiteral(StringLiteralExpr),
    Transfer(TransferExpr),
    FunctionCall(FunctionCallExpr),
    Array(ArrayExpr),
    MemberAccess(MemberAccessExpr),
    IndexAccess(IndexAccessExpr),
    MethodCall(MethodCallExpr),
    BinaryOp(BinaryOpExpr),
    UnaryOp(UnaryOpExpr),
    Borrow(BorrowExpr),
    Dereference(DereferenceExpr),
    UnsafeBlock(UnsafeBlockExpr),
    ComptimeBlock(ComptimeBlockExpr),
    StructInit(StructInitExpr),
    MemorySpace(MemorySpaceExpr),
    Topology(TopologyExpr),
    If(IfExpr),
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
            Expr::Array(e) => e.span.clone(),
            Expr::MemberAccess(e) => e.span.clone(),
            Expr::IndexAccess(e) => e.span.clone(),
            Expr::MethodCall(e) => e.span.clone(),
            Expr::BinaryOp(e) => e.span.clone(),
            Expr::UnaryOp(e) => e.span.clone(),
            Expr::Borrow(e) => e.span.clone(),
            Expr::Dereference(e) => e.span.clone(),
            Expr::UnsafeBlock(e) => e.span.clone(),
            Expr::ComptimeBlock(e) => e.span.clone(),
            Expr::StructInit(e) => e.span.clone(),
            Expr::MemorySpace(e) => e.span.clone(),
            Expr::Topology(e) => e.span.clone(),
            Expr::If(e) => e.span.clone(),
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
            Expr::FunctionCall(e) => Expr::FunctionCall(FunctionCallExpr {
                name: e.name.clone(),
                args: e.args.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::Array(e) => Expr::Array(ArrayExpr {
                elements: e.elements.iter().map(|a| a.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Expr::MemberAccess(e) => Expr::MemberAccess(MemberAccessExpr {
                base: Box::new(e.base.substitute(mapping)),
                member: e.member.clone(),
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
                span: e.span.clone(),
            }),
            Expr::UnsafeBlock(e) => Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                ret: e.ret.as_ref().map(|r| Box::new(r.substitute(mapping))),
                span: e.span.clone(),
            }),
            Expr::StructInit(e) => Expr::StructInit(StructInitExpr {
                name: e.name.clone(),
                fields: e
                    .fields
                    .iter()
                    .map(|(n, ex)| (n.clone(), ex.substitute(mapping)))
                    .collect(),
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
            Expr::Identifier(_)
            | Expr::EnumVariant(_)
            | Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::MemorySpace(_)
            | Expr::Topology(_) => self.clone(),
        }
    }
}

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
pub struct SpawnOnStmt {
    pub top: Topology,
    pub stmts: Vec<Statement>,
    pub span: Span,
}
impl SpawnOnStmt {
    pub fn new(top: Topology, stmts: Vec<Statement>, span: Span) -> Self {
        Self { top, stmts, span }
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
    pub start: Box<Expr>,
    pub end: Box<Expr>,
    pub body: Vec<Statement>,
    pub span: Span,
}
impl ForLoopStmt {
    pub fn new(
        iter: String,
        start: Box<Expr>,
        end: Box<Expr>,
        body: Vec<Statement>,
        span: Span,
    ) -> Self {
        Self {
            iter,
            start,
            end,
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
pub enum Statement {
    LetDecl(LetDeclStmt),
    Return(ReturnStmt),
    SpawnOn(SpawnOnStmt),
    ExprStmt(ExprStmtStmt),
    ForLoop(ForLoopStmt),
    Assign(AssignStmt),
    CompoundAssign(CompoundAssignStmt),
    Assert(AssertStmt),
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
            Statement::SpawnOn(e) => Statement::SpawnOn(SpawnOnStmt {
                top: e.top.clone(),
                stmts: e.stmts.iter().map(|s| s.substitute(mapping)).collect(),
                span: e.span.clone(),
            }),
            Statement::ExprStmt(e) => Statement::ExprStmt(ExprStmtStmt {
                expr: e.expr.substitute(mapping),
                has_semi: e.has_semi,
                span: e.span.clone(),
            }),
            Statement::ForLoop(e) => Statement::ForLoop(ForLoopStmt {
                iter: e.iter.clone(),
                start: Box::new(e.start.substitute(mapping)),
                end: Box::new(e.end.substitute(mapping)),
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
pub struct ImportDecl {
    pub path: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub module_path: String,
    pub imports: Vec<ImportDecl>,
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub enums: Vec<EnumDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
}

pub type VxModule = Program;
pub type VxFunction = Function;

impl Program {
    pub fn add(&mut self, func: VxFunction) {
        self.functions.push(func);
    }
}

impl Type {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        match self {
            Type::Struct(name, id) | Type::Enum(name, id) | Type::Generic(name, id) => {
                if let Some(mod_syms) = symbol_map.get(current_module) {
                    if let Some(tid) = mod_syms.get(name) {
                        *id = Some(*tid);
                    }
                }
            }
            Type::Tensor(_, dims, top) => {
                for dim in dims {
                    dim.resolve_names(current_module, symbol_map);
                }
                if let Some(t) = top {
                    t.resolve_names(current_module, symbol_map);
                }
            }
            Type::Ref(inner, _)
            | Type::Borrow(inner, _, _)
            | Type::Pointer(inner, _, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => {
                inner.resolve_names(current_module, symbol_map);
            }
            Type::GenericInstance(base, args) => {
                base.resolve_names(current_module, symbol_map);
                for arg in args {
                    arg.resolve_names(current_module, symbol_map);
                }
            }
            Type::Module(_, exported) => {
                for ty in exported.values_mut() {
                    ty.resolve_names(current_module, symbol_map);
                }
            }
            Type::Matrix | Type::Scalar(_) | Type::Simd(_, _) => {}
        }
    }
}

impl Topology {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        match self {
            Topology::NPU(e) | Topology::AccCore(e) => e.resolve_names(current_module, symbol_map),
            Topology::Slice(t, e1, e2) => {
                t.resolve_names(current_module, symbol_map);
                e1.resolve_names(current_module, symbol_map);
                e2.resolve_names(current_module, symbol_map);
            }
            _ => {}
        }
    }
}

impl Expr {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        match self {
            Expr::Transfer(e) => e.expr.resolve_names(current_module, symbol_map),
            Expr::MemberAccess(e) => e.base.resolve_names(current_module, symbol_map),
            Expr::UnaryOp(e) => e.expr.resolve_names(current_module, symbol_map),
            Expr::Borrow(e) => e.expr.resolve_names(current_module, symbol_map),
            Expr::Dereference(e) => e.expr.resolve_names(current_module, symbol_map),
            Expr::FunctionCall(e) => {
                for a in &mut e.args {
                    a.resolve_names(current_module, symbol_map);
                }
            }
            Expr::MethodCall(e) => {
                e.base.resolve_names(current_module, symbol_map);
                for a in &mut e.args {
                    a.resolve_names(current_module, symbol_map);
                }
            }
            Expr::Array(e) => {
                for a in &mut e.elements {
                    a.resolve_names(current_module, symbol_map);
                }
            }
            Expr::IndexAccess(e) => {
                e.base.resolve_names(current_module, symbol_map);
                e.index.resolve_names(current_module, symbol_map);
            }
            Expr::BinaryOp(e) => {
                e.lhs.resolve_names(current_module, symbol_map);
                e.rhs.resolve_names(current_module, symbol_map);
            }
            Expr::StructInit(e) => {
                for (_, ex) in &mut e.fields {
                    ex.resolve_names(current_module, symbol_map);
                }
            }
            Expr::UnsafeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(current_module, symbol_map);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(current_module, symbol_map);
                }
            }
            Expr::ComptimeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(current_module, symbol_map);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(current_module, symbol_map);
                }
            }
            Expr::If(e) => {
                e.cond.resolve_names(current_module, symbol_map);
                for s in &mut e.then_block {
                    s.resolve_names(current_module, symbol_map);
                }
                if let Some(eb) = &mut e.else_block {
                    for s in eb {
                        s.resolve_names(current_module, symbol_map);
                    }
                }
            }
            Expr::Topology(e) => e.top.resolve_names(current_module, symbol_map),
            _ => {}
        }
    }
}

impl Statement {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        match self {
            Statement::LetDecl(e) => {
                if let Some(t) = &mut e.ty_ann {
                    t.resolve_names(current_module, symbol_map);
                }
                e.expr.resolve_names(current_module, symbol_map);
            }
            Statement::Return(e) => e.expr.resolve_names(current_module, symbol_map),
            Statement::ExprStmt(e) => e.expr.resolve_names(current_module, symbol_map),
            Statement::Assert(e) => e.expr.resolve_names(current_module, symbol_map),
            Statement::SpawnOn(e) => {
                e.top.resolve_names(current_module, symbol_map);
                for s in &mut e.stmts {
                    s.resolve_names(current_module, symbol_map);
                }
            }
            Statement::ForLoop(e) => {
                e.start.resolve_names(current_module, symbol_map);
                e.end.resolve_names(current_module, symbol_map);
                for s in &mut e.body {
                    s.resolve_names(current_module, symbol_map);
                }
            }
            Statement::Assign(e) => {
                e.lhs.resolve_names(current_module, symbol_map);
                e.rhs.resolve_names(current_module, symbol_map);
            }
            Statement::CompoundAssign(e) => {
                e.lhs.resolve_names(current_module, symbol_map);
                e.rhs.resolve_names(current_module, symbol_map);
            }
        }
    }
}

impl Function {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        for (_, ty) in &mut self.params {
            ty.resolve_names(current_module, symbol_map);
        }
        self.return_type.resolve_names(current_module, symbol_map);
        for s in &mut self.body {
            s.resolve_names(current_module, symbol_map);
        }
    }
}

impl StructDecl {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        for (_, ty) in &mut self.fields {
            ty.resolve_names(current_module, symbol_map);
        }
    }
}

impl EnumDecl {
    pub fn resolve_names(
        &mut self,
        _current_module: &str,
        _symbol_map: &crate::resolver::SymbolMap,
    ) {
    }
}

impl TraitDecl {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        for (_, params, ret_ty) in &mut self.methods {
            for (_, ty) in params {
                ty.resolve_names(current_module, symbol_map);
            }
            ret_ty.resolve_names(current_module, symbol_map);
        }
    }
}

impl ImplBlock {
    pub fn resolve_names(&mut self, current_module: &str, symbol_map: &crate::resolver::SymbolMap) {
        self.target_type.resolve_names(current_module, symbol_map);
        for f in &mut self.methods {
            f.resolve_names(current_module, symbol_map);
        }
    }
}

impl Program {
    pub fn resolve_names(&mut self, symbol_map: &crate::resolver::SymbolMap) {
        let current_module = self.module_path.clone();
        for s in &mut self.structs {
            s.resolve_names(&current_module, symbol_map);
        }
        for e in &mut self.enums {
            e.resolve_names(&current_module, symbol_map);
        }
        for t in &mut self.traits {
            t.resolve_names(&current_module, symbol_map);
        }
        for i in &mut self.impls {
            i.resolve_names(&current_module, symbol_map);
        }
        for f in &mut self.functions {
            f.resolve_names(&current_module, symbol_map);
        }
    }
}
