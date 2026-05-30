use super::*;

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub generics: Vec<(String, Option<String>)>, // (TypeParamName, OptionalTraitBound)
    pub params: Vec<(String, Type)>,
    pub topology: Topology,
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
    pub generics: Vec<(String, Option<String>)>,
    pub variants: Vec<(String, Option<Vec<Type>>)>,
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
