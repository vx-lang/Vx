//===- decl.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for declarations, including traits, structs, and functions.
//
//===----------------------------------------------------------------------===//

use super::*;

#[derive(Debug, PartialEq, Clone)]
pub enum GenericParam {
    Type { name: String, bound: Option<String> },
    Const { name: String, ty: Type },
}

impl GenericParam {
    pub fn name(&self) -> &str {
        match self {
            GenericParam::Type { name, .. } => name,
            GenericParam::Const { name, .. } => name,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub params: Vec<(String, Type)>,
    pub topology: Topology,
    pub return_type: Type,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub body: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub generics: Vec<GenericParam>,
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
    pub generics: Vec<GenericParam>,
    // (method_name, params, return_type)
    #[allow(clippy::type_complexity)]
    pub methods: Vec<(String, Vec<(String, Type)>, Type)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImplBlock {
    pub generics: Vec<GenericParam>,
    pub trait_name: Option<String>,
    pub target_type: Type,
    pub methods: Vec<Function>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportDecl {
    pub path: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroDefDecl {
    pub name: String,
    pub rules: Vec<MacroRule>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroRule {
    pub matcher: Vec<TokenTree>,
    pub transcriber: Vec<TokenTree>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub module_path: String,
    pub imports: Vec<ImportDecl>,
    pub macros: Vec<MacroDefDecl>,
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
