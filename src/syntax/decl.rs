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
use crate::symbol::Symbol;

#[derive(Debug, PartialEq, Clone)]
pub enum GenericParam {
    Type { name: Symbol, bound: Option<Symbol> },
    Const { name: Symbol, ty: Type },
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
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub params: Vec<(Symbol, Type)>,
    pub topology: Topology,
    pub return_type: Type,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    /// `where Transfer<S, D>` constraints: pairs of topology names (generic topology
    /// variables or concrete topologies) that must have a transfer path in the cost
    /// graph. Discharged at each generic call once the variables are bound.
    pub where_transfers: Vec<(Symbol, Symbol)>,
    pub body: Vec<Statement>,
    pub doc_comment: Option<String>,
}

impl Function {
    pub fn clone_signature(&self, preserve_body: bool) -> Self {
        Self {
            name: self.name.clone(),
            generics: self.generics.clone(),
            params: self.params.clone(),
            topology: self.topology.clone(),
            return_type: self.return_type.clone(),
            requires: self.requires.clone(),
            ensures: self.ensures.clone(),
            where_transfers: self.where_transfers.clone(),
            body: if preserve_body || !self.generics.is_empty() {
                self.body.clone()
            } else {
                Vec::new()
            },
            doc_comment: self.doc_comment.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<(Symbol, Type)>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<(Symbol, Option<Vec<Type>>)>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExternDecl {
    pub name: Symbol,
    pub is_safe: bool,
    pub params: Vec<(Symbol, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MethodSignature {
    pub name: Symbol,
    pub params: Vec<(Symbol, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TraitDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub methods: Vec<MethodSignature>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImplBlock {
    pub generics: Vec<GenericParam>,
    pub trait_name: Option<Symbol>,
    pub target_type: Type,
    pub methods: Vec<Function>,
}

impl ImplBlock {
    pub fn clone_signature(&self) -> Self {
        Self {
            generics: self.generics.clone(),
            trait_name: self.trait_name.clone(),
            target_type: self.target_type.clone(),
            // Always preserve method bodies: methods only exist in impl blocks, so
            // method-call monomorphization clones the body from the type-check env
            // (env.impls). Dropping it (as the signature clone does for free
            // functions, whose bodies come from `program.functions`) yields an
            // empty method body and invalid IR. See GitHub #146.
            methods: self
                .methods
                .iter()
                .map(|m| m.clone_signature(true))
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportDecl {
    pub path: Vec<Symbol>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroDefDecl {
    pub name: Symbol,
    pub rules: Vec<MacroRule>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroRule {
    pub matcher: Vec<TokenTree>,
    pub transcriber: Vec<TokenTree>,
}

/// A byte quantity, e.g. `256 KB`, normalized to bytes (binary multipliers: `KB` = 1024).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ByteSize(pub u64);

/// The denominator of a bandwidth rate.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RatePer {
    /// `/s`
    Second,
    /// `/cyc`
    Cycle,
}

/// A bandwidth, e.g. `8 TB/s` or `128 B/cyc`; the numerator is normalized to bytes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Bandwidth {
    pub bytes: u64,
    pub per: RatePer,
}

/// How a memory space is managed (M5 uses this to decide the transfer obligation).
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Management {
    /// Programmer-managed: movement in/out requires an explicit `transfer`.
    Explicit,
    /// Hardware-cached: movement may be implicit. The default.
    #[default]
    Cached,
}

/// A first-class memory-space declaration:
/// `Memory <Name> { within:, capacity:, bandwidth:, managed:, granule: }`.
///
/// Carried on the AST (`Program.memories`) and indexed by `GlobalAstEnv` for semantic
/// analysis; codegen reads it from the `Program`. Deliberately *not* registered in a
/// process-global registry (unlike topologies), so declarations cannot leak between
/// compilations. See `docs/discussions/implementation_plans/first_class_memory_spaces.md`.
#[derive(Debug, PartialEq, Clone)]
pub struct MemoryDecl {
    pub name: Symbol,
    /// `within: Memory::X` — the parent space in the hierarchy tree; `None` for a root.
    pub parent: Option<MemorySpace>,
    pub capacity: Option<ByteSize>,
    pub bandwidth: Option<Bandwidth>,
    pub managed: Management,
    pub granule: Option<ByteSize>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub module_path: Symbol,
    pub imports: Vec<ImportDecl>,
    pub macros: Vec<MacroDefDecl>,
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub enums: Vec<EnumDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    /// Names of user-defined topologies declared in this program (`Topology <Name> { ... }`).
    /// Descriptors live in the (process-global) topology registry; this per-program list
    /// scopes coherence checking to the topologies this compilation actually declared, so
    /// one program's declarations don't leak diagnostics into another's.
    pub topologies: Vec<Symbol>,
    /// User-defined memory spaces declared in this program (`Memory <Name> { ... }`). Unlike
    /// topologies, the full descriptors live here on the AST (not a global registry); sema
    /// indexes them via `GlobalAstEnv` and codegen reads them from the `Program`.
    pub memories: Vec<MemoryDecl>,
}

pub type VxModule = Program;
pub type VxFunction = Function;

impl Program {
    pub fn add(&mut self, func: VxFunction) {
        self.functions.push(func);
    }

    pub fn clone_signature(&self) -> Self {
        Self {
            module_path: self.module_path.clone(),
            imports: self.imports.clone(),
            macros: self.macros.clone(),
            externs: self.externs.clone(),
            structs: self.structs.clone(),
            enums: self.enums.clone(),
            traits: self.traits.clone(),
            impls: self.impls.iter().map(|i| i.clone_signature()).collect(),
            functions: self
                .functions
                .iter()
                .map(|f| f.clone_signature(false))
                .collect(),
            topologies: self.topologies.clone(),
            memories: self.memories.clone(),
        }
    }
}
