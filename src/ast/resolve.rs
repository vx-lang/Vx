//===- resolve.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Name resolution passes for the AST, resolving variable shadowing and lifetimes.
//
//===----------------------------------------------------------------------===//

use super::*;

impl Type {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        match self {
            Type::Struct(name, id) | Type::Enum(name, id) | Type::Generic(name, id) => {
                if let Some(tid) = mod_syms.and_then(|m| m.get(name)) {
                    *id = Some(*tid);
                }
            }
            Type::Tensor(_, dims, top) => {
                for dim in dims {
                    dim.resolve_names(mod_syms, symbol_map);
                }
                if let Some(t) = top {
                    t.resolve_names(mod_syms, symbol_map);
                }
            }
            Type::Ref(inner, _)
            | Type::Borrow(inner, _, _, _)
            | Type::Pointer(inner, _, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => {
                inner.resolve_names(mod_syms, symbol_map);
            }
            Type::GenericInstance(base, args) => {
                base.resolve_names(mod_syms, symbol_map);
                for arg in args {
                    arg.resolve_names(mod_syms, symbol_map);
                }
            }
            Type::Module(_, exported) => {
                for ty in exported.values_mut() {
                    ty.resolve_names(mod_syms, symbol_map);
                }
            }
            Type::Function(arg_tys, ret_ty) => {
                for t in arg_tys {
                    t.resolve_names(mod_syms, symbol_map);
                }
                ret_ty.resolve_names(mod_syms, symbol_map);
            }
            Type::Closure(arg_tys, ret_ty) => {
                for t in arg_tys {
                    t.resolve_names(mod_syms, symbol_map);
                }
                ret_ty.resolve_names(mod_syms, symbol_map);
            }
            Type::Const(_) => {}
            Type::Matrix | Type::Scalar(_) | Type::Simd(_, _) => {}
            Type::Unknown => {}
        }
    }
}

impl Topology {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        match self {
            Topology::NPU(e) | Topology::AccCore(e) => e.resolve_names(mod_syms, symbol_map),
            Topology::Slice(t, e1, e2) => {
                t.resolve_names(mod_syms, symbol_map);
                e1.resolve_names(mod_syms, symbol_map);
                e2.resolve_names(mod_syms, symbol_map);
            }
            _ => {}
        }
    }
}

impl Expr {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        match self {
            Expr::Transfer(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Expr::MemberAccess(e) => e.base.resolve_names(mod_syms, symbol_map),
            Expr::UnaryOp(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Expr::Borrow(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Expr::Dereference(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Expr::FunctionCall(e) => {
                for a in &mut e.args {
                    a.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::MethodCall(e) => {
                e.base.resolve_names(mod_syms, symbol_map);
                for a in &mut e.args {
                    a.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::Array(e) => {
                for a in &mut e.elements {
                    a.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::IndexAccess(e) => {
                e.base.resolve_names(mod_syms, symbol_map);
                e.index.resolve_names(mod_syms, symbol_map);
            }
            Expr::BinaryOp(e) => {
                e.lhs.resolve_names(mod_syms, symbol_map);
                e.rhs.resolve_names(mod_syms, symbol_map);
            }
            Expr::StructInit(e) => {
                for (_, ex) in &mut e.fields {
                    ex.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::UnsafeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(mod_syms, symbol_map);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::ComptimeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(mod_syms, symbol_map);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(mod_syms, symbol_map);
                }
            }
            Expr::If(e) => {
                e.cond.resolve_names(mod_syms, symbol_map);
                for s in &mut e.then_block {
                    s.resolve_names(mod_syms, symbol_map);
                }
                if let Some(eb) = &mut e.else_block {
                    for s in eb {
                        s.resolve_names(mod_syms, symbol_map);
                    }
                }
            }
            Expr::Topology(e) => e.top.resolve_names(mod_syms, symbol_map),
            Expr::SpawnOn(e) => {
                e.top.resolve_names(mod_syms, symbol_map);
                for s in &mut e.stmts {
                    s.resolve_names(mod_syms, symbol_map);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(mod_syms, symbol_map);
                }
            }
            _ => {}
        }
    }
}

impl Statement {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        match self {
            Statement::LetDecl(e) => {
                if let Some(t) = &mut e.ty_ann {
                    t.resolve_names(mod_syms, symbol_map);
                }
                e.expr.resolve_names(mod_syms, symbol_map);
            }
            Statement::Return(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Statement::ExprStmt(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Statement::Assert(e) => e.expr.resolve_names(mod_syms, symbol_map),
            Statement::ForLoop(e) => {
                e.iterable.resolve_names(mod_syms, symbol_map);
                for s in &mut e.body {
                    s.resolve_names(mod_syms, symbol_map);
                }
            }
            Statement::Assign(e) => {
                e.lhs.resolve_names(mod_syms, symbol_map);
                e.rhs.resolve_names(mod_syms, symbol_map);
            }
            Statement::CompoundAssign(e) => {
                e.lhs.resolve_names(mod_syms, symbol_map);
                e.rhs.resolve_names(mod_syms, symbol_map);
            }
            Statement::Loop(e) => {
                for s in &mut e.body {
                    s.resolve_names(mod_syms, symbol_map);
                }
            }
            Statement::Break(_) => {}
            Statement::Continue(_) => {}
            Statement::MacroCall(_) => panic!("Macros should be expanded before name resolution"),
        }
    }
}

impl Function {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        for (_, ty) in &mut self.params {
            ty.resolve_names(mod_syms, symbol_map);
        }
        self.return_type.resolve_names(mod_syms, symbol_map);
        for s in &mut self.body {
            s.resolve_names(mod_syms, symbol_map);
        }
    }
}

impl StructDecl {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        for (_, ty) in &mut self.fields {
            ty.resolve_names(mod_syms, symbol_map);
        }
    }
}

impl EnumDecl {
    pub fn resolve_names(
        &mut self,
        _mod_syms: Option<&crate::resolver::SymbolTable>,
        _symbol_map: &crate::resolver::SymbolMap,
    ) {
        // Enum variants are resolved during type checking or
        // as part of the Type::Enum resolution.
    }
}

impl TraitDecl {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        for method in &mut self.methods {
            for (_, p_ty) in &mut method.params {
                p_ty.resolve_names(mod_syms, symbol_map);
            }
            method.return_type.resolve_names(mod_syms, symbol_map);
        }
    }
}

impl ImplBlock {
    pub fn resolve_names(
        &mut self,
        mod_syms: Option<&crate::resolver::SymbolTable>,
        symbol_map: &crate::resolver::SymbolMap,
    ) {
        self.target_type.resolve_names(mod_syms, symbol_map);
        for f in &mut self.methods {
            f.resolve_names(mod_syms, symbol_map);
        }
    }
}

impl Program {
    pub fn resolve_names(&mut self, symbol_map: &crate::resolver::SymbolMap) {
        let mod_syms = symbol_map.get(&self.module_path);
        for s in &mut self.structs {
            s.resolve_names(mod_syms, symbol_map);
        }
        for e in &mut self.enums {
            e.resolve_names(mod_syms, symbol_map);
        }
        for t in &mut self.traits {
            t.resolve_names(mod_syms, symbol_map);
        }
        for i in &mut self.impls {
            i.resolve_names(mod_syms, symbol_map);
        }
        for f in &mut self.functions {
            f.resolve_names(mod_syms, symbol_map);
        }
    }
}
