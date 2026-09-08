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
use crate::gid::TypeId;

use crate::symbol::Symbol;
use std::collections::HashMap;

/// Everything the name-resolution walk needs to attach a GID to a nominal type: the *current*
/// module's symbol table (local defs), the full cross-module symbol map, and this module's import
/// index. Bundled into one reference so the recursive `resolve_names` walk threads a single value
/// and cross-module lookup is centralized in [`ResolutionScope::resolve_nominal`] rather than
/// bolted onto each AST arm.
pub struct ResolutionScope<'a> {
    current: Option<&'a SymbolTable>,
    /// The current module's path (`symbol_map` key for `current`). Lets a qualified reference to the
    /// *current* module (`A::Foo` inside module `A`) short-circuit to the local table instead of a
    /// second `symbol_map` lookup — module-local references are the common case. Owned (a cheap
    /// `Arc<str>` bump) so the scope doesn't borrow the `Program` during the mutable walk.
    current_path: Option<Symbol>,
    symbol_map: &'a SymbolMap,
    imports: ImportIndex,
    /// Every topology this compilation declares, indexed by kind, so a placement written as a
    /// device can be given the space that device actually holds. Owned for the same reason
    /// `ImportIndex` is: the scope must not borrow the `Program` during the mutable walk, and a
    /// `--machine` file's declarations arrive from a *different* program in the array anyway.
    topologies: HashMap<TopologyKind, crate::arch::TopologyDescriptor>,
}

/// A module's `import` declarations, pre-indexed for the two resolution shapes.
struct ImportIndex {
    /// `import a::b::Name;` -> an *unqualified* `Name` resolves to module `a::b`.
    leaf_to_module: HashMap<Symbol, Symbol>,
    /// `import a::b;` -> the last segment `b` aliases module path `a::b`; used to expand the leading
    /// segment of a *qualified* reference (`b::Name`) to its full defining-module path.
    alias_to_module: HashMap<Symbol, Symbol>,
}

fn join_path(segs: &[Symbol]) -> String {
    segs.iter()
        .map(|s| s.as_ref())
        .collect::<Vec<_>>()
        .join("::")
}

impl<'a> ResolutionScope<'a> {
    fn new(
        current: Option<&'a SymbolTable>,
        current_path: Option<Symbol>,
        symbol_map: &'a SymbolMap,
        imports: &[crate::syntax::ImportDecl],
        topologies: &[crate::arch::TopologyDecl],
    ) -> Self {
        let mut leaf_to_module = HashMap::new();
        let mut alias_to_module = HashMap::new();
        for imp in imports {
            // `import a::b::Name;` -> unqualified `Name` comes from module `a::b`.
            if let Some((leaf, module_segs)) = imp.path.split_last() {
                if !module_segs.is_empty() {
                    leaf_to_module
                        .insert(leaf.clone(), Symbol::from(join_path(module_segs).as_ref()));
                }
                // The whole path also names a module reachable by its last segment as an alias:
                // `import a::b;` -> alias `b` = module `a::b`.
                alias_to_module.insert(leaf.clone(), Symbol::from(join_path(&imp.path).as_ref()));
            }
        }
        Self {
            current,
            current_path,
            symbol_map,
            imports: ImportIndex {
                leaf_to_module,
                alias_to_module,
            },
            topologies: topologies
                .iter()
                .map(|d| (TopologyKind::Custom(d.name.clone()), d.descriptor.clone()))
                .collect(),
        }
    }

    /// Resolve a nominal type name (possibly a `::`-qualified path) to its *defining* module's GID.
    /// Qualified names route straight to the named module; unqualified names take local definitions
    /// first, then imported names. Returns `None` (leaving the id unresolved) if nothing matches.
    fn resolve_nominal(&self, name: &Symbol) -> Option<TypeId> {
        let s = name.as_ref();
        if let Some(idx) = s.rfind("::") {
            // Qualified `Mod::...::Name`: the leaf is the type, the prefix names the module. Never
            // fall back to a same-named local type -- a qualified reference is explicit (#194).
            let module_part = &s[..idx];
            let leaf = Symbol::from(&s[idx + 2..]);
            let module_key = self.resolve_module_path(module_part);
            // Fast path: a qualified reference to the *current* module hits the local table
            // directly, skipping the outer `symbol_map` lookup (`current` *is*
            // `symbol_map[current_path]`). Module-local references dominate.
            if self.current_path.as_deref() == Some(&*module_key) {
                return self.current.and_then(|m| m.get(&leaf)).copied();
            }
            return self.symbol_map.get(&module_key)?.get(&leaf).copied();
        }
        // Unqualified: a local definition wins (as before), else an imported name.
        if let Some(id) = self.current.and_then(|m| m.get(name)) {
            return Some(*id);
        }
        let module_key = self.imports.leaf_to_module.get(name)?;
        self.symbol_map.get(module_key)?.get(name).copied()
    }

    /// Turn the module portion of a qualified path into a real `symbol_map` key: try it literally
    /// (already fully qualified, e.g. `crate::a`), else expand its leading segment through an
    /// `import a::b;` alias (`b::...` -> `a::b::...`).
    fn resolve_module_path(&self, module_part: &str) -> Symbol {
        let literal = Symbol::from(module_part);
        if self.symbol_map.contains_key(&literal) {
            return literal;
        }
        let mut segs = module_part.split("::");
        if let Some(head) = segs.next() {
            if let Some(full_head) = self.imports.alias_to_module.get(&Symbol::from(head)) {
                let tail: Vec<&str> = segs.collect();
                let expanded = if tail.is_empty() {
                    full_head.as_ref().to_string()
                } else {
                    format!("{}::{}", full_head.as_ref(), tail.join("::"))
                };
                return Symbol::from(expanded.as_ref());
            }
        }
        literal
    }
}

impl Type {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        match self {
            Type::Struct(name, id) | Type::Enum(name, id) => {
                if let Some(tid) = scope.resolve_nominal(name) {
                    *id = Some(tid);
                }
            }
            // A `Type::Generic` is a generic *parameter* — a type variable the parser already
            // classified as such from the in-scope generic list (`parser/types.rs`). It is not a
            // nominal type, so it must NOT bind to a same-named struct/enum (that would conflate a
            // type variable with a concrete type). Its `id` is unused downstream (every consumer
            // binds it to `_`); leave it None.
            Type::Generic(..) => {}
            Type::Tensor(_, dims, top) => {
                for dim in dims.iter_mut().filter_map(|d| match d {
                    Dim::Static(e) => Some(e),
                    Dim::Dyn => None,
                }) {
                    dim.resolve_names(scope);
                }
                if let Some(p) = top {
                    p.resolve_names(scope);
                }
            }
            Type::Ref(inner, _)
            | Type::Borrow { inner, .. }
            | Type::Pointer(inner, _, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => {
                inner.resolve_names(scope);
            }
            Type::GenericInstance(base, args) => {
                base.resolve_names(scope);
                for arg in args {
                    arg.resolve_names(scope);
                }
            }
            Type::Module(_, exported) => {
                for ty in exported.values_mut() {
                    ty.resolve_names(scope);
                }
            }
            Type::Function(arg_tys, ret_ty, _) => {
                for t in arg_tys {
                    t.resolve_names(scope);
                }
                ret_ty.resolve_names(scope);
            }
            Type::Closure(arg_tys, ret_ty) => {
                for t in arg_tys {
                    t.resolve_names(scope);
                }
                ret_ty.resolve_names(scope);
            }
            Type::Const(_) => {}
            Type::Matrix | Type::Scalar(_) | Type::Simd(_, _) => {}
            Type::Unknown => {}
        }
    }
}

impl Placement {
    /// Resolve the names inside the device, then fill in whichever projection the source did not
    /// write. This is the pass with program-wide scope, so it is the first point at which
    /// `Topology SmemDev { memory: Memory::SMEM }` can say what `Topology::SmemDev` holds.
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        self.topology.resolve_names(scope);
        self.complete(&scope.topologies);
    }
}

impl Topology {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        match self {
            Topology::NPU(e) | Topology::AccCore(e) => e.resolve_names(scope),
            Topology::Slice(t, e1, e2) => {
                t.resolve_names(scope);
                e1.resolve_names(scope);
                e2.resolve_names(scope);
            }
            _ => {}
        }
    }
}

impl Expr {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        match self {
            Expr::Transfer(e) => e.expr.resolve_names(scope),
            Expr::MemberAccess(e) => e.base.resolve_names(scope),
            Expr::UnaryOp(e) => e.expr.resolve_names(scope),
            Expr::Borrow(e) => e.expr.resolve_names(scope),
            Expr::Dereference(e) => e.expr.resolve_names(scope),
            Expr::FunctionCall(e) => {
                for a in &mut e.args {
                    a.resolve_names(scope);
                }
            }
            Expr::MethodCall(e) => {
                e.base.resolve_names(scope);
                for a in &mut e.args {
                    a.resolve_names(scope);
                }
            }
            Expr::Array(e) => {
                for a in &mut e.elements {
                    a.resolve_names(scope);
                }
            }
            Expr::IndexAccess(e) => {
                e.base.resolve_names(scope);
                e.index.resolve_names(scope);
            }
            Expr::BinaryOp(e) => {
                e.lhs.resolve_names(scope);
                e.rhs.resolve_names(scope);
            }
            Expr::StructInit(e) => {
                for (_, ex) in &mut e.fields {
                    ex.resolve_names(scope);
                }
            }
            Expr::UnsafeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(scope);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(scope);
                }
            }
            Expr::ComptimeBlock(e) => {
                for s in &mut e.stmts {
                    s.resolve_names(scope);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(scope);
                }
            }
            Expr::If(e) => {
                e.cond.resolve_names(scope);
                for s in &mut e.then_block {
                    s.resolve_names(scope);
                }
                if let Some(eb) = &mut e.else_block {
                    for s in eb {
                        s.resolve_names(scope);
                    }
                }
            }
            Expr::Topology(e) => e.top.resolve_names(scope),
            Expr::SpawnOn(e) => {
                e.top.resolve_names(scope);
                for s in &mut e.stmts {
                    s.resolve_names(scope);
                }
                if let Some(r) = &mut e.ret {
                    r.resolve_names(scope);
                }
            }
            _ => {}
        }
    }
}

impl Statement {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        match self {
            Statement::LetDecl(e) => {
                if let Some(t) = &mut e.ty_ann {
                    t.resolve_names(scope);
                }
                e.expr.resolve_names(scope);
            }
            Statement::Return(e) => e.expr.resolve_names(scope),
            Statement::ExprStmt(e) => e.expr.resolve_names(scope),
            Statement::Assert(e) => e.expr.resolve_names(scope),
            Statement::ForLoop(e) => {
                e.iterable.resolve_names(scope);
                for s in &mut e.body {
                    s.resolve_names(scope);
                }
            }
            Statement::Assign(e) => {
                e.lhs.resolve_names(scope);
                e.rhs.resolve_names(scope);
            }
            Statement::CompoundAssign(e) => {
                e.lhs.resolve_names(scope);
                e.rhs.resolve_names(scope);
            }
            Statement::Loop(e) => {
                for s in &mut e.body {
                    s.resolve_names(scope);
                }
            }
            Statement::Break(_) => {}
            Statement::Continue(_) => {}
            Statement::MacroCall(_) => panic!("Macros should be expanded before name resolution"),
            Statement::Error(_) => {}
        }
    }
}

impl Function {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        for (_, ty) in &mut self.params {
            ty.resolve_names(scope);
        }
        self.return_type.resolve_names(scope);
        for s in &mut self.body {
            s.resolve_names(scope);
        }
    }
}

impl StructDecl {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        for (_, ty) in &mut self.fields {
            ty.resolve_names(scope);
        }
    }
}

impl EnumDecl {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        // Resolve nominal GIDs inside variant payloads (`Node(Tree, i32)`), so the frozen registry
        // sees enum by-value dependencies (recursive-layout / cross-module cycle detection) and the
        // flat type stream carries the payload identity. Previously a no-op — fine for the AST
        // type-checker, which re-resolves, but the flat pipeline relies on these GIDs being attached.
        for (_, payload) in &mut self.variants {
            if let Some(types) = payload {
                for ty in types {
                    ty.resolve_names(scope);
                }
            }
        }
    }
}

impl TraitDecl {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        for method in &mut self.methods {
            for (_, p_ty) in &mut method.params {
                p_ty.resolve_names(scope);
            }
            method.return_type.resolve_names(scope);
        }
    }
}

impl ImplBlock {
    pub fn resolve_names(&mut self, scope: &ResolutionScope) {
        self.target_type.resolve_names(scope);
        for f in &mut self.methods {
            f.resolve_names(scope);
        }
    }
}

impl Program {
    /// `topologies` is every topology *the compilation* declares, which is not the same as this
    /// module's: a `--machine` file is a separate program in the array and is where a fleet's
    /// topologies are declared. This module's own are folded in here, so a caller with a single
    /// module can pass `&[]`.
    pub fn resolve_names(
        &mut self,
        symbol_map: &crate::syntax::SymbolMap,
        topologies: &[crate::arch::TopologyDecl],
    ) {
        // Build the scope up front: it borrows only `symbol_map` (and owns indexes cloned from the
        // imports and topologies), so it no longer borrows `self` and we can mutably walk the
        // declarations below.
        let current = symbol_map.get(&self.module_path);
        let mut decls = topologies.to_vec();
        decls.extend(self.topologies.iter().cloned());
        let scope = ResolutionScope::new(
            current,
            Some(self.module_path.clone()),
            symbol_map,
            &self.imports,
            &decls,
        );
        for s in &mut self.structs {
            s.resolve_names(&scope);
        }
        for e in &mut self.enums {
            e.resolve_names(&scope);
        }
        for t in &mut self.traits {
            t.resolve_names(&scope);
        }
        for i in &mut self.impls {
            i.resolve_names(&scope);
        }
        for f in &mut self.functions {
            f.resolve_names(&scope);
        }
    }
}
