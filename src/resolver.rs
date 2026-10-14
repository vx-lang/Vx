//===- resolver.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Name Resolution pass for the Vx compiler.
// It maps local and global identifiers to their corresponding definitions in the AST,
// handling scope shadowing, namespace resolution, and ensuring that all referenced
// variables and functions actually exist.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hash::{compute_module_hash, DefPath};
use crate::symbol::Symbol;
use crate::syntax::{
    BinaryOp, ConstDecl, Dim, ElementType, Expr, GenericParam, NumberExpr, Span, SymbolMap, Type,
    UnaryOp, VxModule,
};
use std::collections::{HashMap, HashSet};

/// Iterates through all parsed modules sequentially and computes their deterministic
/// 256-bit TypeId for every top-level struct, enum, and trait.
pub fn build_symbol_map(modules: &[VxModule]) -> SymbolMap {
    use rayon::prelude::*;

    modules
        .par_iter()
        .map(|module| {
            let mut module_symbols = HashMap::new();
            let module_hash = compute_module_hash(&module.module_path);

            let mut process_decl = |name: &crate::symbol::Symbol| {
                let sym_hash = DefPath::Named(name.as_ref()).compute_symbol_hash();
                let tid = TypeId::new(module_hash, sym_hash, 0, 0);
                module_symbols.insert(name.clone(), tid);
            };

            module.structs.iter().for_each(|d| process_decl(&d.name));
            module.enums.iter().for_each(|d| process_decl(&d.name));
            module.traits.iter().for_each(|d| process_decl(&d.name));

            (module.module_path.clone(), module_symbols)
        })
        .collect()
}

/// Every topology the compilation declares, gathered across its modules.
///
/// Companion to `build_symbol_map` and used the same way: computed once from `&[VxModule]`, then
/// handed to each module's `resolve_names` during the mutable walk. Owned because that walk takes
/// the array by `&mut`, and because a topology declared in a `--machine` file has to reach the
/// module that names it, which a per-module view cannot do.
pub fn collect_topologies(modules: &[VxModule]) -> Vec<crate::arch::TopologyDecl> {
    modules
        .iter()
        .flat_map(|m| m.topologies.iter().cloned())
        .collect()
}

/// Replace `Self` in a trait method's signature with the type the impl is for. A trait
/// writes `fn eq(self : Self, other : Self) -> bool`; the copy handed to `impl Eq for i32`
/// has to say `i32`, because from here on it is an ordinary method and nothing downstream
/// knows what `Self` was.
fn substitute_self(ty: &crate::syntax::Type, target: &crate::syntax::Type) -> crate::syntax::Type {
    use crate::syntax::Type;
    let is_self = |name: &crate::symbol::Symbol| name.as_ref() == "Self";
    match ty {
        Type::Struct(name, _) | Type::Generic(name, _) if is_self(name) => target.clone(),
        Type::Ref(inner, space) => {
            Type::Ref(Box::new(substitute_self(inner, target)), space.clone())
        }
        Type::Borrow {
            inner,
            mem_space,
            is_mut,
            region_id,
        } => Type::Borrow {
            inner: Box::new(substitute_self(inner, target)),
            mem_space: mem_space.clone(),
            is_mut: *is_mut,
            region_id: *region_id,
        },
        Type::Pointer(inner, space, is_mut) => Type::Pointer(
            Box::new(substitute_self(inner, target)),
            space.clone(),
            *is_mut,
        ),
        Type::Verified(inner) => Type::Verified(Box::new(substitute_self(inner, target))),
        Type::Pinned(inner, topo) => {
            Type::Pinned(Box::new(substitute_self(inner, target)), topo.clone())
        }
        Type::GenericInstance(base, args) => Type::GenericInstance(
            Box::new(substitute_self(base, target)),
            args.iter().map(|a| substitute_self(a, target)).collect(),
        ),
        // Nothing else can hold a `Self`: a scalar, a tensor and a function type are built
        // from element types and shapes rather than from nominal names.
        other => other.clone(),
    }
}

/// Give every impl of a trait the trait's default method bodies, for the methods it does
/// not write itself.
///
/// This runs over the parsed modules before anything reads them, so from here on an impl
/// block holds every method it is supposed to have and nothing downstream -- name
/// resolution, the checker, either code generator -- needs to know that defaults exist.
/// That is also why it is a rewrite rather than a fallback at method-lookup time: the
/// lookup is not the only reader, and codegen walks the impl's methods directly.
///
/// A trait's defaults are collected across every module first, because the trait and the
/// impl need not be in the same one.
#[derive(Default)]
pub struct TraitDefaults {
    /// Each trait's parameters and the methods it gives a default body.
    pub methods: std::collections::HashMap<
        crate::symbol::Symbol,
        (
            Vec<crate::syntax::GenericParam>,
            Vec<crate::syntax::MethodSignature>,
        ),
    >,
    /// Every impl's associated-type bindings, by the type it is for. `Self::Item` in an impl
    /// that binds no `Item` of its own -- `impl DoubleEndedIterator for Range` -- means the one
    /// another impl for the same type binds, as a supertrait's does in Rust.
    pub bindings: Vec<(
        crate::syntax::Type,
        Vec<(crate::symbol::Symbol, crate::syntax::Type)>,
    )>,
}

/// The trait methods that carry a default body, across every module.
///
/// Global on purpose: an impl and the trait it implements need not share a module, and an
/// impl a macro produces is as entitled to the defaults as one written by hand.
pub fn collect_trait_defaults<'p>(
    programs: impl IntoIterator<Item = &'p crate::syntax::Program>,
) -> TraitDefaults {
    let mut defaults = TraitDefaults::default();
    for program in programs {
        for block in &program.impls {
            if !block.assoc_bindings.is_empty() {
                defaults
                    .bindings
                    .push((block.target_type.clone(), block.assoc_bindings.clone()));
            }
        }
        for decl in &program.traits {
            let with_bodies: Vec<crate::syntax::MethodSignature> = decl
                .methods
                .iter()
                .filter(|m| m.default_body.is_some())
                .cloned()
                .collect();
            if !with_bodies.is_empty() {
                defaults
                    .methods
                    .insert(decl.name.clone(), (decl.generics.clone(), with_bodies));
            }
        }
    }
    defaults
}

/// What an impl binds each of the trait's associated types to, keyed by the way a signature
/// spells it: `type Item = i64;` becomes `Self::Item -> i64`.
///
/// A name the block does not bind comes from another impl for the same type, when one binds it.
fn assoc_substitution(
    block: &crate::syntax::ImplBlock,
    defaults: &TraitDefaults,
) -> std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Type> {
    let others = defaults
        .bindings
        .iter()
        .filter(|(target, _)| *target == block.target_type)
        .flat_map(|(_, bound)| bound.iter());
    let mut subst = std::collections::HashMap::new();
    for (name, bound) in others.chain(block.assoc_bindings.iter()) {
        subst.insert(
            crate::symbol::Symbol::from(format!("Self::{}", name.as_ref()).as_str()),
            bound.clone(),
        );
    }
    subst
}

/// Replace `Self::Item` in an impl's methods with the type that impl bound it to.
///
/// This runs beside the trait-default copy and for the same reason `Self` itself is
/// substituted there: from here on the impl's methods are ordinary methods, and nothing
/// downstream -- name resolution, the checker, either code generator -- needs to know that
/// an associated type was ever written.
///
/// Bodies as well as signatures, so a default can write `Option<Self::Item>::None()`.
pub fn bind_associated_types_in(program: &mut crate::syntax::Program, defaults: &TraitDefaults) {
    for block in &mut program.impls {
        let subst = assoc_substitution(block, defaults);
        if subst.is_empty() {
            continue;
        }
        for method in &mut block.methods {
            for (_, param_ty) in &mut method.params {
                *param_ty = param_ty.substitute(&subst);
            }
            method.return_type = method.return_type.substitute(&subst);
            method.body = method.body.iter().map(|s| s.substitute(&subst)).collect();
        }
    }
}

pub fn fill_trait_defaults_in(program: &mut crate::syntax::Program, defaults: &TraitDefaults) {
    if defaults.methods.is_empty() {
        return;
    }
    for block in &mut program.impls {
        let Some(trait_name) = block.trait_name.clone() else {
            continue;
        };
        let Some((trait_generics, trait_methods)) = defaults.methods.get(&trait_name) else {
            continue;
        };
        // `impl Iterator<i64> for Range` binds the trait's `Item` to `i64`. Without this
        // the copied signature still said `Option<Item>` while the body it came with
        // produced `Option<i64>`, so the default did not type-check in the impl.
        let mut trait_subst: std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Type> =
            std::collections::HashMap::new();
        for (param, arg) in trait_generics.iter().zip(block.trait_args.iter()) {
            trait_subst.insert(param.name().into(), arg.clone());
        }
        for signature in trait_methods {
            if block
                .methods
                .iter()
                .any(|m| m.name.as_ref() == signature.name.as_ref())
            {
                continue;
            }
            // A method of a generic impl has the block's parameters in scope, which the
            // parser arranges for a hand-written one. A default arrives here after that,
            // so it is done again: without it the copied body is checked with `I`
            // unbound and reports "Method 'next' not found on type I". A method parameter
            // named like one of the block's is renamed, or `zip<U>` copied into
            // `impl<I, U> Iterator for Map<I, ..U..>` would make the two `U`s one type.
            let in_block = |n: &str| block.generics.iter().any(|b| b.name() == n);
            let taken = |n: &str| in_block(n) || signature.generics.iter().any(|g| g.name() == n);
            let mut renames = std::collections::HashMap::new();
            for g in &signature.generics {
                if in_block(g.name()) {
                    assert!(
                        matches!(g, crate::syntax::GenericParam::Type { .. }),
                        "a const parameter of a trait default is named like one of the impl's"
                    );
                    let mut fresh = format!("{}_{}", g.name(), signature.name.as_ref());
                    while taken(&fresh) {
                        fresh.push('_');
                    }
                    renames.insert(
                        crate::symbol::Symbol::from(g.name()),
                        crate::syntax::Type::Generic(fresh.as_str().into(), None),
                    );
                }
            }
            let mut generics: Vec<crate::syntax::GenericParam> = block.generics.clone();
            generics.extend(signature.generics.iter().map(|g| {
                match g {
                    crate::syntax::GenericParam::Type { name, bounds } => {
                        crate::syntax::GenericParam::Type {
                            name: match renames.get(name) {
                                Some(crate::syntax::Type::Generic(fresh, _)) => fresh.clone(),
                                _ => name.clone(),
                            },
                            bounds: bounds
                                .iter()
                                .map(|b| crate::syntax::TraitBound {
                                    trait_name: b.trait_name.clone(),
                                    args: b.args.iter().map(|a| a.substitute(&renames)).collect(),
                                    bindings: b
                                        .bindings
                                        .iter()
                                        .map(|(n, t)| (n.clone(), t.substitute(&renames)))
                                        .collect(),
                                })
                                .collect(),
                        }
                    }
                    other => other.clone(),
                }
            }));
            // Renamed first, because `Self` names the block's own `U`.
            let signature_type = |t: &crate::syntax::Type| {
                substitute_self(
                    &t.substitute(&renames).substitute(&trait_subst),
                    &block.target_type,
                )
            };
            // `Self` in the body too, where an adaptor names its own type: `Map<Self, ..> {..}`,
            // and the trait's own parameters, as the signature gets them: `Option<Item>::None()`.
            let mut self_subst = trait_subst.clone();
            self_subst.insert("Self".into(), block.target_type.clone());
            let body = signature
                .default_body
                .clone()
                .expect("only methods with a default body are collected")
                .iter()
                .map(|s| s.substitute(&renames).substitute(&self_subst))
                .collect();
            block.copied_defaults.push(signature.name.clone());
            block.methods.push(crate::syntax::Function {
                name: signature.name.clone(),
                generics,
                params: signature
                    .params
                    .iter()
                    .map(|(n, t)| (n.clone(), signature_type(t)))
                    .collect(),
                mut_params: signature.mut_params.clone(),
                topology: crate::syntax::Topology::CPU,
                return_type: signature_type(&signature.return_type),
                requires: Vec::new(),
                ensures: Vec::new(),
                where_transfers: Vec::new(),
                is_unsafe: false,
                body,
                doc_comment: None,
            });
        }
    }
}

/// Replace each integer module `const` written in a declaration's tensor extents, or as a const
/// generic argument of a type, with its value. With `const N : i32 = 4;`, `fn f(t : Tensor<f32,
/// [N]>)` becomes `[4]`: the checker and both code generators read an extent as a number.
///
/// Only signatures, struct fields and enum payloads are folded here, where no local variable can
/// be in scope. A generic parameter with the same name as a `const` keeps its meaning. The
/// checker folds the types written inside function bodies, where it knows the local names.
pub fn fold_const_extents(modules: &mut [VxModule]) {
    let values = const_extent_values(modules);
    if values.is_empty() {
        return;
    }
    let type_names: HashSet<Symbol> = modules
        .iter()
        .flat_map(|m| {
            let structs = m.structs.iter().map(|s| s.name.clone());
            let enums = m.enums.iter().map(|e| e.name.clone());
            structs
                .chain(enums)
                .chain(m.traits.iter().map(|t| t.name.clone()))
        })
        .collect();
    let fold = |generics: &[&[GenericParam]]| ConstExtents {
        values: &values,
        type_names: &type_names,
        hidden: generics
            .iter()
            .flat_map(|list| list.iter())
            .map(|g| match g {
                GenericParam::Type { name, .. } | GenericParam::Const { name, .. } => name.clone(),
            })
            .collect(),
    };
    for m in modules.iter_mut() {
        for s in &mut m.structs {
            let f = fold(&[&s.generics]);
            for (_, ty) in &mut s.fields {
                f.fold_type(ty);
            }
        }
        for e in &mut m.enums {
            let f = fold(&[&e.generics]);
            for ty in e
                .variants
                .iter_mut()
                .filter_map(|(_, p)| p.as_mut())
                .flatten()
            {
                f.fold_type(ty);
            }
        }
        for t in &mut m.traits {
            for sig in &mut t.methods {
                let f = fold(&[&t.generics, &sig.generics]);
                for (_, ty) in &mut sig.params {
                    f.fold_type(ty);
                }
                f.fold_type(&mut sig.return_type);
            }
        }
        for i in &mut m.impls {
            fold(&[&i.generics]).fold_type(&mut i.target_type);
            for func in &mut i.methods {
                fold(&[&i.generics, &func.generics]).fold_signature(func);
            }
        }
        for func in &mut m.functions {
            fold(&[&func.generics]).fold_signature(func);
        }
        for ext in &mut m.externs {
            let f = fold(&[]);
            for (_, ty) in &mut ext.params {
                f.fold_type(ty);
            }
            f.fold_type(&mut ext.return_type);
        }
    }
}

/// Each integer module `const` by name, with its value. A name declared twice with different
/// values is left out: which one is meant is for the checker to report.
fn const_extent_values(modules: &[VxModule]) -> HashMap<Symbol, i128> {
    let mut decls: HashMap<Symbol, Option<&ConstDecl>> = HashMap::new();
    for c in modules.iter().flat_map(|m| m.consts.iter()) {
        decls
            .entry(c.name.clone())
            .and_modify(|d| {
                if *d != Some(c) {
                    *d = None;
                }
            })
            .or_insert(Some(c));
    }
    let decls: HashMap<Symbol, &ConstDecl> = decls
        .into_iter()
        .filter_map(|(name, d)| Some((name, d?)))
        .collect();
    decls
        .keys()
        .filter_map(|name| Some((name.clone(), const_int(name, &decls, &mut Vec::new())?)))
        .collect()
}

/// The value of the integer `const` `name`, when it fits the type it is declared with.
/// `visiting` holds the `const`s being worked out, so one defined through itself has none.
fn const_int(
    name: &Symbol,
    decls: &HashMap<Symbol, &ConstDecl>,
    visiting: &mut Vec<Symbol>,
) -> Option<i128> {
    let decl = decls.get(name)?;
    let Type::Scalar(elem) = &decl.ty else {
        return None;
    };
    let bits = elem.int_bits().filter(|_| *elem != ElementType::Bool)?;
    if visiting.contains(name) {
        return None;
    }
    visiting.push(name.clone());
    let value = int_value(&decl.expr, decls, visiting);
    visiting.pop();
    let value = value?;
    let fits = match (elem.is_signed_int(), bits) {
        (true, 128) => true,
        (false, 128) => value >= 0,
        (true, b) => (-(1i128 << (b - 1))..(1i128 << (b - 1))).contains(&value),
        (false, b) => (0..(1i128 << b)).contains(&value),
    };
    fits.then_some(value)
}

/// `expr` worked out as an integer, reading the integer `const`s it names from `decls`.
/// Overflow and division by zero give `None`.
fn int_value(
    expr: &Expr,
    decls: &HashMap<Symbol, &ConstDecl>,
    visiting: &mut Vec<Symbol>,
) -> Option<i128> {
    match expr {
        Expr::Number(n) => n.value.parse::<i128>().ok(),
        Expr::Identifier(id) => const_int(&id.name, decls, visiting),
        Expr::UnaryOp(u) if matches!(u.op, UnaryOp::Neg) => {
            int_value(&u.expr, decls, visiting)?.checked_neg()
        }
        Expr::BinaryOp(b) => {
            let l = int_value(&b.lhs, decls, visiting)?;
            let r = int_value(&b.rhs, decls, visiting)?;
            match b.op {
                BinaryOp::Add => l.checked_add(r),
                BinaryOp::Sub => l.checked_sub(r),
                BinaryOp::Mul => l.checked_mul(r),
                BinaryOp::Div => l.checked_div(r),
                BinaryOp::Rem => l.checked_rem(r),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Folds the module `const`s in the types of one declaration.
struct ConstExtents<'a> {
    values: &'a HashMap<Symbol, i128>,
    type_names: &'a HashSet<Symbol>,
    /// The declaration's generic parameters. A `const` with the same name does not replace one.
    hidden: Vec<Symbol>,
}

impl ConstExtents<'_> {
    /// The value an extent may use for `name`. A negative one is left as the name, for the
    /// checker to report.
    fn value(&self, name: &Symbol) -> Option<i128> {
        if self.hidden.contains(name) {
            return None;
        }
        self.values.get(name).copied().filter(|v| *v >= 0)
    }

    fn fold_signature(&self, func: &mut crate::syntax::Function) {
        for (_, ty) in &mut func.params {
            self.fold_type(ty);
        }
        self.fold_type(&mut func.return_type);
    }

    fn fold_type(&self, ty: &mut Type) {
        match ty {
            Type::Tensor(_, dims, _) => {
                for dim in dims {
                    if let Dim::Static(e) = dim {
                        self.fold_extent(e, true);
                    }
                }
            }
            Type::Pointer(inner, ..)
            | Type::Ref(inner, _)
            | Type::Borrow { inner, .. }
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => self.fold_type(inner),
            Type::GenericInstance(base, args) => {
                self.fold_type(base);
                for arg in args {
                    self.fold_generic_arg(arg);
                }
            }
            Type::Function(params, ret, _) | Type::Closure(params, ret) => {
                for p in params {
                    self.fold_type(p);
                }
                self.fold_type(ret);
            }
            Type::Const(e) => self.fold_extent(e, false),
            _ => {}
        }
    }

    /// A name parses as a type, so `Grid<N>` holds the `const` `N` as a struct name.
    fn fold_generic_arg(&self, arg: &mut Type) {
        if let Type::Struct(name, None) = arg {
            if !self.type_names.contains(name) {
                if let Some(v) = self.value(name) {
                    *arg = Type::Const(Box::new(number(v, Span::default(), false)));
                    return;
                }
            }
        }
        self.fold_type(arg);
    }

    /// Replace the `const`s in an extent with their values, then the whole extent with its
    /// value when nothing in it is left unknown. `typed` gives the numbers the type the parser
    /// gives a number written in a tensor's shape.
    fn fold_extent(&self, e: &mut Expr, typed: bool) {
        if !self.substitute(e, typed) || matches!(e, Expr::Number(_)) {
            return;
        }
        if let Some(v) = int_value(e, &HashMap::new(), &mut Vec::new()).filter(|v| *v >= 0) {
            *e = number(v, e.span(), typed);
        }
    }

    /// Whether any `const` in `e` was replaced.
    fn substitute(&self, e: &mut Expr, typed: bool) -> bool {
        match e {
            Expr::Identifier(id) => match self.value(&id.name) {
                Some(v) => {
                    *e = number(v, id.span, typed);
                    true
                }
                None => false,
            },
            Expr::BinaryOp(b) => {
                let l = self.substitute(&mut b.lhs, typed);
                let r = self.substitute(&mut b.rhs, typed);
                l || r
            }
            Expr::UnaryOp(u) => self.substitute(&mut u.expr, typed),
            _ => false,
        }
    }
}

fn number(value: i128, span: Span, typed: bool) -> Expr {
    let value = value.to_string();
    Expr::Number(NumberExpr {
        ty: typed.then(|| crate::parser::expr::default_number_elem(&value)),
        value: value.into(),
        span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    fn parse_module(path: &str, src: &str) -> VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut program = parser.parse().expect("parse failed");
        program.module_path = path.into();
        program
    }

    /// The GID is used as intended (docs/parallel_compiler_architecture.md §2): each top-level
    /// symbol's `TypeId` is minted from *content hashes* -- word 0 = module hash, word 1 = symbol
    /// hash -- not a monotonic counter. So the same symbol name in two modules gets distinct GIDs
    /// (module isolation), the same name hashes identically (word 1), and minting is deterministic
    /// across runs. This is the property that makes cross-module identity lock-free and stable.
    #[test]
    fn gids_are_module_isolated_symbol_hashed_and_deterministic() {
        let a = parse_module("crate::a", "struct Foo { x: i32 }");
        let b = parse_module("crate::b", "struct Foo { y: i32 }");
        let map = build_symbol_map(&[a, b]);

        let a_foo = map[&Symbol::from("crate::a")][&Symbol::from("Foo")];
        let b_foo = map[&Symbol::from("crate::b")][&Symbol::from("Foo")];

        // Word 0 (module hash) isolates same-named symbols across modules, and is exactly the
        // module-path content hash.
        assert_ne!(a_foo.module_id(), b_foo.module_id());
        assert_eq!(a_foo.module_id(), compute_module_hash("crate::a"));
        assert_eq!(b_foo.module_id(), compute_module_hash("crate::b"));

        // Word 1 (symbol hash) is the content hash of the name -- identical for both `Foo`s.
        assert_eq!(a_foo.symbol_id(), b_foo.symbol_id());
        assert_eq!(
            a_foo.symbol_id(),
            DefPath::Named("Foo").compute_symbol_hash()
        );

        // Deterministic: re-minting the same module yields the identical GID (content hashes, not
        // position-dependent counters -- the incremental-stability guarantee).
        let a2 = parse_module("crate::a", "struct Foo { x: i32 }");
        let map2 = build_symbol_map(&[a2]);
        assert_eq!(map2[&Symbol::from("crate::a")][&Symbol::from("Foo")], a_foo);
    }

    /// `build_symbol_map` mints GIDs in parallel (`par_iter`); doing so must be deterministic --
    /// the same inputs produce byte-identical GID streams regardless of thread scheduling. This is
    /// the parallel-architecture promise that GIDs "mean the same thing on every thread".
    #[test]
    fn parallel_gid_minting_is_deterministic_across_threads() {
        let modules: Vec<VxModule> = (0..32)
            .map(|i| {
                parse_module(
                    &format!("crate::m{i}"),
                    "struct S { v: i32 }\nenum E { A, B }",
                )
            })
            .collect();

        let run = || {
            let map = build_symbol_map(&modules);
            let mut flat: Vec<(u64, u64)> = map
                .values()
                .flat_map(|t| t.values().map(|id| (id.module_id(), id.symbol_id())))
                .collect();
            flat.sort_unstable();
            flat
        };

        // Two independent parallel builds agree exactly.
        assert_eq!(run(), run());
    }

    /// The extents of the first parameter of function `name`, each as the number it is written
    /// as, or `None` where it is not a number.
    fn first_param_extents(module: &VxModule, name: &str) -> Vec<Option<String>> {
        let func = module
            .functions
            .iter()
            .find(|f| f.name.as_ref() == name)
            .expect("no such function");
        match &func.params[0].1 {
            Type::Tensor(_, dims, _) => dims
                .iter()
                .map(|d| d.literal().map(str::to_string))
                .collect(),
            other => panic!("not a tensor: {other:?}"),
        }
    }

    #[test]
    fn module_consts_in_a_signature_become_numbers() {
        let mut modules = vec![parse_module(
            "crate::a",
            "const N : i32 = 4;\n\
             fn f(t : Tensor<f32, [N, N * 2]>) -> void { return; }\n\
             fn g<const N : i32>(t : Tensor<f32, [N]>) -> void { return; }",
        )];
        fold_const_extents(&mut modules);
        let four_and_eight = vec![Some("4".to_string()), Some("8".to_string())];
        assert_eq!(first_param_extents(&modules[0], "f"), four_and_eight);
        // `g`'s own parameter `N` is not the module `const`.
        assert_eq!(first_param_extents(&modules[0], "g"), vec![None]);
    }

    /// `Grid<N>` holds `N` as a struct name, since a name parses as a type, and `Grid<N * 2>`
    /// holds an expression. Both become numbers.
    #[test]
    fn module_consts_as_generic_arguments_become_numbers() {
        let mut modules = vec![parse_module(
            "crate::a",
            "const N : i32 = 4;\n\
             struct Grid<const S : i32> { cells : Tensor<f32, [S]> }\n\
             fn f(a : Grid<N>, b : Grid<N * 2>) -> void { return; }",
        )];
        fold_const_extents(&mut modules);
        let args: Vec<Option<String>> = modules[0].functions[0]
            .params
            .iter()
            .map(|(_, ty)| match ty {
                Type::GenericInstance(_, args) => match &args[0] {
                    Type::Const(e) => match &**e {
                        Expr::Number(n) => Some(n.value.to_string()),
                        _ => None,
                    },
                    _ => None,
                },
                other => panic!("not a generic instance: {other:?}"),
            })
            .collect();
        assert_eq!(args, vec![Some("4".to_string()), Some("8".to_string())]);
    }

    #[test]
    fn a_const_from_another_module_is_folded() {
        let mut modules = vec![
            parse_module("crate::a", "const N : i32 = 3 * 5;"),
            parse_module("crate::b", "fn f(t : Tensor<f32, [N]>) -> void { return; }"),
        ];
        fold_const_extents(&mut modules);
        assert_eq!(
            first_param_extents(&modules[1], "f"),
            vec![Some("15".to_string())]
        );
    }

    /// A float `const`, one defined through itself and one too large for its type have no
    /// integer value, and a negative one is no extent. They stay names for the checker to
    /// report.
    #[test]
    fn a_const_that_cannot_be_an_extent_stays_a_name() {
        let mut modules = vec![parse_module(
            "crate::a",
            "const F : f32 = 4.0;\n\
             const L : i32 = L + 1;\n\
             const BIG : u8 = 300;\n\
             const NEG : i32 = 0 - 2;\n\
             fn f(t : Tensor<f32, [F, L, BIG, NEG]>) -> void { return; }",
        )];
        fold_const_extents(&mut modules);
        assert_eq!(first_param_extents(&modules[0], "f"), vec![None; 4]);
    }
}
