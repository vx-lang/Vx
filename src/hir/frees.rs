//===- frees.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `!frees(p)` walk: whether a call can free memory that it reaches through the parameter
// `p`, or move that memory. The walk follows `p` and each value that the body gets from it: a
// copy, a pointer that `offset` moves, a reference into its memory, and a value read through it
// that holds a pointer. The body keeps `!frees(p)` when it only reads and writes values without
// a drop through them, compares them, returns them, and gives them to parameters that have
// `!frees`. Any other use, such as a store or a method call, can free memory.
// The compiler runs the walk on the syntax before it checks any body, as it does for `!keeps`,
// and on a checked body to check a written `!frees(p)`.
// docs/discussions/raw_pointer_facts.md, section 5, has the rules.
//
//===----------------------------------------------------------------------===//

use crate::hir::provenance::Decls;
use crate::symbol::Symbol;
use crate::syntax::{
    ElementType, Expr, Function, FunctionCallExpr, Span, Statement, Type, UnaryOp,
};
use std::collections::HashMap;

/// What the walk knows about a function that the body calls.
pub struct Callee {
    /// The parameters that the function says `!frees` for, as bits.
    pub not_freed: u32,
    pub returns: Type,
    /// How a message names the function.
    pub shown: String,
}

/// A use through which a body can free memory that a parameter reaches.
#[derive(Clone, Debug)]
pub struct FreeingUse {
    pub at: Span,
    /// What the use does, such as "it gives `p` to 'dealloc', ...".
    pub why: String,
}

/// The parameters that a value comes from, as bits, and its type when the walk knows it.
type Value = (u32, Option<Type>);

/// The functions of `core::ptr` that compute an address from their first argument.
const MOVES: [&str; 6] = ["offset", "offset_mut", "add", "add_mut", "sub", "sub_mut"];
/// The functions of `core::ptr` that compare pointers.
const COMPARES: [&str; 3] = ["eq", "is_null", "is_null_mut"];

/// For each parameter of `f`, the first use in the body through which a call can free memory
/// that the parameter reaches. `None` in the list for a parameter that the body frees nothing
/// through, and for one whose type cannot free memory. `None` for a body that the walk cannot
/// read. `callee` gives what the walk knows about each function that the body calls.
pub fn freeing_uses(
    f: &Function,
    decls: &Decls,
    callee: &dyn Fn(&str) -> Option<Callee>,
) -> Option<Vec<Option<FreeingUse>>> {
    if f.body.is_empty() || f.params.len() > 32 {
        return None;
    }
    let mut walk = FreesWalk {
        decls,
        callee,
        vars: HashMap::new(),
        uses: Vec::new(),
        at: f.span,
        changed: false,
    };
    // A variable only gets more parameters that it comes from, so the walk settles. The last
    // pass finds the uses with everything that the walk knows.
    for _ in 0..16 {
        walk.changed = false;
        walk.uses = vec![None; f.params.len()];
        for (i, (name, ty)) in f.params.iter().enumerate() {
            if !decls.can_free_through(ty, 0) {
                // It is still a variable, which hides a function of the same name.
                walk.assign(name, 0, Some(ty.clone()));
                continue;
            }
            walk.assign(name, 1 << i, Some(ty.clone()));
            // The function owns a value that it gets by value. It drops it or moves it.
            if decls.has_drop(ty, 0) {
                walk.uses[i] = Some(FreeingUse {
                    at: f.span,
                    why: format!("it gets `{name}` by value, so it drops it or moves it"),
                });
            }
        }
        walk.block(&f.body);
        if !walk.changed {
            return Some(walk.uses);
        }
    }
    None
}

/// The `!frees` bits that the walk proves for `f`: each parameter whose type can free memory,
/// when the body frees nothing through it.
pub fn inferred_not_freed(
    f: &Function,
    decls: &Decls,
    callee: &dyn Fn(&str) -> Option<Callee>,
) -> u32 {
    let Some(uses) = freeing_uses(f, decls, callee) else {
        return 0;
    };
    f.params
        .iter()
        .zip(&uses)
        .enumerate()
        .filter(|(_, ((_, ty), use_))| use_.is_none() && decls.can_free_through(ty, 0))
        .fold(0, |bits, (i, _)| bits | 1 << i)
}

/// The name of the function that the call `name` calls, without its module path and type
/// arguments.
fn plain_name(name: &str) -> &str {
    let name = name.split(['<', '$']).next().unwrap_or(name);
    name.rsplit("::").next().unwrap_or(name)
}

/// `e` as the program wrote it, for a message.
fn shown(e: &Expr) -> String {
    fn text(e: &Expr) -> Option<String> {
        match e {
            Expr::Borrow(b) => {
                let mutable = if b.is_mut { "mut " } else { "" };
                Some(format!("&{mutable}{}", text(&b.expr)?))
            }
            Expr::Dereference(d) => Some(format!("*{}", text(&d.expr)?)),
            Expr::IndexAccess(i) => Some(format!("{}[{}]", text(&i.base)?, text(&i.index)?)),
            Expr::MemberAccess(m) => match &*m.base {
                Expr::Dereference(_) => Some(format!("({}).{}", text(&m.base)?, m.member)),
                base => Some(format!("{}.{}", text(base)?, m.member)),
            },
            e => crate::hir::prover::to_source(e),
        }
    }
    text(crate::syntax::bare_unsafe_value(e))
        .map_or_else(|| "a value".to_string(), |t| format!("`{t}`"))
}

/// What a raw pointer or a reference of type `ty` points at.
fn pointee(ty: &Type) -> Option<Type> {
    match ty {
        Type::Pointer(inner, ..) | Type::Borrow { inner, .. } | Type::Ref(inner, _) => {
            Some((**inner).clone())
        }
        _ => None,
    }
}

/// The type of an element of `ty`, for `a[i]`. An index into another type, such as a `Vec`,
/// calls a method, so its type stays unknown.
fn element(ty: &Type) -> Option<Type> {
    match ty {
        Type::Pointer(inner, ..) => Some((**inner).clone()),
        Type::Tensor(e, dims, _) if dims.len() == 1 => Some(Type::Scalar(e.clone())),
        _ => None,
    }
}

/// The struct that a field access on a value of type `ty` reads: through each reference and
/// raw pointer.
fn through(ty: &Type) -> &Type {
    match ty {
        Type::Pointer(inner, ..) | Type::Borrow { inner, .. } | Type::Ref(inner, _) => {
            through(inner)
        }
        ty => ty,
    }
}

struct FreesWalk<'a> {
    decls: &'a Decls<'a>,
    callee: &'a dyn Fn(&str) -> Option<Callee>,
    /// What each variable comes from. A name that two `let`s declare is one variable here.
    vars: HashMap<Symbol, Value>,
    uses: Vec<Option<FreeingUse>>,
    /// The place of the statement that the walk is in, for an expression with no place.
    at: Span,
    changed: bool,
}

impl FreesWalk<'_> {
    fn var(&self, name: &str) -> Value {
        self.vars.get(name).cloned().unwrap_or((0, None))
    }

    /// `name` gets a value that comes from the parameters in `bits`. A variable comes from what
    /// each of its values comes from. It has a type only while each value has that type.
    fn assign(&mut self, name: &Symbol, bits: u32, ty: Option<Type>) {
        let Some(entry) = self.vars.get_mut(name) else {
            self.vars.insert(name.clone(), (bits, ty));
            self.changed |= bits != 0;
            return;
        };
        if entry.0 | bits != entry.0 {
            entry.0 |= bits;
            self.changed = true;
        }
        if entry.1.is_some() && entry.1 != ty {
            entry.1 = None;
            self.changed = true;
        }
    }

    /// The use at `at` can free memory that the parameters in `bits` reach.
    fn stop(&mut self, bits: u32, at: Span, why: impl FnOnce() -> String) {
        let at = if at == Span::default() { self.at } else { at };
        let mut why = Some(why);
        let mut text: Option<String> = None;
        for (i, use_) in self.uses.iter_mut().enumerate() {
            if bits & (1 << i) != 0 && use_.is_none() {
                let why = text.get_or_insert_with(|| (why.take().expect("set once"))());
                *use_ = Some(FreeingUse {
                    at,
                    why: why.clone(),
                });
            }
        }
    }

    /// The parameters that the variables named in `e` come from, and the first such variable.
    fn mentioned(&self, e: &Expr) -> (u32, String) {
        let mut bits = 0;
        let mut first = None;
        crate::syntax::visit_expr(e, &mut |x| {
            if let Expr::Identifier(id) = x {
                let from = self.var(&id.name).0;
                if from != 0 {
                    bits |= from;
                    first.get_or_insert_with(|| id.name.to_string());
                }
            }
        });
        (bits, first.unwrap_or_default())
    }

    fn block(&mut self, stmts: &[Statement]) -> Value {
        let mut last = (0, None);
        for (i, s) in stmts.iter().enumerate() {
            match s {
                Statement::ExprStmt(e) if !e.has_semi && i + 1 == stmts.len() => {
                    self.at = e.span;
                    last = self.value(&e.expr);
                }
                s => self.stmt(s),
            }
        }
        last
    }

    fn block_value(&mut self, stmts: &[Statement], ret: Option<&Expr>) -> Value {
        let tail = self.block(stmts);
        match ret {
            Some(r) => self.value(r),
            None => tail,
        }
    }

    fn stmt(&mut self, s: &Statement) {
        self.at = s.span();
        match s {
            Statement::LetDecl(l) => {
                let (bits, ty) = self.value(&l.expr);
                self.assign(&l.name, bits, l.ty_ann.clone().or(ty));
            }
            Statement::Return(r) => {
                if let Some(e) = &r.expr {
                    self.value(e);
                }
            }
            Statement::ExprStmt(e) => {
                self.value(&e.expr);
            }
            Statement::Assign(a) => self.store(&a.lhs, &a.rhs),
            Statement::CompoundAssign(a) => {
                self.value(&a.rhs);
                if !matches!(a.lhs, Expr::Identifier(_)) {
                    let (bits, ty) = self.place(&a.lhs);
                    self.write(bits, ty, &a.lhs);
                }
            }
            Statement::ForLoop(l) => {
                match &*l.iterable {
                    Expr::Range(r) => {
                        self.value(&r.start);
                        self.value(&r.end);
                    }
                    e => {
                        let (bits, _) = self.value(e);
                        self.stop(bits, e.span(), || {
                            format!("it loops over {}, which calls 'next' on it", shown(e))
                        });
                    }
                }
                self.block(&l.body);
            }
            Statement::Assert(a) => {
                self.value(&a.expr);
            }
            Statement::Loop(l) => {
                self.block(&l.body);
            }
            Statement::Drop(d) if !d.ends_lifetime => {
                let (bits, _) = self.var(&d.name);
                self.stop(bits, d.span, || format!("it drops `{}`", d.name));
            }
            Statement::Drop(_)
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::MacroCall(_)
            | Statement::Error(_) => {}
        }
    }

    /// `lhs = rhs`. A store of a value that comes from a parameter anywhere but in a variable
    /// can free memory, because the walk does not follow the place.
    fn store(&mut self, lhs: &Expr, rhs: &Expr) {
        let (bits, ty) = self.value(rhs);
        if let Expr::Identifier(id) = lhs {
            self.assign(&id.name, bits, ty);
            return;
        }
        let (to, to_ty) = self.place(lhs);
        self.write(to, to_ty, lhs);
        self.stop(bits, rhs.span(), || {
            format!("it stores {} in {}", shown(rhs), shown(lhs))
        });
    }

    /// A write to the place `lhs` drops its old value, which frees memory when the type of the
    /// place has a drop.
    fn write(&mut self, bits: u32, ty: Option<Type>, lhs: &Expr) {
        if bits == 0 || ty.as_ref().is_some_and(|t| !self.decls.has_drop(t, 0)) {
            return;
        }
        self.stop(bits, lhs.span(), || match &ty {
            Some(t) => format!(
                "it writes {}, a place of `{t}`, which has a drop, so the write drops the old \
                 value",
                shown(lhs)
            ),
            None => format!(
                "it writes {}, whose type the compiler does not know here",
                shown(lhs)
            ),
        });
    }

    /// The place `e`: the parameters that its address comes from, and its type. A place inside
    /// a variable comes from what the variable comes from, and a place through a pointer from
    /// what the pointer comes from.
    fn place(&mut self, e: &Expr) -> Value {
        match e {
            Expr::Identifier(id) => self.var(&id.name),
            Expr::Dereference(d) => {
                let (bits, ty) = self.place(&d.expr);
                (bits, ty.as_ref().and_then(pointee))
            }
            Expr::IndexAccess(i) => {
                self.value(&i.index);
                let (bits, ty) = self.place(&i.base);
                (bits, ty.as_ref().and_then(element))
            }
            Expr::MemberAccess(m) => {
                let (bits, ty) = self.place(&m.base);
                let field = ty.and_then(|t| self.decls.field_type(through(&t), &m.member));
                (bits, field)
            }
            Expr::UnsafeBlock(u) if u.stmts.is_empty() && u.ret.is_some() => {
                self.place(u.ret.as_deref().expect("checked above"))
            }
            e => self.value(e),
        }
    }

    /// A read of the place `e`. The value comes from what the place comes from when its type
    /// can free memory. A read of a value of a type with a drop moves the value, and its new
    /// owner can free the memory that the value owns.
    fn read(&mut self, e: &Expr) -> Value {
        let (bits, ty) = self.place(e);
        if bits == 0 {
            return (0, ty);
        }
        match &ty {
            Some(t) if !self.decls.has_drop(t, 0) => {
                let from = if self.decls.can_free_through(t, 0) {
                    bits
                } else {
                    0
                };
                (from, ty)
            }
            Some(t) => {
                self.stop(bits, e.span(), || {
                    format!(
                        "it reads {}, a value of `{t}`, which has a drop, so the read moves it",
                        shown(e)
                    )
                });
                (0, ty)
            }
            None => {
                self.stop(bits, e.span(), || {
                    format!(
                        "it reads {}, whose type the compiler does not know here",
                        shown(e)
                    )
                });
                (0, None)
            }
        }
    }

    /// The value of `e`, and the uses in it that can free memory.
    fn value(&mut self, e: &Expr) -> Value {
        match e {
            Expr::Identifier(id) => self.var(&id.name),
            Expr::Dereference(_) | Expr::IndexAccess(_) | Expr::MemberAccess(_) => self.read(e),
            Expr::Borrow(b) => {
                let (bits, ty) = self.place(&b.expr);
                let ty = ty.map(|inner| Type::Borrow {
                    inner: Box::new(inner),
                    mem_space: None,
                    is_mut: b.is_mut,
                    region_id: 0,
                });
                (bits, ty)
            }
            Expr::FunctionCall(fc) => self.call(fc),
            Expr::MethodCall(m) => {
                for a in std::iter::once(&*m.base).chain(&m.args) {
                    let (bits, _) = self.value(a);
                    self.stop(bits, m.span, || {
                        format!(
                            "it gives {} to the method '{}', and the compiler does not look \
                             into a method call",
                            shown(a),
                            m.method_name
                        )
                    });
                }
                (0, None)
            }
            Expr::IndirectCall(ic) => {
                for a in std::iter::once(&*ic.callee).chain(&ic.args) {
                    let (bits, _) = self.value(a);
                    self.stop(bits, ic.span, || {
                        format!(
                            "it gives {} to a call through a function value, which has no \
                             clauses",
                            shown(a)
                        )
                    });
                }
                (0, None)
            }
            Expr::BinaryOp(b) => {
                self.value(&b.lhs);
                self.value(&b.rhs);
                (0, None)
            }
            Expr::RelationalOp(r) => {
                self.value(&r.lhs);
                self.value(&r.rhs);
                (0, Some(Type::Scalar(ElementType::Bool)))
            }
            Expr::LogicalOp(l) => {
                self.value(&l.lhs);
                self.value(&l.rhs);
                (0, Some(Type::Scalar(ElementType::Bool)))
            }
            Expr::UnaryOp(u) => {
                let (bits, _) = self.value(&u.expr);
                match u.op {
                    UnaryOp::Try => (bits, None),
                    UnaryOp::Not | UnaryOp::Neg => (0, None),
                }
            }
            Expr::AsCast(c) => {
                let (bits, _) = self.value(&c.expr);
                if matches!(c.target_ty, Type::Pointer(..)) {
                    return (bits, Some(c.target_ty.clone()));
                }
                self.stop(bits, c.span, || {
                    format!(
                        "it casts {} to `{}`, and the compiler does not follow an address in a \
                         number",
                        shown(&c.expr),
                        c.target_ty
                    )
                });
                (0, Some(c.target_ty.clone()))
            }
            Expr::StructInit(s) => {
                for (_, v) in &s.fields {
                    let (bits, _) = self.value(v);
                    self.stop(bits, v.span(), || {
                        format!("it puts {} in a value of `{}`", shown(v), s.name)
                    });
                }
                (0, Some(Type::Struct(s.name.clone(), None)))
            }
            Expr::EnumVariant(v) => {
                for p in v.payload.iter().flatten() {
                    let (bits, _) = self.value(p);
                    self.stop(bits, p.span(), || {
                        format!("it puts {} in a value of `{}`", shown(p), v.enum_name)
                    });
                }
                (0, None)
            }
            Expr::Array(a) => {
                self.put_in(&a.elements, "an array");
                (0, None)
            }
            Expr::VecMacro(v) => {
                self.put_in(&v.elements, "a vector");
                (0, None)
            }
            Expr::If(i) => {
                self.value(&i.cond);
                let a = self.block(&i.then_block);
                let b = match &i.else_block {
                    Some(b) => self.block(b),
                    None => (0, None),
                };
                (a.0 | b.0, if a.1 == b.1 { a.1 } else { None })
            }
            Expr::Match(m) => {
                let (bits, _) = self.value(&m.expr);
                let mut out: Option<Value> = None;
                for arm in &m.arms {
                    let mut names = Vec::new();
                    crate::hir::check::control::pattern_names(&arm.pattern, &mut names);
                    for name in &names {
                        self.assign(name, bits, None);
                    }
                    let v = self.block(&arm.body);
                    out = Some(match out {
                        None => v,
                        Some(o) => (o.0 | v.0, if o.1 == v.1 { o.1 } else { None }),
                    });
                }
                out.unwrap_or((0, None))
            }
            Expr::UnsafeBlock(u) => self.block_value(&u.stmts, u.ret.as_deref()),
            Expr::ComptimeBlock(c) => self.block_value(&c.stmts, c.ret.as_deref()),
            Expr::Closure(c) => {
                let (bits, name) = self.mentioned(&c.body);
                self.stop(bits, c.span, || format!("a closure captures `{name}`"));
                (0, None)
            }
            Expr::InlineMlir(m) => {
                for (_, v, _) in &m.inputs {
                    let (bits, _) = self.value(v);
                    self.stop(bits, m.span, || {
                        format!("it gives {} to an `mlir!` block", shown(v))
                    });
                }
                for c in &m.clobbers {
                    let (bits, _) = self.place(c);
                    self.stop(bits, m.span, || {
                        format!("an `mlir!` block writes {}", shown(c))
                    });
                }
                (0, m.returns.clone())
            }
            Expr::Print(p) => {
                p.args.iter().for_each(|a| {
                    self.value(a);
                });
                (0, None)
            }
            Expr::Println(p) => {
                p.args.iter().for_each(|a| {
                    self.value(a);
                });
                (0, None)
            }
            Expr::Range(r) => {
                self.value(&r.start);
                self.value(&r.end);
                (0, None)
            }
            Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::SizeOf(_)
            | Expr::MemorySpace(_)
            | Expr::Topology(_)
            | Expr::TransferPredicate(_) => (0, None),
            Expr::Transfer(_)
            | Expr::Grad(_)
            | Expr::Vjp(_)
            | Expr::Jvp(_)
            | Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::MacroCall(_) => {
                let (bits, name) = self.mentioned(e);
                self.stop(bits, e.span(), || {
                    format!("the compiler does not follow `{name}` into this expression")
                });
                (0, None)
            }
        }
    }

    /// Each value of `elements` that comes from a parameter goes into `what`, a new value.
    fn put_in(&mut self, elements: &[Expr], what: &str) {
        for v in elements {
            let (bits, _) = self.value(v);
            self.stop(bits, v.span(), || format!("it puts {} in {what}", shown(v)));
        }
    }

    /// A call. Each argument that comes from a parameter can free memory, unless the callee says
    /// `!frees` for its parameter. The result can be a value that the callee got.
    fn call(&mut self, fc: &FunctionCallExpr) -> Value {
        let args: Vec<Value> = fc.args.iter().map(|a| self.value(a)).collect();
        let given = args.iter().fold(0, |bits, (b, _)| bits | b);
        let name = fc.name.as_ref();
        // A variable of the same name, such as a closure, hides the function.
        let hidden = self.vars.contains_key(name);
        let plain = plain_name(name);
        if !hidden && MOVES.contains(&plain) {
            return args.into_iter().next().unwrap_or((0, None));
        }
        if !hidden && COMPARES.contains(&plain) {
            return (0, Some(Type::Scalar(ElementType::Bool)));
        }
        let callee = if hidden { None } else { (self.callee)(name) };
        let (not_freed, returns) = match &callee {
            Some(c) => (c.not_freed, Some(c.returns.clone())),
            None => (0, None),
        };
        let to = match callee {
            Some(c) => format!("'{}', which has no `!frees` for that parameter", c.shown),
            None if hidden => format!("the call through `{name}`, which has no clauses"),
            None => format!("'{name}', whose clauses the compiler does not know"),
        };
        for (i, (a, (bits, _))) in fc.args.iter().zip(&args).enumerate() {
            if i >= 32 || not_freed & (1 << i) == 0 {
                self.stop(*bits, a.span(), || format!("it gives {} to {to}", shown(a)));
            }
        }
        let from = match &returns {
            Some(t) if !self.decls.can_free_through(t, 0) => 0,
            _ => given,
        };
        (from, returns)
    }
}
