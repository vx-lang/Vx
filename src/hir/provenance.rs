//===- provenance.rs - Vx Compiler -----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Return-provenance summary: for each function whose result can hold a reference, where the
// result may point, by parameter. It is computed from the syntax of each body before the checker
// checks any body, so the parallel checker only reads it. A call uses it to decide which argument
// borrows the result keeps, and whose held borrows the result holds too. A case the walk cannot
// follow gives `AnyParam`, which keeps every argument.
//
// The path of a returned reference decides the answer:
// - through a shared reference field, the result points where the field points: what the
//   argument holds. `VecIter::next` returns `&(*self.vec).data[i]`, so an item holds the borrow
//   of the vector that the iterator holds, and not the iterator.
// - through a `&mut` field, the result keeps the argument itself.
// - through a raw pointer field, the result points into the struct that holds the field.
//
//===----------------------------------------------------------------------===//
use crate::symbol::Symbol;
use crate::syntax::{
    ElementType, EnumDecl, Expr, Function, FunctionCallExpr, ImplBlock, Pattern, Statement,
    StructDecl, Type,
};
use std::collections::{HashMap, HashSet};

/// Where a function's result may point, by parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnProvenance {
    /// The result holds no reference.
    NotAReference,
    /// The result points into what the parameters in `into` point at, or where a reference that
    /// the parameters in `held` hold points. Bit `i` is parameter `i`, for up to 32 parameters.
    FromParams { into: u32, held: u32 },
    /// The result may point anywhere any parameter reaches. The answer for each case the walk
    /// cannot follow: a method call, recursion, more than 32 parameters, or a missing summary.
    AnyParam,
    /// The result points into the function's own variables, which is an `E4005` in the
    /// function. A caller treats it as `AnyParam`.
    Local,
}

fn has(bits: u32, slot: usize) -> bool {
    slot < 32 && bits & (1u32 << slot) != 0
}

impl ReturnProvenance {
    /// Does a call whose result is kept keep the borrow that the argument at `slot` makes?
    pub fn includes(&self, slot: usize) -> bool {
        match self {
            ReturnProvenance::NotAReference => false,
            ReturnProvenance::FromParams { into, .. } => has(*into, slot),
            ReturnProvenance::AnyParam | ReturnProvenance::Local => true,
        }
    }

    /// Can the result point where a reference that the argument at `slot` holds points? Then the
    /// result holds each borrow that the argument holds.
    pub fn holds(&self, slot: usize) -> bool {
        match self {
            ReturnProvenance::NotAReference => false,
            ReturnProvenance::FromParams { into, held } => has(*into | *held, slot),
            ReturnProvenance::AnyParam | ReturnProvenance::Local => true,
        }
    }
}

/// Pack a return-provenance summary into the 3-bit slot-0 code used by the inline `TypeId` encoding
/// (#265; see [`crate::gid::TypeId::set_return_provenance`]). The inline field can name a single
/// parameter (`1..=4`) or fall back to the conservative top (`7`); a *multi-parameter* union, a
/// slot `>= 4` or a `held` set does not fit the fixed width and encodes as top. It therefore always
/// yields a **conservative refinement** of the summary — for every parameter the summary flags as
/// an alias source, the packed code flags it too — so an over-budget signature degrades gracefully
/// to today's all-arguments behaviour rather than becoming unsound. `NotAReference` (no reference
/// returned) encodes as `0`; `Local` matches the summary's conservative-top reading (it is an
/// `E4005` in the callee, so a caller never actually sees the reference, but the encoding must not
/// read narrower than [`ReturnProvenance::includes`]). A code has no `held` set, so a call reads
/// [`inline_prov_includes`] for both questions.
pub fn encode_return_provenance(prov: &ReturnProvenance) -> u8 {
    match prov {
        ReturnProvenance::NotAReference | ReturnProvenance::FromParams { into: 0, held: 0 } => 0,
        ReturnProvenance::FromParams { into, held: 0 }
            if into.count_ones() == 1 && into.trailing_zeros() < 4 =>
        {
            (into.trailing_zeros() + 1) as u8
        }
        _ => 7,
    }
}

/// Does the packed slot-0 code `prov` (see [`encode_return_provenance`]) treat parameter `slot` as a
/// possible alias source? The packed-form mirror of [`ReturnProvenance::includes`]: `0` excludes all,
/// `7` (and the reserved `5`/`6`) includes all, `1..=4` includes exactly parameter slot `0..=3`.
pub fn inline_prov_includes(prov: u8, slot: usize) -> bool {
    match prov {
        0 => false,
        1..=4 => slot == (prov as usize - 1),
        _ => true, // 7 = conservative top; 5/6 are reserved and treated conservatively.
    }
}

fn type_is_ref(ty: &Type) -> bool {
    matches!(ty, Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..))
}

/// Whether a value of `ty` holds no reference: a number, a tensor or a function pointer.
fn holds_nothing(ty: &Type) -> bool {
    match ty {
        Type::Scalar(e) => !matches!(e, ElementType::Generic(_)),
        Type::Tensor(..) | Type::Matrix | Type::Simd(..) | Type::Function(..) => true,
        _ => false,
    }
}

/// Whether the walk looks at a function that returns `ty`.
fn can_hold_reference(ty: &Type) -> bool {
    match ty {
        Type::Scalar(e) => matches!(e, ElementType::Generic(_)),
        Type::Borrow { .. }
        | Type::Pointer(..)
        | Type::Ref(..)
        | Type::Struct(..)
        | Type::Enum(..)
        | Type::GenericInstance(..)
        | Type::Generic(..) => true,
        _ => false,
    }
}

/// What a reference or raw pointer of type `ty` points at.
fn pointee(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Borrow { inner, .. } | Type::Pointer(inner, _, _) | Type::Ref(inner, _) => {
            Some(inner)
        }
        _ => None,
    }
}

/// The struct and enum declarations that the walk reads.
#[derive(Default)]
pub struct Decls<'d> {
    structs: HashMap<&'d str, &'d StructDecl>,
    enums: HashMap<&'d str, &'d EnumDecl>,
}

impl<'d> Decls<'d> {
    pub fn new(
        structs: impl IntoIterator<Item = &'d StructDecl>,
        enums: impl IntoIterator<Item = &'d EnumDecl>,
    ) -> Self {
        Decls {
            structs: structs.into_iter().map(|s| (s.name.as_ref(), s)).collect(),
            enums: enums.into_iter().map(|e| (e.name.as_ref(), e)).collect(),
        }
    }

    /// The declaration named `name`, also when it is written `Vec<T>` or `std::vec::Vec`.
    fn lookup<V: Copy>(map: &HashMap<&'d str, V>, name: &str) -> Option<V> {
        let name = name.split('<').next().unwrap_or(name);
        map.get(name)
            .or_else(|| map.get(name.rsplit("::").next()?))
            .copied()
    }

    fn type_name(ty: &Type) -> Option<&str> {
        match ty {
            Type::Struct(name, _) | Type::Enum(name, _) => Some(name.as_ref()),
            Type::GenericInstance(base, _) => Self::type_name(base),
            _ => None,
        }
    }

    /// The type of field `field` of the struct type `ty`.
    fn field(&self, ty: &Type, field: &str) -> Option<&'d Type> {
        let decl = Self::lookup(&self.structs, Self::type_name(ty)?)?;
        decl.fields
            .iter()
            .find(|(name, _)| name.as_ref() == field)
            .map(|(_, ty)| ty)
    }

    /// Whether `ty` names a declared struct or enum. Any other name may be a type parameter.
    fn is_declared(&self, ty: &Type) -> bool {
        Self::type_name(ty).is_some_and(|name| {
            Self::lookup(&self.structs, name).is_some() || Self::lookup(&self.enums, name).is_some()
        })
    }

    /// Whether the call `name(..)` makes an enum value, as `Option::Some(r)` does.
    fn is_variant(&self, name: &str) -> bool {
        let Some((head, variant)) = name.rsplit_once("::") else {
            return false;
        };
        Self::lookup(&self.enums, head)
            .is_some_and(|e| e.variants.iter().any(|(v, _)| v.as_ref() == variant))
    }
}

/// Where a value may point, by parameter. For a value that holds references, where they may
/// point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Prov {
    /// Into what these reference parameters point at.
    into: u32,
    /// Where a reference that these parameters hold points.
    held: u32,
    /// Into this function's own variables.
    local: bool,
    /// Anywhere: the walk cannot follow it.
    any: bool,
}

impl Prov {
    const NONE: Prov = Prov {
        into: 0,
        held: 0,
        local: false,
        any: false,
    };
    const ANY: Prov = Prov {
        any: true,
        ..Prov::NONE
    };
    const LOCAL: Prov = Prov {
        local: true,
        ..Prov::NONE
    };

    fn into(slot: usize) -> Prov {
        Prov {
            into: 1 << slot,
            ..Prov::NONE
        }
    }

    fn held(slot: usize) -> Prov {
        Prov {
            held: 1 << slot,
            ..Prov::NONE
        }
    }

    fn join(self, other: Prov) -> Prov {
        Prov {
            into: self.into | other.into,
            held: self.held | other.held,
            local: self.local || other.local,
            any: self.any || other.any,
        }
    }

    /// Where a reference stored in the memory that `self` points at points.
    fn deeper(self) -> Prov {
        Prov {
            into: 0,
            held: self.into | self.held,
            local: false,
            any: self.any || self.local,
        }
    }
}

/// A place: where its memory is, and where a reference stored in it points.
#[derive(Clone)]
struct Place {
    at: Prov,
    holds: Prov,
    /// The variable whose own memory holds the place, if no pointer is on the way to it.
    owner: Option<String>,
    /// The place is the whole variable `owner`.
    whole: bool,
}

impl Place {
    /// The place that a pointer to `v` points at.
    fn pointed_at(v: Prov) -> Place {
        Place {
            at: v,
            holds: v.deeper(),
            owner: None,
            whole: false,
        }
    }
}

struct ProvWalk<'a> {
    /// Each variable, parameters too: where its value may point, joined over every value it is
    /// given, and its type when the walk knows it.
    vars: HashMap<String, (Prov, Option<Type>)>,
    /// The variables whose address the body takes. A call may change them through it.
    borrowed: HashSet<String>,
    /// Where each returned value may point, joined.
    returned: Prov,
    saw_return: bool,
    /// The body stores a value that may hold a reference through a pointer, or gives one to a
    /// call. Memory that a parameter reaches may then hold anything.
    stores: bool,
    /// The body has something the walk does not read, such as a macro that was not expanded.
    unknown: bool,
    /// What the walk knows grew in this pass, so it walks again.
    changed: bool,
    decls: &'a Decls<'a>,
    /// The summaries known so far for the functions this body calls.
    callee: &'a dyn Fn(&str) -> Option<ReturnProvenance>,
}

/// The variable at the root of `x`, `x.f` and `x[i]`.
fn root_var(e: &Expr) -> Option<&str> {
    match e {
        Expr::Identifier(id) => Some(id.name.as_ref()),
        Expr::MemberAccess(m) => root_var(&m.base),
        Expr::IndexAccess(ix) => root_var(&ix.base),
        _ => None,
    }
}

impl<'a> ProvWalk<'a> {
    fn new(
        func: &Function,
        decls: &'a Decls<'a>,
        callee: &'a dyn Fn(&str) -> Option<ReturnProvenance>,
    ) -> Self {
        let mut walk = ProvWalk {
            vars: HashMap::new(),
            borrowed: HashSet::new(),
            returned: Prov::NONE,
            saw_return: false,
            stores: false,
            unknown: false,
            changed: false,
            decls,
            callee,
        };
        for (slot, (name, ty)) in func.params.iter().enumerate() {
            let v = walk.param_value(slot, ty);
            walk.vars.insert(name.to_string(), (v, Some(ty.clone())));
        }
        walk
    }

    /// What parameter `slot` of type `ty` points at, or what the references in it point at.
    fn param_value(&self, slot: usize, ty: &Type) -> Prov {
        if type_is_ref(ty) {
            Prov::into(slot)
        } else if holds_nothing(ty) {
            Prov::NONE
        } else if self.decls.is_declared(ty) || matches!(ty, Type::Closure(..)) {
            Prov::held(slot)
        } else {
            // A type parameter may be a reference.
            Prov::into(slot).join(Prov::held(slot))
        }
    }

    /// Give the variable `name` the value `v` too.
    fn assign(&mut self, name: &str, v: Prov, ty: Option<Type>) {
        match self.vars.get_mut(name) {
            Some(entry) => {
                if entry.0.join(v) != entry.0 {
                    entry.0 = entry.0.join(v);
                    self.changed = true;
                }
                if entry.1.is_none() && ty.is_some() {
                    entry.1 = ty;
                    self.changed = true;
                }
            }
            None => {
                self.vars.insert(name.to_string(), (v, ty));
                self.changed = true;
            }
        }
    }

    fn var(&self, name: &str) -> Option<(Prov, Option<Type>)> {
        let (v, ty) = self.vars.get(name)?.clone();
        // A call or a store through a pointer may have changed a variable whose address was taken.
        if self.stores && self.borrowed.contains(name) && !ty.as_ref().is_some_and(holds_nothing) {
            return Some((Prov::ANY, ty));
        }
        Some((v, ty))
    }

    /// `&mut x.f`, or `x.m()` with a method that may take `&mut self`, lets a call change `x`.
    /// Through a reference variable it takes the address of what the variable points at.
    fn note_borrowed(&mut self, e: &Expr) {
        let Some(root) = root_var(e) else {
            return;
        };
        let is_ref = self
            .vars
            .get(root)
            .is_some_and(|(_, ty)| ty.as_ref().is_some_and(type_is_ref));
        if !is_ref && self.borrowed.insert(root.to_string()) {
            self.changed = true;
        }
    }

    /// The body stores `v` where the walk does not follow it.
    fn store_somewhere(&mut self, v: Prov) {
        if v != Prov::NONE && !self.stores {
            self.stores = true;
            self.changed = true;
        }
    }

    fn note_return(&mut self, v: Prov) {
        self.returned = self.returned.join(v);
        self.saw_return = true;
    }

    /// The value read from `place`, which has type `ty`.
    fn read(&self, place: &Place, ty: Option<&Type>) -> Prov {
        let Some(ty) = ty else {
            return Prov::ANY;
        };
        if holds_nothing(ty) {
            return Prov::NONE;
        }
        let at = place.at;
        if at == Prov::LOCAL {
            // A variable knows where its value points. A raw pointer inside a local struct
            // points into memory that the walk does not follow.
            return match ty {
                Type::Pointer(..) | Type::Ref(..) if !place.whole => Prov::ANY,
                _ => place.holds,
            };
        }
        if at.local || at.any || self.stores {
            return Prov::ANY;
        }
        match ty {
            // A `&mut` read out of what a parameter reaches keeps that parameter, so that two
            // calls cannot give two `&mut` to one place.
            Type::Borrow { is_mut: true, .. } => Prov {
                into: at.into | at.held,
                ..Prov::NONE
            },
            // A raw pointer field points into memory that the struct holding it owns.
            Type::Pointer(..) => at,
            Type::Ref(..) => Prov::ANY,
            _ => place.holds,
        }
    }

    /// The place that the reference or raw pointer in `place` points at, for `p.f` and `p[i]`.
    /// Any other place is its own.
    fn through(&self, place: Place, ty: Type) -> (Place, Option<Type>) {
        match pointee(&ty) {
            Some(inner) => {
                let v = self.read(&place, Some(&ty));
                (Place::pointed_at(v), Some(inner.clone()))
            }
            None => (place, Some(ty)),
        }
    }

    /// The place that `e` names, with its type when the walk knows it.
    fn place(&mut self, e: &Expr) -> Option<(Place, Option<Type>)> {
        match e {
            Expr::Identifier(id) => {
                let (holds, ty) = self.var(id.name.as_ref())?;
                let place = Place {
                    at: Prov::LOCAL,
                    holds,
                    owner: Some(id.name.to_string()),
                    whole: true,
                };
                Some((place, ty))
            }
            Expr::Dereference(d) => {
                let (v, ty) = self.value(&d.expr);
                Some((Place::pointed_at(v), ty.as_ref().and_then(pointee).cloned()))
            }
            Expr::MemberAccess(m) => {
                let (base, ty) = self.place(&m.base)?;
                let (base, ty) = self.through(base, ty?);
                // A field is in the memory of its struct, whatever its type.
                let field = ty.and_then(|ty| self.decls.field(&ty, m.member.as_ref()).cloned());
                Some((
                    Place {
                        whole: false,
                        ..base
                    },
                    field,
                ))
            }
            Expr::IndexAccess(ix) => {
                self.value(&ix.index);
                let (base, ty) = self.place(&ix.base)?;
                match ty? {
                    // `p[i]` is the element `i` places after the one that `p` points at.
                    ptr @ Type::Pointer(..) => Some(self.through(base, ptr)),
                    ty => match self.through(base, ty) {
                        (base, Some(Type::Tensor(el, ..))) => Some((
                            Place {
                                whole: false,
                                ..base
                            },
                            Some(Type::Scalar(el)),
                        )),
                        _ => None,
                    },
                }
            }
            _ => {
                self.value(e);
                None
            }
        }
    }

    fn join_all<'e>(&mut self, es: impl IntoIterator<Item = &'e Expr>) -> Prov {
        es.into_iter()
            .fold(Prov::NONE, |acc, e| acc.join(self.value(e).0))
    }

    /// Where the value of `e` may point, with its type when the walk knows it. Each expression
    /// inside `e` is walked too, so that no `return` and no call is missed.
    fn value(&mut self, e: &Expr) -> (Prov, Option<Type>) {
        let bool_ty = Some(Type::Scalar(ElementType::Bool));
        match e {
            Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::SizeOf(_)
            | Expr::MemorySpace(_)
            | Expr::Topology(_)
            | Expr::TransferPredicate(_) => (Prov::NONE, None),
            Expr::Identifier(id) => match self.var(id.name.as_ref()) {
                Some(v) => v,
                None if matches!(id.name.as_ref(), "true" | "false") => (Prov::NONE, bool_ty),
                None => (Prov::ANY, None),
            },
            Expr::Borrow(b) => {
                // A `&x` cannot change `x`.
                if b.is_mut {
                    self.note_borrowed(&b.expr);
                }
                match self.place(&b.expr) {
                    Some((place, ty)) => (
                        place.at,
                        ty.map(|t| Type::Borrow {
                            inner: Box::new(t),
                            mem_space: None,
                            is_mut: b.is_mut,
                            region_id: 0,
                        }),
                    ),
                    None => (Prov::ANY, None),
                }
            }
            Expr::Dereference(_) | Expr::MemberAccess(_) | Expr::IndexAccess(_) => {
                match self.place(e) {
                    Some((place, ty)) => (self.read(&place, ty.as_ref()), ty),
                    None => (Prov::ANY, None),
                }
            }
            Expr::AsCast(c) => {
                let (v, _) = self.value(&c.expr);
                // An address made from a number may point anywhere.
                let made_from_number = type_is_ref(&c.target_ty) && v == Prov::NONE;
                let v = if made_from_number { Prov::ANY } else { v };
                (v, Some(c.target_ty.clone()))
            }
            Expr::StructInit(s) => {
                let v = self.join_all(s.fields.iter().map(|(_, f)| f));
                (v, Some(Type::Struct(s.name.clone(), None)))
            }
            Expr::EnumVariant(v) => (self.join_all(v.payload.iter().flatten()), None),
            Expr::Array(a) => (self.join_all(&a.elements), None),
            Expr::VecMacro(v) => (self.join_all(&v.elements), None),
            Expr::FunctionCall(fc) => (self.call(fc), None),
            Expr::MethodCall(m) => {
                self.note_borrowed(&m.base);
                let v = self.value(&m.base).0.join(self.join_all(&m.args));
                self.store_somewhere(v);
                (Prov::ANY, None)
            }
            Expr::IndirectCall(c) => {
                let v = self.value(&c.callee).0.join(self.join_all(&c.args));
                self.store_somewhere(v);
                (Prov::ANY, None)
            }
            Expr::Grad(g) => {
                let v = self.join_all(&g.args);
                self.store_somewhere(v);
                (Prov::ANY, None)
            }
            Expr::Vjp(v) => {
                let given = self.join_all(&v.args).join(self.value(&v.cotangent).0);
                self.store_somewhere(given);
                (Prov::ANY, None)
            }
            Expr::Jvp(j) => {
                let given = self.join_all(&j.args).join(self.value(&j.tangent).0);
                self.store_somewhere(given);
                (Prov::ANY, None)
            }
            Expr::BinaryOp(b) => (self.value(&b.lhs).0.join(self.value(&b.rhs).0), None),
            Expr::RelationalOp(r) => {
                self.value(&r.lhs);
                self.value(&r.rhs);
                (Prov::NONE, bool_ty)
            }
            Expr::LogicalOp(l) => {
                self.value(&l.lhs);
                self.value(&l.rhs);
                (Prov::NONE, bool_ty)
            }
            Expr::UnaryOp(u) => (self.value(&u.expr).0, None),
            Expr::Range(r) => {
                self.value(&r.start);
                self.value(&r.end);
                (Prov::NONE, None)
            }
            Expr::Transfer(t) => {
                self.value(&t.expr);
                (Prov::ANY, None)
            }
            Expr::Print(p) => {
                self.join_all(&p.args);
                (Prov::NONE, None)
            }
            Expr::Println(p) => {
                self.join_all(&p.args);
                (Prov::NONE, None)
            }
            Expr::UnsafeBlock(b) => (self.block_value(&b.stmts, b.ret.as_deref()), None),
            Expr::ComptimeBlock(b) => (self.block_value(&b.stmts, b.ret.as_deref()), None),
            Expr::TaskBlock(b) => {
                self.block_value(&b.stmts, b.ret.as_deref());
                (Prov::ANY, None)
            }
            Expr::SpawnOn(b) => {
                self.block_value(&b.stmts, b.ret.as_deref());
                (Prov::ANY, None)
            }
            Expr::If(i) => {
                self.value(&i.cond);
                let then = self.walk_block(&i.then_block);
                let other = match &i.else_block {
                    Some(b) => self.walk_block(b),
                    None => Prov::ANY,
                };
                (then.join(other), None)
            }
            Expr::Match(m) => {
                let (v, _) = self.value(&m.expr);
                // A name an arm binds is a part of the value, or a reference into it.
                let part = v.join(v.deeper());
                let mut value = Prov::NONE;
                for arm in &m.arms {
                    self.bind(&arm.pattern, part);
                    value = value.join(self.walk_block(&arm.body));
                }
                (value, None)
            }
            // A `return` in a closure leaves the closure, so its body is not walked here.
            Expr::Closure(_) => (Prov::ANY, None),
            Expr::MacroCall(_) => {
                self.unknown = true;
                (Prov::ANY, None)
            }
            Expr::InlineMlir(m) => {
                for (_, input, _) in &m.inputs {
                    self.value(input);
                }
                self.join_all(&m.clobbers);
                self.unknown = true;
                (Prov::ANY, None)
            }
        }
    }

    fn bind(&mut self, pattern: &Pattern, v: Prov) {
        match pattern {
            Pattern::Identifier(name) => self.assign(name.as_ref(), v, None),
            Pattern::EnumVariant(_, _, Some(parts)) => {
                for p in parts {
                    self.bind(p, v);
                }
            }
            Pattern::EnumVariant(_, _, None) | Pattern::Wildcard | Pattern::Literal(_) => {}
        }
    }

    fn call(&mut self, fc: &FunctionCallExpr) -> Prov {
        let typed: Vec<(Prov, Option<Type>)> = fc.args.iter().map(|a| self.value(a)).collect();
        let args: Vec<Prov> = typed.iter().map(|(v, _)| *v).collect();
        let given = args.iter().fold(Prov::NONE, |acc, a| acc.join(*a));
        // `Option::Some(r)` makes a value that holds `r`.
        if self.decls.is_variant(fc.name.as_ref()) {
            return given;
        }
        // The function may store what it is given where a parameter reaches.
        self.store_somewhere(given);
        let name = fc.name.as_ref();
        let name = match name.find('<') {
            Some(i) if name.ends_with('>') => &name[..i],
            _ => name,
        };
        // A variable of the same name, a closure, hides the function.
        if self.vars.contains_key(name) {
            return Prov::ANY;
        }
        let (into, held) = match (self.callee)(name) {
            Some(ReturnProvenance::NotAReference) => return Prov::NONE,
            Some(ReturnProvenance::FromParams { into, held }) => (into, held),
            _ => return Prov::ANY,
        };
        let mut v = Prov::NONE;
        for i in (0..32).filter(|&i| has(into | held, i)) {
            let a = args.get(i).copied().unwrap_or(Prov::ANY);
            if has(into, i) {
                v = v.join(a);
            }
            if has(held, i) {
                // What `&x` holds is what `x` holds, and a reference holds what it points at
                // holds. Any other argument may be a reference or the value that holds one.
                v = v.join(match (fc.args.get(i), &typed[i].1) {
                    (Some(Expr::Borrow(b)), _) => {
                        self.place(&b.expr).map_or(Prov::ANY, |(p, _)| p.holds)
                    }
                    (_, Some(ty)) if type_is_ref(ty) => a.deeper(),
                    _ => a.join(a.deeper()),
                });
            }
        }
        v
    }

    /// Walk the statements of a block and its last expression.
    fn block_value(&mut self, stmts: &[Statement], ret: Option<&Expr>) -> Prov {
        let tail = self.walk_block(stmts);
        match ret {
            Some(e) => self.value(e).0,
            None => tail,
        }
    }

    /// Walk a block. Its value is the value of its last statement, when that is an expression
    /// with no `;`. A block with no such statement gives no value: a branch used as a value
    /// either ends the function, as `return` or `panic` does, or does not type-check.
    fn walk_block(&mut self, stmts: &[Statement]) -> Prov {
        let mut tail = Prov::NONE;
        for (i, s) in stmts.iter().enumerate() {
            match s {
                Statement::ExprStmt(e) if i + 1 == stmts.len() && !e.has_semi => {
                    tail = self.value(&e.expr).0;
                }
                _ => self.walk_stmt(s),
            }
        }
        tail
    }

    fn walk_stmt(&mut self, stmt: &Statement) {
        match stmt {
            Statement::LetDecl(l) => {
                let (v, ty) = self.value(&l.expr);
                self.assign(l.name.as_ref(), v, l.ty_ann.clone().or(ty));
            }
            Statement::Return(r) => {
                if let Some(e) = &r.expr {
                    let (v, _) = self.value(e);
                    self.note_return(v);
                }
            }
            Statement::ExprStmt(e) => {
                self.value(&e.expr);
            }
            Statement::ForLoop(f) => {
                let (v, _) = self.value(&f.iterable);
                // The loop calls `next` on the iterator, which may store what it holds.
                self.store_somewhere(v);
                self.assign(&f.iter, Prov::ANY, None);
                self.join_all(&f.invariants);
                self.walk_block(&f.body);
            }
            Statement::Assign(a) => self.walk_store(&a.lhs, &a.rhs),
            Statement::CompoundAssign(a) => self.walk_store(&a.lhs, &a.rhs),
            Statement::Assert(a) => {
                self.value(&a.expr);
            }
            Statement::Loop(l) => {
                self.join_all(&l.invariants);
                self.walk_block(&l.body);
            }
            Statement::Break(b) => {
                if let Some(v) = &b.tasks_value {
                    self.value(v);
                }
            }
            Statement::Continue(_) | Statement::Drop(_) => {}
            Statement::MacroCall(_) | Statement::Error(_) => self.unknown = true,
        }
    }

    /// `lhs = rhs`. A variable, or a part of one, holds the value too. A store anywhere else
    /// is not followed.
    fn walk_store(&mut self, lhs: &Expr, rhs: &Expr) {
        let (v, _) = self.value(rhs);
        match self.place(lhs) {
            Some((
                Place {
                    owner: Some(name), ..
                },
                _,
            )) => self.assign(&name, v, None),
            _ => self.store_somewhere(v),
        }
    }
}

/// Compute the return-provenance summary for `func` (#243). Pure and body-only — safe to call from
/// any worker on any function, in any order. It reads no struct field and no call; see
/// [`compute_return_provenance_with`] for that.
pub fn compute_return_provenance(func: &Function) -> ReturnProvenance {
    compute_return_provenance_with(func, &Decls::default(), &|_| None)
}

/// As [`compute_return_provenance`], reading the fields of structs from `decls` and what a called
/// function returns from `callee`.
pub fn compute_return_provenance_with(
    func: &Function,
    decls: &Decls,
    callee: &dyn Fn(&str) -> Option<ReturnProvenance>,
) -> ReturnProvenance {
    if !can_hold_reference(&func.return_type) {
        return ReturnProvenance::NotAReference;
    }
    // The exact bitsets hold 32 parameters. A function with no body, an `extern`, cannot be read.
    if func.params.len() > 32 || func.body.is_empty() {
        return ReturnProvenance::AnyParam;
    }
    let mut walk = ProvWalk::new(func, decls, callee);
    // Each pass can only add to what the walk knows, so the walk settles. The returned values
    // of the last pass are read with everything it knows.
    for _ in 0..16 {
        walk.changed = false;
        walk.returned = Prov::NONE;
        walk.saw_return = false;
        let tail = walk.walk_block(&func.body);
        if matches!(func.body.last(), Some(Statement::ExprStmt(e)) if !e.has_semi) {
            walk.note_return(tail);
        }
        if !walk.changed {
            break;
        }
    }
    let r = walk.returned;
    if walk.changed || walk.unknown || r.any || !walk.saw_return {
        return ReturnProvenance::AnyParam;
    }
    if r.into | r.held != 0 {
        // A return of a local is an `E4005` in the function, and adds nothing here.
        return ReturnProvenance::FromParams {
            into: r.into,
            held: r.held,
        };
    }
    if r.local {
        return ReturnProvenance::Local;
    }
    ReturnProvenance::NotAReference
}

/// Summarize each function in `funcs` that has a body. Functions with an empty body
/// (signature-stripped, or genuinely bodiless like `extern`) are skipped, so the map carries an
/// entry only where the summary is meaningful; a missing lookup means `AnyParam`.
pub fn provenance_map<'a>(
    funcs: impl Iterator<Item = &'a Function>,
    decls: &Decls,
) -> HashMap<Symbol, ReturnProvenance> {
    let mut map = HashMap::new();
    for f in funcs {
        if f.body.is_empty() {
            continue;
        }
        map.insert(
            f.name.clone(),
            compute_return_provenance_with(f, decls, &|_| None),
        );
    }
    map
}

/// Recompute the summaries in `map` with each call read through its callee's summary, until none
/// changes. Every summary starts sound and each round only narrows it, so the result is sound
/// after any number of rounds; recursion keeps `AnyParam`.
pub fn refine_through_calls(
    funcs: &[&Function],
    decls: &Decls,
    map: &mut HashMap<Symbol, ReturnProvenance>,
) {
    let funcs: Vec<&Function> = funcs
        .iter()
        .copied()
        .filter(|f| map.contains_key(&f.name) && can_hold_reference(&f.return_type))
        .collect();
    // Each round settles at least one more level of the call graph; deeper chains stay
    // conservative.
    for _ in 0..16 {
        let mut changed = false;
        for f in &funcs {
            let prov = compute_return_provenance_with(f, decls, &|name| map.get(name).copied());
            if map.get(&f.name) != Some(&prov) {
                map.insert(f.name.clone(), prov);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// The name a method's summary is kept under: the impl's target and trait, and the method.
pub fn method_key(ib: &ImplBlock, method: &str) -> String {
    crate::syntax::types::mangle_method(
        &ib.target_type,
        ib.trait_name.as_ref(),
        &ib.trait_args,
        method,
    )
}

/// Summarize each method of `impls`, reading a call to a free function through `functions`. A
/// method calls another method through a receiver, which the walk does not follow, so one pass is
/// enough.
pub fn method_provenance_map<'i>(
    impls: impl Iterator<Item = &'i ImplBlock>,
    decls: &Decls,
    functions: &HashMap<Symbol, ReturnProvenance>,
) -> HashMap<String, ReturnProvenance> {
    let callee = |name: &str| functions.get(name).copied();
    let mut map = HashMap::new();
    for ib in impls {
        for m in ib.methods.iter().filter(|m| !m.body.is_empty()) {
            let prov = compute_return_provenance_with(m, decls, &callee);
            // Two impls under one name are an error. Neither summary is chosen.
            map.entry(method_key(ib, m.name.as_ref()))
                .and_modify(|p| {
                    if *p != prov {
                        *p = ReturnProvenance::AnyParam;
                    }
                })
                .or_insert(prov);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> crate::syntax::Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        parser.parse().expect("parses")
    }

    fn summarize(src: &str, fname: &str) -> ReturnProvenance {
        let program = parse(src);
        let f = program
            .functions
            .iter()
            .find(|f| f.name.as_ref() == fname)
            .expect("function present");
        let decls = Decls::new(&program.structs, &program.enums);
        compute_return_provenance_with(f, &decls, &|_| None)
    }

    fn from(into: u32, held: u32) -> ReturnProvenance {
        ReturnProvenance::FromParams { into, held }
    }

    #[test]
    fn single_source_names_only_its_param() {
        // `pick` returns from `b` (slot 1) — the signature Rust cannot elide.
        let src = "fn pick(a : &i32, b : &i32) -> &i32 { return b; }";
        assert_eq!(summarize(src, "pick"), from(0b10, 0));
    }

    #[test]
    fn reborrow_through_param_field_names_that_param() {
        let src = "struct M { slot : i32 } fn f(a : &M, b : &M) -> &i32 { return &b.slot; }";
        assert_eq!(summarize(src, "f"), from(0b10, 0));
    }

    #[test]
    fn value_return_is_not_a_reference() {
        let src = "fn f(a : &i32) -> i32 { return 3; }";
        assert_eq!(summarize(src, "f"), ReturnProvenance::NotAReference);
    }

    #[test]
    fn multi_source_unions_the_params() {
        // params c=0, a=1, b=2 -> {1, 2}
        let src =
            "fn choose(c : i32, a : &i32, b : &i32) -> &i32 { if c == 0 { return a; } return b; }";
        assert_eq!(summarize(src, "choose"), from(0b110, 0));
    }

    #[test]
    fn transitive_local_binding_is_threaded() {
        let src =
            "struct M { slot : i32 } fn via(a : &M, b : &M) -> &i32 { let t = &b.slot; return t; }";
        assert_eq!(summarize(src, "via"), from(0b10, 0));
    }

    #[test]
    fn returning_a_local_reference_is_local() {
        let src = "fn dangle() -> &i32 { let x = 5; return &x; }";
        assert_eq!(summarize(src, "dangle"), ReturnProvenance::Local);
    }

    #[test]
    fn the_last_expression_of_the_body_is_a_return() {
        let src = "fn second(a : &i32, b : &i32) -> &i32 { b }";
        assert_eq!(summarize(src, "second"), from(0b10, 0));
    }

    #[test]
    fn a_variable_given_a_second_value_points_at_both() {
        let src = "fn either(a : &i32, b : &i32) -> &i32 { let mut r = a; r = b; return r; }";
        assert_eq!(summarize(src, "either"), from(0b11, 0));
    }

    #[test]
    fn a_reference_made_from_a_number_points_anywhere() {
        let src = "fn at(a : &i32) -> &i32 { return unsafe { &*(64 as *mut i32) }; }";
        assert_eq!(summarize(src, "at"), ReturnProvenance::AnyParam);
    }

    #[test]
    fn call_in_return_position_is_unknown_without_the_callee() {
        let src = "fn probe(m : &i32) -> &i32 { return m; } fn wrap(a : &i32, b : &i32) -> &i32 { return probe(b); }";
        assert_eq!(summarize(src, "wrap"), ReturnProvenance::AnyParam);
    }

    /// `Vec` and its iterators, as `stdlib/std/vec.vx` writes them.
    const VEC: &str = "
        enum Option<T> { Some(T), None }
        struct Vec<T> { data : *mut T, len : i32, capacity : i32 }
        struct VecIter<T> { vec : &Vec<T>, current : i32, back : i32 }
        struct VecIterMut<T> { vec : &mut Vec<T>, current : i32, back : i32 }
        fn next(self : &mut VecIter<T>) -> Option<&T> {
          let mut ret = Option<&T>::None();
          if self.current < self.back {
            let r : &T = unsafe { &(*self.vec).data[self.current] };
            self.current = self.current + 1;
            ret = Option<&T>::Some(r);
          }
          return ret;
        }
        fn next_mut(self : &mut VecIterMut<T>) -> Option<&mut T> {
          if self.current >= self.back {
            return Option<&mut T>::None();
          }
          let r : &mut T = unsafe { &mut (*self.vec).data[self.current] };
          self.current = self.current + 1;
          return Option<&mut T>::Some(r);
        }
        fn get_ref(self : &Vec<T>, index : i32) -> &T {
          check(index as i64, self.len as i64);
          return unsafe { &self.data[index] };
        }
        fn as_slice(self : &Vec<T>) -> &T {
          let ptr : *mut T = self.data;
          return unsafe { &*ptr };
        }
        fn leak(self : Vec<T>) -> &mut T {
          let ptr : *mut T = self.data;
          return unsafe { &mut *ptr };
        }
        fn iter(self : &Vec<T>) -> VecIter<T> {
          return VecIter<T> { vec : self, current : 0, back : self.len };
        }
    ";

    #[test]
    fn an_item_through_a_shared_reference_field_points_where_the_field_points() {
        assert_eq!(summarize(VEC, "next"), from(0, 0b1));
    }

    #[test]
    fn an_item_through_a_mut_reference_field_keeps_the_receiver() {
        assert_eq!(summarize(VEC, "next_mut"), from(0b1, 0));
    }

    #[test]
    fn an_element_through_a_raw_pointer_field_keeps_the_struct_that_holds_it() {
        assert_eq!(summarize(VEC, "get_ref"), from(0b1, 0));
        assert_eq!(summarize(VEC, "as_slice"), from(0b1, 0));
    }

    #[test]
    fn a_raw_pointer_in_a_struct_given_by_value_points_anywhere() {
        assert_eq!(summarize(VEC, "leak"), ReturnProvenance::AnyParam);
    }

    #[test]
    fn a_struct_that_holds_the_receiver_keeps_it() {
        assert_eq!(summarize(VEC, "iter"), from(0b1, 0));
    }

    #[test]
    fn a_reference_read_back_after_a_store_points_anywhere() {
        // `h.r = b` puts `b` into what `h` holds. Reading it back must not answer "what `h` held".
        let src = "struct H { r : &i32 } \
                   fn stash(h : &mut H, b : &i32) -> &i32 { h.r = b; return h.r; }";
        assert_eq!(summarize(src, "stash"), ReturnProvenance::AnyParam);
    }

    #[test]
    fn a_reference_read_back_after_a_call_points_anywhere() {
        let src = "struct H { r : &i32 } \
                   fn stash(h : &mut H, b : &i32) -> &i32 { put(h, b); return h.r; }";
        assert_eq!(summarize(src, "stash"), ReturnProvenance::AnyParam);
    }

    #[test]
    fn a_reference_field_read_through_a_parameter_points_where_the_field_points() {
        let src = "struct H { r : &i32 } fn get(h : &H) -> &i32 { return h.r; }";
        assert_eq!(summarize(src, "get"), from(0, 0b1));
    }

    #[test]
    fn a_name_bound_by_a_match_arm_hides_the_variable_of_that_name() {
        // The `r` the arm returns is the payload of `o`, not the `r` made from `a`.
        let src = "enum Option<T> { Some(T), None } \
                   fn f(a : &i32, o : Option<&i32>) -> &i32 { \
                     let r = a; \
                     match o { Option::Some(r) => { return r; } Option::None => { return a; } } \
                     return r; }";
        assert_eq!(summarize(src, "f"), from(0b01, 0b10));
    }

    #[test]
    fn a_type_parameter_given_by_value_may_be_a_reference() {
        let src = "fn id<T>(x : T) -> T { return x; }";
        assert_eq!(summarize(src, "id"), from(0b1, 0b1));
    }

    #[test]
    fn a_match_whose_arms_return_or_stop_adds_no_value() {
        // `Option::unwrap`: the `match` is the last statement, and no arm ends in a value.
        let src = "enum Option<T> { Some(T), None } \
                   fn unwrap(self : Option<&i32>) -> &i32 { \
                     match self { Option::Some(v) => { return v; } Option::None => { stop(); } } }";
        assert_eq!(summarize(src, "unwrap"), from(0, 0b1));
    }

    #[test]
    fn a_shared_borrow_given_to_a_call_does_not_change_the_variable() {
        let src = "enum Option<T> { Some(T), None } \
                   fn keep(o : Option<&i32>, b : &i32) -> Option<&i32> { \
                     let s = o; look(&s, b); return s; }";
        assert_eq!(summarize(src, "keep"), from(0, 0b1));
        let src = "enum Option<T> { Some(T), None } \
                   fn keep(o : Option<&i32>, b : &i32) -> Option<&i32> { \
                     let mut s = o; put(&mut s, b); return s; }";
        assert_eq!(summarize(src, "keep"), ReturnProvenance::AnyParam);
    }

    /// Every function in `src`, summarized with calls read through their callees.
    fn summarize_all(src: &str) -> HashMap<Symbol, ReturnProvenance> {
        let program = parse(src);
        let decls = Decls::new(&program.structs, &program.enums);
        let funcs: Vec<&Function> = program.functions.iter().collect();
        let mut map = provenance_map(funcs.iter().copied(), &decls);
        refine_through_calls(&funcs, &decls, &mut map);
        map
    }

    #[test]
    fn a_call_maps_the_callee_parameters_onto_its_arguments() {
        // `wrap` passes its `b` as `probe`'s only parameter, and swaps them for `second`.
        let map = summarize_all(
            "fn probe(m : &i32) -> &i32 { return m; } \
             fn second(a : &i32, b : &i32) -> &i32 { return b; } \
             fn wrap(a : &i32, b : &i32) -> &i32 { return probe(b); } \
             fn swap(a : &i32, b : &i32) -> &i32 { return second(b, a); }",
        );
        assert_eq!(map[&Symbol::from("wrap")], from(0b10, 0));
        assert_eq!(map[&Symbol::from("swap")], from(0b01, 0));
    }

    #[test]
    fn a_chain_of_calls_is_followed_to_the_end() {
        let map = summarize_all(
            "fn three(a : &i32, b : &i32) -> &i32 { return two(a, b); } \
             fn two(a : &i32, b : &i32) -> &i32 { return one(b); } \
             fn one(m : &i32) -> &i32 { return m; }",
        );
        assert_eq!(map[&Symbol::from("three")], from(0b10, 0));
    }

    #[test]
    fn what_an_argument_holds_follows_how_it_is_passed() {
        let map = summarize_all(
            "struct H { r : &i32 } \
             fn get(h : &H) -> &i32 { return h.r; } \
             fn through_a_reference(h : &H) -> &i32 { return get(h); } \
             fn through_a_borrow(h : H) -> &i32 { return get(&h); }",
        );
        // `h` is a reference, so what `*h` holds is what the caller's argument holds.
        assert_eq!(map[&Symbol::from("through_a_reference")], from(0, 0b1));
        // `&h` borrows a parameter given by value, so the result holds what that parameter holds.
        assert_eq!(map[&Symbol::from("through_a_borrow")], from(0, 0b1));
    }

    #[test]
    fn recursion_and_a_closure_named_like_a_function_stay_unknown() {
        let map = summarize_all(
            "fn probe(m : &i32) -> &i32 { return m; } \
             fn again(a : &i32, b : &i32) -> &i32 { return again(a, b); } \
             fn hidden(a : &i32, b : &i32, probe : i32) -> &i32 { return probe(b); }",
        );
        assert_eq!(map[&Symbol::from("again")], ReturnProvenance::AnyParam);
        assert_eq!(map[&Symbol::from("hidden")], ReturnProvenance::AnyParam);
    }

    #[test]
    fn a_method_is_summarized_under_its_impl() {
        let program = parse(&format!(
            "{VEC} impl<T> Iterator for VecIter<T> {{ fn next(self : &mut VecIter<T>) -> Option<&T> {{ \
               return Option<&T>::Some(unsafe {{ &(*self.vec).data[self.current] }}); }} }}"
        ));
        let decls = Decls::new(&program.structs, &program.enums);
        let map = method_provenance_map(program.impls.iter(), &decls, &HashMap::new());
        let key = method_key(&program.impls[0], "next");
        assert_eq!(map[&key], from(0, 0b1));
    }

    /// The inline slot-0 code (#265) names a single in-budget parameter as `slot + 1` and degrades
    /// every wider or unknown case to the conservative top; a non-reference return encodes as none.
    #[test]
    fn encode_maps_single_param_and_degrades_the_rest() {
        assert_eq!(encode_return_provenance(&from(0b0001, 0)), 1);
        assert_eq!(encode_return_provenance(&from(0b0010, 0)), 2);
        assert_eq!(encode_return_provenance(&from(0b0100, 0)), 3);
        assert_eq!(encode_return_provenance(&from(0b1000, 0)), 4);
        // Over budget -> conservative top.
        assert_eq!(encode_return_provenance(&from(0b0110, 0)), 7); // two sources
        assert_eq!(encode_return_provenance(&from(1 << 4, 0)), 7); // slot >= 4
        assert_eq!(encode_return_provenance(&from(0, 0b1)), 7); // a held set
        assert_eq!(encode_return_provenance(&ReturnProvenance::AnyParam), 7);
        assert_eq!(encode_return_provenance(&ReturnProvenance::Local), 7);
        // No reference returned -> no alias source.
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::NotAReference),
            0
        );
    }

    /// The packed code is always a *conservative refinement* of the summary it replaces: for every
    /// parameter the summary flags as an alias source, the packed code flags it too. This is the
    /// soundness property that makes the fixed-width inline encoding safe — it may lose precision (a
    /// multi-source union widens to "any"), never soundness (#265). A call reads the code for both
    /// questions, so it must cover the held set too.
    #[test]
    fn inline_code_conservatively_refines_the_summary() {
        let cases = [
            ReturnProvenance::NotAReference,
            ReturnProvenance::Local,
            ReturnProvenance::AnyParam,
            from(0b0001, 0),
            from(0b1000, 0),
            from(0b0110, 0), // multi-source -> widens to top
            from(1 << 5, 0), // out-of-budget slot -> top
            from(0, 0b0001), // held -> top
            from(0b0001, 0b0010),
        ];
        for prov in cases {
            let code = encode_return_provenance(&prov);
            for slot in 0..32 {
                assert!(
                    !prov.holds(slot) || inline_prov_includes(code, slot),
                    "summary {prov:?} flags slot {slot} but the packed code {code} does not"
                );
                assert!(
                    !prov.includes(slot) || inline_prov_includes(code, slot),
                    "summary {prov:?} flags slot {slot} but the packed code {code} does not"
                );
            }
        }
    }
}
