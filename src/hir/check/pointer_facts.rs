//===- pointer_facts.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The raw-pointer facts of a body, and the proof of a pointer `requires` at a call. A pointer
// value is a base and an offset in elements. A fact says that a range of elements at a base is
// in one live allocation (`Valid`), holds values (`Init`), or shares no byte with another range
// (`Apart`). docs/discussions/raw_pointer_facts.md has the design.
//
// Each event that can free memory ends each fact about memory that the body does not name: a
// call with an argument that can free memory, an assignment that drops a value, the drop of a
// local, an `mlir!` block with such an input or a clobber, and a `spawn`. No callee is known to
// keep memory alive yet, so a call that gets a raw pointer is such an event. Each ended fact
// keeps the place and the reason of its event, for the note of E8006.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::check::raw::expr_mentions;
use std::collections::HashMap;

/// Where the memory of a base is.
#[derive(Clone, Debug, PartialEq)]
enum Memory {
    /// The storage of the local variable `var`. `drops` is true when its type has a drop.
    Local { var: String, drops: bool },
    /// Memory that the body does not name.
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Valid,
    Init,
}

/// `Valid` or `Init` for the elements `lo` to `hi - 1` at `base`.
#[derive(Clone, Debug)]
struct Range {
    id: usize,
    kind: Kind,
    base: usize,
    lo: Expr,
    hi: Expr,
}

/// The `len` elements from `lo` at each of two bases share no byte.
#[derive(Clone, Debug)]
struct Apart {
    id: usize,
    first: (usize, Expr),
    second: (usize, Expr),
    len: Expr,
}

/// A fact that an event ended, with the place and the reason of the event, for the note of
/// E8006.
#[derive(Clone, Debug)]
struct Ended {
    range: Range,
    at: Span,
    why: String,
}

/// A call as the program wrote it, before the check can turn it into another call, such as a
/// method call into a call of its function.
pub(crate) struct WrittenCall {
    /// The name of the function or the method.
    name: Option<crate::symbol::Symbol>,
    /// For a call of a local variable, such as a closure: whether its value can free memory.
    local_can_free: Option<bool>,
    at: Span,
}

/// A pointer value: a base, and an offset in elements that is the exact sum of its terms.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PointerValue {
    base: usize,
    offset: Vec<Expr>,
}

/// What the checker knows about the raw pointers of the body that it checks.
#[derive(Clone, Debug, Default)]
pub(crate) struct PointerFacts {
    /// The memory of each base.
    memory: HashMap<usize, Memory>,
    /// The value of each local variable of a raw pointer type.
    entries: HashMap<String, PointerValue>,
    ranges: Vec<Range>,
    aparts: Vec<Apart>,
    ended: Vec<Ended>,
}

impl PointerFacts {
    pub(crate) fn entry_of(&self, name: &str) -> Option<PointerValue> {
        self.entries.get(name).cloned()
    }

    /// Drops each fact and each entry whose expressions name `name`, and the entry of `name`.
    /// Called where the variable `name` can change or stops being the same variable.
    pub(crate) fn forget(&mut self, name: &str) {
        self.entries.remove(name);
        self.entries
            .retain(|_, v| !v.offset.iter().any(|t| expr_mentions(t, name)));
        let names = |r: &Range| expr_mentions(&r.lo, name) || expr_mentions(&r.hi, name);
        self.ranges.retain(|r| !names(r));
        self.ended.retain(|e| !names(&e.range));
        self.aparts.retain(|a| {
            ![&a.first.1, &a.second.1, &a.len]
                .iter()
                .any(|e| expr_mentions(e, name))
        });
    }

    /// Ends each range that `ends` selects, and records the event for the note of E8006.
    fn end_ranges(&mut self, at: Span, why: &str, ends: impl Fn(&Range, Option<&Memory>) -> bool) {
        let memory = &self.memory;
        let (gone, kept) = std::mem::take(&mut self.ranges)
            .into_iter()
            .partition(|r| ends(r, memory.get(&r.base)));
        self.ranges = kept;
        self.ended.extend(gone.into_iter().map(|range| Ended {
            range,
            at,
            why: why.to_string(),
        }));
    }

    /// An event that can free memory that the body does not name. It also ends `Init` for a
    /// local variable whose type has a drop: the event can move its value out and drop it. No
    /// event ends an `Apart` fact: a free does not change an address.
    fn end_other_memory(&mut self, at: Span, why: &str) {
        self.end_ranges(at, why, |r, memory| match memory {
            Some(Memory::Local { drops, .. }) => *drops && r.kind == Kind::Init,
            _ => true,
        });
    }

    /// The storage of the local variable `var` ends.
    fn end_local(&mut self, var: &str, at: Span, why: &str) {
        self.end_ranges(
            at,
            why,
            |_, memory| matches!(memory, Some(Memory::Local { var: v, .. }) if v == var),
        );
    }

    /// A move takes the value out of the local variable `var`.
    fn end_init_of_local(&mut self, var: &str, at: Span, why: &str) {
        self.end_ranges(at, why, |r, memory| {
            r.kind == Kind::Init && matches!(memory, Some(Memory::Local { var: v, .. }) if v == var)
        });
    }

    fn has_local(&self, var: &str) -> bool {
        self.ranges.iter().any(
            |r| matches!(self.memory.get(&r.base), Some(Memory::Local { var: v, .. }) if v == var),
        )
    }

    /// The facts that hold at the end of both `self` and `other`, each of which started from
    /// the same state.
    pub(crate) fn join(mut self, other: &PointerFacts) -> PointerFacts {
        self.entries
            .retain(|name, v| other.entries.get(name) == Some(v));
        self.ranges
            .retain(|r| other.ranges.iter().any(|o| o.id == r.id));
        self.aparts
            .retain(|a| other.aparts.iter().any(|o| o.id == a.id));
        // A fact that ended in either branch keeps the record of its event.
        for e in &other.ended {
            if !self.ended.iter().any(|s| s.range.id == e.range.id) {
                self.ended.push(e.clone());
            }
        }
        for (base, memory) in &other.memory {
            self.memory.entry(*base).or_insert_with(|| memory.clone());
        }
        self
    }
}

/// The functions of `core::ptr` that compute an address and free nothing.
const ADDRESS_FUNCTIONS: [&str; 11] = [
    "offset",
    "offset_mut",
    "add",
    "add_mut",
    "sub",
    "sub_mut",
    "eq",
    "is_null",
    "is_null_mut",
    "null",
    "null_mut",
];

/// The name of the function that the call `name` calls, without its module path and type
/// arguments. An instance of a generic function is named `offset$i64`.
fn plain_name(name: &str) -> &str {
    let name = name.split(['<', '$']).next().unwrap_or(name);
    name.rsplit("::").next().unwrap_or(name)
}

/// A term of an exact goal for the prover: program values are compared and added as exact
/// integers, while each program value keeps the meaning that its own operations give it.
pub(crate) enum Exact {
    Term(Box<Expr>),
    Sum(Vec<Exact>),
    Le(Box<Exact>, Box<Exact>),
    And(Vec<Exact>),
    Or(Vec<Exact>),
}

fn sum(terms: &[Expr], more: &[&Expr]) -> Exact {
    Exact::Sum(
        terms
            .iter()
            .chain(more.iter().copied())
            .map(|t| term(t.clone()))
            .collect(),
    )
}

fn term(e: Expr) -> Exact {
    Exact::Term(Box::new(e))
}

fn le(a: Exact, b: Exact) -> Exact {
    Exact::Le(Box::new(a), Box::new(b))
}

fn zero() -> Expr {
    Expr::Number(NumberExpr {
        value: "0".to_string().into(),
        ty: None,
        span: Span::default(),
    })
}

/// The clause `valid(p, n)`, `initialized(p, n)` or `!overlaps(p, q, n)`, as its function, its
/// pointer arguments and its length.
fn read_clause(e: &Expr) -> Option<(&str, Vec<&Expr>, &Expr)> {
    let call = match e {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => &*u.expr,
        e => e,
    };
    let Expr::FunctionCall(fc) = call else {
        return None;
    };
    let (pointers, len) = fc.args.split_at(fc.args.len().checked_sub(1)?);
    let name = fc.name.as_ref();
    let want = match name {
        "valid" | "initialized" => 1,
        "overlaps" => 2,
        _ => return None,
    };
    (pointers.len() == want).then(|| (name, pointers.iter().collect(), &len[0]))
}

impl<'a> TypeChecker<'a> {
    fn new_base(&mut self, memory: Memory) -> usize {
        self.pointer_fact_ids += 1;
        self.pointer_facts
            .memory
            .insert(self.pointer_fact_ids, memory);
        self.pointer_fact_ids
    }

    fn add_range(&mut self, kind: Kind, base: usize, lo: Expr, hi: Expr) {
        self.pointer_fact_ids += 1;
        let id = self.pointer_fact_ids;
        self.pointer_facts.ranges.push(Range {
            id,
            kind,
            base,
            lo,
            hi,
        });
    }

    /// The facts at the start of the body of `func`: each raw pointer parameter has a base of
    /// its own, and each pointer `requires` is a fact. `valid(p, n)` also says that `n >= 0`.
    pub(crate) fn start_pointer_facts(&mut self, func: &Function) {
        self.pointer_facts = PointerFacts::default();
        for (name, ty) in &func.params {
            if matches!(ty, Type::Pointer(..)) {
                let base = self.new_base(Memory::Other);
                let entry = PointerValue {
                    base,
                    offset: Vec::new(),
                };
                self.pointer_facts.entries.insert(name.to_string(), entry);
            }
        }
        let base_of = |facts: &PointerFacts, p: &Expr| match p {
            Expr::Identifier(id) => facts.entries.get(id.name.as_ref()).map(|v| v.base),
            _ => None,
        };
        for clause in &func.requires {
            let Some((name, pointers, len)) = read_clause(clause) else {
                continue;
            };
            let bases: Vec<usize> = pointers
                .iter()
                .filter_map(|p| base_of(&self.pointer_facts, p))
                .collect();
            if bases.len() != pointers.len() {
                continue;
            }
            match name {
                "valid" | "initialized" => {
                    let kind = if name == "valid" {
                        Kind::Valid
                    } else {
                        Kind::Init
                    };
                    self.add_range(kind, bases[0], zero(), len.clone());
                    if kind == Kind::Valid && crate::hir::prover::is_modelled(len) {
                        self.consteval
                            .constraints
                            .push(Expr::RelationalOp(RelationalOpExpr {
                                lhs: Box::new(len.clone()),
                                op: RelationalOp::Ge,
                                rhs: Box::new(zero()),
                                span: Span::default(),
                                operand_ty: None,
                            }));
                    }
                }
                _ => {
                    self.pointer_fact_ids += 1;
                    let id = self.pointer_fact_ids;
                    self.pointer_facts.aparts.push(Apart {
                        id,
                        first: (bases[0], zero()),
                        second: (bases[1], zero()),
                        len: len.clone(),
                    });
                }
            }
        }
    }

    /// The pointer value of `e`, a value of the type `*T` with `T` the type `pointee`.
    fn pointer_value(&mut self, e: &Expr, pointee: Option<&Type>) -> PointerValue {
        let fresh = |c: &mut Self| PointerValue {
            base: c.new_base(Memory::Other),
            offset: Vec::new(),
        };
        match e {
            Expr::Identifier(id) => {
                if let Some(v) = self.pointer_facts.entries.get(id.name.as_ref()) {
                    return v.clone();
                }
                let v = fresh(self);
                if self.lookup(id.name.as_ref()).is_some() {
                    self.pointer_facts
                        .entries
                        .insert(id.name.to_string(), v.clone());
                }
                v
            }
            Expr::FunctionCall(fc) if fc.args.len() == 2 => {
                let negative = match plain_name(fc.name.as_ref()) {
                    "offset" | "offset_mut" | "add" | "add_mut" => false,
                    "sub" | "sub_mut" => true,
                    _ => return fresh(self),
                };
                let mut v = self.pointer_value(&fc.args[0], pointee);
                let count = &fc.args[1];
                if !crate::hir::prover::is_modelled(count) {
                    return fresh(self);
                }
                v.offset.push(if negative {
                    Expr::UnaryOp(UnaryOpExpr {
                        op: UnaryOp::Neg,
                        expr: Box::new(count.clone()),
                        span: Span::default(),
                    })
                } else {
                    count.clone()
                });
                v
            }
            // A cast between `*const T` and `*mut T` keeps the value.
            Expr::AsCast(c) => match (&c.source_ty, &c.target_ty) {
                (Some(Type::Pointer(from, ..)), Type::Pointer(to, ..)) if from == to => {
                    self.pointer_value(&c.expr, pointee)
                }
                _ => fresh(self),
            },
            Expr::Borrow(b) => self.pointer_to_a_local(&b.expr, pointee),
            _ => fresh(self),
        }
    }

    /// A pointer made from `&place`. When `place` is a local variable or a field of one, the
    /// pointer points at its storage, which holds one value of the type: `Valid(b, 0, 1)` and
    /// `Init(b, 0, 1)`.
    fn pointer_to_a_local(&mut self, place: &Expr, pointee: Option<&Type>) -> PointerValue {
        fn root(e: &Expr) -> Option<&str> {
            match e {
                Expr::Identifier(id) => Some(id.name.as_ref()),
                Expr::MemberAccess(m) => root(&m.base),
                _ => None,
            }
        }
        let place_ty = self.check_expr_type_probe(place);
        let local = root(place).filter(|r| {
            self.lookup(r).is_some_and(|(t, _)| {
                !matches!(t, Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..))
            })
        });
        let Some(var) = local.filter(|_| pointee == Some(&place_ty)) else {
            return PointerValue {
                base: self.new_base(Memory::Other),
                offset: Vec::new(),
            };
        };
        let var_ty = self.lookup(var).map(|(t, _)| t.clone());
        let drops = var_ty
            .is_none_or(|t| matches!(t, Type::Generic(..)) || !self.drops_glue(&t).is_empty());
        let memory = Memory::Local {
            var: var.to_string(),
            drops,
        };
        let base = self.new_base(memory);
        let one = Expr::Number(NumberExpr {
            value: "1".to_string().into(),
            ty: None,
            span: Span::default(),
        });
        self.add_range(Kind::Valid, base, zero(), one.clone());
        self.add_range(Kind::Init, base, zero(), one);
        PointerValue {
            base,
            offset: Vec::new(),
        }
    }

    /// `let name = init` or `name = init` gave `name` a new value.
    pub(crate) fn note_pointer_value(&mut self, name: &str, value: Option<PointerValue>) {
        if let Some(v) = value {
            self.pointer_facts.entries.insert(name.to_string(), v);
        }
    }

    /// The value of `init` for the variable `name`, when `name` is a raw pointer. Computed before
    /// the facts about `name` are dropped, because `init` can use the old value.
    pub(crate) fn pointer_value_for(&mut self, name: &str, init: &Expr) -> Option<PointerValue> {
        match self.lookup(name).map(|(t, _)| t.clone()) {
            Some(Type::Pointer(pointee, ..)) => Some(self.pointer_value(init, Some(&pointee))),
            _ => None,
        }
    }

    /// Whether a value of the type `ty` can free memory when the program passes it to a call:
    /// it holds a raw pointer or a type with a drop, at any depth.
    pub(crate) fn type_can_free(&mut self, ty: &Type, depth: usize) -> bool {
        if depth > 8 {
            return true;
        }
        match ty {
            Type::Scalar(_) | Type::Simd(..) | Type::Function(..) | Type::Const(_) => false,
            Type::Pointer(..) | Type::Closure(..) | Type::Generic(..) | Type::Unknown => true,
            Type::Borrow { inner, .. }
            | Type::Ref(inner, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => self.type_can_free(inner, depth + 1),
            Type::Struct(name, _) if name.as_ref() == "void" => false,
            Type::Struct(name, _) | Type::Enum(name, _) => {
                !self.drops_glue(ty).is_empty()
                    || self
                        .declared_part_types(name, &[])
                        .iter()
                        .any(|t| self.type_can_free(t, depth + 1))
            }
            Type::GenericInstance(base, args) => match &**base {
                Type::Struct(name, _) | Type::Enum(name, _) => {
                    !self.drops_glue(ty).is_empty()
                        || self
                            .declared_part_types(name, args)
                            .iter()
                            .any(|t| self.type_can_free(t, depth + 1))
                }
                _ => true,
            },
            Type::Tensor(..) | Type::Matrix | Type::Module(..) => true,
        }
    }

    /// An event at `at` that can free memory that the body does not name. `why` gives the
    /// event for the note of E8006, such as "the call to 'push'".
    pub(crate) fn pointer_facts_event(&mut self, at: Span, why: impl FnOnce(&Self) -> String) {
        if !self.pointer_facts.ranges.is_empty() {
            let why = why(self);
            self.pointer_facts.end_other_memory(at, &why);
        }
    }

    /// The variables of a scope end. Their storage ends, and the drop of a value of a type with a
    /// drop can free memory.
    pub(crate) fn pointer_facts_end_scope(
        &mut self,
        vars: &HashMap<crate::symbol::Symbol, (Type, Topology)>,
    ) {
        let at = self.current_statement_span;
        let mut dropped: Vec<&str> = Vec::new();
        for (name, (ty, _)) in vars {
            if self.pointer_facts.has_local(name.as_ref()) {
                let why = format!("the end of the scope of `{name}`");
                self.pointer_facts.end_local(name.as_ref(), at, &why);
            }
            self.pointer_facts.forget(name.as_ref());
            if !self.drops_glue(ty).is_empty() || matches!(ty, Type::Generic(..)) {
                dropped.push(name.as_ref());
            }
        }
        if let Some(name) = dropped.iter().min() {
            self.pointer_facts_event(at, |_| {
                format!("the end of the scope of `{name}`, which drops its value")
            });
        }
    }

    /// A move took the value out of the local variable `var`, or out of a field of it.
    pub(crate) fn pointer_facts_move_out(&mut self, var: &str) {
        if self.pointer_facts.has_local(var) {
            let why = format!("the move out of `{var}`");
            let at = self.current_statement_span;
            self.pointer_facts.end_init_of_local(var, at, &why);
        }
    }

    /// At the start of a loop: a fact holds in the body only when no pass ends it, so an event
    /// that the body can have ends the facts before the body.
    pub(crate) fn end_pointer_facts_a_loop_can_end(&mut self, body: &[Statement]) {
        if self.loop_body_can_free(body) {
            let at = self.current_statement_span;
            self.pointer_facts_event(at, |_| "the loop, whose body can free memory".to_string());
        }
    }

    /// After a loop: the checker made one pass over the body, and a variable that the body
    /// writes can hold the value of any pass, or its value before the loop. The events of the
    /// body ended their facts before the body.
    pub(crate) fn pointer_facts_after_a_loop(&mut self, body: &[Statement]) {
        for name in crate::hir::check::raw::names_a_body_writes(body) {
            self.pointer_facts.forget(&name);
        }
    }

    /// Whether a pass of `body` can have an event. The body is not checked yet, so this reads
    /// its syntax, and anything that it cannot show to free nothing counts.
    fn loop_body_can_free(&self, body: &[Statement]) -> bool {
        let mut frees = self.statements_can_free(body);
        crate::syntax::visit_exprs(body, &mut |e| {
            frees |= match e {
                Expr::FunctionCall(fc) => {
                    !ADDRESS_FUNCTIONS.contains(&plain_name(fc.name.as_ref()))
                }
                Expr::MethodCall(_)
                | Expr::IndirectCall(_)
                | Expr::InlineMlir(_)
                | Expr::SpawnOn(_)
                | Expr::TaskBlock(_) => true,
                Expr::If(i) => {
                    self.statements_can_free(&i.then_block)
                        || i.else_block
                            .as_ref()
                            .is_some_and(|b| self.statements_can_free(b))
                }
                Expr::Match(m) => m.arms.iter().any(|a| self.statements_can_free(&a.body)),
                Expr::UnsafeBlock(u) => self.statements_can_free(&u.stmts),
                Expr::ComptimeBlock(c) => self.statements_can_free(&c.stmts),
                _ => false,
            };
        });
        frees
    }

    /// Whether a statement of `stmts`, or of a loop in it, can drop a value: a `drop`, a `let`
    /// whose value is not a plain number, or an assignment to a variable that can hold memory.
    fn statements_can_free(&self, stmts: &[Statement]) -> bool {
        stmts.iter().any(|s| match s {
            Statement::Drop(_) => true,
            Statement::LetDecl(l) => !self.is_plain_number(&l.expr),
            Statement::Assign(a) => !self.is_plain_number(&a.lhs),
            Statement::CompoundAssign(a) => !self.is_plain_number(&a.lhs),
            Statement::ForLoop(l) => self.statements_can_free(&l.body),
            Statement::Loop(l) => self.statements_can_free(&l.body),
            _ => false,
        })
    }

    /// Whether `e` is a number that the syntax shows: a literal, a variable of a scalar type, an
    /// element of a pointer to a scalar, or an operation on those.
    fn is_plain_number(&self, e: &Expr) -> bool {
        let scalar = |name: &str| matches!(self.lookup(name), Some((Type::Scalar(_), _)));
        match crate::syntax::bare_unsafe_value(e) {
            Expr::Number(_) => true,
            Expr::Identifier(id) => scalar(id.name.as_ref()),
            Expr::IndexAccess(ix) => match &*ix.base {
                Expr::Identifier(id) => {
                    matches!(
                        self.lookup(id.name.as_ref()),
                        Some((Type::Pointer(inner, ..), _)) if matches!(**inner, Type::Scalar(_))
                    ) && self.is_plain_number(&ix.index)
                }
                _ => false,
            },
            Expr::BinaryOp(b) => self.is_plain_number(&b.lhs) && self.is_plain_number(&b.rhs),
            Expr::RelationalOp(r) => self.is_plain_number(&r.lhs) && self.is_plain_number(&r.rhs),
            Expr::LogicalOp(l) => self.is_plain_number(&l.lhs) && self.is_plain_number(&l.rhs),
            Expr::UnaryOp(u) => self.is_plain_number(&u.expr),
            _ => false,
        }
    }

    /// The call `expr` as the program wrote it, read before the check.
    pub(crate) fn written_call(&mut self, expr: &Expr) -> WrittenCall {
        let (name, at) = match (self.method_call_written.take(), expr) {
            (Some((name, at)), _) => (Some(name), at),
            (None, Expr::FunctionCall(fc)) => (Some(fc.name.clone()), fc.span),
            (None, Expr::MethodCall(m)) => (Some(m.method_name.clone()), m.span),
            (None, e) => (None, e.span()),
        };
        // The parser gives some calls no place, such as `(h.f)()`.
        let at = if at == Span::default() {
            self.current_statement_span
        } else {
            at
        };
        let local = match expr {
            Expr::FunctionCall(fc) => self.lookup(fc.name.as_ref()).map(|(t, _)| t.clone()),
            _ => None,
        };
        let local_can_free = local.map(|t| self.type_can_free(&t, 0));
        WrittenCall {
            name,
            local_can_free,
            at,
        }
    }

    /// After the call `expr` is checked and its value `requires` is proved: prove each pointer
    /// `requires` of the callee in safe code, and then apply the events of the call.
    pub(crate) fn pointer_facts_at_call(&mut self, expr: &Expr, written: WrittenCall) {
        let WrittenCall {
            name: written,
            local_can_free,
            at,
        } = written;
        let written = written.as_deref();
        // The receiver of a method call that the check kept, and the function value of an
        // indirect call, are arguments too.
        let (name, args, callee): (&str, &[Expr], Option<&Expr>) = match expr {
            Expr::FunctionCall(fc) => (fc.name.as_ref(), &fc.args, None),
            Expr::MethodCall(m) => (m.method_name.as_ref(), &m.args, Some(&*m.base)),
            Expr::IndirectCall(ic) => ("", &ic.args, Some(&*ic.callee)),
            _ => return,
        };
        if !self.speculating && !self.in_unsafe_block && !self.in_unsafe_fn {
            self.prove_pointer_requires(name, args, &at);
        }
        if ADDRESS_FUNCTIONS.contains(&plain_name(name)) {
            return;
        }
        let shown = written.unwrap_or(name);
        // A call of a local closure gets the values that the closure captures. Its type does not
        // show them, so the call can free memory.
        let local_call = local_can_free.is_some();
        let why = if local_can_free == Some(true) {
            Some(format!(
                "the call of the closure `{shown}`, which can free memory through what it captures"
            ))
        } else {
            let params = self.call_param_types(name, expr);
            let mut through = callee.filter(|c| self.argument_can_free(c, None));
            for (i, a) in args.iter().enumerate() {
                if through.is_none()
                    && self.argument_can_free(a, params.as_ref().and_then(|p| p.get(i)))
                {
                    through = Some(a);
                }
            }
            // The check gives a call of a closure the closure value as an argument.
            let is_the_closure = |a: &Expr| match a {
                Expr::Borrow(b) => {
                    matches!(&*b.expr, Expr::Identifier(id) if id.name.as_ref() == shown)
                }
                Expr::Identifier(id) => id.name.as_ref() == shown,
                _ => false,
            };
            let closure = |c: String| {
                format!(
                    "the call of the closure {c}, which can free memory through what it captures"
                )
            };
            through.map(|a| match expr {
                Expr::IndirectCall(ic) if std::ptr::eq(&*ic.callee, a) => {
                    closure(shown_argument(a))
                }
                Expr::IndirectCall(ic) => format!(
                    "the call through {}, which can free memory through {}",
                    shown_argument(&ic.callee),
                    shown_argument(a)
                ),
                _ if local_call && is_the_closure(a) => closure(format!("`{shown}`")),
                _ if local_call => format!(
                    "the call through `{shown}`, which can free memory through {}",
                    shown_argument(a)
                ),
                _ => format!(
                    "the call to '{shown}', which can free memory through {}",
                    shown_argument(a)
                ),
            })
        };
        if let Some(why) = why {
            self.pointer_facts_event(at, |_| why);
        }
    }

    /// The parameter types of the function that the call `expr` to `name` calls, when the
    /// checker knows them.
    fn call_param_types(&self, name: &str, expr: &Expr) -> Option<Vec<Type>> {
        let from_type = |t: Type| match t {
            Type::Function(params, ..) | Type::Closure(params, _) => Some(params),
            _ => None,
        };
        match expr {
            Expr::IndirectCall(ic) => from_type(self.check_expr_type_probe(&ic.callee)),
            Expr::FunctionCall(_) => match self.lookup(name) {
                Some((t, _)) => from_type(t.clone()),
                None => self
                    .env
                    .functions
                    .get(name)
                    .map(|f| f.2.clone())
                    .or_else(|| {
                        self.mono
                            .functions
                            .iter()
                            .find(|(f, _)| f.name.as_ref() == name)
                            .map(|(f, _)| f.params.iter().map(|(_, t)| t.clone()).collect())
                    }),
            },
            _ => None,
        }
    }

    /// Whether the argument `arg` can free memory. The type of its parameter `param` says it
    /// when it has no type parameter, because an argument has the type of its parameter.
    fn argument_can_free(&mut self, arg: &Expr, param: Option<&Type>) -> bool {
        match param {
            Some(p) if !p.has_generic_params() => self.type_can_free(p, 0),
            _ => {
                let ty = self.check_expr_type_probe(arg);
                self.type_can_free(&ty, 0)
            }
        }
    }

    /// E8006 for each pointer `requires` of the callee `name` that the facts do not prove.
    fn prove_pointer_requires(&mut self, name: &str, args: &[Expr], span: &Span) {
        let callee = self.env.syntax_functions.get(name).copied().or_else(|| {
            self.mono
                .functions
                .iter()
                .find(|(f, _)| f.name.as_ref() == name)
                .map(|(f, _)| f)
        });
        let Some((params, clauses)) = callee.map(|f| {
            let clauses: Vec<Expr> = f
                .requires
                .iter()
                .filter(|e| read_clause(e).is_some())
                .cloned()
                .collect();
            (f.params.clone(), clauses)
        }) else {
            return;
        };
        if clauses.is_empty() {
            return;
        }
        let mut names = HashMap::new();
        for (i, ((param, _), arg)) in params.iter().zip(args).enumerate() {
            let value = if crate::hir::prover::is_modelled(arg) {
                arg.clone()
            } else {
                Expr::Identifier(IdentifierExpr {
                    name: format!("$arg_{}_{}_{i}", span.line, span.column).into(),
                    span: *span,
                })
            };
            names.insert(param.clone(), value);
        }
        let arg_of = |p: &Expr| match p {
            Expr::Identifier(id) => params
                .iter()
                .position(|(n, _)| n == &id.name)
                .and_then(|i| args.get(i)),
            _ => None,
        };
        for clause in &clauses {
            let Some((function, pointers, len)) = read_clause(clause) else {
                continue;
            };
            let Some(values): Option<Vec<PointerValue>> = pointers
                .iter()
                .map(|p| arg_of(p).map(|a| self.pointer_value(a, None)))
                .collect()
            else {
                continue;
            };
            let m = crate::hir::prover::substitute_names(len, &names);
            let goal = self.pointer_goal(function, &values, &m, None);
            if self.prove_exact(&goal) {
                continue;
            }
            let ended = self.ended_fact_that_proves(function, &values, &m);
            let error = self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E8006,
                format!(
                    "this call to '{}' may not meet its precondition `{}`: the compiler cannot \
                     prove it from what is known here",
                    self.shown_callee(name),
                    clause_source(clause)
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
            if let Some((at, why)) = ended {
                error.notes.push(crate::diagnostic::Note {
                    message: format!("a fact that proves it ended at {why}").into(),
                    span: Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
                });
            }
        }
    }

    /// The last event that ended a fact that would prove the clause, as its place and reason.
    fn ended_fact_that_proves(
        &mut self,
        function: &str,
        values: &[PointerValue],
        m: &Expr,
    ) -> Option<(Span, String)> {
        let ended: Vec<Ended> = self
            .pointer_facts
            .ended
            .iter()
            .filter(|e| values.iter().any(|v| v.base == e.range.base))
            .cloned()
            .collect();
        ended.into_iter().rev().find_map(|e| {
            let goal = self.pointer_goal(function, values, m, Some(&e.range));
            self.prove_exact(&goal).then_some((e.at, e.why))
        })
    }

    /// The callee `name` as the program wrote it: an instance of a generic function, such as
    /// `sum$i64`, is shown as `sum`.
    fn shown_callee(&self, name: &str) -> String {
        match name.split_once('$') {
            Some((first, _)) if self.env.generic_functions.contains_key(first) => first.to_string(),
            _ => name.to_string(),
        }
    }

    /// The goal that proves `valid`, `initialized` or `!overlaps` for the pointer values
    /// `values` and the length `m`, from one fact. `extra` is one more range to use.
    fn pointer_goal(
        &self,
        function: &str,
        values: &[PointerValue],
        m: &Expr,
        extra: Option<&Range>,
    ) -> Exact {
        let facts = &self.pointer_facts;
        let ranges = || facts.ranges.iter().chain(extra);
        let within = |r: &Range, v: &PointerValue| {
            Exact::And(vec![
                le(term(r.lo.clone()), sum(&v.offset, &[])),
                le(sum(&v.offset, &[m]), term(r.hi.clone())),
            ])
        };
        match function {
            "valid" | "initialized" => {
                let kind = if function == "valid" {
                    Kind::Valid
                } else {
                    Kind::Init
                };
                let v = &values[0];
                let goals = ranges()
                    .filter(|r| r.kind == kind && r.base == v.base)
                    .map(|r| within(r, v))
                    .collect();
                Exact::And(vec![le(term(zero()), term(m.clone())), Exact::Or(goals)])
            }
            _ => {
                let (a, b) = (&values[0], &values[1]);
                let mut goals = vec![le(term(m.clone()), term(zero()))];
                if a.base == b.base {
                    for r in ranges().filter(|r| r.kind == Kind::Valid && r.base == a.base) {
                        goals.push(Exact::And(vec![
                            within(r, a),
                            within(r, b),
                            Exact::Or(vec![
                                le(sum(&a.offset, &[m]), sum(&b.offset, &[])),
                                le(sum(&b.offset, &[m]), sum(&a.offset, &[])),
                            ]),
                        ]));
                    }
                }
                for apart in &facts.aparts {
                    for (x, y) in [(a, b), (b, a)] {
                        if apart.first.0 == x.base && apart.second.0 == y.base {
                            let window = |(_, lo): &(usize, Expr), v: &PointerValue| {
                                Exact::And(vec![
                                    le(term(lo.clone()), sum(&v.offset, &[])),
                                    le(
                                        sum(&v.offset, &[m]),
                                        sum(std::slice::from_ref(lo), &[&apart.len]),
                                    ),
                                ])
                            };
                            goals.push(Exact::And(vec![
                                window(&apart.first, x),
                                window(&apart.second, y),
                            ]));
                        }
                    }
                }
                Exact::Or(goals)
            }
        }
    }
}

/// The argument `e` as the program wrote it, for the note of E8006.
fn shown_argument(e: &Expr) -> String {
    fn text(e: &Expr) -> Option<String> {
        match e {
            Expr::Borrow(b) => {
                let mutable = if b.is_mut { "mut " } else { "" };
                Some(format!("&{mutable}{}", text(&b.expr)?))
            }
            Expr::Dereference(d) => Some(format!("*{}", text(&d.expr)?)),
            Expr::AsCast(c) => text(&c.expr),
            e => crate::hir::prover::to_source(e),
        }
    }
    text(crate::syntax::bare_unsafe_value(e))
        .map_or_else(|| "an argument".to_string(), |t| format!("`{t}`"))
}

/// The clause as the program wrote it, for an error.
fn clause_source(e: &Expr) -> String {
    let Some((function, pointers, len)) = read_clause(e) else {
        return String::new();
    };
    let args: Vec<String> = pointers
        .iter()
        .copied()
        .chain(std::iter::once(len))
        .map(|a| crate::hir::prover::to_source(a).unwrap_or_else(|| "..".to_string()))
        .collect();
    let not = if function == "overlaps" { "!" } else { "" };
    format!("{not}{function}({})", args.join(", "))
}
