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
// keeps the place and the reason of its event, for the note of E8006. Before a loop body, a scan
// of the body finds its first event, which ends the facts for each pass. After a loop, the facts
// are those that hold at each place where the loop ends.
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

/// `Valid` or `Init` for the elements `lo` to `hi - 1` at `base`. Each end is a sum of terms, as
/// an offset is, so that the prover adds them with no wrap.
#[derive(Clone, Debug)]
struct Range {
    id: usize,
    kind: Kind,
    base: usize,
    lo: Vec<Expr>,
    hi: Vec<Expr>,
    /// Whether the checker made this `Init` fact from two others. A later `Init` fact that
    /// touches it grows it in place. The other facts stay, so that the end of an `if` keeps them
    /// when the other branch has them.
    merged: bool,
}

/// The `len` elements from `lo` at each of two bases share no byte.
#[derive(Clone, Debug)]
struct Apart {
    id: usize,
    first: (usize, Vec<Expr>),
    second: (usize, Vec<Expr>),
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

/// The facts at the `break`s of a loop that the checker is in.
pub(crate) struct LoopExit {
    /// The number of scopes outside the loop body.
    depth: usize,
    at_break: Option<PointerFacts>,
}

/// The raw-pointer state of the code around a body that starts with none of it: a closure body, a
/// task body or the body of another function. It also holds the invariants of the loops around
/// the body, which a `continue` in the body does not reach.
pub(crate) struct OuterPointerState(
    PointerFacts,
    Vec<LoopExit>,
    Vec<String>,
    PointerFunction,
    Vec<crate::hir::stmt::InvariantLoop>,
);

/// The function whose body the checker checks.
#[derive(Clone, Default)]
pub(crate) struct PointerFunction {
    /// How a message names the function.
    name: String,
    /// Its pointer `ensures`. Each `return` must prove them. A closure or a task body has none.
    ensures: Vec<Expr>,
    /// The index of the scope that holds the parameters.
    scope: usize,
    /// The parameters that the body assigns or takes a `&mut` to.
    params_changed: std::collections::HashSet<String>,
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
    /// The memories of each base. A call result can point into other memory and into the local
    /// memory of each variable that an argument points into.
    memory: HashMap<usize, Vec<Memory>>,
    /// The value of each local variable of a raw pointer type.
    entries: HashMap<String, PointerValue>,
    ranges: Vec<Range>,
    aparts: Vec<Apart>,
    ended: Vec<Ended>,
    /// The value of each call result that a pointer `ensures` is about, by the line and the
    /// column of the call.
    results: HashMap<(usize, usize), PointerValue>,
    /// The tasks that a `spawn` started and that no `join()` or end of a `tasks` block ended
    /// yet, by an id and the name of the handle. While one runs, it can free memory, so no fact
    /// about other memory starts.
    tasks: Vec<(usize, Option<String>, Span)>,
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
        let names = |r: &Range| r.lo.iter().chain(&r.hi).any(|t| expr_mentions(t, name));
        self.ranges.retain(|r| !names(r));
        self.ended.retain(|e| !names(&e.range));
        self.aparts.retain(|a| {
            !a.first
                .1
                .iter()
                .chain(&a.second.1)
                .chain([&a.len])
                .any(|e| expr_mentions(e, name))
        });
    }

    /// Gives each name of `names` its new name in each fact and each entry.
    fn rename_all(&mut self, names: &HashMap<crate::symbol::Symbol, Expr>) {
        let rename = |t: &mut Expr| *t = crate::hir::prover::substitute_names(t, names);
        for v in self.entries.values_mut() {
            v.offset.iter_mut().for_each(rename);
        }
        for r in self
            .ranges
            .iter_mut()
            .chain(self.ended.iter_mut().map(|e| &mut e.range))
        {
            r.lo.iter_mut().chain(r.hi.iter_mut()).for_each(rename);
        }
        for a in &mut self.aparts {
            let terms = a.first.1.iter_mut().chain(a.second.1.iter_mut());
            terms.chain([&mut a.len]).for_each(rename);
        }
    }

    /// Ends each range that `ends` selects, and records the event for the note of E8006.
    fn end_ranges(&mut self, at: Span, why: &str, ends: impl Fn(&Range, &Memory) -> bool) {
        let memory = &self.memory;
        let (gone, kept) = std::mem::take(&mut self.ranges).into_iter().partition(|r| {
            memory
                .get(&r.base)
                .is_some_and(|ms| ms.iter().any(|m| ends(r, m)))
        });
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
            Memory::Local { drops, .. } => *drops && r.kind == Kind::Init,
            Memory::Other => true,
        });
    }

    /// The storage of the local variable `var` ends.
    fn end_local(&mut self, var: &str, at: Span, why: &str) {
        self.end_ranges(
            at,
            why,
            |_, memory| matches!(memory, Memory::Local { var: v, .. } if v == var),
        );
    }

    /// A move takes the value out of the local variable `var`.
    fn end_init_of_local(&mut self, var: &str, at: Span, why: &str) {
        self.end_ranges(at, why, |r, memory| {
            r.kind == Kind::Init && matches!(memory, Memory::Local { var: v, .. } if v == var)
        });
    }

    fn has_local(&self, var: &str) -> bool {
        self.ranges.iter().any(|r| {
            self.memory.get(&r.base).is_some_and(|ms| {
                ms.iter()
                    .any(|m| matches!(m, Memory::Local { var: v, .. } if v == var))
            })
        })
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
        for t in &other.tasks {
            if !self.tasks.contains(t) {
                self.tasks.push(t.clone());
            }
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

/// The parameters of a function, with their types.
type Params = Vec<(crate::symbol::Symbol, Type)>;

/// A `let` of a loop body: its name, its written type and its value.
type BodyLet = (String, Option<Type>, Expr);

/// Whether `body` declares a variable named `name`, at any depth.
pub(crate) fn body_declares(body: &[Statement], name: &str) -> bool {
    declarations_of_a_body(body).0.contains_key(name)
}

/// The variables that `body` declares at any depth, with the number of declarations of each:
/// each `let`, the variable of each `for`, each name that a pattern binds and each parameter of a
/// closure. Also each `let`.
fn declarations_of_a_body(body: &[Statement]) -> (HashMap<String, usize>, Vec<BodyLet>) {
    fn own(stmts: &[Statement], out: &mut HashMap<String, usize>, lets: &mut Vec<BodyLet>) {
        for s in stmts {
            match s {
                Statement::LetDecl(l) => {
                    *out.entry(l.name.to_string()).or_default() += 1;
                    lets.push((l.name.to_string(), l.ty_ann.clone(), l.expr.clone()));
                }
                Statement::ForLoop(l) => {
                    *out.entry(l.iter.clone()).or_default() += 1;
                    own(&l.body, out, lets);
                }
                Statement::Loop(l) => own(&l.body, out, lets),
                _ => {}
            }
        }
    }
    let mut out = HashMap::new();
    let mut lets = Vec::new();
    own(body, &mut out, &mut lets);
    crate::syntax::visit_exprs(body, &mut |e| match e {
        Expr::If(i) => {
            own(&i.then_block, &mut out, &mut lets);
            if let Some(b) = &i.else_block {
                own(b, &mut out, &mut lets);
            }
        }
        Expr::Match(m) => {
            for arm in &m.arms {
                let mut bound = Vec::new();
                crate::hir::check::control::pattern_names(&arm.pattern, &mut bound);
                for n in bound {
                    *out.entry(n.to_string()).or_default() += 1;
                }
                own(&arm.body, &mut out, &mut lets);
            }
        }
        Expr::UnsafeBlock(u) => own(&u.stmts, &mut out, &mut lets),
        Expr::ComptimeBlock(c) => own(&c.stmts, &mut out, &mut lets),
        Expr::TaskBlock(t) => own(&t.stmts, &mut out, &mut lets),
        Expr::SpawnOn(s) => own(&s.stmts, &mut out, &mut lets),
        Expr::Closure(c) => {
            for (n, _) in &c.params {
                *out.entry(n.to_string()).or_default() += 1;
            }
        }
        _ => {}
    });
    (out, lets)
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

/// The name of the result of the call at `at` in the facts of its `ensures`. A `let` of the
/// result gives the facts its own name.
fn result_name(at: Span) -> Expr {
    Expr::Identifier(IdentifierExpr {
        name: format!("$result_{}_{}", at.line, at.column).into(),
        span: at,
    })
}

fn zero() -> Expr {
    Expr::Number(NumberExpr {
        value: "0".to_string().into(),
        ty: None,
        span: Span::default(),
    })
}

fn one() -> Expr {
    Expr::Number(NumberExpr {
        value: "1".to_string().into(),
        ty: None,
        span: Span::default(),
    })
}

/// The clause `valid(p, n)`, `initialized(p, n)` or `!overlaps(p, q, n)`, as its function, its
/// pointer arguments and its length.
pub(crate) fn read_clause(e: &Expr) -> Option<(&str, Vec<&Expr>, &Expr)> {
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
        self.new_base_in(vec![memory])
    }

    fn new_base_in(&mut self, memory: Vec<Memory>) -> usize {
        self.pointer_fact_ids += 1;
        self.pointer_facts
            .memory
            .insert(self.pointer_fact_ids, memory);
        self.pointer_fact_ids
    }

    fn add_range(&mut self, kind: Kind, base: usize, lo: Vec<Expr>, hi: Vec<Expr>) {
        if kind == Kind::Init && self.merge_init(base, &lo, &hi) {
            return;
        }
        self.pointer_fact_ids += 1;
        let id = self.pointer_fact_ids;
        self.pointer_facts.ranges.push(Range {
            id,
            kind,
            base,
            lo,
            hi,
            merged: false,
        });
    }

    /// Before an assignment to the number `name`, after the prover's facts gave its old value
    /// the new name in `names`: the pointer facts name the old value in the same way. A fact
    /// that changes gets a new id, so the end of an `if` keeps it only when both branches have
    /// it. A fact that still names `name` ends.
    pub(crate) fn pointer_facts_name_the_old_value(
        &mut self,
        name: &str,
        names: &HashMap<crate::symbol::Symbol, Expr>,
    ) {
        // Renames each term that names `name`, and says whether one did.
        fn rename<'e>(
            terms: impl Iterator<Item = &'e mut Expr>,
            name: &str,
            names: &HashMap<crate::symbol::Symbol, Expr>,
        ) -> bool {
            let mut changed = false;
            for t in terms.filter(|t| expr_mentions(t, name)) {
                *t = crate::hir::prover::substitute_names(t, names);
                changed = true;
            }
            changed
        }
        let mut ids = self.pointer_fact_ids;
        let facts = &mut self.pointer_facts;
        for v in facts.entries.values_mut() {
            rename(v.offset.iter_mut(), name, names);
        }
        let ended = facts.ended.iter_mut().map(|e| &mut e.range);
        for r in facts.ranges.iter_mut().chain(ended) {
            if rename(r.lo.iter_mut().chain(r.hi.iter_mut()), name, names) {
                ids += 1;
                r.id = ids;
            }
        }
        for a in facts.aparts.iter_mut() {
            let terms = a.first.1.iter_mut().chain(a.second.1.iter_mut());
            if rename(terms.chain([&mut a.len]), name, names) {
                ids += 1;
                a.id = ids;
            }
        }
        self.pointer_fact_ids = ids;
        self.pointer_facts.forget(name);
    }

    /// The facts at the start of the body of `func`: each raw pointer parameter has a base of
    /// its own, and each pointer `requires` is a fact. `valid(p, n)` also says that `n >= 0`.
    pub(crate) fn start_pointer_facts(&mut self, func: &Function) {
        self.pointer_facts = PointerFacts::default();
        let well_formed = |e: &&Expr| {
            read_clause(e).is_some()
                && crate::hir::check::pointer_clauses::read_pointer_clause(
                    "",
                    &func.params,
                    &func.return_type,
                    true,
                    e,
                )
                .is_ok()
        };
        self.pointer_function = PointerFunction {
            name: self.clause_owner_name(func),
            ensures: func.ensures.iter().filter(well_formed).cloned().collect(),
            scope: self.scopes.len() - 1,
            params_changed: crate::hir::check::raw::names_a_body_writes(&func.body),
        };
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
                    self.add_range(kind, bases[0], Vec::new(), vec![len.clone()]);
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
                        first: (bases[0], Vec::new()),
                        second: (bases[1], Vec::new()),
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
            Expr::FunctionCall(fc)
                if self
                    .pointer_facts
                    .results
                    .contains_key(&(fc.span.line, fc.span.column)) =>
            {
                self.pointer_facts.results[&(fc.span.line, fc.span.column)].clone()
            }
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
        self.add_range(Kind::Valid, base, Vec::new(), vec![one()]);
        self.add_range(Kind::Init, base, Vec::new(), vec![one()]);
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
            let why = self.in_branch(why(self));
            self.pointer_facts.end_other_memory(at, &why);
        }
    }

    /// `why` with the branch of an `if` or the arm of a `match` that the checker is in.
    fn in_branch(&self, why: String) -> String {
        match self.pointer_branches.last() {
            Some(branch) => format!("{why}, {branch}"),
            None => why,
        }
    }

    /// The names of the two branches of `if_expr`, for the note of E8006. The `if` of a `while`
    /// has none: its `else` only leaves the loop.
    pub(crate) fn pointer_branch_names(if_expr: &IfExpr) -> Option<(String, String)> {
        let is_a_while = matches!(
            if_expr.else_block.as_deref(),
            Some([Statement::Break(b)]) if b.span == if_expr.span
        );
        if is_a_while {
            return None;
        }
        Some(match crate::hir::prover::to_source(&if_expr.cond) {
            Some(c) => (
                format!("in the branch where `{c}` is true"),
                format!("in the branch where `{c}` is false"),
            ),
            None => (
                "in the first branch of the `if`".to_string(),
                "in the `else` branch of the `if`".to_string(),
            ),
        })
    }

    /// The name of the `match` arm with the pattern `pattern`, for the note of E8006.
    pub(crate) fn pointer_arm_name(pattern: &Pattern) -> String {
        match pattern {
            Pattern::EnumVariant(_, variant, _) => format!("in the arm for `{variant}`"),
            Pattern::Wildcard => "in the `_` arm of the `match`".to_string(),
            _ => "in an arm of the `match`".to_string(),
        }
    }

    /// The variables of a scope end. Their storage ends, and the drop of a value of a type with a
    /// drop can free memory.
    pub(crate) fn pointer_facts_end_scope(
        &mut self,
        vars: &HashMap<crate::symbol::Symbol, (Type, Topology)>,
    ) {
        self.pointer_facts_end_storage(vars);
        for name in vars.keys() {
            self.pointer_facts.forget(name.as_ref());
        }
    }

    /// The storage of the variables `vars` ends, and the program drops each value of a type with
    /// a drop.
    fn pointer_facts_end_storage(
        &mut self,
        vars: &HashMap<crate::symbol::Symbol, (Type, Topology)>,
    ) {
        let at = self.current_statement_span;
        let mut dropped: Vec<&str> = Vec::new();
        for (name, (ty, _)) in vars {
            if self.pointer_facts.has_local(name.as_ref()) {
                let why = self.in_branch(format!("the end of the scope of `{name}`"));
                self.pointer_facts.end_local(name.as_ref(), at, &why);
            }
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

    /// At a `return` of `value`, or at the end of the body for `None`: E8012 for each pointer
    /// `ensures` that the facts do not prove after the end of the storage of each local variable
    /// and parameter, and their drops. `place` says where, as in "at this `return`".
    pub(crate) fn prove_pointer_ensures(&mut self, value: Option<&Expr>, at: Span, place: &str) {
        if self.pointer_function.ensures.is_empty() || self.speculating {
            return;
        }
        let returned = match value {
            Some(v) if matches!(self.check_expr_type_probe(v), Type::Pointer(..)) => {
                Some(self.pointer_value(v, None))
            }
            _ => None,
        };
        let saved = self.pointer_facts.clone();
        for i in (self.pointer_function.scope..self.scopes.len()).rev() {
            let vars = self.scopes[i].clone();
            self.pointer_facts_end_storage(&vars);
        }
        let mut names = HashMap::new();
        if let Some(v) = value.filter(|v| crate::hir::prover::is_modelled(v)) {
            names.insert(crate::symbol::Symbol::from("return"), v.clone());
        }
        let PointerFunction {
            name: shown,
            ensures,
            ..
        } = self.pointer_function.clone();
        for clause in &ensures {
            let Some((function, pointers, len)) = read_clause(clause) else {
                continue;
            };
            let values: Option<Vec<PointerValue>> = pointers
                .iter()
                .map(|p| match p {
                    Expr::Identifier(id) if id.name.as_ref() == "return" => returned.clone(),
                    p => Some(self.pointer_value(p, None)),
                })
                .collect();
            let m = if expr_mentions(len, "return") && !names.contains_key("return") {
                result_name(at)
            } else {
                crate::hir::prover::substitute_names(len, &names)
            };
            let proved = values.as_ref().is_some_and(|v| {
                let goal = self.pointer_goal(function, v, &m, None);
                self.prove_exact(&goal)
            });
            if proved {
                continue;
            }
            let ended = values
                .as_ref()
                .and_then(|v| self.ended_fact_that_proves(function, v, &m));
            let error = self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E8012,
                format!(
                    "`{shown}` says `{}`, but the compiler cannot prove it {place}",
                    clause_source(clause)
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
            );
            if let Some((at, why)) = ended {
                error.notes.push(crate::diagnostic::Note {
                    message: format!("a fact that proves it ended at {why}").into(),
                    span: Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
                });
            }
        }
        self.pointer_facts = saved;
    }

    /// `assume fact;` at `at`: the fact without a proof. Only an `unsafe` block can hold it. The
    /// fact is a pointer clause, or a boolean expression for the prover.
    pub(crate) fn check_assume(&mut self, fact: &Expr, at: Span) {
        if self.unsafe_is_missing() {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E5002,
                "`assume` adds a fact that the compiler does not prove, so it must be in an \
                 `unsafe` block"
                    .to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
            );
        }
        let what = format!(
            "an `assume` trusts `{}`",
            crate::hir::check::audit::clause_text(fact)
        );
        self.note_trusted(at, what);
        let Some((function, pointers, len)) = read_clause(fact) else {
            if crate::hir::check::pointer_clauses::first_pointer_call(fact).is_some() {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8011,
                    "write the fact of `assume` as one pointer clause, as in `assume valid(p, n)`"
                        .to_string(),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
                );
                return;
            }
            let mut fact = fact.clone();
            let ty = self.check_expr_type_flag(&mut fact, false);
            if ty != Type::Scalar(ElementType::Bool) && ty != Type::Unknown {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3003,
                    format!("the fact of `assume` must be a `bool`, but it is `{ty}`"),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
                );
            } else if !self.speculating {
                self.consteval.constraints.push(fact);
            }
            return;
        };
        for p in &pointers {
            let mut arg = (*p).clone();
            let ty = self.check_expr_type_flag(&mut arg, false);
            if !matches!(ty, Type::Pointer(..) | Type::Unknown) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8011,
                    format!(
                        "`{function}` needs a raw pointer, but `{}` is `{ty}`",
                        crate::hir::prover::to_source(p).unwrap_or_default()
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
                );
                return;
            }
        }
        let mut n = len.clone();
        self.check_expr_type_flag(&mut n, false);
        if self.speculating {
            return;
        }
        let values: Vec<PointerValue> = pointers
            .iter()
            .map(|p| self.pointer_value(p, None))
            .collect();
        self.add_pointer_fact(function, &values, len.clone());
    }

    /// The `spawn` at `at` started a task. Its handle has no name yet.
    pub(crate) fn pointer_facts_task_started(&mut self, at: Span) {
        self.pointer_fact_ids += 1;
        self.pointer_facts
            .tasks
            .push((self.pointer_fact_ids, None, at));
    }

    /// `let name = spawn { .. }`: the task that the `spawn` started has the handle `name`.
    pub(crate) fn pointer_facts_name_the_task(&mut self, name: &str) {
        if let Some(t) = self
            .pointer_facts
            .tasks
            .iter_mut()
            .rev()
            .find(|t| t.1.is_none())
        {
            t.1 = Some(name.to_string());
        }
    }

    /// `name.join()` ended the task of the handle `name`.
    pub(crate) fn pointer_facts_task_joined(&mut self, name: &str) {
        self.pointer_facts
            .tasks
            .retain(|t| t.1.as_deref() != Some(name));
    }

    /// The ids of the tasks that run, for the start of a `tasks` block.
    pub(crate) fn pointer_facts_running_tasks(&self) -> Vec<usize> {
        self.pointer_facts.tasks.iter().map(|t| t.0).collect()
    }

    /// The end of a `tasks` block, which `running` ran before: each task started in it ended.
    pub(crate) fn pointer_facts_tasks_ended(&mut self, running: &[usize]) {
        self.pointer_facts.tasks.retain(|t| running.contains(&t.0));
    }

    /// `let name = f(..)`: the facts of the `ensures` of `f` call its result `name`.
    pub(crate) fn pointer_facts_name_the_result(&mut self, name: &str, init: &Expr) {
        let Expr::FunctionCall(fc) = crate::syntax::bare_unsafe_value(init) else {
            return;
        };
        let Expr::Identifier(from) = result_name(fc.span) else {
            return;
        };
        let to = Expr::Identifier(IdentifierExpr {
            name: name.into(),
            span: fc.span,
        });
        let names = HashMap::from([(from.name, to)]);
        let rename = |e: &mut Expr| *e = crate::hir::prover::substitute_names(e, &names);
        for r in &mut self.pointer_facts.ranges {
            r.lo.iter_mut().chain(&mut r.hi).for_each(rename);
        }
        for a in &mut self.pointer_facts.aparts {
            a.first.1.iter_mut().chain(&mut a.second.1).for_each(rename);
            rename(&mut a.len);
        }
    }

    /// A move took the value out of the local variable `var`, or out of a field of it.
    pub(crate) fn pointer_facts_move_out(&mut self, var: &str) {
        if self.pointer_facts.has_local(var) {
            let why = self.in_branch(format!("the move out of `{var}`"));
            let at = self.current_statement_span;
            self.pointer_facts.end_init_of_local(var, at, &why);
        }
    }

    /// The write `p[k] = x` or `*p = x` gives the element a value, so a pointer `p` at the base
    /// `b` and the offset `j` gets `Init(b, j + k, j + k + 1)`. The compiler trusts the write, as
    /// it trusts the rest of the `unsafe` block. Inside a call, the facts change after the call,
    /// so a write there gives no fact.
    pub(crate) fn pointer_facts_note_a_write(&mut self, place: &Expr) {
        let (pointer, index) = match place {
            Expr::IndexAccess(a) => (&*a.base, Some(&*a.index)),
            Expr::Dereference(d) => (&*d.expr, None),
            _ => return,
        };
        let Expr::Identifier(p) = pointer else {
            return;
        };
        if !self.fact_change_frames.is_empty()
            || !matches!(self.lookup(p.name.as_ref()), Some((Type::Pointer(..), _)))
            || index.is_some_and(|k| !crate::hir::prover::is_modelled(k))
        {
            return;
        }
        let Some(value) = self.pointer_facts.entry_of(p.name.as_ref()) else {
            return;
        };
        let mut lo = value.offset;
        lo.extend(index.cloned());
        let mut hi = lo.clone();
        hi.push(one());
        self.add_range(Kind::Init, value.base, lo, hi);
    }

    /// At the start of a loop: a fact holds in the body only when no pass ends it, so an event
    /// that the body can have ends the facts before the body.
    pub(crate) fn end_pointer_facts_a_loop_can_end(&mut self, body: &[Statement]) {
        if self.pointer_facts.ranges.is_empty() {
            return;
        }
        if let Some((at, why)) = self.loop_body_event(body) {
            self.pointer_facts_event(at, |_| format!("{why}, in a pass of the loop"));
        }
    }

    /// The start of a body that starts with no raw-pointer fact, no loop and no branch of the code
    /// around it.
    pub(crate) fn pointer_facts_enter_body(&mut self) -> OuterPointerState {
        let params_changed = self.pointer_function.params_changed.clone();
        OuterPointerState(
            std::mem::take(&mut self.pointer_facts),
            std::mem::take(&mut self.pointer_loops),
            std::mem::take(&mut self.pointer_branches),
            // A `return` in the body returns from the body, so the function's `ensures` do not
            // apply there.
            std::mem::replace(
                &mut self.pointer_function,
                PointerFunction {
                    params_changed,
                    ..Default::default()
                },
            ),
            std::mem::take(&mut self.invariant_loops),
        )
    }

    /// The end of a body that `pointer_facts_enter_body` started.
    pub(crate) fn pointer_facts_leave_body(&mut self, outer: OuterPointerState) {
        let OuterPointerState(facts, loops, branches, function, invariant_loops) = outer;
        self.pointer_facts = facts;
        self.pointer_loops = loops;
        self.pointer_branches = branches;
        self.pointer_function = function;
        self.invariant_loops = invariant_loops;
    }

    /// The start of a loop body. `self.scopes` holds the scopes outside the body.
    pub(crate) fn pointer_facts_enter_loop(&mut self) {
        self.pointer_loops.push(LoopExit {
            depth: self.scopes.len(),
            at_break: None,
        });
    }

    /// A `break` leaves the innermost loop, and each scope of the loop body ends there.
    pub(crate) fn pointer_facts_at_break(&mut self) {
        let Some(depth) = self.pointer_loops.last().map(|l| l.depth) else {
            return;
        };
        let before = self.pointer_facts.clone();
        for i in (depth..self.scopes.len()).rev() {
            let vars = self.scopes[i].clone();
            self.pointer_facts_end_scope(&vars);
        }
        let state = std::mem::replace(&mut self.pointer_facts, before);
        let exit = self.pointer_loops.last_mut().expect("checked above");
        exit.at_break = Some(match exit.at_break.take() {
            Some(joined) => joined.join(&state),
            None => state,
        });
    }

    /// After a loop: a fact holds when it holds at each place where the loop ends. A `for` loop
    /// also ends at its head, where `head` holds. A `loop` ends only at a `break`.
    /// `renamed` gives the variable of a `for` loop its value where the loop ended.
    pub(crate) fn pointer_facts_after_a_loop(
        &mut self,
        head: Option<PointerFacts>,
        renamed: Option<(&str, &Expr)>,
    ) {
        let exit = self
            .pointer_loops
            .pop()
            .expect("pushed at the start of the loop");
        let names = renamed.map(|(var, value)| HashMap::from([(var.into(), value.clone())]));
        let rename = |mut facts: PointerFacts| {
            if let Some(names) = &names {
                facts.rename_all(names);
            }
            facts
        };
        self.pointer_facts = match (head.map(rename), exit.at_break.map(rename)) {
            (Some(head), Some(at_break)) => head.join(&at_break),
            (Some(state), None) | (None, Some(state)) => state,
            // No path leaves the loop, so no code after it runs.
            (None, None) => PointerFacts::default(),
        };
    }

    /// The value of each raw pointer variable that the pointer clauses of a loop invariant name,
    /// before the start of the loop body ends the facts of the body.
    pub(crate) fn invariant_pointers(&self, clauses: &[Expr]) -> Vec<(String, PointerValue)> {
        let mut values: Vec<(String, PointerValue)> = self
            .pointer_facts
            .entries
            .iter()
            .filter(|(name, _)| clauses.iter().any(|c| expr_mentions(c, name)))
            .map(|(name, v)| (name.clone(), v.clone()))
            .collect();
        values.sort_by(|a, b| a.0.cmp(&b.0));
        values
    }

    /// At the start of a loop body: each pointer clause of the invariant adds its facts, as a
    /// `requires` does at the start of a function. `before` holds the values from
    /// `invariant_pointers`. A pointer that the body assigns lost its value, so it gets a new base
    /// with the memory of its base when the loop started. Returns each such pointer with its new
    /// base.
    pub(crate) fn pointer_facts_assume_invariants(
        &mut self,
        clauses: &[Expr],
        before: &[(String, PointerValue)],
    ) -> Vec<(String, usize)> {
        let mut pass_bases = Vec::new();
        for (name, value) in before {
            if self.pointer_facts.entries.contains_key(name) {
                continue;
            }
            let memory = self.pointer_facts.memory.get(&value.base).cloned();
            let base = self.new_base_in(memory.unwrap_or_else(|| vec![Memory::Other]));
            let entry = PointerValue {
                base,
                offset: Vec::new(),
            };
            self.pointer_facts.entries.insert(name.clone(), entry);
            pass_bases.push((name.clone(), base));
        }
        for clause in clauses {
            let (function, pointers, len) = read_clause(clause).expect("a pointer clause");
            let values: Vec<PointerValue> = pointers
                .iter()
                .map(|p| self.pointer_value(p, None))
                .collect();
            self.add_pointer_fact(function, &values, len.clone());
        }
        pass_bases
    }

    /// Proves the pointer clause `clause` of a loop invariant from the facts here, as at a call.
    /// On a failure, gives the event that ended a fact that would prove it, if there is one.
    pub(crate) fn prove_pointer_clause_here(
        &mut self,
        clause: &Expr,
    ) -> Result<(), Option<(Span, String)>> {
        let (function, pointers, len) = read_clause(clause).expect("a pointer clause");
        let values: Vec<PointerValue> = pointers
            .iter()
            .map(|p| self.pointer_value(p, None))
            .collect();
        let goal = self.pointer_goal(function, &values, len, None);
        if self.prove_exact(&goal) {
            return Ok(());
        }
        Err(self.ended_fact_that_proves(function, &values, len))
    }

    /// At the end of a pass or at a `continue`: whether the pointer clause `clause` holds for the
    /// next pass. `names` gives the variables of the body new names, as in the prover's facts. A
    /// pointer that the body assigns must also point into the memory of the base that the pass
    /// started with. Otherwise the next pass could use a fact that a later event does not end.
    pub(crate) fn pointer_clause_holds_in_the_next_pass(
        &mut self,
        clause: &Expr,
        names: &HashMap<crate::symbol::Symbol, Expr>,
        pass_bases: &[(String, usize)],
    ) -> bool {
        let here = self.pointer_facts.clone();
        self.pointer_facts.rename_all(names);
        let memory_kept = pass_bases.iter().all(|(name, start)| {
            if !expr_mentions(clause, name) {
                return true;
            }
            let facts = &self.pointer_facts;
            let Some(now) = facts.entries.get(name).map(|v| v.base) else {
                return false;
            };
            match (facts.memory.get(&now), facts.memory.get(start)) {
                _ if now == *start => true,
                (Some(now), Some(start)) => now.iter().all(|m| start.contains(m)),
                _ => false,
            }
        });
        let holds = memory_kept && self.prove_pointer_clause_here(clause).is_ok();
        self.pointer_facts = here;
        holds
    }

    /// The first event that a pass of the loop `body` can have, as its place and its reason.
    /// The body is not checked yet. So the type of an argument comes from the parameter of the
    /// callee, or from the variables before the loop and the `let`s of the body. A value of a type
    /// that the scan does not know can free memory.
    ///
    /// An event of a value whose type the scan does not know comes after the other events,
    /// because the value can have a type with no drop.
    fn loop_body_event(&mut self, body: &[Statement]) -> Option<(Span, String)> {
        let declared = self.types_of_body_variables(body);
        let mut events = Vec::new();
        self.statement_events(body, &declared, &mut events);
        // The fact of an `assume` runs no code.
        let mut in_an_assume = std::collections::HashSet::new();
        crate::syntax::visit_exprs(body, &mut |e| match e {
            Expr::FunctionCall(fc) if fc.name.as_ref() == crate::syntax::stmt::ASSUME => {
                for a in &fc.args {
                    crate::syntax::visit_expr(a, &mut |x| {
                        let at = x.span();
                        if at != Span::default() {
                            in_an_assume.insert((at.line, at.column));
                        }
                    });
                }
            }
            _ => {}
        });
        for s in body {
            // The parser gives some calls no place. Their note gives the line of the statement.
            let first = events.len();
            crate::syntax::visit_exprs(std::slice::from_ref(s), &mut |e| {
                let at = e.span();
                if !in_an_assume.contains(&(at.line, at.column)) {
                    self.expression_events(e, &declared, &mut events)
                }
            });
            for event in &mut events[first..] {
                if event.1 == Span::default() {
                    event.1 = s.span();
                }
            }
        }
        events
            .into_iter()
            .min_by_key(|(unknown, at, _)| (*unknown, at.line, at.column))
            .map(|(_, at, why)| (at, why))
    }

    /// The type of each variable that the loop `body` declares, before the body is checked. A
    /// variable that the body declares once, with a `let`, has the type of its value, when a
    /// variable of the same name before the loop has that type too. Each other one is `Unknown`.
    fn types_of_body_variables(&self, body: &[Statement]) -> HashMap<String, Type> {
        let (counts, lets) = declarations_of_a_body(body);
        let mut types: HashMap<String, Type> =
            counts.keys().map(|n| (n.clone(), Type::Unknown)).collect();
        // A value can name the variable of another `let`. So the scan repeats until it finds no
        // new type.
        for _ in 0..lets.len() {
            let mut found = false;
            for (name, written, value) in &lets {
                if counts[name] != 1 || types[name] != Type::Unknown {
                    continue;
                }
                let ty = match written {
                    Some(t) => t.clone(),
                    None => self.scan_type(value, &types),
                };
                let outer = self.lookup(name).map(|(t, _)| t.clone());
                if ty != Type::Unknown && outer.is_none_or(|o| o == ty) {
                    types.insert(name.clone(), ty);
                    found = true;
                }
            }
            if !found {
                break;
            }
        }
        types
    }

    /// The events of the statements of one block, and of the loops in it, that are not in an
    /// expression: a drop, the end of the scope of a `let` whose value has a drop, an assignment
    /// that drops the old value, and the call of `next` in a `for` loop over an iterator.
    fn statement_events(
        &mut self,
        stmts: &[Statement],
        declared: &HashMap<String, Type>,
        events: &mut Vec<(bool, Span, String)>,
    ) {
        for s in stmts {
            match s {
                // The checker writes a drop after it checks a block. So the scan sees one only
                // when the driver checks a body again, for the flat code generator. A drop that
                // only ends the lifetime of a value frees nothing.
                Statement::Drop(d) if !d.ends_lifetime => {
                    let at = self.current_statement_span;
                    events.push((false, at, format!("the drop of `{}`", d.name)))
                }
                Statement::LetDecl(l) => {
                    let ty = match &l.ty_ann {
                        Some(t) => t.clone(),
                        None => self.scan_type(&l.expr, declared),
                    };
                    if self.scan_type_drops(&ty) {
                        let why = format!(
                            "the end of the scope of `{}`, which drops its value",
                            l.name
                        );
                        events.push((ty == Type::Unknown, l.span, why));
                    }
                }
                Statement::Assign(AssignStmt { lhs, span, .. })
                | Statement::CompoundAssign(CompoundAssignStmt { lhs, span, .. }) => {
                    let ty = self.scan_type(lhs, declared);
                    if self.scan_type_drops(&ty) {
                        let why = "an assignment that drops the old value".to_string();
                        events.push((ty == Type::Unknown, *span, why));
                    }
                }
                Statement::ForLoop(l) => {
                    // A `for` loop over a tensor reads it by index, with no call of `next`.
                    let ty = self.scan_type(&l.iterable, declared);
                    if !matches!(*l.iterable, Expr::Range(_))
                        && !matches!(ty, Type::Tensor(..))
                        && self.type_can_free(&ty, 0)
                    {
                        let why = format!(
                            "the call to 'next' of the `for` loop, which can free memory through \
                             {}",
                            shown_argument(&l.iterable)
                        );
                        events.push((false, l.span, why));
                    }
                    self.statement_events(&l.body, declared, events);
                }
                Statement::Loop(l) => self.statement_events(&l.body, declared, events),
                _ => {}
            }
        }
    }

    /// The events of the expression `e` alone, and of the statements of each block in it.
    fn expression_events(
        &mut self,
        e: &Expr,
        declared: &HashMap<String, Type>,
        events: &mut Vec<(bool, Span, String)>,
    ) {
        let spawn = "the `spawn`, whose task can free memory";
        match e {
            Expr::FunctionCall(fc) => {
                if let Some(why) = self.scan_call_event(fc, declared) {
                    events.push((false, fc.span, why));
                }
            }
            Expr::MethodCall(m) => {
                // The receiver and an argument for a parameter with a written `!frees` free
                // nothing.
                let ty = self.scan_type(&m.base, declared);
                let speculating = std::mem::replace(&mut self.speculating, true);
                let not_freed = self
                    .resolve_method_in_impls(&ty, &m.method_name, &mut HashMap::new())
                    .map_or(0, |(f, _)| {
                        crate::hir::check::pointer_clauses::not_freed_params(&f.params, &f.ensures)
                    });
                self.speculating = speculating;
                let arg = std::iter::once(&*m.base)
                    .chain(&m.args)
                    .enumerate()
                    .find(|&(i, a)| {
                        (i >= 32 || not_freed & (1 << i) == 0)
                            && self.scan_can_free(a, None, declared)
                    })
                    .map(|(_, a)| a);
                if let Some(a) = arg {
                    let why = format!(
                        "the call to '{}', which can free memory through {}",
                        m.method_name,
                        shown_argument(a)
                    );
                    events.push((false, m.span, why));
                }
            }
            Expr::IndirectCall(ic) => {
                let callee = self.scan_type(&ic.callee, declared);
                let params = match &callee {
                    Type::Function(params, ..) => Some(params.clone()),
                    _ => None,
                };
                let why = if params.is_none() && self.type_can_free(&callee, 0) {
                    Some(format!(
                        "the call of the closure {}, which can free memory through what it \
                         captures",
                        shown_argument(&ic.callee)
                    ))
                } else {
                    self.scan_argument_that_can_free(&ic.args, params.as_deref(), 0, declared)
                        .map(|a| {
                            format!(
                                "the call through {}, which can free memory through {a}",
                                shown_argument(&ic.callee)
                            )
                        })
                };
                if let Some(why) = why {
                    events.push((false, ic.span, why));
                }
            }
            Expr::InlineMlir(m)
                if !m.clobbers.is_empty()
                    || m.inputs
                        .iter()
                        .any(|(_, a, _)| self.scan_can_free(a, None, declared)) =>
            {
                let why = "the `mlir!` block, which can free memory".to_string();
                events.push((false, m.span, why));
            }
            Expr::SpawnOn(s) => events.push((false, s.span, spawn.to_string())),
            Expr::TaskBlock(t) if t.kind == TaskBlockKind::Spawn => {
                events.push((false, t.span, spawn.to_string()))
            }
            Expr::TaskBlock(t) => self.statement_events(&t.stmts, declared, events),
            Expr::If(i) => {
                self.statement_events(&i.then_block, declared, events);
                if let Some(b) = &i.else_block {
                    self.statement_events(b, declared, events);
                }
            }
            Expr::Match(m) => {
                for arm in &m.arms {
                    self.statement_events(&arm.body, declared, events);
                }
            }
            Expr::UnsafeBlock(u) => self.statement_events(&u.stmts, declared, events),
            // A `comptime` block runs when the program compiles, so its statements drop nothing
            // when the program runs.
            _ => {}
        }
    }

    /// The reason that the call `fc` in a loop body can free memory, if it can.
    fn scan_call_event(
        &mut self,
        fc: &FunctionCallExpr,
        declared: &HashMap<String, Type>,
    ) -> Option<String> {
        let name = fc.name.as_ref();
        if ADDRESS_FUNCTIONS.contains(&plain_name(name)) || name == crate::syntax::stmt::ASSUME {
            return None;
        }
        // A call of a local variable, such as a closure.
        let local = match declared.get(name) {
            Some(t) => Some(t.clone()),
            None => self.lookup(name).map(|(t, _)| t.clone()),
        };
        let params = match &local {
            Some(Type::Function(params, ..)) => Some(params.clone()),
            Some(t) if self.type_can_free(t, 0) => {
                return Some(format!(
                    "the call of the closure `{name}`, which can free memory through what it \
                     captures"
                ));
            }
            Some(_) => None,
            None => self.function_param_types(name),
        };
        let not_freed = if local.is_none() {
            self.not_freed_by(name)
        } else {
            0
        };
        let arg =
            self.scan_argument_that_can_free(&fc.args, params.as_deref(), not_freed, declared)?;
        Some(match local {
            Some(_) => format!("the call through `{name}`, which can free memory through {arg}"),
            None => format!("the call to '{name}', which can free memory through {arg}"),
        })
    }

    /// The first argument of `args` that can free memory, as the program wrote it. An argument
    /// for a parameter in `not_freed`, the `!frees` bits of the callee, frees nothing.
    fn scan_argument_that_can_free(
        &mut self,
        args: &[Expr],
        params: Option<&[Type]>,
        not_freed: u32,
        declared: &HashMap<String, Type>,
    ) -> Option<String> {
        args.iter()
            .enumerate()
            .filter(|(i, _)| *i >= 32 || not_freed & (1 << i) == 0)
            .find(|(i, a)| self.scan_can_free(a, params.and_then(|p| p.get(*i)), declared))
            .map(|(_, a)| shown_argument(a))
    }

    /// Whether the argument `arg` in a loop body that is not checked yet can free memory. The
    /// type of its parameter `param` says it, and a type parameter counts as one that can.
    fn scan_can_free(
        &mut self,
        arg: &Expr,
        param: Option<&Type>,
        declared: &HashMap<String, Type>,
    ) -> bool {
        let ty = match param {
            Some(p) => p.clone(),
            None => self.scan_type(arg, declared),
        };
        self.type_can_free(&ty, 0)
    }

    /// Whether the drop of a value of the type `ty` can free memory.
    fn scan_type_drops(&mut self, ty: &Type) -> bool {
        matches!(ty, Type::Unknown | Type::Generic(..)) || !self.drops_glue(ty).is_empty()
    }

    /// The type of `e` in a loop body that is not checked yet, or `Unknown`. `declared` has the
    /// type of each variable of the body.
    fn scan_type(&self, e: &Expr, declared: &HashMap<String, Type>) -> Type {
        let scalar = |t: &Type| matches!(t, Type::Scalar(_));
        let both_scalar = |a: &Expr, b: &Expr| {
            scalar(&self.scan_type(a, declared)) && scalar(&self.scan_type(b, declared))
        };
        match crate::syntax::bare_unsafe_value(e) {
            Expr::Number(n) => Type::Scalar(n.ty.clone().unwrap_or(ElementType::I64)),
            Expr::Identifier(id) => match declared.get(id.name.as_ref()) {
                Some(t) => t.clone(),
                None => self.check_expr_type_probe(e),
            },
            Expr::BinaryOp(b) if both_scalar(&b.lhs, &b.rhs) => self.scan_type(&b.lhs, declared),
            Expr::RelationalOp(r) if both_scalar(&r.lhs, &r.rhs) => Type::Scalar(ElementType::Bool),
            Expr::LogicalOp(l) if both_scalar(&l.lhs, &l.rhs) => Type::Scalar(ElementType::Bool),
            Expr::UnaryOp(u) => match self.scan_type(&u.expr, declared) {
                t if scalar(&t) => t,
                _ => Type::Unknown,
            },
            // `offset(p, k)` has the type of `p`.
            Expr::FunctionCall(fc)
                if matches!(
                    plain_name(fc.name.as_ref()),
                    "offset" | "offset_mut" | "add" | "add_mut" | "sub" | "sub_mut"
                ) && !fc.args.is_empty() =>
            {
                self.scan_type(&fc.args[0], declared)
            }
            // A function that is not a variable has the return type that it declares.
            Expr::FunctionCall(fc)
                if !declared.contains_key(fc.name.as_ref())
                    && self.lookup(fc.name.as_ref()).is_none() =>
            {
                self.check_expr_type_probe(e)
            }
            Expr::Dereference(d) => match self.scan_type(&d.expr, declared) {
                Type::Pointer(inner, ..) | Type::Borrow { inner, .. } | Type::Ref(inner, _) => {
                    *inner
                }
                _ => Type::Unknown,
            },
            Expr::IndexAccess(ix) => match self.scan_type(&ix.base, declared) {
                Type::Pointer(inner, ..) if scalar(&inner) => *inner,
                Type::Tensor(el, dims, _) if dims.len() <= 1 => Type::Scalar(el),
                _ => Type::Unknown,
            },
            // A block with a value has the type of its last expression.
            Expr::UnsafeBlock(UnsafeBlockExpr { ret: Some(r), .. })
            | Expr::ComptimeBlock(ComptimeBlockExpr { ret: Some(r), .. }) => {
                self.scan_type(r, declared)
            }
            Expr::InlineMlir(m) => m
                .returns
                .clone()
                .unwrap_or_else(|| Type::Struct("void".into(), None)),
            e => {
                let mut names_the_body = false;
                crate::syntax::visit_expr(e, &mut |x| {
                    if let Expr::Identifier(id) = x {
                        names_the_body |= declared.contains_key(id.name.as_ref());
                    }
                });
                if names_the_body {
                    Type::Unknown
                } else {
                    self.check_expr_type_probe(e)
                }
            }
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

    /// Runs `events`, and then `changes`, the changes at the same place, as of a call or of a
    /// loop. The changes keep the facts that the events ended, so that the note of E8006 can name
    /// the event, as for a fact about `v.len` and `v.push(1i64)`.
    pub(crate) fn pointer_facts_events_then_changes(
        &mut self,
        events: impl FnOnce(&mut Self),
        changes: impl FnOnce(&mut Self),
    ) {
        let from = self.pointer_facts.ended.len();
        events(self);
        let ended = self.pointer_facts.ended[from..].to_vec();
        changes(self);
        for e in ended {
            if !self
                .pointer_facts
                .ended
                .iter()
                .any(|x| x.range.id == e.range.id)
            {
                self.pointer_facts.ended.push(e);
            }
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
        if !self.speculating {
            let trusted = self.in_unsafe_block || self.in_unsafe_fn;
            // A clause that a call in `unsafe` code trusts needs the `unsafe` (W1010).
            if self.prove_pointer_requires(name, args, &at, trusted) {
                self.unsafe_op_seen = true;
            }
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
            // An argument for a parameter with `!frees` frees nothing in the call.
            let not_freed = match expr {
                Expr::FunctionCall(_) if !local_call => self.not_freed_by(name),
                _ => 0,
            };
            let mut through = callee.filter(|c| self.argument_can_free(c, None));
            for (i, a) in args.iter().enumerate() {
                if through.is_none()
                    && (i >= 32 || not_freed & (1 << i) == 0)
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
        if !self.speculating && !local_call && matches!(expr, Expr::FunctionCall(_)) {
            self.add_facts_of_ensures(name, args, at);
        }
    }

    /// The parameters and the pointer `ensures` of the function `name`, when the checker knows
    /// it: a Vx function, an instance or an `extern` function.
    fn pointer_ensures_of(&self, name: &str) -> Option<(Params, Vec<Expr>)> {
        let clauses = |params: &[(crate::symbol::Symbol, Type)], ret: &Type, ensures: &[Expr]| {
            ensures
                .iter()
                .filter(|e| {
                    read_clause(e).is_some()
                        && crate::hir::check::pointer_clauses::read_pointer_clause(
                            "", params, ret, true, e,
                        )
                        .is_ok()
                })
                .cloned()
                .collect::<Vec<Expr>>()
        };
        let function = self.env.syntax_functions.get(name).copied().or_else(|| {
            self.mono
                .functions
                .iter()
                .find(|(f, _)| f.name.as_ref() == name)
                .map(|(f, _)| f)
        });
        if let Some(f) = function {
            return Some((
                f.params.clone(),
                clauses(&f.params, &f.return_type, &f.ensures),
            ));
        }
        let ext = self.env.externs.iter().find(|e| e.name.as_ref() == name)?;
        let ensures = clauses(&ext.params, &ext.return_type, &ext.ensures);
        Some((ext.params.clone(), ensures))
    }

    /// After the events of the call at `at` to `name`: the facts of its pointer `ensures`. In
    /// each clause, each parameter stands for its argument, and `return` for the result. The
    /// result gets a new base.
    fn add_facts_of_ensures(&mut self, name: &str, args: &[Expr], at: Span) {
        let Some((params, clauses)) = self.pointer_ensures_of(name) else {
            return;
        };
        if clauses.is_empty() {
            return;
        }
        let mut names = HashMap::new();
        for (i, ((param, _), arg)) in params.iter().zip(args).enumerate() {
            let arg = crate::hir::prover::without_borrow(arg);
            let value = if crate::hir::prover::is_modelled(arg) {
                arg.clone()
            } else {
                Expr::Identifier(IdentifierExpr {
                    name: format!("$arg_{}_{}_{i}", at.line, at.column).into(),
                    span: at,
                })
            };
            names.insert(param.clone(), value);
        }
        names.insert("return".into(), result_name(at));
        let is_return =
            |p: &&Expr| matches!(p, Expr::Identifier(id) if id.name.as_ref() == "return");
        let about_the_result = clauses
            .iter()
            .any(|c| read_clause(c).is_some_and(|(_, pointers, _)| pointers.iter().any(is_return)));
        let result = if about_the_result {
            self.call_result_value(args)
        } else {
            None
        };
        if let Some(v) = &result {
            self.pointer_facts
                .results
                .insert((at.line, at.column), v.clone());
        }
        for clause in &clauses {
            let Some((function, pointers, len)) = read_clause(clause) else {
                continue;
            };
            let values: Option<Vec<PointerValue>> = pointers
                .iter()
                .map(|p| match p {
                    Expr::Identifier(id) if id.name.as_ref() == "return" => result.clone(),
                    Expr::Identifier(id) => {
                        let i = params.iter().position(|(n, _)| n == &id.name)?;
                        let arg = args.get(i)?.clone();
                        Some(self.pointer_value(&arg, None))
                    }
                    _ => None,
                })
                .collect();
            let Some(values) = values else {
                continue;
            };
            let n = crate::hir::prover::substitute_names(len, &names);
            self.add_pointer_fact(function, &values, n);
        }
    }

    /// The fact `function(values, n)`, such as `valid(p, n)`, from an `ensures` or an `assume`.
    fn add_pointer_fact(&mut self, function: &str, values: &[PointerValue], n: Expr) {
        if function == "overlaps" {
            self.pointer_fact_ids += 1;
            let id = self.pointer_fact_ids;
            self.pointer_facts.aparts.push(Apart {
                id,
                first: (values[0].base, values[0].offset.clone()),
                second: (values[1].base, values[1].offset.clone()),
                len: n,
            });
            return;
        }
        let v = &values[0];
        let kind = if function == "valid" {
            Kind::Valid
        } else {
            Kind::Init
        };
        if kind == Kind::Valid && crate::hir::prover::is_modelled(&n) {
            self.consteval
                .constraints
                .push(Expr::RelationalOp(RelationalOpExpr {
                    lhs: Box::new(n.clone()),
                    op: RelationalOp::Ge,
                    rhs: Box::new(zero()),
                    span: Span::default(),
                    operand_ty: None,
                }));
        }
        let other_memory = self
            .pointer_facts
            .memory
            .get(&v.base)
            .is_some_and(|ms| ms.contains(&Memory::Other));
        let lo = v.offset.clone();
        let hi = lo.iter().cloned().chain([n.clone()]).collect();
        if let Some(&(_, _, spawn)) = self.pointer_facts.tasks.last().filter(|_| other_memory) {
            // The fact does not start, and the note of E8006 names the `spawn`.
            self.pointer_fact_ids += 1;
            let range = Range {
                id: self.pointer_fact_ids,
                kind,
                base: v.base,
                lo,
                hi,
                merged: false,
            };
            let why = "the `spawn`, whose task can free memory until its `join()`".to_string();
            self.pointer_facts.ended.push(Ended {
                range,
                at: spawn,
                why,
            });
            return;
        }
        self.add_range(kind, v.base, lo, hi);
    }

    /// A new base for the result of a call with the arguments `args`. Its memory is other memory
    /// and the local memory of each variable that an argument can point into. `None` when the
    /// checker does not know where an argument can point.
    fn call_result_value(&mut self, args: &[Expr]) -> Option<PointerValue> {
        let mut memory = vec![Memory::Other];
        for arg in args {
            for m in self.locals_an_argument_reaches(arg)? {
                if !memory.contains(&m) {
                    memory.push(m);
                }
            }
        }
        Some(PointerValue {
            base: self.new_base_in(memory),
            offset: Vec::new(),
        })
    }

    /// The local memory of each variable that the argument `arg` can point into, or `None` when
    /// the checker does not know it. A reference parameter that the body does not assign or hide
    /// points into the memory of the caller.
    fn locals_an_argument_reaches(&mut self, arg: &Expr) -> Option<Vec<Memory>> {
        fn root(e: &Expr) -> Option<&str> {
            match e {
                Expr::Identifier(id) => Some(id.name.as_ref()),
                Expr::MemberAccess(m) => root(&m.base),
                _ => None,
            }
        }
        // A number, and arithmetic on numbers, points nowhere.
        if crate::hir::prover::is_modelled(arg) && !matches!(arg, Expr::Identifier(_)) {
            return Some(Vec::new());
        }
        let ty = self.check_expr_type_probe(arg);
        if matches!(ty, Type::Scalar(ref e) if !matches!(e, ElementType::Generic(_)))
            || matches!(ty, Type::Function(..) | Type::Simd(..))
        {
            return Some(Vec::new());
        }
        let a_caller_reference = |c: &Self, name: &str| {
            c.borrow.current_params.contains_key(name)
                && !c.declared_vars.iter().any(|(v, _)| v.as_ref() == name)
                && !c.pointer_function.params_changed.contains(name)
                && matches!(c.lookup(name), Some((Type::Borrow { .. }, _)))
        };
        match arg {
            Expr::Borrow(b) => {
                let var = root(&b.expr)?;
                if a_caller_reference(self, var) {
                    return Some(Vec::new());
                }
                let var_ty = self.lookup(var).map(|(t, _)| t.clone())?;
                if matches!(
                    var_ty,
                    Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..)
                ) {
                    return None;
                }
                let drops =
                    matches!(var_ty, Type::Generic(..)) || !self.drops_glue(&var_ty).is_empty();
                Some(vec![Memory::Local {
                    var: var.to_string(),
                    drops,
                }])
            }
            Expr::Identifier(id) if a_caller_reference(self, id.name.as_ref()) => Some(Vec::new()),
            _ if matches!(ty, Type::Pointer(..)) => {
                let v = self.pointer_value(arg, None);
                let memory = self
                    .pointer_facts
                    .memory
                    .get(&v.base)
                    .cloned()
                    .unwrap_or_default();
                Some(memory.into_iter().filter(|m| *m != Memory::Other).collect())
            }
            _ => None,
        }
    }

    /// The parameter types of the function that the call `expr` to `name` calls, when the
    /// checker knows them.
    pub(crate) fn call_param_types(&self, name: &str, expr: &Expr) -> Option<Vec<Type>> {
        let from_type = |t: Type| match t {
            Type::Function(params, ..) | Type::Closure(params, _) => Some(params),
            _ => None,
        };
        match expr {
            Expr::IndirectCall(ic) => from_type(self.check_expr_type_probe(&ic.callee)),
            Expr::FunctionCall(_) => match self.lookup(name) {
                Some((t, _)) => from_type(t.clone()),
                None => self.function_param_types(name),
            },
            _ => None,
        }
    }

    /// The parameter types of the function `name`, when the checker knows them.
    fn function_param_types(&self, name: &str) -> Option<Vec<Type>> {
        self.env
            .functions
            .get(name)
            .map(|f| f.2.clone())
            .or_else(|| {
                self.mono
                    .functions
                    .iter()
                    .find(|(f, _)| f.name.as_ref() == name)
                    .map(|(f, _)| f.params.iter().map(|(_, t)| t.clone()).collect())
            })
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

    /// E8006 for each pointer `requires` of the callee `name` that the facts do not prove. With
    /// `trusted`, for a call in `unsafe` code, each such clause goes in the audit instead, and the
    /// result says whether there is one.
    fn prove_pointer_requires(
        &mut self,
        name: &str,
        args: &[Expr],
        span: &Span,
        trusted: bool,
    ) -> bool {
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
            return false;
        };
        if clauses.is_empty() {
            return false;
        }
        let mut any_trusted = false;
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
            if trusted {
                let what = format!(
                    "a call in `unsafe` code trusts `{}` of `{}`, which the compiler cannot prove",
                    clause_source(clause),
                    self.shown_callee(name)
                );
                self.note_trusted(*span, what);
                any_trusted = true;
                continue;
            }
            let ended = self.ended_fact_that_proves(function, &values, &m);
            let lost = match ended {
                Some(_) => None,
                None => self.note_of_a_lost_invariant(|c| c.prove_exact(&goal)),
            };
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
            error.notes.extend(lost);
        }
        any_trusted
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
            // The fact held, so its range was not negative. A change at its event can hide that.
            let negative = le(sum(&e.range.hi, &[&one()]), sum(&e.range.lo, &[]));
            self.prove_exact(&Exact::Or(vec![negative, goal]))
                .then_some((e.at, e.why))
        })
    }

    /// The callee `name` as the program wrote it: an instance of a generic function, such as
    /// `sum$i64`, is shown as `sum`.
    pub(crate) fn shown_callee(&self, name: &str) -> String {
        match name.split_once('$') {
            Some((first, _)) if self.env.generic_functions.contains_key(first) => first.to_string(),
            _ => name.to_string(),
        }
    }

    /// Before a new fact `Init(base, lo, hi)`: whether another `Init` fact of the base takes its
    /// place. A fact that covers the new range does. So does a fact whose range touches or
    /// overlaps it: the two become one merged fact, from the lower `lo` to the higher `hi`. Then
    /// no chain of three facts forms, and a goal needs at most two facts.
    fn merge_init(&mut self, base: usize, lo: &[Expr], hi: &[Expr]) -> bool {
        let mut same: Vec<Range> = self
            .pointer_facts
            .ranges
            .iter()
            .filter(|r| r.kind == Kind::Init && r.base == base)
            .cloned()
            .collect();
        let at_most = |a: &[Expr], b: &[Expr]| le(sum(a, &[]), sum(b, &[]));
        for r in &same {
            if self.prove_exact(&Exact::And(vec![at_most(&r.lo, lo), at_most(hi, &r.hi)])) {
                return true;
            }
        }
        // A merged fact grows in place, so it comes first.
        same.sort_by_key(|r| !r.merged);
        for r in same {
            let new_after = [at_most(&r.lo, lo), at_most(lo, &r.hi), at_most(&r.hi, hi)];
            let new_before = [at_most(lo, &r.lo), at_most(&r.lo, hi), at_most(hi, &r.hi)];
            let new_covers = [at_most(lo, &r.lo), at_most(&r.hi, hi)];
            let (new_lo, new_hi) = if self.prove_exact(&Exact::And(new_after.into())) {
                (r.lo.clone(), hi.to_vec())
            } else if self.prove_exact(&Exact::And(new_before.into())) {
                (lo.to_vec(), r.hi.clone())
            } else if self.prove_exact(&Exact::And(new_covers.into())) {
                (lo.to_vec(), hi.to_vec())
            } else {
                continue;
            };
            self.pointer_fact_ids += 1;
            let merged = Range {
                id: self.pointer_fact_ids,
                kind: Kind::Init,
                base,
                lo: new_lo,
                hi: new_hi,
                merged: true,
            };
            let ranges = &mut self.pointer_facts.ranges;
            match ranges.iter_mut().find(|f| f.id == r.id && f.merged) {
                Some(f) => *f = merged,
                None => ranges.push(merged),
            }
            return true;
        }
        false
    }

    /// The goal that proves `valid`, `initialized` or `!overlaps` for the pointer values
    /// `values` and the length `m`, from one fact or two. `extra` is one more range to use.
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
                le(sum(&r.lo, &[]), sum(&v.offset, &[])),
                le(sum(&v.offset, &[m]), sum(&r.hi, &[])),
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
                let mine: Vec<&Range> = ranges()
                    .filter(|r| r.kind == kind && r.base == v.base)
                    .collect();
                let mut goals: Vec<Exact> = mine.iter().map(|r| within(r, v)).collect();
                // Two facts: the first holds the start, and the second the end. `Init` ranges
                // can touch. `Valid` ranges must share an element, because two allocations can
                // sit next to each other.
                for a in &mine {
                    for b in mine.iter().filter(|b| b.id != a.id) {
                        let joins = match kind {
                            Kind::Init => le(sum(&b.lo, &[]), sum(&a.hi, &[])),
                            Kind::Valid => le(sum(&b.lo, &[&one()]), sum(&a.hi, &[])),
                        };
                        goals.push(Exact::And(vec![
                            le(sum(&a.lo, &[]), sum(&v.offset, &[])),
                            le(sum(&a.lo, &[]), sum(&b.lo, &[])),
                            joins,
                            le(sum(&v.offset, &[m]), sum(&b.hi, &[])),
                        ]));
                    }
                }
                // Zero elements hold values. `valid(p, 0)` still says that `p` is not null.
                if kind == Kind::Init {
                    goals.push(le(term(m.clone()), term(zero())));
                }
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
                            let window = |(_, lo): &(usize, Vec<Expr>), v: &PointerValue| {
                                Exact::And(vec![
                                    le(sum(lo, &[]), sum(&v.offset, &[])),
                                    le(sum(&v.offset, &[m]), sum(lo, &[&apart.len])),
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
pub(crate) fn clause_source(e: &Expr) -> String {
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
