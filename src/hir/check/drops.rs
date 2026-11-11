//===- drops.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where each tensor a function owns is dropped: `docs/implementation_plans/drop_semantics.md`.
// The checker writes them into the program as `Statement::Drop`, which the code generators
// lower. With `VX_PRINT_DROPS=1` the points are also printed.
//
// An owner is a `let` bound to a tensor that is not a view, or a tensor parameter taken by
// value. It is dropped after the statement of its block that last uses it, or anything that
// borrows it: a view, a reference, a closure. A use inside a nested `if` or loop counts as a
// use by that whole statement. A `return`, `break` or `continue` first drops the owners it
// leaves that are still alive. An owner moved at its own level is not dropped; one moved
// inside a nested block may or may not have been, so its drop is under a flag. An owner a
// raw pointer was taken from (`t.as_ptr()`) waits for the end of its block: no borrow
// checker sees what a raw pointer is used for. Assigning a new tensor to an owner drops the
// old one first, unless it was moved, or `c = a @ b` writes the product into `c`'s buffer.
// A tensor operator moves its operands, as in Rust, so it drops them once it has its result.
//
// Written into the program, a drop never frees what is still to be read. A drop placed before
// a `return`, a block's last expression or an assignment is marked `after_value`: it runs once
// that statement has computed its value, which may read the tensor, and before it is returned
// or stored. A flag is a `bool` local, set after the statement that moves the owner, in the
// block where it moves, and named by the drop, which then frees nothing once it is set.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Left in a block's uses by a `Statement::Drop`, so a body already rewritten is known: a
/// generic instantiation can be checked again from a copy of it.
pub(crate) const DROP_MARK: &str = "\u{1}drop";

/// The default of `Drop::wait_limit` in `stdlib/core/ops.vx`, in seconds.
pub(crate) const DEFAULT_WAIT_LIMIT_SECS: u64 = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moved {
    No,
    Maybe,
    Yes,
}

#[derive(Clone)]
struct Owner {
    name: String,
    /// What gave it to its block, `the function` or `the arm`; `None` for a `let`.
    given_by: Option<&'static str>,
    /// The statement that declares it in its block; `None` for a parameter.
    decl: Option<usize>,
    /// A raw pointer was taken from it: dropped at the end of its block, not its last use.
    to_block_end: bool,
    /// Its drop can read a reference it holds, so what it borrows stays borrowed until then.
    drop_reads_borrows: bool,
    /// The scope it was declared in: a move from the same scope is certain.
    scope: usize,
    moved: Moved,
    /// The statement of its block that last moved it for certain.
    moved_at: Option<usize>,
    /// The `bool` local that says it was moved, once a move inside a nested block needed one.
    flag: Option<String>,
    /// What dropping it does, in order.
    glue: Vec<GluePart>,
    /// Fields moved out of it, `let x = p.a`: their parts of `glue` are not dropped with it.
    fields: Vec<FieldMove>,
    /// A struct or enum in a stack slot: its last drop also ends the lifetime of the slot.
    ends_lifetime: bool,
    ty: Type,
}

impl Owner {
    /// An owner with nothing to drop. It is an owner only so that its slot's lifetime ends.
    fn only_ends_lifetime(&self) -> bool {
        self.glue.is_empty()
    }
}

/// A field's path, whether it was moved out, and the flag that says so when it may have been.
pub(crate) type MovedField = (Vec<String>, Moved, Option<String>);

/// An owner that the `let` at statement `at` of its block hid. Its new name is `to`.
#[derive(Clone)]
struct Hidden {
    from: String,
    to: String,
    /// The `let` that declares it, or `None` for a parameter or a name of a `match` arm.
    decl: Option<usize>,
    at: usize,
    ty: Type,
}

/// A field moved out of an owner, with the same states the owner itself has.
#[derive(Clone)]
struct FieldMove {
    path: Vec<String>,
    moved: Moved,
    moved_at: Option<usize>,
    flag: Option<String>,
}

impl Owner {
    /// The move state of the owner, or with `field`, of that field of it.
    fn state(
        &mut self,
        field: Option<&[String]>,
    ) -> (&mut Moved, &mut Option<usize>, &mut Option<String>) {
        let Some(path) = field else {
            return (&mut self.moved, &mut self.moved_at, &mut self.flag);
        };
        let k = match self.fields.iter().position(|f| f.path == path) {
            Some(k) => k,
            None => {
                self.fields.push(FieldMove {
                    path: path.to_vec(),
                    moved: Moved::No,
                    moved_at: None,
                    flag: None,
                });
                self.fields.len() - 1
            }
        };
        let f = &mut self.fields[k];
        (&mut f.moved, &mut f.moved_at, &mut f.flag)
    }
}

/// One step of dropping a value: a `Drop` impl's `drop` to call, or a tensor to free, at a
/// field path inside it.
#[derive(Clone)]
pub(crate) struct GluePart {
    /// Each field followed, with the type of the struct it is a field of.
    path: Vec<(crate::symbol::Symbol, String)>,
    call: Option<crate::symbol::Symbol>,
    /// With `call`: the mangled name of the type's `Drop::wait_limit_nanos`.
    limit: Option<crate::symbol::Symbol>,
    /// For an enum: what dropping each variant's payload does, by element. Dropping it
    /// matches on the value.
    payload: Option<EnumGlue>,
    /// For a tensor: the memory its type names.
    placement: Option<crate::syntax::Placement>,
}

#[derive(Clone)]
pub(crate) struct EnumGlue {
    /// The enum's type as a pattern names it: `Opt<Noisy>`.
    name: String,
    /// The variants with something to drop: each payload element's glue, or `None`.
    arms: Vec<(crate::symbol::Symbol, Vec<Option<Vec<GluePart>>>)>,
}

impl GluePart {
    /// Whether dropping this runs a `Drop` impl anywhere.
    fn calls(&self) -> bool {
        self.call.is_some()
            || self.payload.as_ref().is_some_and(|p| {
                p.arms
                    .iter()
                    .flat_map(|(_, elems)| elems.iter().flatten().flatten())
                    .any(GluePart::calls)
            })
    }
}

/// One walk over a statement's calls, in the order they are evaluated.
struct TempScan {
    /// The calls seen so far.
    calls: usize,
    /// A branch, a loop or a call of unknown type was seen: nothing after it moves out.
    blocked: bool,
    /// Probing: the last call that is a temporary which could move out.
    last_temp: Option<usize>,
    /// Moving out: every call up to this one.
    upto: Option<usize>,
    /// Each call moved out: its local, type, value and drops.
    found: Vec<(String, Type, Expr, Vec<GluePart>)>,
    /// A `&&` or `||` became an `if`, so that a side drops its temporaries when it ends.
    short_circuit_rewritten: bool,
}

impl TempScan {
    fn new(upto: Option<usize>) -> Self {
        Self {
            calls: 0,
            blocked: false,
            last_temp: None,
            upto,
            found: Vec::new(),
            short_circuit_rewritten: false,
        }
    }
}

/// What the checker writes into one block, by statement.
#[derive(Clone, Default)]
struct Edits {
    /// The `let`s of the statement's temporaries, which go first.
    lets: BTreeMap<usize, Vec<Statement>>,
    before: BTreeMap<usize, Vec<Statement>>,
    after: BTreeMap<usize, Vec<Statement>>,
    /// `t = e`: the drops of the old value, if it was not moved, and the flag to clear once
    /// everything else at the statement has run.
    assign: HashMap<usize, (Vec<Statement>, Option<String>)>,
    /// `*call() = e`: the reference the call returned and the new value, held in the
    /// temporaries the statement now reads in place of the call and `e`.
    reroot: HashMap<usize, (String, String)>,
}

/// Where a `return`, `break`, `continue`, or the leave of a failed join goes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExitTo {
    Function,
    Loop,
    TasksBlock,
}

#[derive(Clone)]
enum FrameKind {
    Function {
        /// Whether drops are written into this body.
        rewrite: bool,
        /// The body checked so far works with a raw pointer. A raw pointer into a struct
        /// keeps no borrow, so the struct's slot then lives to the end of its block.
        raw_pointers: bool,
    },
    Loop,
    /// A `spawn` region. A tensor it captures stays owned by the enclosing function, so a
    /// move inside the region does not reach owners outside it.
    Spawn,
    /// A `tasks` block. An exit that leaves it waits for its tasks before its drops.
    Tasks,
    /// The body of a task. A `return` in it leaves the task, so it drops only the owners of the
    /// task. A move in it still reaches the owners outside, which the task takes.
    Task,
    Block {
        owners: Vec<Owner>,
        /// What each statement of the block uses, and where each name has a `let`.
        uses: BlockUseList,
        /// The first line of each statement, and of the block's separate value.
        lines: Vec<usize>,
        /// The statement being checked.
        stmt: usize,
        edits: Box<Edits>,
        /// The position in `uses` of a separate value, `v` in `unsafe { ..; v }`: one
        /// after the last statement. What it uses is dropped after it.
        value_at: Option<usize>,
        /// The owners that a later `let` of the block hid, now under a name of their own.
        hidden: Vec<Hidden>,
    },
}

/// What each statement of a block uses, nested uses included, and where each name has a
/// `let` in the block. A block's separate value is one more position after its statements.
#[derive(Clone, Default)]
pub(crate) struct BlockUseList {
    used: Vec<HashSet<String>>,
    lets: HashMap<String, Vec<usize>>,
}

impl BlockUseList {
    fn new(body: &[Statement], value: Option<&Expr>) -> Self {
        let mut list = BlockUseList::default();
        for (i, stmt) in body.iter().enumerate() {
            let mut used = HashSet::new();
            TypeChecker::extract_uses_stmt(stmt, &mut used);
            list.used.push(used);
            if let Statement::LetDecl(l) = stmt {
                list.lets.entry(l.name.to_string()).or_default().push(i);
            }
        }
        if let Some(value) = value {
            let mut used = HashSet::new();
            TypeChecker::extract_uses_expr(value, &mut used);
            list.used.push(used);
        }
        // An empty block still has one position, so a drop has somewhere to go.
        if list.used.is_empty() {
            list.used.push(HashSet::new());
        }
        list
    }

    /// The `let` that next hides `name` after `decl`, its own `let` (`None` for a parameter).
    fn hidden_at(&self, name: &str, decl: Option<usize>) -> Option<usize> {
        self.lets
            .get(name)?
            .iter()
            .copied()
            .find(|&i| decl.is_none_or(|d| i > d))
    }
}

/// The drop-point state of the function being checked: one frame per open block, loop and
/// function or closure body.
#[derive(Clone, Default)]
pub(crate) struct DropFrames {
    frames: Vec<FrameKind>,
    /// Owners to add to the next block: a function's parameters, or the names a `match` arm
    /// binds by value, that need dropping. `pending_by` is what gave them.
    pending: Vec<(String, usize, Vec<GluePart>, bool, Type)>,
    pending_by: &'static str,
    /// A `match` arm takes the value that is moving, and another arm may not. The move is not
    /// certain, so the owner gets a flag. `pending_sets` set the flags at the start of the arm.
    arm_takes: bool,
    pending_sets: Vec<Statement>,
    /// The names moved by the statement being checked. A `return t` moves `t` on its way out.
    moved_now: HashSet<String>,
    /// Makes the names of the locals the rewrite adds unique.
    counter: usize,
    /// For each open `tasks` block, the statement of its body that joins its tasks before the
    /// drops of what they use, if it has one.
    tasks_joins: Vec<Option<usize>>,
}

impl DropFrames {
    /// Whether `name` is a live value whose drop at the end of its block can read a reference
    /// it holds: that drop is a use of what it borrows.
    pub(crate) fn drop_reads_borrows(&self, name: &str) -> bool {
        for f in self.frames.iter().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return false,
                FrameKind::Loop | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, .. } => {
                    if let Some(o) = owners.iter().rev().find(|o| o.name == name) {
                        return o.drop_reads_borrows && o.moved != Moved::Yes;
                    }
                }
            }
        }
        false
    }

    /// The body being checked works with a raw pointer. A closure's pointer can point into
    /// the function around it, so every function open now is marked.
    pub(crate) fn note_raw_pointer(&mut self) {
        for f in &mut self.frames {
            if let FrameKind::Function { raw_pointers, .. } = f {
                *raw_pointers = true;
            }
        }
    }

    fn uses_raw_pointers(&self) -> bool {
        self.frames
            .iter()
            .rev()
            .find_map(|f| match f {
                FrameKind::Function { raw_pointers, .. } => Some(*raw_pointers),
                _ => None,
            })
            .unwrap_or(false)
    }
}

fn printing_requested() -> bool {
    std::env::var_os("VX_PRINT_DROPS").is_some()
}

fn ident(name: &str) -> Expr {
    Expr::Identifier(IdentifierExpr::new(name.into(), Span::default()))
}

fn drop_stmt(name: &str, flag: Option<&str>) -> Statement {
    Statement::Drop(DropStmt {
        name: name.into(),
        path: Vec::new(),
        call: None,
        limit: None,
        expr: None,
        flag: flag.map(Into::into),
        after_value: false,
        ends_lifetime: false,
        span: Span::default(),
        placement: None,
    })
}

/// The end of the lifetime of `name`'s stack slot. It needs no flag: after its last use, the
/// slot is not read again, whether or not the value was moved out of it.
fn lifetime_end(name: &str) -> Statement {
    Statement::Drop(DropStmt {
        name: name.into(),
        path: Vec::new(),
        call: None,
        limit: None,
        expr: None,
        flag: None,
        after_value: false,
        ends_lifetime: true,
        span: Span::default(),
        placement: None,
    })
}

/// `*call() = ..` becomes `*reference = ..`, and `(*call()).f = ..` becomes `reference.f = ..`.
fn reroot_place(place: &mut Expr, reference: &str) {
    match place {
        Expr::MemberAccess(m) if matches!(*m.base, Expr::Dereference(_)) => {
            *m.base = ident(reference)
        }
        Expr::MemberAccess(m) => reroot_place(&mut m.base, reference),
        Expr::Dereference(d) => {
            assert!(
                matches!(*d.expr, Expr::FunctionCall(_)),
                "not a write through a call"
            );
            *d.expr = ident(reference);
        }
        _ => unreachable!("not a write through a call"),
    }
}

/// A drop that runs once the statement it precedes has computed its value.
fn after_value(mut s: Statement) -> Statement {
    if let Statement::Drop(d) = &mut s {
        d.after_value = true;
    }
    s
}

fn assign_stmt(name: &str, value: Expr) -> Statement {
    Statement::Assign(AssignStmt::new(ident(name), value, Span::default()))
}

fn let_stmt(name: &str, is_mut: bool, ty: Type, value: Expr) -> Statement {
    Statement::LetDecl(LetDeclStmt {
        name: name.into(),
        is_mut,
        ty_ann: Some(ty),
        expr: value,
        span: Span::default(),
    })
}

/// Renames the variable `from` to `to` everywhere in `stmts`: each use, each `let`, each name
/// that a pattern, a loop or a closure binds, and each drop. A nested `let from` gets the new
/// name too, so it still hides the outer one.
fn rename_in_stmts(stmts: &mut [Statement], from: &str, to: &str) {
    crate::syntax::visit_exprs_mut(stmts, &mut |e| rename_in_node(e, from, to));
    rename_stmt_names(stmts, from, to);
}

fn rename_in_expr(e: &mut Expr, from: &str, to: &str) {
    crate::syntax::visit_expr_mut(e, &mut |e| rename_in_node(e, from, to));
}

/// The names that statements hold outside their expressions: a `let`, a `for` variable and a
/// drop. The expressions of a drop are not visited by `visit_exprs_mut`.
fn rename_stmt_names(stmts: &mut [Statement], from: &str, to: &str) {
    for s in stmts {
        match s {
            Statement::LetDecl(l) if l.name.as_ref() == from => l.name = to.into(),
            Statement::ForLoop(l) => {
                if l.iter == from {
                    l.iter = to.to_string();
                }
                rename_stmt_names(&mut l.body, from, to);
            }
            Statement::Loop(l) => rename_stmt_names(&mut l.body, from, to),
            Statement::Drop(d) => {
                if d.name.as_ref() == from {
                    d.name = to.into();
                }
                for e in d.expr.iter_mut().chain(d.limit.iter_mut()) {
                    rename_in_expr(e, from, to);
                }
            }
            _ => {}
        }
    }
}

/// One expression: a use of `from`, or the names that it binds and that its blocks hold.
fn rename_in_node(e: &mut Expr, from: &str, to: &str) {
    match e {
        Expr::Identifier(id) if id.name.as_ref() == from => id.name = to.into(),
        Expr::Closure(c) => {
            for (n, _) in c.params.iter_mut().chain(c.captures.iter_mut()) {
                if n.as_ref() == from {
                    *n = to.into();
                }
            }
        }
        Expr::Match(m) => {
            for arm in &mut m.arms {
                rename_pattern(&mut arm.pattern, from, to);
                rename_stmt_names(&mut arm.body, from, to);
            }
        }
        Expr::If(i) => {
            rename_stmt_names(&mut i.then_block, from, to);
            if let Some(b) = &mut i.else_block {
                rename_stmt_names(b, from, to);
            }
        }
        Expr::UnsafeBlock(b) => rename_stmt_names(&mut b.stmts, from, to),
        Expr::ComptimeBlock(b) => rename_stmt_names(&mut b.stmts, from, to),
        Expr::SpawnOn(b) => rename_stmt_names(&mut b.stmts, from, to),
        Expr::TaskBlock(b) => rename_stmt_names(&mut b.stmts, from, to),
        _ => {}
    }
}

fn rename_pattern(p: &mut Pattern, from: &str, to: &str) {
    match p {
        Pattern::Identifier(n) if n.as_ref() == from => *n = to.into(),
        Pattern::EnumVariant(_, _, Some(parts)) => {
            parts.iter_mut().for_each(|p| rename_pattern(p, from, to))
        }
        _ => {}
    }
}

/// Gives the owner `h` its new name in the code from where it is declared to the `let` that
/// hides it, and in that `let`'s value, which still reads it. A parameter or a name of an arm
/// is declared outside the block, so the `let` that moves it to its new name is returned.
fn rename_hidden_owner(body: &mut [Statement], edits: &mut Edits, h: &Hidden) -> Option<Statement> {
    let start = match h.decl {
        Some(d) => {
            let Statement::LetDecl(l) = &mut body[d] else {
                panic!("an owner with a statement is declared by a `let`");
            };
            assert_eq!(l.name.as_ref(), h.from, "the `let` declares the owner");
            l.name = h.to.as_str().into();
            d + 1
        }
        None => 0,
    };
    rename_in_stmts(&mut body[start..h.at], &h.from, &h.to);
    let Statement::LetDecl(l) = &mut body[h.at] else {
        panic!("only a `let` hides an owner");
    };
    rename_in_expr(&mut l.expr, &h.from, &h.to);
    let range = start..h.at;
    for (_, stmts) in edits
        .lets
        .range_mut(range.clone())
        .chain(edits.before.range_mut(range.clone()))
        .chain(edits.after.range_mut(range.clone()))
    {
        rename_in_stmts(stmts, &h.from, &h.to);
    }
    for (k, (stmts, _)) in edits.assign.iter_mut() {
        if range.contains(k) {
            rename_in_stmts(stmts, &h.from, &h.to);
        }
    }
    h.decl
        .is_none()
        .then(|| let_stmt(&h.to, true, h.ty.clone(), ident(&h.from)))
}

/// `drop owner`, naming its flag when it may have been moved, and then the end of its slot's
/// lifetime. In a function with raw pointers, an owner with something to drop keeps its old
/// drop point, which can come before the end of its block, so its slot gets no end.
fn owner_drop(owner: &Owner, raw_pointers: bool) -> Vec<Statement> {
    let flag = match owner.moved {
        Moved::Maybe => owner.flag.as_deref(),
        _ => None,
    };
    let mut out = owner_glue_drops(owner, flag);
    if owner.ends_lifetime && (owner.only_ends_lifetime() || !raw_pointers) {
        out.push(lifetime_end(&owner.name));
    }
    out
}

/// Whether `owner` is dropped at the end of its block. In a function with raw pointers, a
/// struct with nothing to drop waits for it, as in Rust.
fn waits_for_block_end(owner: &Owner, raw_pointers: bool) -> bool {
    owner.to_block_end || (raw_pointers && owner.only_ends_lifetime())
}

/// The drops of what `owner` still holds: a field moved out is left out, and one that may
/// have been is dropped under its own flag.
fn owner_glue_drops(owner: &Owner, flag: Option<&str>) -> Vec<Statement> {
    let mut out = Vec::new();
    for part in &owner.glue {
        let moved = owner.fields.iter().find(|f| {
            f.moved != Moved::No
                && part.path.len() >= f.path.len()
                && part
                    .path
                    .iter()
                    .zip(&f.path)
                    .all(|((field, _), moved)| field.as_ref() == moved)
        });
        match moved {
            Some(f) if f.moved == Moved::Yes => {}
            Some(f) => out.extend(glue_drops(
                &owner.name,
                std::slice::from_ref(part),
                f.flag.as_deref(),
            )),
            None => out.extend(glue_drops(&owner.name, std::slice::from_ref(part), flag)),
        }
    }
    out
}

/// `match place { E::V(p0, _) => { drops of p0 } .. _ => {} }`: the active variant's payload
/// dropped, element by element. `name` keeps the names it binds apart from others.
fn payload_match(name: &str, place: Expr, payload: &EnumGlue) -> Expr {
    let mut arms = Vec::new();
    for (variant, elems) in &payload.arms {
        let mut patterns = Vec::new();
        let mut body = Vec::new();
        for (k, glue) in elems.iter().enumerate() {
            match glue {
                Some(glue) => {
                    let bound = format!("{name}__payload{k}");
                    patterns.push(Pattern::Identifier(bound.as_str().into()));
                    body.extend(glue_drops(&bound, glue, None));
                }
                None => patterns.push(Pattern::Wildcard),
            }
        }
        arms.push(MatchArm {
            pattern: Pattern::EnumVariant(
                payload.name.as_str().into(),
                variant.clone(),
                Some(patterns),
            ),
            body,
            span: Span::default(),
        });
    }
    arms.push(MatchArm {
        pattern: Pattern::Wildcard,
        body: Vec::new(),
        span: Span::default(),
    });
    Expr::Match(MatchExpr::new(Box::new(place), arms, Span::default()))
}

/// The drops that run `glue` on `name`.
fn glue_drops(name: &str, glue: &[GluePart], flag: Option<&str>) -> Vec<Statement> {
    glue_drops_at(name, None, glue, flag)
}

/// The drops that run `glue` on the value at `base`, a place reached from the variable `name`
/// (`b.v`, `*out`), or on `name` itself when `base` is `None`.
fn glue_drops_at(
    name: &str,
    base: Option<&Expr>,
    glue: &[GluePart],
    flag: Option<&str>,
) -> Vec<Statement> {
    glue.iter()
        .map(|part| {
            let mut place = base.cloned().unwrap_or_else(|| ident(name));
            for (field, struct_ty) in &part.path {
                let mut access =
                    MemberAccessExpr::new(Box::new(place), field.clone(), Span::default());
                access.struct_name = Some(struct_ty.as_str().into());
                place = Expr::MemberAccess(access);
            }
            if let Some(payload) = &part.payload {
                return Statement::Drop(DropStmt {
                    name: name.into(),
                    path: part.path.iter().map(|(f, _)| f.clone()).collect(),
                    call: None,
                    limit: None,
                    expr: Some(Box::new(payload_match(name, place, payload))),
                    flag: flag.map(Into::into),
                    after_value: false,
                    ends_lifetime: false,
                    span: Span::default(),
                    placement: None,
                });
            }
            // `wait_limit_nanos(&place)`, which the code before the call of `drop` asks. An impl
            // has a copy of it only when the resolver saw `core::ops`, so without one the limit
            // is the default itself.
            let limit = part.call.as_ref().map(|_| {
                Box::new(match &part.limit {
                    Some(limit) => Expr::FunctionCall(FunctionCallExpr::new(
                        limit.clone(),
                        None,
                        vec![Expr::Borrow(BorrowExpr::new(
                            Box::new(place.clone()),
                            false,
                            Span::default(),
                        ))],
                        Span::default(),
                    )),
                    None => Expr::Number(NumberExpr::new(
                        (DEFAULT_WAIT_LIMIT_SECS * 1_000_000_000).to_string(),
                        Some(ElementType::I64),
                        Span::default(),
                    )),
                })
            });
            let expr = match &part.call {
                Some(call) => Some(Expr::FunctionCall(FunctionCallExpr::new(
                    call.clone(),
                    None,
                    vec![Expr::Borrow(BorrowExpr::new(
                        Box::new(place),
                        true,
                        Span::default(),
                    ))],
                    Span::default(),
                ))),
                None if part.path.is_empty() && base.is_none() => None,
                None => Some(place),
            };
            Statement::Drop(DropStmt {
                name: name.into(),
                path: part.path.iter().map(|(f, _)| f.clone()).collect(),
                call: part.call.clone(),
                limit,
                expr: expr.map(Box::new),
                flag: flag.map(Into::into),
                after_value: false,
                ends_lifetime: false,
                span: Span::default(),
                placement: part.placement.clone(),
            })
        })
        .collect()
}

impl<'a> TypeChecker<'a> {
    /// Whether drops are written into the function being checked.
    /// A body that already holds its drops is checked again without being rewritten; its drops
    /// were printed the first time.
    fn printing(&self) -> bool {
        printing_requested() && self.drops_rewriting()
    }

    pub(crate) fn drops_rewriting(&self) -> bool {
        !self.speculating
            && self.borrow.drops.frames.iter().rev().find_map(|f| match f {
                FrameKind::Function { rewrite, .. } => Some(*rewrite),
                _ => None,
            }) == Some(true)
    }

    /// `for n in it { .. }` over an iterator whose items need dropping: the loop binds a hidden
    /// name, and the body starts with `let n = <hidden>;`. `n` is then a variable of the body,
    /// dropped at the end of each pass and on `break`, `continue` and `return`, as in Rust.
    pub(crate) fn drops_bind_loop_item(
        &mut self,
        iter: &mut String,
        body: &mut Vec<Statement>,
        item_ty: &Type,
    ) {
        if !self.drops_rewriting() || self.drops_glue(item_ty).is_empty() {
            return;
        }
        let item = self.drops_fresh("item");
        let user = std::mem::replace(iter, item.clone());
        let is_mut = crate::hir::check::raw::body_reassigns(body, &user);
        body.insert(0, let_stmt(&user, is_mut, item_ty.clone(), ident(&item)));
    }

    pub(crate) fn drops_fresh(&mut self, what: &str) -> String {
        self.borrow.drops.counter += 1;
        format!("__vx_{what}_{}", self.borrow.drops.counter)
    }

    /// What dropping a value of type `ty` does, in order: its `Drop` impl's `drop`, then each
    /// field's, in declaration order. Empty for a type with nothing to drop.
    pub(crate) fn drops_glue(&mut self, ty: &Type) -> Vec<GluePart> {
        self.drops_glue_at(ty, 0)
    }

    fn drops_glue_at(&mut self, ty: &Type, depth: usize) -> Vec<GluePart> {
        assert!(depth < 64, "a struct that holds itself: {ty:?}");
        match ty {
            Type::Tensor(..) => vec![GluePart {
                path: Vec::new(),
                call: None,
                limit: None,
                payload: None,
                placement: ty.placement().cloned(),
            }],
            // The same value as the type inside, with a proof attached.
            Type::Verified(inner) => self.drops_glue_at(inner, depth),
            // A tensor pinned to a device is in that device's memory, so its plugin frees it.
            Type::Pinned(inner, topology) => {
                let mut glue = self.drops_glue_at(inner, depth);
                for part in glue.iter_mut().filter(|p| p.call.is_none()) {
                    if part.placement.is_none() {
                        part.placement = Some(crate::syntax::Placement::on(topology.clone()));
                    }
                }
                glue
            }
            // A type is often written as a struct before anything knows it names an enum, so
            // the declarations decide.
            _ if self.drops_enum_decl(ty).is_some() => self.drops_enum_glue(ty, depth),
            Type::Struct(..) | Type::GenericInstance(..) => {
                if self.is_copy(ty) {
                    return Vec::new();
                }
                // A closure's environment holds what it captured, views among them, and the
                // function that made it still owns those: nothing to drop.
                if ty.to_string().starts_with("Closure_") {
                    return Vec::new();
                }
                // `core::mem::ManuallyDrop` is the one struct whose value is never dropped:
                // `forget` is built on it, as in Rust.
                let name = ty.to_string();
                if name == "ManuallyDrop" || name.starts_with("ManuallyDrop<") {
                    return Vec::new();
                }
                let Some(fields) = self.drops_struct_fields(ty) else {
                    return Vec::new();
                };
                let mut glue = Vec::new();
                if let Some(call) = self.drops_impl_call(ty) {
                    glue.push(GluePart {
                        path: Vec::new(),
                        call: Some(call),
                        limit: self.drops_impl_method(ty, "wait_limit_nanos"),
                        payload: None,
                        placement: None,
                    });
                }
                let struct_ty = ty.to_string();
                for (field, field_ty) in fields {
                    for mut part in self.drops_glue_at(&field_ty, depth + 1) {
                        part.path.insert(0, (field.clone(), struct_ty.clone()));
                        glue.push(part);
                    }
                }
                glue
            }
            _ => Vec::new(),
        }
    }

    /// The declaration of the enum `ty` names, and its type arguments.
    fn drops_enum_decl(&self, ty: &Type) -> Option<(decl::EnumDecl, Vec<Type>)> {
        let (name, args): (&crate::symbol::Symbol, Vec<Type>) = match ty {
            Type::Enum(name, _) | Type::Struct(name, _) => (name, Vec::new()),
            Type::GenericInstance(base, args) => match base.as_ref() {
                Type::Enum(name, _) | Type::Struct(name, _) => (name, args.clone()),
                _ => return None,
            },
            _ => return None,
        };
        let base = name.split('<').next().unwrap_or(name);
        let decl = (*self.env.enums.get(base)?).clone();
        Some((decl, args))
    }

    /// An enum's glue: its `Drop` impl's `drop`, then the active variant's payload.
    fn drops_enum_glue(&mut self, ty: &Type, depth: usize) -> Vec<GluePart> {
        if self.is_copy(ty) {
            return Vec::new();
        }
        let Some((decl, args)) = self.drops_enum_decl(ty) else {
            return Vec::new();
        };
        let mapping: HashMap<crate::symbol::Symbol, Type> = decl
            .generics
            .iter()
            .map(|g| crate::symbol::Symbol::from(g.name()))
            .zip(args.iter().cloned())
            .collect();
        let mut arms = Vec::new();
        for (variant, payload) in &decl.variants {
            let elems: Vec<Option<Vec<GluePart>>> = payload
                .iter()
                .flatten()
                .map(|t| {
                    let glue = self.drops_glue_at(&t.substitute(&mapping), depth + 1);
                    (!glue.is_empty()).then_some(glue)
                })
                .collect();
            if elems.iter().any(Option::is_some) {
                arms.push((variant.clone(), elems));
            }
        }
        let mut glue = Vec::new();
        if let Some(call) = self.drops_impl_call(ty) {
            glue.push(GluePart {
                path: Vec::new(),
                call: Some(call),
                limit: self.drops_impl_method(ty, "wait_limit_nanos"),
                payload: None,
                placement: None,
            });
        }
        if !arms.is_empty() {
            glue.push(GluePart {
                path: Vec::new(),
                call: None,
                limit: None,
                payload: Some(EnumGlue {
                    name: ty.to_string(),
                    arms,
                }),
                placement: None,
            });
        }
        glue
    }

    /// The fields of struct type `ty`, with its type arguments put in. `None` when `ty` is not
    /// a struct.
    fn drops_struct_fields(&self, ty: &Type) -> Option<Vec<(crate::symbol::Symbol, Type)>> {
        let (name, args): (&crate::symbol::Symbol, &[Type]) = match ty {
            Type::Struct(name, _) => (name, &[]),
            Type::GenericInstance(base, args) => match base.as_ref() {
                Type::Struct(name, _) => (name, args.as_slice()),
                _ => return None,
            },
            _ => return None,
        };
        let base = name.split('<').next().unwrap_or(name);
        let decl = match self.env.structs.get(base) {
            Some(s) => (*s).clone(),
            None => self
                .mono
                .generated_structs
                .iter()
                .find(|s| s.name.as_ref() == base)?
                .clone(),
        };
        let mapping: HashMap<crate::symbol::Symbol, Type> = decl
            .generics
            .iter()
            .map(|g| crate::symbol::Symbol::from(g.name()))
            .zip(args.iter().cloned())
            .collect();
        Some(
            decl.fields
                .iter()
                .map(|(n, t)| (n.clone(), t.substitute(&mapping)))
                .collect(),
        )
    }

    /// The mangled name of `ty`'s `Drop::drop`, instantiated for `ty` when the impl is
    /// generic, or `None` when `ty` does not implement `Drop`.
    pub(crate) fn drops_impl_call(&mut self, ty: &Type) -> Option<crate::symbol::Symbol> {
        self.drops_impl_method(ty, "drop")
    }

    /// The mangled name of the method `method` of `ty`'s `Drop` impl, as for `drop`. A default
    /// method, such as `wait_limit`, is in the impl too.
    pub(crate) fn drops_impl_method(
        &mut self,
        ty: &Type,
        method: &str,
    ) -> Option<crate::symbol::Symbol> {
        let impls: Vec<&decl::ImplBlock> = self.env.impls.get("Drop")?.clone();
        let base_name = |t: &Type| -> Option<String> {
            match t {
                Type::Struct(n, _) | Type::Enum(n, _) => {
                    Some(n.split('<').next().unwrap_or(n).to_string())
                }
                Type::GenericInstance(b, _) => match b.as_ref() {
                    Type::Struct(n, _) | Type::Enum(n, _) => {
                        Some(n.split('<').next().unwrap_or(n).to_string())
                    }
                    _ => None,
                },
                _ => None,
            }
        };
        let want = base_name(ty)?;
        let ib = impls
            .into_iter()
            .find(|ib| base_name(&ib.target_type).as_deref() == Some(want.as_str()))?;
        let method = ib
            .methods
            .iter()
            .find(|m| m.name.as_ref() == method)?
            .clone();
        let mut mapping = HashMap::new();
        self.unify_types(&ib.target_type, ty, &mut mapping);
        let (_, name) = self.instantiate_impl_method(&method, &mapping, ty, ib)?;
        Some(name.into())
    }

    /// A function or closure body begins. `params` are its parameters taken by value.
    pub(crate) fn drops_enter_function(&mut self, params: Vec<(String, Type)>, body: &[Statement]) {
        let scope = self.scopes.len() - 1;
        let params: Vec<(String, Vec<GluePart>, bool, Type)> = params
            .into_iter()
            .map(|(name, ty)| {
                let glue = self.drops_glue(&ty);
                (name, glue, self.drop_reads_a_reference(&ty), ty)
            })
            .filter(|(_, glue, _, _)| !glue.is_empty())
            .collect();
        let already = body.iter().any(|s| {
            let mut uses = HashSet::new();
            Self::extract_uses_stmt(s, &mut uses);
            uses.contains(DROP_MARK)
        });
        let rewrite = !already;
        let d = &mut self.borrow.drops;
        d.frames.push(FrameKind::Function {
            rewrite,
            raw_pointers: false,
        });
        d.pending = params
            .into_iter()
            .map(|(p, glue, reads, ty)| (p, scope, glue, reads, ty))
            .collect();
        d.pending_by = "the function";
    }

    /// A `match` arm takes the value it matched: the names it binds own their parts of it, and
    /// the arm's block drops them, as a function body drops its parameters.
    pub(crate) fn drops_bind_arm(&mut self, names: &[crate::symbol::Symbol]) {
        let scope = self.scopes.len() - 1;
        let mut pending = Vec::new();
        for name in names {
            let Some(ty) = self.lookup(name).map(|(t, _)| t.clone()) else {
                continue;
            };
            let glue = self.drops_glue(&ty);
            if !glue.is_empty() {
                let reads = self.drop_reads_a_reference(&ty);
                pending.push((name.to_string(), scope, glue, reads, ty));
            }
        }
        let d = &mut self.borrow.drops;
        d.pending = pending;
        d.pending_by = "the arm";
    }

    /// A `match` arm takes the value that `take` moves, and another arm may not.
    pub(crate) fn drops_arm_takes(&mut self, take: impl FnOnce(&mut Self)) {
        self.borrow.drops.arm_takes = true;
        take(self);
        self.borrow.drops.arm_takes = false;
    }

    pub(crate) fn drops_exit_function(&mut self) {
        let d = &mut self.borrow.drops;
        while let Some(f) = d.frames.pop() {
            if matches!(f, FrameKind::Function { .. }) {
                break;
            }
        }
        d.pending.clear();
    }

    pub(crate) fn drops_enter_loop(&mut self) {
        self.borrow.drops.frames.push(FrameKind::Loop);
    }

    pub(crate) fn drops_enter_spawn(&mut self) {
        self.borrow.drops.frames.push(FrameKind::Spawn);
    }

    pub(crate) fn drops_exit_spawn(&mut self) {
        let popped = self.borrow.drops.frames.pop();
        assert!(
            matches!(popped, Some(FrameKind::Spawn)),
            "a spawn frame closes a spawn region"
        );
    }

    pub(crate) fn drops_enter_task(&mut self) {
        self.borrow.drops.frames.push(FrameKind::Task);
    }

    pub(crate) fn drops_exit_task(&mut self) {
        let popped = self.borrow.drops.frames.pop();
        assert!(
            matches!(popped, Some(FrameKind::Task)),
            "a task frame closes a task body"
        );
    }

    pub(crate) fn drops_enter_tasks(&mut self) {
        self.borrow.drops.frames.push(FrameKind::Tasks);
        self.borrow.drops.tasks_joins.push(None);
    }

    /// Returns the statement of the block's body that joins its tasks, if it has one.
    pub(crate) fn drops_exit_tasks(&mut self) -> Option<usize> {
        let popped = self.borrow.drops.frames.pop();
        assert!(
            matches!(popped, Some(FrameKind::Tasks)),
            "a tasks frame closes a tasks block"
        );
        self.borrow
            .drops
            .tasks_joins
            .pop()
            .expect("each tasks block has an entry")
    }

    pub(crate) fn drops_exit_loop(&mut self) {
        let popped = self.borrow.drops.frames.pop();
        assert!(
            matches!(popped, Some(FrameKind::Loop)),
            "a loop frame closes a loop"
        );
    }

    pub(crate) fn drops_enter_block(&mut self, body: &[Statement], value: Option<&Expr>) {
        let d = &mut self.borrow.drops;
        let given_by = d.pending_by;
        let mut edits: Box<Edits> = Box::default();
        let sets = std::mem::take(&mut d.pending_sets);
        if !sets.is_empty() {
            edits.before.insert(0, sets);
        }
        let owners = std::mem::take(&mut d.pending)
            .into_iter()
            .map(|(name, scope, glue, drop_reads_borrows, ty)| Owner {
                name,
                given_by: Some(given_by),
                decl: None,
                to_block_end: glue.iter().any(GluePart::calls),
                drop_reads_borrows,
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
                glue,
                fields: Vec::new(),
                ends_lifetime: false,
                ty,
            })
            .collect();
        d.frames.push(FrameKind::Block {
            owners,
            uses: BlockUseList::new(body, value),
            lines: body
                .iter()
                .map(|s| s.span().line)
                .chain(value.map(|v| v.span().line))
                .collect(),
            stmt: 0,
            edits,
            value_at: value.map(|_| body.len()),
            hidden: Vec::new(),
        });
    }

    /// The drop state, to put back after checking code that never runs.
    pub(crate) fn drops_save(&self) -> DropFrames {
        self.borrow.drops.clone()
    }

    /// Put back a state from [`Self::drops_save`]. The counter stays, so the names of the
    /// locals added since are not given out again.
    pub(crate) fn drops_restore(&mut self, saved: DropFrames) {
        let counter = self.borrow.drops.counter;
        self.borrow.drops = saved;
        self.borrow.drops.counter = counter;
    }

    pub(crate) fn drops_set_stmt(&mut self, i: usize) {
        if let Some(FrameKind::Block { stmt, .. }) = self.borrow.drops.frames.last_mut() {
            *stmt = i;
        }
        self.borrow.drops.moved_now.clear();
    }

    /// `tensor_view_2d(..)`, also as the value of an `unsafe` block: a view of memory the
    /// program got from elsewhere, which no drop frees.
    pub(crate) fn views_foreign_memory(expr: &Expr) -> bool {
        match expr {
            Expr::FunctionCall(c) => c.name.as_ref() == "tensor_view_2d",
            Expr::UnsafeBlock(u) => u.ret.as_deref().is_some_and(Self::views_foreign_memory),
            _ => false,
        }
    }

    /// `let name = ..` bound a value that owns something to drop: a tensor that is not a
    /// view, or a struct that implements `Drop` or holds one. A value whose drop runs a `Drop`
    /// impl waits for the end of its block, as in Rust, since what the impl does may be part
    /// of what the program means; one that only frees memory is freed after its last use.
    /// A struct or enum with nothing to drop is an owner too, so that its stack slot's
    /// lifetime ends after its last use.
    pub(crate) fn drops_note_owner(&mut self, name: &str, ty: &Type) {
        // `let (a, b) = e` holds `e` in a `$tuple` local and moves each part out of it, so the
        // parts are dropped as `a` and `b`, and the local owns nothing.
        // The same for the `$value` an `unsafe` block's value is moved into.
        if self.borrow.views.contains_key(name)
            || name.starts_with("$tuple")
            || name.starts_with("$value")
        {
            return;
        }
        let glue = self.drops_glue(ty);
        let ends_lifetime = Self::lives_in_a_stack_slot(ty);
        if glue.is_empty() && !ends_lifetime {
            return;
        }
        let scope = self.scopes.len() - 1;
        let drop_reads_borrows = self.drop_reads_a_reference(ty);
        self.drops_hide_owner(name);
        if let Some(FrameKind::Block { owners, stmt, .. }) = self.borrow.drops.frames.last_mut() {
            owners.push(Owner {
                name: name.to_string(),
                given_by: None,
                decl: Some(*stmt),
                to_block_end: glue.iter().any(GluePart::calls),
                drop_reads_borrows,
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
                glue,
                fields: Vec::new(),
                ends_lifetime,
                ty: ty.clone(),
            });
        }
    }

    /// A `let name` hides an owner of the same block. A drop of the owner after the `let` would
    /// name the new value: its drop at the end of the block, or its drop after the last use of
    /// something that borrows it. So the owner gets a name of its own, and the end of the block
    /// renames it in the code up to the `let`. An owner that something borrows waits for the end
    /// of the block, since a borrow of it can be used after the `let`.
    fn drops_hide_owner(&mut self, name: &str) {
        let borrowed = !self.borrow.borrowers_of(name).is_empty();
        let hides =
            |o: &Owner| o.name == name && o.moved != Moved::Yes && (o.to_block_end || borrowed);
        let found = matches!(self.borrow.drops.frames.last(), Some(FrameKind::Block { owners, .. })
            if owners.iter().any(hides));
        if !found {
            return;
        }
        let to = self.drops_fresh("hidden");
        let Some(FrameKind::Block {
            owners,
            stmt,
            hidden,
            ..
        }) = self.borrow.drops.frames.last_mut()
        else {
            unreachable!("checked above");
        };
        let owner = owners
            .iter_mut()
            .rev()
            .find(|o| hides(o))
            .expect("checked above");
        hidden.push(Hidden {
            from: name.to_string(),
            to: to.clone(),
            decl: owner.decl,
            at: *stmt,
            ty: owner.ty.clone(),
        });
        owner.name = to;
        owner.to_block_end = true;
    }

    /// Whether a local of type `ty` lives in a stack slot of its own: a struct or an enum. A
    /// number lives in a register, and a tensor's buffer has lifetime markers of its own.
    fn lives_in_a_stack_slot(ty: &Type) -> bool {
        match ty {
            Type::Verified(inner) | Type::Pinned(inner, _) => Self::lives_in_a_stack_slot(inner),
            // A closure's environment holds borrows of what it captured. A `void` holds nothing.
            Type::Struct(name, _) => {
                !name.starts_with("Closure_") && !crate::syntax::is_void_ty(ty)
            }
            Type::Enum(..) => true,
            Type::GenericInstance(base, _) => {
                matches!(base.as_ref(), Type::Struct(..) | Type::Enum(..))
            }
            _ => false,
        }
    }

    pub(crate) fn is_raw_pointer(ty: &Type) -> bool {
        match ty {
            Type::Verified(inner) | Type::Pinned(inner, _) => Self::is_raw_pointer(inner),
            Type::Pointer(..) => true,
            _ => false,
        }
    }

    /// `name` was moved, or (`again`) given a new value after a move.
    pub(crate) fn drops_note_move(&mut self, name: &str, again: bool) {
        self.drops_note_move_at(name, None, again);
    }

    /// The field `name.path` was moved out, or (`again`) given a new value after that.
    pub(crate) fn drops_note_field_move(&mut self, name: &str, path: &[String], again: bool) {
        self.drops_note_move_at(name, Some(path), again);
    }

    fn drops_note_move_at(&mut self, name: &str, field: Option<&[String]>, again: bool) {
        // A speculative check only asks what a type is; it moves nothing.
        if self.speculating {
            return;
        }
        // A move at the top of a task body happens at its `spawn` (T4). The task runs on a
        // thread of its own, so a flag that the task set would come too late for a drop after
        // the `spawn`.
        let mut here = self.scopes.len() - 1;
        while let Some(outer) = self.tasks.spawn_scope_of_body(here) {
            here = outer;
        }
        let rewriting = self.drops_rewriting();
        let arm = self.borrow.drops.arm_takes;
        if !again && field.is_none() {
            self.borrow.drops.moved_now.insert(name.to_string());
        }
        let top = self.borrow.drops.frames.len();
        let mut found = None;
        for (k, f) in self.borrow.drops.frames.iter_mut().enumerate().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return,
                FrameKind::Loop | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, stmt, .. } => {
                    if let Some(o) = owners.iter_mut().rev().find(|o| o.name == name) {
                        let (scope, decl) = (o.scope, o.decl);
                        let needs_no_flag = o.only_ends_lifetime();
                        // `o.p = v` after `let y = o.p.a`: the new `o.p` holds an `a` of its own.
                        if let (true, Some(path)) = (again && scope == here, field) {
                            o.fields.retain(|f| {
                                f.path.len() <= path.len() || !f.path.starts_with(path)
                            });
                        }
                        let (moved, moved_at, flag) = o.state(field);
                        // A value given in a nested block only reaches the paths through it:
                        // a move before it stays possible.
                        *moved = if again {
                            if scope == here || *moved == Moved::No {
                                Moved::No
                            } else {
                                Moved::Maybe
                            }
                        } else if scope == here && !arm {
                            *moved_at = Some(*stmt);
                            Moved::Yes
                        } else {
                            Moved::Maybe
                        };
                        found = Some((k, *moved, flag.clone(), decl, *moved_at, needs_no_flag));
                        break;
                    }
                }
            }
        }
        let Some((k, Moved::Maybe, flag, decl, moved_at, needs_no_flag)) = found else {
            return;
        };
        // The end of a lifetime frees nothing, so it needs no flag (see `lifetime_end`).
        if !rewriting || (again && flag.is_some()) || needs_no_flag {
            return;
        }
        // The first move inside a nested block gives the owner a flag, declared beside it.
        let flag = match flag {
            Some(f) => f,
            None => {
                let f = self.drops_fresh("moved");
                let declare = let_stmt(&f, true, Type::Scalar(ElementType::Bool), ident("false"));
                if let FrameKind::Block { owners, edits, .. } = &mut self.borrow.drops.frames[k] {
                    if let Some(o) = owners.iter_mut().rev().find(|o| o.name == name) {
                        *o.state(field).2 = Some(f.clone());
                    }
                    match decl {
                        Some(i) => edits.after.entry(i).or_default().push(declare),
                        None => edits.before.entry(0).or_default().insert(0, declare),
                    }
                }
                f
            }
        };
        // Given a value again in a nested block after a certain move: the flag is set at that
        // move, and the assignment clears it.
        if again {
            if let (Some(i), FrameKind::Block { edits, .. }) =
                (moved_at, &mut self.borrow.drops.frames[k])
            {
                edits
                    .after
                    .entry(i)
                    .or_default()
                    .push(assign_stmt(&flag, ident("true")));
            }
            return;
        }
        // A move in a task happens at its `spawn`, so the flag is set after the `spawn`, in the
        // block around it. In the task the flag would be set too late, or not at all when the
        // move is in the task's last value, which is checked after the task's block closes.
        let first_task =
            (k + 1..top).find(|&i| matches!(self.borrow.drops.frames[i], FrameKind::Task));
        let at = match first_task {
            Some(t) => (k..t)
                .rev()
                .find(|&i| matches!(self.borrow.drops.frames[i], FrameKind::Block { .. }))
                .expect("the owner's block is below the task"),
            None => top - 1,
        };
        // The arm takes the value before its first statement, in a block not opened yet.
        if arm && first_task.is_none() {
            let set = assign_stmt(&flag, ident("true"));
            self.borrow.drops.pending_sets.push(set);
            return;
        }
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.get_mut(at) {
            edits
                .after
                .entry(*stmt)
                .or_default()
                .push(assign_stmt(&flag, ident("true")));
        }
    }

    /// E4015 when `value` names an owner from outside a `spawn` region, or a field of one, and
    /// the place it goes to owns it: a variable, a struct field or a parameter that takes it by
    /// value. That owner would drop it in the region, and the function after the region.
    pub(crate) fn check_spawn_capture_moved(&mut self, value: &Expr) {
        if self.speculating {
            return;
        }
        let Some((name, _)) = Self::extract_base_and_path(value) else {
            return;
        };
        if !matches!(value, Expr::Identifier(_) | Expr::MemberAccess(_))
            || !self.drops_owner_outside_spawn(&name)
        {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4015,
            format!(
                "this `spawn` region moves '{name}', which belongs to the code around the \
                 region. The function drops '{name}' after the region, so it would be dropped \
                 twice. Borrow it (`&{name}`), or clone it (`{name}.clone()`)"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(&value.span())),
        );
    }

    /// Whether `name` is an owner from outside a `spawn` region that the code is in.
    fn drops_owner_outside_spawn(&self, name: &str) -> bool {
        let mut crossed = false;
        for f in self.borrow.drops.frames.iter().rev() {
            match f {
                FrameKind::Function { .. } => return false,
                FrameKind::Spawn => crossed = true,
                FrameKind::Loop | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, .. } => {
                    // A struct with nothing to drop is not dropped twice.
                    if owners
                        .iter()
                        .any(|o| o.name == name && !o.only_ends_lifetime())
                    {
                        return crossed;
                    }
                }
            }
        }
        false
    }

    /// `obj.as_ptr()`: the owner behind `obj` now waits for the end of its block.
    pub(crate) fn drops_note_raw_pointer(&mut self, obj: &Expr) {
        let Some((root, _)) = Self::extract_base_and_path(obj) else {
            return;
        };
        let mut names = vec![root.clone()];
        if let Some(view) = self.borrow.views.get(root.as_str()) {
            names.extend(
                view.owners
                    .iter()
                    .chain(&view.behind)
                    .map(|(o, _)| o.clone()),
            );
        }
        names.extend(self.borrow.borrowed_by(&root).into_iter().map(|(o, _)| o));
        for f in self.borrow.drops.frames.iter_mut().rev() {
            match f {
                FrameKind::Function { .. } => return,
                // A raw pointer taken inside a region still points into the owner outside it.
                FrameKind::Loop | FrameKind::Spawn | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, .. } => {
                    for o in owners.iter_mut().filter(|o| names.contains(&o.name)) {
                        o.to_block_end = true;
                    }
                }
            }
        }
    }

    /// Whether `name` is an owner, and whether its value was moved. Read before an assignment
    /// to it clears the mark.
    /// Whether dropping a `ty` can run a `drop` that reads a reference the value holds. A
    /// reference behind a type's own raw pointer, as in `Vec<&i32>`, counts only when the
    /// element has a `drop` of its own that can read it, as Rust's `#[may_dangle]` allows.
    pub(crate) fn drop_reads_a_reference(&mut self, ty: &Type) -> bool {
        self.drop_reads_a_reference_in(ty, &mut HashSet::new())
    }

    fn drop_reads_a_reference_in(&mut self, ty: &Type, seen: &mut HashSet<String>) -> bool {
        let parts = match ty {
            Type::Pointer(inner, ..) | Type::Verified(inner) | Type::Pinned(inner, _) => {
                return self.drop_reads_a_reference_in(inner, seen);
            }
            Type::Struct(name, _) | Type::Enum(name, _) => self.declared_part_types(name, &[]),
            Type::GenericInstance(base, args) => match &**base {
                Type::Struct(name, _) | Type::Enum(name, _) => self.declared_part_types(name, args),
                _ => return false,
            },
            _ => return false,
        };
        if !seen.insert(ty.to_string()) {
            return false;
        }
        if self.drops_impl_call(ty).is_some()
            && parts
                .iter()
                .any(|p| !matches!(p, Type::Pointer(..)) && self.type_can_hold_reference(p))
        {
            return true;
        }
        parts
            .iter()
            .any(|p| self.drop_reads_a_reference_in(p, seen))
    }

    pub(crate) fn drops_owner_state(&self, name: &str) -> Option<Moved> {
        self.drops_owner(name).map(|o| o.moved)
    }

    /// `name` holds a whole new value: none of its fields is moved out any more.
    fn drops_clear_fields(&mut self, name: &str) {
        for f in self.borrow.drops.frames.iter_mut().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return,
                FrameKind::Loop | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, .. } => {
                    if let Some(o) = owners.iter_mut().rev().find(|o| o.name == name) {
                        o.fields.clear();
                        return;
                    }
                }
            }
        }
    }

    /// The fields of `name` moved out of it, or that may have been, with the flag of each.
    /// Read before `p.a = v` gives one of them a value again, which clears its mark.
    pub(crate) fn drops_moved_fields(&self, name: &str) -> Vec<MovedField> {
        self.drops_owner(name)
            .map(|o| {
                o.fields
                    .iter()
                    .filter(|f| f.moved != Moved::No)
                    .map(|f| (f.path.clone(), f.moved, f.flag.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn drops_owner(&self, name: &str) -> Option<&Owner> {
        for f in self.borrow.drops.frames.iter().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return None,
                FrameKind::Loop | FrameKind::Tasks | FrameKind::Task => {}
                FrameKind::Block { owners, .. } => {
                    if let Some(o) = owners.iter().rev().find(|o| o.name == name) {
                        return Some(o);
                    }
                }
            }
        }
        None
    }

    /// `name = rhs` on `line`, where `before` is what `drops_owner_state` said first: the old
    /// value is dropped once `rhs` is computed, before it is stored.
    pub(crate) fn drops_note_assign(
        &mut self,
        name: &str,
        before: Option<Moved>,
        rhs: &Expr,
        line: usize,
    ) {
        let Some(moved) = before else {
            return;
        };
        // The old value has nothing to drop, and the slot keeps its lifetime.
        if self
            .drops_owner(name)
            .is_some_and(Owner::only_ends_lifetime)
        {
            return;
        }
        // `t = t + b` moved the old value into the operator, which queued its drop after the
        // statement, where it would free the new value. The drop moves before the store.
        let operand_drop = !self.speculating
            && self.drops_rewriting()
            && self.borrow.drops.moved_now.contains(name)
            && match self.borrow.drops.frames.last_mut() {
                Some(FrameKind::Block { stmt, edits, .. }) => {
                    let queued = edits.after.entry(*stmt).or_default();
                    let n = queued.len();
                    queued.retain(|s| !matches!(s, Statement::Drop(d) if d.name.as_ref() == name));
                    queued.len() != n
                }
                _ => false,
            };
        // `t = f(t)` moved the old value into the call; a flag still has to be cleared.
        if !operand_drop && (moved == Moved::Yes || self.borrow.drops.moved_now.contains(name)) {
            if self.speculating || !self.drops_rewriting() {
                return;
            }
            let flag = self.drops_owner(name).and_then(|o| o.flag.clone());
            if let (Some(f), Some(FrameKind::Block { stmt, edits, .. })) =
                (flag, self.borrow.drops.frames.last_mut())
            {
                edits.assign.insert(*stmt, (Vec::new(), Some(f)));
            }
            return;
        }
        // `c = a @ b` with neither operand `c`: the product is written into `c`'s buffer, the
        // same test the code generators make.
        if let Expr::BinaryOp(b) = rhs {
            if matches!(b.op, BinaryOp::MatMul) {
                let plain_root = |e: &Expr| -> Option<String> {
                    let r = crate::syntax::matmul_operand_root(e)?;
                    let through = matches!(
                        self.lookup(r).map(|(t, _)| t),
                        Some(Type::Borrow { .. } | Type::Pointer(..))
                    );
                    (!through).then(|| r.to_string())
                };
                if let (Some(l), Some(r)) = (plain_root(&b.lhs), plain_root(&b.rhs)) {
                    if l != name && r != name {
                        return;
                    }
                }
            }
        }
        if self.speculating {
            return;
        }
        // The operator's drop was printed where the operator moved it.
        if self.printing() && !operand_drop {
            let flag = if moved == Moved::Maybe {
                " if it was not moved"
            } else {
                ""
            };
            eprintln!(
                "drop in {}: the old value of {name}{flag}, before the assignment on line {line}",
                self.current_function
            );
        }
        if !self.drops_rewriting() {
            return;
        }
        let Some(owner) = self.drops_owner(name) else {
            return;
        };
        let flag = owner.flag.clone();
        let old_flag = flag
            .as_deref()
            .filter(|_| moved == Moved::Maybe && !operand_drop);
        let drop_old: Vec<Statement> = owner_glue_drops(owner, old_flag)
            .into_iter()
            .map(after_value)
            .collect();
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.last_mut() {
            edits.assign.insert(*stmt, (drop_old, flag));
        }
        self.drops_clear_fields(name);
    }

    /// `place = rhs` where `place` is a field (`b.v`, `r.v`) or a dereference (`*out`): the
    /// value it held is dropped once `rhs` is computed, before it is stored, as for a variable.
    /// A tensor is left alone, since a store into one may write into its buffer, and so is a
    /// place behind a raw pointer, whose memory may hold no value yet.
    pub(crate) fn drops_note_place_assign(
        &mut self,
        place: &Expr,
        rhs: &Expr,
        ty: &Type,
        line: usize,
        moved_before: &[MovedField],
    ) {
        if self.speculating || matches!(ty, Type::Tensor(..)) {
            return;
        }
        let mut root = place;
        let mut fields = Vec::new();
        while let Expr::MemberAccess(m) = root {
            fields.push(m.member.to_string());
            root = &m.base;
        }
        fields.reverse();
        // `*f() = v`, and `p[i] = v` through `IndexMut`, which is `*p.index_mut(i) = v`. The old
        // value is dropped through the reference the call returns, so the call runs once.
        let call = match root {
            Expr::Dereference(d) if matches!(*d.expr, Expr::FunctionCall(_)) => Some(&*d.expr),
            _ => None,
        };
        let (name, through_reference) = match root {
            Expr::Identifier(id) => (id.name.to_string(), false),
            Expr::Dereference(d) => match &*d.expr {
                Expr::Identifier(id) => (id.name.to_string(), true),
                Expr::FunctionCall(f) => (format!("{}(..)", f.name), true),
                _ => return,
            },
            _ => return,
        };
        let root_ty = match call {
            Some(call) => Some(self.check_expr_type_probe(call)),
            None => self.lookup(&name).map(|(t, _)| t.clone()),
        };
        let is_mut_borrow = matches!(root_ty, Some(Type::Borrow { is_mut: true, .. }));
        if call.is_some() && !is_mut_borrow {
            return;
        }
        // The old value lives in a variable this function owns, or behind a `&mut`, which
        // always points at a value.
        let flag = if is_mut_borrow {
            None
        } else if through_reference {
            return;
        } else {
            match self.drops_owner(&name).map(|o| (o.moved, o.flag.clone())) {
                Some((Moved::No, _)) => None,
                Some((Moved::Maybe, Some(f))) => Some(f),
                _ => return,
            }
        };
        // A part of the old value moved out (`let x = p.a; p.a = v;`) is not dropped again, and
        // one that may have been is dropped under its own flag.
        let mut parts = Vec::new();
        for part in self.drops_glue(ty) {
            let at: Vec<&str> = fields
                .iter()
                .map(String::as_str)
                .chain(part.path.iter().map(|(field, _)| field.as_ref()))
                .collect();
            let moved = moved_before.iter().find(|(path, _, _)| {
                at.len() >= path.len() && at.iter().zip(path).all(|(a, m)| *a == m)
            });
            match moved {
                Some((_, Moved::Yes, _)) => {}
                Some((_, _, f)) => parts.push((part, f.clone())),
                None => parts.push((part, flag.clone())),
            }
        }
        if parts.is_empty() {
            return;
        }
        if self.printing() {
            eprintln!(
                "drop in {}: the old value of {}{}, before the assignment on line {line}",
                self.current_function,
                if through_reference { "*" } else { "" },
                crate::hir::places::display_place(&name, &fields)
            );
        }
        if !self.drops_rewriting() {
            return;
        }
        let mut place = place.clone();
        let mut lets = Vec::new();
        let mut reroot = None;
        if let (Some(call), Some(ref_ty)) = (call, root_ty) {
            let (reference, value) = (self.drops_fresh("element"), self.drops_fresh("value"));
            lets.push(let_stmt(&value, false, ty.clone(), rhs.clone()));
            lets.push(let_stmt(&reference, false, ref_ty, call.clone()));
            reroot_place(&mut place, &reference);
            reroot = Some((reference, value));
        }
        let drop_old: Vec<Statement> = parts
            .iter()
            .flat_map(|(part, f)| {
                glue_drops_at(
                    &name,
                    Some(&place),
                    std::slice::from_ref(part),
                    f.as_deref(),
                )
            })
            .map(after_value)
            .collect();
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.last_mut() {
            edits.assign.insert(*stmt, (drop_old, None));
            if let Some(r) = reroot {
                edits.lets.entry(*stmt).or_default().extend(lets);
                edits.reroot.insert(*stmt, r);
            }
        }
    }

    /// The operands of a tensor operator, `a @ b`, `a + b` or `-a`, that it moved: dropped
    /// right after it.
    pub(crate) fn drops_note_operands(&mut self, operands: &[(&Expr, &Type)]) {
        if self.speculating {
            return;
        }
        let rewriting = self.drops_rewriting();
        let Some(FrameKind::Block { lines, stmt, .. }) = self.borrow.drops.frames.last() else {
            return;
        };
        let (line, at) = (lines.get(*stmt).copied().unwrap_or(0), *stmt);
        let mut dropped = Vec::new();
        for &(e, ty) in operands {
            let Expr::Identifier(id) = e else { continue };
            if !matches!(ty, Type::Tensor(..))
                || !self.borrow.drops.moved_now.contains(id.name.as_ref())
                || self.drops_owner_state(id.name.as_ref()).is_none()
            {
                continue;
            }
            if self.printing() {
                eprintln!(
                    "drop in {}: {}, after the operator it is moved into on line {line}",
                    self.current_function, id.name
                );
            }
            dropped.push(drop_stmt(id.name.as_ref(), None));
        }
        if rewriting {
            if let Some(FrameKind::Block { edits, .. }) = self.borrow.drops.frames.last_mut() {
                edits.after.entry(at).or_default().extend(dropped);
            }
        }
    }

    /// `name` and everything that borrows it, or borrows something that does. A task holds its
    /// borrows until its join, so the handle of a task counts as well.
    fn drops_users(&self, name: &str) -> Vec<String> {
        let mut seen = vec![name.to_string()];
        let mut i = 0;
        while i < seen.len() {
            let mut next = self.borrow.borrowers_of(&seen[i]);
            next.extend(self.tasks.handles_of(&seen[i]));
            for b in next {
                if !seen.contains(&b) {
                    seen.push(b);
                }
            }
            i += 1;
        }
        seen
    }

    /// Whether a task that is still running borrows `owner`, or something that borrows it.
    fn drops_held_by_running_task(&self, owner: &Owner) -> bool {
        self.drops_users(&owner.name)
            .iter()
            .any(|u| self.borrow.is_running_task(u))
    }

    /// The last statement of a block that uses `owner` or what borrows it. A later `let` of
    /// the same name hides the owner, so only the statements up to that `let` count. The `let`
    /// itself counts, as `let a = f(a)` reads the owner.
    fn drops_last_use(&self, owner: &Owner, uses: &BlockUseList) -> Option<usize> {
        let users = self.drops_users(&owner.name);
        let start = owner.decl.map_or(0, |d| d + 1);
        let end = uses
            .hidden_at(&owner.name, owner.decl)
            .unwrap_or(uses.used.len() - 1);
        (start..=end)
            .rev()
            .find(|&i| users.iter().any(|n| uses.used[i].contains(n)))
    }

    /// Prints the drops that free something or run a `Drop` impl.
    fn drops_print(&self, owner: &Owner, at: String) {
        if !self.printing() || owner.only_ends_lifetime() {
            return;
        }
        let flag = if owner.moved == Moved::Maybe {
            " if it was not moved"
        } else {
            ""
        };
        eprintln!(
            "drop in {}: {}{flag}, {at}",
            self.current_function, owner.name
        );
    }

    /// The block ends. `terminated_at` is its `return`, `break` or `continue`, whose own drops
    /// were found when it was checked. When drops are written into the program, the block's
    /// edits are made now.
    /// `value_used`: something takes the value of the block's last statement.
    pub(crate) fn drops_exit_block(
        &mut self,
        body: &mut Vec<Statement>,
        terminated_at: Option<usize>,
        value_used: bool,
    ) {
        let rewriting = self.drops_rewriting();
        let Some(FrameKind::Block {
            owners,
            uses,
            lines,
            mut edits,
            value_at,
            hidden,
            ..
        }) = self.borrow.drops.frames.pop()
        else {
            panic!("a block frame closes a block");
        };
        if self.speculating {
            return;
        }
        let mut moves = Vec::new();
        if rewriting {
            for h in &hidden {
                moves.extend(rename_hidden_owner(body, &mut edits, h));
            }
            self.drops_hoist_temporaries(body, &mut edits, value_used);
        }
        // A parameter or a name of an arm moves to its new name before the block starts.
        if !moves.is_empty() {
            let lets = edits.lets.entry(0).or_default();
            lets.splice(0..0, moves);
        }
        let raw_pointers = self.borrow.drops.uses_raw_pointers();
        // The body of a `tasks` block: a task that runs until the block ends can use its owners.
        let tasks_body = matches!(self.borrow.drops.frames.last(), Some(FrameKind::Tasks));
        let mut after_tasks = Vec::new();
        for owner in owners.iter().rev() {
            if owner.moved == Moved::Yes {
                continue;
            }
            if tasks_body && self.drops_held_by_running_task(owner) {
                // An exit waited for the tasks and then dropped it.
                if terminated_at.is_some() {
                    continue;
                }
                self.drops_print(owner, "once the tasks of its `tasks` block end".to_string());
                after_tasks.extend(owner_drop(owner, raw_pointers));
                continue;
            }
            let at = if waits_for_block_end(owner, raw_pointers) {
                if terminated_at.is_some() {
                    continue;
                }
                self.drops_print(owner, "at the end of its block".to_string());
                // An empty block, as `fn drop<T>(_x : T) {}` has, drops it where it starts.
                value_at.or(body.len().checked_sub(1))
            } else {
                match self.drops_last_use(owner, &uses) {
                    Some(i) if terminated_at.is_some_and(|t| i >= t) => continue,
                    // `let a = f(a)`: after it, `a` names the new value. The old one is
                    // dropped before it, once its value is computed.
                    Some(i) if uses.hidden_at(&owner.name, owner.decl) == Some(i) => {
                        self.drops_print(
                            owner,
                            format!(
                                "once the `let` on line {} that hides it has its value",
                                lines[i]
                            ),
                        );
                        let stmts = owner_drop(owner, raw_pointers).into_iter().map(after_value);
                        edits.before.entry(i).or_default().extend(stmts);
                        continue;
                    }
                    Some(i) if Some(i) == value_at => {
                        self.drops_print(
                            owner,
                            format!("after the value of its block on line {}", lines[i]),
                        );
                        Some(i)
                    }
                    Some(i) => {
                        self.drops_print(
                            owner,
                            format!("after the statement on line {}", lines[i]),
                        );
                        Some(i)
                    }
                    None => match owner.given_by {
                        Some(by) => {
                            self.drops_print(owner, format!("at the start of {by}"));
                            None
                        }
                        None => {
                            self.drops_print(owner, "where it is made".to_string());
                            owner.decl
                        }
                    },
                }
            };
            let stmts = owner_drop(owner, raw_pointers);
            match at {
                Some(i) => edits.after.entry(i).or_default().extend(stmts),
                None => edits.before.entry(0).or_default().extend(stmts),
            }
        }
        if rewriting {
            // The block's separate value comes after its statements, so what waits for it goes
            // at the end of the block. It runs once the value is computed.
            let mut after_the_value = Vec::new();
            if let Some(v) = value_at {
                after_the_value.extend(edits.before.remove(&v).unwrap_or_default());
                after_the_value.extend(edits.after.remove(&v).unwrap_or_default());
            }
            self.drops_apply(body, *edits);
            if !after_tasks.is_empty() {
                let span = body.last().map_or_else(Span::default, |s| s.span());
                *self
                    .borrow
                    .drops
                    .tasks_joins
                    .last_mut()
                    .expect("a tasks block is open") = Some(body.len());
                body.push(crate::syntax::tasks_join_stmt(span));
                if value_at.is_some() {
                    after_the_value.extend(after_tasks);
                } else {
                    body.extend(after_tasks);
                }
            }
            body.extend(after_the_value.into_iter().map(after_value));
        }
    }

    /// A struct that a statement makes and nothing names -- `make()` in `look(&make())`,
    /// `make().id`, or `make();` -- is dropped at the end of the statement, as in Rust. It is
    /// moved into a `let` of its own just before the statement, and dropped after it. So the
    /// order things happen in does not change, every call the statement makes before it moves
    /// out too, in order, into a `let` that is dropped after the statement only when its
    /// value was just read. Nothing moves out of a branch or a loop, and a temporary after one,
    /// or after a call whose type is not known, stays where it is and is not dropped. The
    /// condition of an `if` and the value of a `match` run before the branch, so their
    /// temporaries move out too. A `&&` or `||` whose sides make temporaries becomes an `if`.
    fn drops_hoist_temporaries(
        &mut self,
        body: &mut [Statement],
        edits: &mut Edits,
        value_used: bool,
    ) {
        let last = body.len().saturating_sub(1);
        for (i, stmt) in body.iter_mut().enumerate() {
            // The block's value goes to whatever takes it, so it is not a temporary.
            let taken =
                value_used && i == last && matches!(stmt, Statement::ExprStmt(e) if !e.has_semi);
            // `for n in it` moves `it` into the loop, which drops what it has not handed out
            // when it ends. The loop advances a variable of its own, made here, in memory.
            if let Statement::ForLoop(f) = stmt {
                if let (Some(_), Expr::Identifier(id)) = (&f.next_fn, &*f.iterable) {
                    let ty = self.lookup(id.name.as_ref()).map(|(t, _)| t.clone());
                    let glue = ty.as_ref().map(|t| self.drops_glue(t)).unwrap_or_default();
                    if let (Some(ty), false) = (ty, glue.is_empty()) {
                        let name = self.drops_fresh("temp");
                        let value = std::mem::replace(&mut *f.iterable, ident(&name));
                        edits
                            .lets
                            .entry(i)
                            .or_default()
                            .push(let_stmt(&name, true, ty, value));
                        let after = edits.after.entry(i).or_default();
                        for (k, drop) in glue_drops(&name, &glue, None).into_iter().enumerate() {
                            after.insert(k, drop);
                        }
                        continue;
                    }
                }
            }
            // Where the last temporary that can move out is, counted in calls.
            let mut probe = TempScan::new(None);
            let mut copy = stmt.clone();
            self.drops_scan_stmt(&mut copy, taken, &mut probe);
            let Some(last_temp) = probe.last_temp else {
                // Only the right side of a `&&` or `||` made a temporary, and its `if` drops it.
                if probe.short_circuit_rewritten {
                    *stmt = copy;
                }
                continue;
            };
            let mut scan = TempScan::new(Some(last_temp));
            self.drops_scan_stmt(stmt, taken, &mut scan);
            for (name, ty, expr, glue) in scan.found {
                // Mutable, so it has a place in memory: the loop advances it there, and its
                // `drop` changes it before its fields are dropped.
                edits
                    .lets
                    .entry(i)
                    .or_default()
                    .push(let_stmt(&name, true, ty, expr));
                let after = edits.after.entry(i).or_default();
                for (k, drop) in glue_drops(&name, &glue, None).into_iter().enumerate() {
                    after.insert(k, drop);
                }
            }
        }
    }

    fn drops_scan_stmt(&mut self, stmt: &mut Statement, taken: bool, scan: &mut TempScan) {
        match stmt {
            Statement::LetDecl(l) => self.drops_scan_temps(&mut l.expr, false, scan),
            Statement::Return(r) => {
                if let Some(e) = &mut r.expr {
                    self.drops_scan_temps(e, false, scan);
                }
            }
            Statement::ExprStmt(e) => self.drops_scan_temps(&mut e.expr, !taken, scan),
            Statement::Assign(a) => self.drops_scan_temps(&mut a.rhs, false, scan),
            Statement::Assert(a) => self.drops_scan_temps(&mut a.expr, false, scan),
            // An iterator the loop makes, `for n in v.into_iter()`, lives until the loop ends.
            Statement::ForLoop(f) => {
                let iterator = f.next_fn.is_some();
                self.drops_scan_temps(&mut f.iterable, iterator, scan)
            }
            _ => {}
        }
    }

    /// Walk `e` in the order it is evaluated, counting its calls. Probing (`scan.upto` is
    /// `None`), note the last temporary every call before which could move out; otherwise move
    /// out each call up to that one. `borrowed`: `e`'s value is only read or discarded.
    fn drops_scan_temps(&mut self, e: &mut Expr, borrowed: bool, scan: &mut TempScan) {
        let ty = match e {
            Expr::FunctionCall(fc) => {
                for arg in fc.args.iter_mut() {
                    self.drops_scan_temps(arg, false, scan);
                }
                self.drops_return_type(&fc.name)
            }
            Expr::StructInit(si) => {
                for (_, field) in si.fields.iter_mut() {
                    self.drops_scan_temps(field, false, scan);
                }
                Some(Type::Struct(si.name.clone(), None))
            }
            Expr::Borrow(b) => return self.drops_scan_temps(&mut b.expr, true, scan),
            Expr::MemberAccess(m) => return self.drops_scan_temps(&mut m.base, true, scan),
            Expr::IndexAccess(ix) => {
                self.drops_scan_temps(&mut ix.base, true, scan);
                return self.drops_scan_temps(&mut ix.index, false, scan);
            }
            Expr::BinaryOp(b) => {
                self.drops_scan_temps(&mut b.lhs, false, scan);
                return self.drops_scan_temps(&mut b.rhs, false, scan);
            }
            Expr::RelationalOp(r) => {
                self.drops_scan_temps(&mut r.lhs, false, scan);
                return self.drops_scan_temps(&mut r.rhs, false, scan);
            }
            Expr::UnaryOp(u) => return self.drops_scan_temps(&mut u.expr, false, scan),
            Expr::Identifier(_) | Expr::Number(_) | Expr::StringLiteral(_) => return,
            // The condition of an `if`, and so of a `while`, runs before either branch. Its
            // temporaries move out, and each branch drops them first, as in Rust.
            Expr::If(i) if !i.is_comptime => {
                let first = scan.found.len();
                self.drops_scan_temps(&mut i.cond, false, scan);
                let mut drops = Vec::new();
                for (name, _, _, glue) in &mut scan.found[first..] {
                    drops.extend(glue_drops(name, &std::mem::take(glue), None));
                }
                if !drops.is_empty() {
                    i.then_block.splice(0..0, drops.clone());
                    i.else_block
                        .get_or_insert_with(Vec::new)
                        .splice(0..0, drops);
                }
                scan.blocked = true;
                return;
            }
            // The value that a `match` matches runs before its arms. Its temporaries drop after
            // the statement, as Rust drops them at the end of the `match`.
            Expr::Match(m) => {
                self.drops_scan_temps(&mut m.expr, false, scan);
                scan.blocked = true;
                return;
            }
            // Each side of `&&` or `||` drops its temporaries when it ends, as in Rust. The right
            // side runs only sometimes, so the expression becomes an `if` that holds the drops:
            // `if a { b } else { false }`, or `if a { true } else { b }`.
            Expr::LogicalOp(l) => {
                let first = scan.found.len();
                self.drops_scan_temps(&mut l.lhs, false, scan);
                let mut drops = Vec::new();
                for (name, _, _, glue) in &mut scan.found[first..] {
                    drops.extend(glue_drops(name, &std::mem::take(glue), None));
                }
                let rhs = std::mem::replace(&mut *l.rhs, ident("false"));
                let mut rhs = vec![Statement::ExprStmt(ExprStmtStmt::new(
                    rhs,
                    false,
                    Span::default(),
                ))];
                let mut edits = Edits::default();
                self.drops_hoist_temporaries(&mut rhs, &mut edits, true);
                self.drops_apply(&mut rhs, edits);
                if drops.is_empty() && rhs.len() == 1 {
                    let Some(Statement::ExprStmt(s)) = rhs.pop() else {
                        unreachable!("the right side is one expression")
                    };
                    *l.rhs = s.expr;
                } else {
                    let constant = |value: &str| {
                        vec![Statement::ExprStmt(ExprStmtStmt::new(
                            ident(value),
                            false,
                            Span::default(),
                        ))]
                    };
                    let (mut then_block, mut else_block) = match l.op {
                        LogicalOp::And => (rhs, constant("false")),
                        LogicalOp::Or => (constant("true"), rhs),
                    };
                    then_block.splice(0..0, drops.clone());
                    else_block.splice(0..0, drops);
                    let lhs = std::mem::replace(&mut *l.lhs, ident("false"));
                    *e = Expr::If(IfExpr::new(
                        false,
                        Box::new(lhs),
                        then_block,
                        Some(else_block),
                        l.span,
                    ));
                    scan.short_circuit_rewritten = true;
                }
                scan.blocked = true;
                return;
            }
            // A branch or a loop runs its parts only sometimes, and a closure later: nothing at
            // or after it moves out.
            _ => {
                scan.blocked = true;
                return;
            }
        };
        let index = scan.calls;
        scan.calls += 1;
        // A `void` call is not a value to keep.
        let ty = ty.filter(|t| !crate::syntax::is_void_ty(t));
        if ty.is_none() {
            scan.blocked = true;
        }
        let glue = match &ty {
            Some(t) if borrowed => self.drops_glue(t),
            _ => Vec::new(),
        };
        let is_struct_temp = !glue.is_empty() && !matches!(ty, Some(Type::Tensor(..)));
        match scan.upto {
            None => {
                if is_struct_temp && !scan.blocked {
                    scan.last_temp = Some(index);
                }
            }
            Some(last) if index <= last => {
                let ty = ty.expect("every call up to the last temporary has a type");
                let name = self.drops_fresh("temp");
                let value = std::mem::replace(e, ident(&name));
                scan.found.push((name, ty, value, glue));
            }
            Some(_) => {}
        }
    }

    /// Whether `e`, a call or struct literal borrowed with no name (`&make()`), is dropped at
    /// the end of its statement, as `drops_scan_temps` arranges for a struct that needs it.
    pub(crate) fn drops_frees_borrowed_temporary(&mut self, e: &Expr) -> bool {
        let ty = match e {
            Expr::FunctionCall(fc) => self.drops_return_type(&fc.name),
            Expr::StructInit(si) => Some(Type::Struct(si.name.clone(), None)),
            _ => None,
        };
        ty.is_some_and(|t| !matches!(t, Type::Tensor(..)) && !self.drops_glue(&t).is_empty())
    }

    /// The declared return type of the function `name`, once the checker has resolved it.
    fn drops_return_type(&self, name: &crate::symbol::Symbol) -> Option<Type> {
        if let Some(f) = self.env.functions.get(name) {
            return Some(f.0.clone());
        }
        self.mono
            .functions
            .iter()
            .find(|(f, _)| &f.name == name)
            .map(|(f, _)| f.return_type.clone())
    }

    /// Make a block's edits, from its last statement back so the indices hold.
    fn drops_apply(&mut self, body: &mut Vec<Statement>, mut edits: Edits) {
        if body.is_empty() {
            body.extend(edits.before.remove(&0).unwrap_or_default());
            return;
        }
        let last = body.len().checked_sub(1);
        let mut at: Vec<usize> = edits
            .lets
            .keys()
            .chain(edits.before.keys())
            .chain(edits.after.keys())
            .chain(edits.assign.keys())
            .copied()
            .collect();
        at.sort_unstable();
        at.dedup();
        for i in at.into_iter().rev() {
            if i >= body.len() {
                continue;
            }
            let lets = edits.lets.remove(&i).unwrap_or_default();
            let before = edits.before.remove(&i).unwrap_or_default();
            let after = edits.after.remove(&i).unwrap_or_default();
            let mut out = before;
            let drops_only =
                |v: Vec<Statement>| v.into_iter().filter(|s| matches!(s, Statement::Drop(_)));
            match &body[i] {
                // Leaving with a value: the drops wait for it. A flag set after the statement
                // could not be reached, and the drops it guards are leaving too.
                Statement::Return(ReturnStmt { expr: Some(_), .. }) => {
                    out = out.into_iter().map(after_value).collect();
                    out.extend(drops_only(after).map(after_value));
                    out.push(body[i].clone());
                }
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    out.extend(drops_only(after));
                    out.push(body[i].clone());
                }
                // The block's value: its drops wait for it, and a flag is set before it, since
                // the value is always computed in full.
                Statement::ExprStmt(e) if Some(i) == last && !e.has_semi => {
                    let (flags, drops): (Vec<_>, Vec<_>) = after
                        .into_iter()
                        .partition(|s| matches!(s, Statement::Assign(_)));
                    out.extend(flags);
                    out.extend(drops.into_iter().map(after_value));
                    out.push(body[i].clone());
                }
                _ => {
                    let assign = edits.assign.remove(&i);
                    if let Some((drop_old, _)) = &assign {
                        out.extend(drop_old.iter().cloned());
                    }
                    if let Some((reference, value)) = edits.reroot.remove(&i) {
                        let Statement::Assign(a) = &mut body[i] else {
                            unreachable!("only an assignment writes through a call")
                        };
                        reroot_place(&mut a.lhs, &reference);
                        a.rhs = ident(&value);
                    }
                    out.push(body[i].clone());
                    out.extend(after);
                    if let Some((_, Some(f))) = assign {
                        out.push(assign_stmt(&f, ident("false")));
                    }
                }
            }
            body.splice(i..=i, lets.into_iter().chain(out));
        }
    }

    /// An exit at `span` to `to`: the owners it leaves that are still alive are dropped first.
    /// When it leaves a `tasks` block, it first waits for the tasks of that block, which can
    /// still use these owners.
    pub(crate) fn drops_exit(&mut self, what: &str, span: &Span, to: ExitTo) {
        if self.speculating {
            return;
        }
        let line = span.line;
        let rewriting = self.drops_rewriting();
        let mut dropped = Vec::new();
        let mut tasks_blocks = 0;
        // An owner outside the inner `tasks` block that the exit leaves.
        let mut outside_inner = None;
        let mut lifetime_ends = Vec::new();
        let d = &self.borrow.drops;
        let raw_pointers = d.uses_raw_pointers();
        'frames: for f in d.frames.iter().rev() {
            match f {
                // A `return` inside a region leaves the region, not the function.
                FrameKind::Function { .. } | FrameKind::Spawn | FrameKind::Task => break 'frames,
                FrameKind::Loop if to == ExitTo::Loop => break 'frames,
                FrameKind::Loop => {}
                FrameKind::Tasks if to == ExitTo::TasksBlock => {
                    tasks_blocks += 1;
                    break 'frames;
                }
                FrameKind::Tasks => tasks_blocks += 1,
                FrameKind::Block {
                    owners, uses, stmt, ..
                } => {
                    for owner in owners.iter().rev() {
                        if owner.moved == Moved::Yes || d.moved_now.contains(&owner.name) {
                            continue;
                        }
                        // Not used from this statement on, or never: dropped already.
                        if !waits_for_block_end(owner, raw_pointers)
                            && !self.drops_held_by_running_task(owner)
                            && self.drops_last_use(owner, uses).is_none_or(|i| i < *stmt)
                        {
                            continue;
                        }
                        if owner.only_ends_lifetime() {
                            lifetime_ends.extend(owner_drop(owner, raw_pointers));
                            continue;
                        }
                        self.drops_print(owner, format!("before the {what} on line {line}"));
                        dropped.extend(owner_drop(owner, raw_pointers));
                        if tasks_blocks > 0 && outside_inner.is_none() {
                            outside_inner = Some(owner.name.clone());
                        }
                    }
                }
            }
        }
        // The wait is for the inner block only. A task of an outer block can still run.
        if let (true, Some(name)) = (tasks_blocks > 1, outside_inner) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E4014,
                format!(
                    "this `{what}` leaves two `tasks` blocks, and it drops '{name}' first. A \
                     task of the outer block can still use '{name}' then. Move the `{what}` out \
                     of the inner `tasks` block"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
        if !rewriting {
            return;
        }
        if tasks_blocks > 0 {
            dropped.insert(0, crate::syntax::tasks_wait_stmt(*span));
        }
        // A task of the outer `tasks` block can still use a struct, so its slot keeps its
        // lifetime on this path.
        if tasks_blocks <= 1 {
            dropped.extend(lifetime_ends);
        }
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.last_mut() {
            edits.before.entry(*stmt).or_default().extend(dropped);
        }
    }
}

#[cfg(test)]
mod tests {
    /// The checker writes the default of `Drop::wait_limit` when an impl has no copy of it, so
    /// the two must say the same number.
    #[test]
    fn the_default_wait_limit_is_the_one_in_core_ops() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/stdlib/core/ops.vx");
        let ops = std::fs::read_to_string(path).expect("stdlib/core/ops.vx can be read");
        let method = ops
            .find("fn wait_limit(self : &Self) -> Duration {")
            .expect("`Drop` has `wait_limit` with a default body");
        let body = ops[method..].lines().nth(1).unwrap_or_default().trim();
        let secs = super::DEFAULT_WAIT_LIMIT_SECS;
        assert_eq!(body, format!("return Duration::from_secs({secs});"));
    }
}
