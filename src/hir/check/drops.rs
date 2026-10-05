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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moved {
    No,
    Maybe,
    Yes,
}

struct Owner {
    name: String,
    is_param: bool,
    /// The statement that declares it in its block; `None` for a parameter.
    decl: Option<usize>,
    /// A raw pointer was taken from it: dropped at the end of its block, not its last use.
    to_block_end: bool,
    /// The scope it was declared in: a move from the same scope is certain.
    scope: usize,
    moved: Moved,
    /// The statement of its block that last moved it for certain.
    moved_at: Option<usize>,
    /// The `bool` local that says it was moved, once a move inside a nested block needed one.
    flag: Option<String>,
    /// What dropping it does, in order.
    glue: Vec<GluePart>,
}

/// One step of dropping a value: a `Drop` impl's `drop` to call, or a tensor to free, at a
/// field path inside it.
#[derive(Clone)]
pub(crate) struct GluePart {
    /// Each field followed, with the type of the struct it is a field of.
    path: Vec<(crate::symbol::Symbol, String)>,
    call: Option<crate::symbol::Symbol>,
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
}

impl TempScan {
    fn new(upto: Option<usize>) -> Self {
        Self {
            calls: 0,
            blocked: false,
            last_temp: None,
            upto,
            found: Vec::new(),
        }
    }
}

/// What the checker writes into one block, by statement.
#[derive(Default)]
struct Edits {
    /// The `let`s of the statement's temporaries, which go first.
    lets: BTreeMap<usize, Vec<Statement>>,
    before: BTreeMap<usize, Vec<Statement>>,
    after: BTreeMap<usize, Vec<Statement>>,
    /// `t = e`: the drops of the old value, if it was not moved, and the flag to clear once
    /// everything else at the statement has run.
    assign: HashMap<usize, (Vec<Statement>, Option<String>)>,
}

enum FrameKind {
    Function {
        /// Whether drops are written into this body.
        rewrite: bool,
    },
    Loop,
    /// A `spawn` region. A tensor it captures stays owned by the enclosing function, so a
    /// move inside the region does not reach owners outside it.
    Spawn,
    Block {
        owners: Vec<Owner>,
        /// The last statement of the block that uses each name, nested uses included.
        last_use: HashMap<crate::symbol::Symbol, usize>,
        /// The first line of each statement.
        lines: Vec<usize>,
        /// The statement being checked.
        stmt: usize,
        edits: Box<Edits>,
    },
}

/// The drop-point state of the function being checked: one frame per open block, loop and
/// function or closure body.
#[derive(Default)]
pub(crate) struct DropFrames {
    frames: Vec<FrameKind>,
    /// Owners to add to the next block: a function's parameters that need dropping.
    pending: Vec<(String, usize, Vec<GluePart>)>,
    /// The names moved by the statement being checked. A `return t` moves `t` on its way out.
    moved_now: HashSet<String>,
    /// Makes the names of the locals the rewrite adds unique.
    counter: usize,
}

fn printing() -> bool {
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
        expr: None,
        flag: flag.map(Into::into),
        after_value: false,
        span: Span::default(),
    })
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

/// `drop owner`, naming its flag when it may have been moved.
fn owner_drop(owner: &Owner) -> Vec<Statement> {
    let flag = match owner.moved {
        Moved::Maybe => owner.flag.as_deref(),
        _ => None,
    };
    glue_drops(&owner.name, &owner.glue, flag)
}

/// The drops that run `glue` on `name`.
fn glue_drops(name: &str, glue: &[GluePart], flag: Option<&str>) -> Vec<Statement> {
    glue.iter()
        .map(|part| {
            let mut place = ident(name);
            for (field, struct_ty) in &part.path {
                let mut access =
                    MemberAccessExpr::new(Box::new(place), field.clone(), Span::default());
                access.struct_name = Some(struct_ty.as_str().into());
                place = Expr::MemberAccess(access);
            }
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
                None if part.path.is_empty() => None,
                None => Some(place),
            };
            Statement::Drop(DropStmt {
                name: name.into(),
                path: part.path.iter().map(|(f, _)| f.clone()).collect(),
                call: part.call.clone(),
                expr: expr.map(Box::new),
                flag: flag.map(Into::into),
                after_value: false,
                span: Span::default(),
            })
        })
        .collect()
}

impl<'a> TypeChecker<'a> {
    /// Whether drops are written into the function being checked.
    fn drops_rewriting(&self) -> bool {
        !self.speculating
            && self.borrow.drops.frames.iter().rev().find_map(|f| match f {
                FrameKind::Function { rewrite } => Some(*rewrite),
                _ => None,
            }) == Some(true)
    }

    fn drops_fresh(&mut self, what: &str) -> String {
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
            }],
            Type::Struct(..) | Type::GenericInstance(..) => {
                if self.is_copy(ty) {
                    return Vec::new();
                }
                // A closure's environment holds what it captured, views among them, and the
                // function that made it still owns those: nothing to drop.
                if ty.to_string().starts_with("Closure_") {
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
    fn drops_impl_call(&mut self, ty: &Type) -> Option<crate::symbol::Symbol> {
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
            .find(|m| m.name.as_ref() == "drop")?
            .clone();
        let mut mapping = HashMap::new();
        self.unify_types(&ib.target_type, ty, &mut mapping);
        let (_, name) = self.instantiate_impl_method(&method, &mapping, ty, ib);
        Some(name.into())
    }

    /// A function or closure body begins. `params` are its parameters taken by value.
    pub(crate) fn drops_enter_function(&mut self, params: Vec<(String, Type)>, body: &[Statement]) {
        let scope = self.scopes.len() - 1;
        let params: Vec<(String, Vec<GluePart>)> = params
            .into_iter()
            .map(|(name, ty)| (name, self.drops_glue(&ty)))
            .filter(|(_, glue)| !glue.is_empty())
            .collect();
        let already = body.iter().any(|s| {
            let mut uses = HashSet::new();
            Self::extract_uses_stmt(s, &mut uses);
            uses.contains(DROP_MARK)
        });
        let rewrite = !already;
        let d = &mut self.borrow.drops;
        d.frames.push(FrameKind::Function { rewrite });
        d.pending = params
            .into_iter()
            .map(|(p, glue)| (p, scope, glue))
            .collect();
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

    pub(crate) fn drops_exit_loop(&mut self) {
        let popped = self.borrow.drops.frames.pop();
        assert!(
            matches!(popped, Some(FrameKind::Loop)),
            "a loop frame closes a loop"
        );
    }

    pub(crate) fn drops_enter_block(&mut self, body: &[Statement]) {
        let d = &mut self.borrow.drops;
        let owners = std::mem::take(&mut d.pending)
            .into_iter()
            .map(|(name, scope, glue)| Owner {
                name,
                is_param: true,
                decl: None,
                to_block_end: glue.iter().any(|g| g.call.is_some()),
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
                glue,
            })
            .collect();
        d.frames.push(FrameKind::Block {
            owners,
            last_use: Self::compute_block_liveness(body),
            lines: body.iter().map(|s| s.span().line).collect(),
            stmt: 0,
            edits: Box::default(),
        });
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
    pub(crate) fn drops_note_owner(&mut self, name: &str, ty: &Type) {
        if self.borrow.views.contains_key(name) {
            return;
        }
        let glue = self.drops_glue(ty);
        if glue.is_empty() {
            return;
        }
        let scope = self.scopes.len() - 1;
        if let Some(FrameKind::Block { owners, stmt, .. }) = self.borrow.drops.frames.last_mut() {
            owners.push(Owner {
                name: name.to_string(),
                is_param: false,
                decl: Some(*stmt),
                to_block_end: glue.iter().any(|g| g.call.is_some()),
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
                glue,
            });
        }
    }

    /// `name` was moved, or (`again`) given a new value after a move.
    pub(crate) fn drops_note_move(&mut self, name: &str, again: bool) {
        let here = self.scopes.len() - 1;
        let rewriting = self.drops_rewriting();
        if !again {
            self.borrow.drops.moved_now.insert(name.to_string());
        }
        let top = self.borrow.drops.frames.len();
        let mut found = None;
        for (k, f) in self.borrow.drops.frames.iter_mut().enumerate().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return,
                FrameKind::Loop => {}
                FrameKind::Block { owners, stmt, .. } => {
                    if let Some(o) = owners.iter_mut().rev().find(|o| o.name == name) {
                        // A value given in a nested block only reaches the paths through it:
                        // a move before it stays possible.
                        o.moved = if again {
                            if o.scope == here || o.moved == Moved::No {
                                Moved::No
                            } else {
                                Moved::Maybe
                            }
                        } else if o.scope == here {
                            o.moved_at = Some(*stmt);
                            Moved::Yes
                        } else {
                            Moved::Maybe
                        };
                        found = Some((k, o.moved, o.flag.clone(), o.decl, o.moved_at));
                        break;
                    }
                }
            }
        }
        let Some((k, Moved::Maybe, flag, decl, moved_at)) = found else {
            return;
        };
        if !rewriting || (again && flag.is_some()) {
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
                        o.flag = Some(f.clone());
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
        if let Some(FrameKind::Block { stmt, edits, .. }) =
            self.borrow.drops.frames.get_mut(top - 1)
        {
            edits
                .after
                .entry(*stmt)
                .or_default()
                .push(assign_stmt(&flag, ident("true")));
        }
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
                FrameKind::Loop | FrameKind::Spawn => {}
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
    pub(crate) fn drops_owner_state(&self, name: &str) -> Option<Moved> {
        self.drops_owner(name).map(|o| o.moved)
    }

    fn drops_owner(&self, name: &str) -> Option<&Owner> {
        for f in self.borrow.drops.frames.iter().rev() {
            match f {
                FrameKind::Function { .. } | FrameKind::Spawn => return None,
                FrameKind::Loop => {}
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
        if printing() && !operand_drop {
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
        let drop_old: Vec<Statement> = glue_drops(name, &owner.glue, old_flag)
            .into_iter()
            .map(after_value)
            .collect();
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.last_mut() {
            edits.assign.insert(*stmt, (drop_old, flag));
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
            if printing() {
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

    /// `name` and everything that borrows it, or borrows something that does.
    fn drops_users(&self, name: &str) -> Vec<String> {
        let mut seen = vec![name.to_string()];
        let mut i = 0;
        while i < seen.len() {
            for b in self.borrow.borrowers_of(&seen[i]) {
                if !seen.contains(&b) {
                    seen.push(b);
                }
            }
            i += 1;
        }
        seen
    }

    /// The last statement of a block that uses `name` or what borrows it.
    fn drops_last_use(
        &self,
        name: &str,
        last_use: &HashMap<crate::symbol::Symbol, usize>,
    ) -> Option<usize> {
        self.drops_users(name)
            .iter()
            .filter_map(|n| last_use.get(n.as_str()).copied())
            .max()
    }

    fn drops_print(&self, owner: &Owner, at: String) {
        if !printing() {
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
    pub(crate) fn drops_exit_block(
        &mut self,
        body: &mut Vec<Statement>,
        terminated_at: Option<usize>,
    ) {
        let rewriting = self.drops_rewriting();
        let Some(FrameKind::Block {
            owners,
            last_use,
            lines,
            mut edits,
            ..
        }) = self.borrow.drops.frames.pop()
        else {
            panic!("a block frame closes a block");
        };
        if self.speculating {
            return;
        }
        if rewriting {
            self.drops_hoist_temporaries(body, &mut edits);
        }
        for owner in owners.iter().rev() {
            if owner.moved == Moved::Yes {
                continue;
            }
            let at = if owner.to_block_end {
                if terminated_at.is_some() || body.is_empty() {
                    continue;
                }
                self.drops_print(owner, "at the end of its block".to_string());
                Some(body.len() - 1)
            } else {
                match self.drops_last_use(&owner.name, &last_use) {
                    Some(i) if terminated_at.is_some_and(|t| i >= t) => continue,
                    Some(i) => {
                        self.drops_print(
                            owner,
                            format!("after the statement on line {}", lines[i]),
                        );
                        Some(i)
                    }
                    None if owner.is_param => {
                        self.drops_print(owner, "at the start of the function".to_string());
                        None
                    }
                    None => {
                        self.drops_print(owner, "where it is made".to_string());
                        owner.decl
                    }
                }
            };
            let stmts = owner_drop(owner);
            match at {
                Some(i) => edits.after.entry(i).or_default().extend(stmts),
                None => edits.before.entry(0).or_default().extend(stmts),
            }
        }
        if rewriting {
            self.drops_apply(body, *edits);
        }
    }

    /// A struct that a statement makes and nothing names -- `make()` in `look(&make())`,
    /// `make().id`, or `make();` -- is dropped at the end of the statement, as in Rust. It is
    /// moved into a `let` of its own just before the statement, and dropped after it. So the
    /// order things happen in does not change, every call the statement makes before it moves
    /// out too, in order, into a `let` that is dropped after the statement only when its
    /// value was just read. Nothing moves out of a branch, a loop or the right of `&&` or `||`,
    /// and a temporary after one, or after a call whose type is not known, stays where it is
    /// and is not dropped.
    fn drops_hoist_temporaries(&mut self, body: &mut [Statement], edits: &mut Edits) {
        for (i, stmt) in body.iter_mut().enumerate() {
            // Where the last temporary that can move out is, counted in calls.
            let mut probe = TempScan::new(None);
            let mut copy = stmt.clone();
            self.drops_scan_stmt(&mut copy, &mut probe);
            let Some(last) = probe.last_temp else {
                continue;
            };
            let mut scan = TempScan::new(Some(last));
            self.drops_scan_stmt(stmt, &mut scan);
            for (name, ty, expr, glue) in scan.found {
                edits
                    .lets
                    .entry(i)
                    .or_default()
                    .push(let_stmt(&name, false, ty, expr));
                let after = edits.after.entry(i).or_default();
                for (k, drop) in glue_drops(&name, &glue, None).into_iter().enumerate() {
                    after.insert(k, drop);
                }
            }
        }
    }

    fn drops_scan_stmt(&mut self, stmt: &mut Statement, scan: &mut TempScan) {
        match stmt {
            Statement::LetDecl(l) => self.drops_scan_temps(&mut l.expr, false, scan),
            Statement::Return(r) => {
                if let Some(e) = &mut r.expr {
                    self.drops_scan_temps(e, false, scan);
                }
            }
            Statement::ExprStmt(e) => self.drops_scan_temps(&mut e.expr, true, scan),
            Statement::Assign(a) => self.drops_scan_temps(&mut a.rhs, false, scan),
            Statement::Assert(a) => self.drops_scan_temps(&mut a.expr, false, scan),
            Statement::ForLoop(f) => self.drops_scan_temps(&mut f.iterable, false, scan),
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
            // A branch, a loop or a short-circuit runs its parts only sometimes, and a
            // closure later: nothing at or after it moves out.
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

    /// A `return` (`out_of_loop` false) or a `break` or `continue` (true) on `line`: the owners
    /// it leaves that are still alive are dropped first.
    pub(crate) fn drops_exit(&mut self, what: &str, line: usize, out_of_loop: bool) {
        if self.speculating {
            return;
        }
        let rewriting = self.drops_rewriting();
        let mut dropped = Vec::new();
        let d = &self.borrow.drops;
        'frames: for f in d.frames.iter().rev() {
            match f {
                // A `return` inside a region leaves the region, not the function.
                FrameKind::Function { .. } | FrameKind::Spawn => break 'frames,
                FrameKind::Loop if out_of_loop => break 'frames,
                FrameKind::Loop => {}
                FrameKind::Block {
                    owners,
                    last_use,
                    stmt,
                    ..
                } => {
                    for owner in owners.iter().rev() {
                        if owner.moved == Moved::Yes || d.moved_now.contains(&owner.name) {
                            continue;
                        }
                        // Not used from this statement on, or never: dropped already.
                        if !owner.to_block_end
                            && self
                                .drops_last_use(&owner.name, last_use)
                                .is_none_or(|i| i < *stmt)
                        {
                            continue;
                        }
                        self.drops_print(owner, format!("before the {what} on line {line}"));
                        dropped.extend(owner_drop(owner));
                    }
                }
            }
        }
        if !rewriting {
            return;
        }
        if let Some(FrameKind::Block { stmt, edits, .. }) = self.borrow.drops.frames.last_mut() {
            edits.before.entry(*stmt).or_default().extend(dropped);
        }
    }
}
