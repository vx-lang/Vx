//===- drops.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where each tensor a function owns is dropped: `docs/implementation_plans/drop_semantics.md`.
// With `VX_PRINT_DROPS=1` the points are printed. With `VX_DROPS=scope` the checker also writes
// them into the program as `Statement::Drop`, which the code generators lower (phase 2).
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
}

/// What the checker writes into one block, by statement.
#[derive(Default)]
struct Edits {
    before: BTreeMap<usize, Vec<Statement>>,
    after: BTreeMap<usize, Vec<Statement>>,
    /// `t = e`: the drop of the old value, if it was not moved, and the flag to clear once
    /// everything else at the statement has run.
    assign: HashMap<usize, (Option<Statement>, Option<String>)>,
}

enum FrameKind {
    Function {
        /// Whether drops are written into this body.
        rewrite: bool,
    },
    Loop,
    Block {
        owners: Vec<Owner>,
        /// The last statement of the block that uses each name, nested uses included.
        last_use: HashMap<crate::symbol::Symbol, usize>,
        /// The first line of each statement.
        lines: Vec<usize>,
        /// The statement being checked.
        stmt: usize,
        edits: Edits,
    },
}

/// The drop-point state of the function being checked: one frame per open block, loop and
/// function or closure body.
#[derive(Default)]
pub(crate) struct DropFrames {
    frames: Vec<FrameKind>,
    /// Owners to add to the next block: a function's tensor parameters.
    pending: Vec<(String, usize)>,
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
fn owner_drop(owner: &Owner) -> Statement {
    let flag = match owner.moved {
        Moved::Maybe => owner.flag.as_deref(),
        _ => None,
    };
    drop_stmt(&owner.name, flag)
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

    /// A function or closure body begins. `params` are its tensor parameters taken by value.
    pub(crate) fn drops_enter_function(&mut self, params: Vec<String>, body: &[Statement]) {
        let scope = self.scopes.len() - 1;
        let already = body.iter().any(|s| {
            let mut uses = HashSet::new();
            Self::extract_uses_stmt(s, &mut uses);
            uses.contains(DROP_MARK)
        });
        let rewrite = !already && std::env::var("VX_DROPS").is_ok_and(|v| v == "scope");
        let d = &mut self.borrow.drops;
        d.frames.push(FrameKind::Function { rewrite });
        d.pending = params.into_iter().map(|p| (p, scope)).collect();
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
            .map(|(name, scope)| Owner {
                name,
                is_param: true,
                decl: None,
                to_block_end: false,
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
            })
            .collect();
        d.frames.push(FrameKind::Block {
            owners,
            last_use: Self::compute_block_liveness(body),
            lines: body.iter().map(|s| s.span().line).collect(),
            stmt: 0,
            edits: Edits::default(),
        });
    }

    pub(crate) fn drops_set_stmt(&mut self, i: usize) {
        if let Some(FrameKind::Block { stmt, .. }) = self.borrow.drops.frames.last_mut() {
            *stmt = i;
        }
        self.borrow.drops.moved_now.clear();
    }

    /// `let name = ..` bound an owned tensor.
    pub(crate) fn drops_note_owner(&mut self, name: &str, ty: &Type) {
        if !matches!(ty, Type::Tensor(..)) || self.borrow.views.contains_key(name) {
            return;
        }
        let scope = self.scopes.len() - 1;
        if let Some(FrameKind::Block { owners, stmt, .. }) = self.borrow.drops.frames.last_mut() {
            owners.push(Owner {
                name: name.to_string(),
                is_param: false,
                decl: Some(*stmt),
                to_block_end: false,
                scope,
                moved: Moved::No,
                moved_at: None,
                flag: None,
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
                FrameKind::Function { .. } => return,
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
                FrameKind::Loop => {}
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
                FrameKind::Function { .. } => return None,
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
                edits.assign.insert(*stmt, (None, Some(f)));
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
        if printing() {
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
        let drop_old = Some(after_value(drop_stmt(name, old_flag)));
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
            let stmt = owner_drop(owner);
            match at {
                Some(i) => edits.after.entry(i).or_default().push(stmt),
                None => edits.before.entry(0).or_default().push(stmt),
            }
        }
        if rewriting {
            self.drops_apply(body, edits);
        }
    }

    /// Make a block's edits, from its last statement back so the indices hold.
    fn drops_apply(&mut self, body: &mut Vec<Statement>, mut edits: Edits) {
        let last = body.len().checked_sub(1);
        let mut at: Vec<usize> = edits
            .before
            .keys()
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
                    if let Some((Some(drop_old), _)) = &assign {
                        out.push(drop_old.clone());
                    }
                    out.push(body[i].clone());
                    out.extend(after);
                    if let Some((_, Some(f))) = assign {
                        out.push(assign_stmt(&f, ident("false")));
                    }
                }
            }
            body.splice(i..=i, out);
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
                FrameKind::Function { .. } => break 'frames,
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
                        dropped.push(owner_drop(owner));
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
