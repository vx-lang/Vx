//===- drops.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where each tensor a function owns is dropped: phase 1 of
// `docs/implementation_plans/drop_semantics.md`. The points are found and, with
// `VX_PRINT_DROPS=1`, printed; nothing acts on them yet.
//
// An owner is a `let` bound to a tensor that is not a view, or a tensor parameter taken by
// value. It is dropped after the statement of its block that last uses it, or anything that
// borrows it: a view, a reference, a closure. A use inside a nested `if` or loop counts as a
// use by that whole statement. A `return`, `break` or `continue` first drops the owners it
// leaves that are still alive. An owner moved at its own level is not dropped; one moved
// inside a nested block may or may not have been, so its drop is under a flag. An owner a
// raw pointer was taken from (`t.as_ptr()`) waits for the end of its block: no borrow
// checker sees what a raw pointer is used for.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    No,
    Maybe,
    Yes,
}

struct Owner {
    name: String,
    is_param: bool,
    /// A raw pointer was taken from it: dropped at the end of its block, not its last use.
    to_block_end: bool,
    /// The scope it was declared in: a move from the same scope is certain.
    scope: usize,
    moved: Moved,
}

enum FrameKind {
    Function,
    Loop,
    Block {
        owners: Vec<Owner>,
        /// The last statement of the block that uses each name, nested uses included.
        last_use: HashMap<crate::symbol::Symbol, usize>,
        /// The first line of each statement.
        lines: Vec<usize>,
        /// The statement being checked.
        stmt: usize,
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
}

fn printing() -> bool {
    std::env::var_os("VX_PRINT_DROPS").is_some()
}

impl<'a> TypeChecker<'a> {
    /// A function or closure body begins. `params` are its tensor parameters taken by value.
    pub(crate) fn drops_enter_function(&mut self, params: Vec<String>) {
        let scope = self.scopes.len() - 1;
        let d = &mut self.borrow.drops;
        d.frames.push(FrameKind::Function);
        d.pending = params.into_iter().map(|p| (p, scope)).collect();
    }

    pub(crate) fn drops_exit_function(&mut self) {
        let d = &mut self.borrow.drops;
        while let Some(f) = d.frames.pop() {
            if matches!(f, FrameKind::Function) {
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
                to_block_end: false,
                scope,
                moved: Moved::No,
            })
            .collect();
        d.frames.push(FrameKind::Block {
            owners,
            last_use: Self::compute_block_liveness(body),
            lines: body.iter().map(|s| s.span().line).collect(),
            stmt: 0,
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
        if let Some(FrameKind::Block { owners, .. }) = self.borrow.drops.frames.last_mut() {
            owners.push(Owner {
                name: name.to_string(),
                is_param: false,
                to_block_end: false,
                scope,
                moved: Moved::No,
            });
        }
    }

    /// `name` was moved, or (`again`) given a new value after a move.
    pub(crate) fn drops_note_move(&mut self, name: &str, again: bool) {
        let here = self.scopes.len() - 1;
        let d = &mut self.borrow.drops;
        if !again {
            d.moved_now.insert(name.to_string());
        }
        for f in d.frames.iter_mut().rev() {
            match f {
                FrameKind::Function => return,
                FrameKind::Loop => {}
                FrameKind::Block { owners, .. } => {
                    if let Some(o) = owners.iter_mut().rev().find(|o| o.name == name) {
                        o.moved = if again {
                            Moved::No
                        } else if o.scope == here {
                            Moved::Yes
                        } else {
                            Moved::Maybe
                        };
                        return;
                    }
                }
            }
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
                FrameKind::Function => return,
                FrameKind::Loop => {}
                FrameKind::Block { owners, .. } => {
                    for o in owners.iter_mut().filter(|o| names.contains(&o.name)) {
                        o.to_block_end = true;
                    }
                }
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
    /// were reported when it was checked.
    pub(crate) fn drops_exit_block(&mut self, terminated_at: Option<usize>) {
        let Some(FrameKind::Block {
            owners,
            last_use,
            lines,
            ..
        }) = self.borrow.drops.frames.pop()
        else {
            panic!("a block frame closes a block");
        };
        if !printing() || self.speculating {
            return;
        }
        for owner in owners.iter().rev() {
            if owner.moved == Moved::Yes {
                continue;
            }
            if owner.to_block_end {
                if terminated_at.is_none() {
                    self.drops_print(owner, "at the end of its block".to_string());
                }
                continue;
            }
            match self.drops_last_use(&owner.name, &last_use) {
                Some(i) if terminated_at.is_some_and(|t| i >= t) => {}
                Some(i) => {
                    self.drops_print(owner, format!("after the statement on line {}", lines[i]))
                }
                None if owner.is_param => {
                    self.drops_print(owner, "at the start of the function".to_string())
                }
                None => self.drops_print(owner, "where it is made".to_string()),
            }
        }
    }

    /// A `return` (`out_of_loop` false) or a `break` or `continue` (true) on `line`: the owners
    /// it leaves that are still alive are dropped first.
    pub(crate) fn drops_exit(&self, what: &str, line: usize, out_of_loop: bool) {
        if !printing() || self.speculating {
            return;
        }
        let d = &self.borrow.drops;
        for f in d.frames.iter().rev() {
            match f {
                FrameKind::Function => return,
                FrameKind::Loop if out_of_loop => return,
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
                    }
                }
            }
        }
    }
}
