//! Borrow-checking state whose coherence the frontend depends on: the active borrow records and the
//! per-block liveness that decides which of them are dead (NLL). This is the chokepoint the frontend
//! previously lacked (frontend_refactoring_borrow_checker.md, R1; #279).
//!
//! The point of the module boundary is that `active_borrows` is **private** — no code outside this file
//! can read it. The only conflict-check read path is [`BorrowCx::live_borrows`], which sweeps dead
//! borrows first, so a *new* access path physically cannot forget the sweep the way `check_borrow_expr`
//! once did (#276): "remember to sweep" is now a type-system guarantee, not a convention. Save/restore
//! and scope bookkeeping go through the other methods.
//!
//! Per-worker, owned by value inside `TypeChecker` — no shared state and no lock, so it composes with
//! the parallel pipeline's phase-3 isolation invariant.

use crate::hir::env::{BorrowRecord, RefProvenance};
use crate::symbol::Symbol;
use crate::syntax::Type;
use std::collections::{HashMap, HashSet};

/// The active borrow table plus the NLL liveness that keeps it honest, and the rest of the frontend's
/// borrow/move-checking state (moves, reference provenance, current parameters). See the module docs.
pub(crate) struct BorrowCx {
    /// `variable -> its borrow records`. **Private**: readable for a conflict check only via
    /// [`Self::live_borrows`], which sweeps dead borrows before returning.
    active_borrows: HashMap<Symbol, Vec<BorrowRecord>>,
    /// Per-block last-use index of each local (NLL). The innermost block is `.last()`.
    block_liveness: Vec<HashMap<Symbol, usize>>,
    /// The statement index currently being checked, one entry per open block. Innermost is `.last()`.
    current_stmt_idx: Vec<usize>,
    /// Whether borrow-conflict checks are suppressed (e.g. while typing a member-access receiver,
    /// which must not itself trip the borrow rules it is only being read to resolve).
    pub(crate) skip_borrow_check: bool,
    /// Moved/consumed linear locals, one `HashSet` per lexical scope, kept in lock-step with
    /// `TypeChecker::scopes` (pushed/popped together). Innermost is `.last()`; starts with one scope
    /// so a top-level `consume` always has a slot to mark.
    pub(crate) moved_vars: Vec<HashSet<String>>,
    /// The bindings declared `mut`, one set per lexical scope, kept in step with `moved_vars`.
    /// Every other binding is read-only.
    pub(crate) mutable_vars: Vec<HashSet<String>>,
    /// Provenance of each reference-typed *local* binding (#243): does it root in caller memory
    /// (`External`) or a function-local slot (`Local`)? Reset per function.
    pub(crate) ref_provenance: HashMap<Symbol, RefProvenance>,
    /// Parameters (name -> declared type) of the function currently being checked (#243). Reset per
    /// function; lets the return-escape analysis tell a caller-owned reference parameter apart from a
    /// local binding of the same reference type.
    pub(crate) current_params: HashMap<Symbol, Type>,
    /// The moves at each `break` and `continue` of the loops being checked, innermost last.
    pub(crate) loop_exits: Vec<LoopExits>,
    /// Every borrower each variable has had in the function, kept after the borrow ends: a
    /// drop point is after the last use of all of them.
    borrowers_ever: HashMap<String, HashSet<String>>,
    /// Where each owned tensor is dropped. See `check/drops.rs`.
    pub(crate) drops: crate::hir::check::drops::DropFrames,
    /// What each view variable (`let r = q[i]`) borrows. Reset per function.
    pub(crate) views: HashMap<Symbol, crate::hir::check::views::View>,
}

/// What a loop's `break`s carry out of it and its `continue`s carry into its next pass: the
/// moved marks, joined over every such statement.
#[derive(Default)]
pub(crate) struct LoopExits {
    pub(crate) at_break: Option<Vec<HashSet<String>>>,
    pub(crate) at_continue: Option<Vec<HashSet<String>>>,
}

impl Default for BorrowCx {
    fn default() -> Self {
        Self {
            active_borrows: HashMap::new(),
            block_liveness: Vec::new(),
            current_stmt_idx: Vec::new(),
            skip_borrow_check: false,
            // One scope so `consume`'s `moved_vars.last_mut()` is always `Some` at top level.
            moved_vars: vec![HashSet::new()],
            mutable_vars: vec![HashSet::new()],
            ref_provenance: HashMap::new(),
            current_params: HashMap::new(),
            loop_exits: Vec::new(),
            views: HashMap::new(),
            borrowers_ever: HashMap::new(),
            drops: Default::default(),
        }
    }
}

impl BorrowCx {
    // --- NLL block liveness ---

    /// Enter a block: install its precomputed last-use map and start at statement 0.
    pub(crate) fn enter_block(&mut self, liveness: HashMap<Symbol, usize>) {
        self.block_liveness.push(liveness);
        self.current_stmt_idx.push(0);
    }

    /// Advance the current block's cursor to statement `i`.
    pub(crate) fn set_stmt(&mut self, i: usize) {
        if let Some(slot) = self.current_stmt_idx.last_mut() {
            *slot = i;
        }
    }

    /// Leave the current block, discarding its liveness and cursor.
    pub(crate) fn exit_block(&mut self) {
        self.block_liveness.pop();
        self.current_stmt_idx.pop();
    }

    /// Whether `name` is read at all in the innermost block.
    ///
    /// A binding nobody reads has no last use, so it is absent from the block's liveness map. The
    /// lowering releases such a tile where it is made -- nothing can observe it -- and the capacity
    /// check needs the same answer to agree with what runs. Unknown means read, which keeps the
    /// tile resident and is the conservative direction.
    pub(crate) fn is_variable_ever_read(&self, name: &str) -> bool {
        match self.block_liveness.last() {
            Some(liveness) => liveness.contains_key(&Symbol::from(name)),
            None => true,
        }
    }

    /// Whether `name` is still read at a statement *after* the one currently being checked, in the
    /// innermost block. The NLL predicate the dead-borrow sweep turns on.
    fn is_variable_used_after(&self, name: &str) -> bool {
        if let (Some(liveness), Some(&current_idx)) =
            (self.block_liveness.last(), self.current_stmt_idx.last())
        {
            return liveness
                .get(name)
                .map(|&u| u > current_idx)
                .unwrap_or(false);
        }
        false
    }

    // --- borrow records ---

    /// The **only** way to read `base`'s borrow records for a conflict check. Sweeps dead borrows (NLL)
    /// first — dropping every record whose borrower local is no longer used past the current statement —
    /// so no caller can observe a stale record. Removing records can only *remove* diagnostics, never add
    /// an unsound accept (the #269/#276 safety argument).
    pub(crate) fn live_borrows(&mut self, base: &str) -> &[BorrowRecord] {
        self.sweep(base);
        self.active_borrows
            .get(base)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// NLL dead-borrow cleanup for `base`. Private — folded into `live_borrows` so it is never skipped.
    fn sweep(&mut self, base: &str) {
        let mut dead_borrowers = HashSet::new();
        if let Some(borrows) = self.active_borrows.get(base) {
            for b in borrows {
                if let Some(borrower) = &b.borrower_name {
                    if !self.is_variable_used_after(borrower) {
                        dead_borrowers.insert(borrower.clone());
                    }
                }
            }
        }
        if dead_borrowers.is_empty() {
            return;
        }
        if let Some(borrows) = self.active_borrows.get_mut(base) {
            borrows.retain(|b| match &b.borrower_name {
                Some(borrower) => !dead_borrowers.contains(borrower),
                None => true,
            });
        }
    }

    /// Each variable `borrower` holds a borrow of, with the path inside it: what `rq` points
    /// at after `let rq = &q.f`.
    pub(crate) fn borrowed_by(&self, borrower: &str) -> Vec<(String, Vec<String>)> {
        let mut found = Vec::new();
        for (base, records) in &self.active_borrows {
            for r in records {
                if r.borrower_name.as_deref() == Some(borrower) {
                    found.push((base.to_string(), r.path.clone()));
                }
            }
        }
        found.sort();
        found
    }

    /// `to = from`: `to` now holds every borrow `from` holds, so each stays alive while either
    /// is used. A reference or a closure copied to another variable keeps what it points at
    /// borrowed.
    pub(crate) fn copy_borrows(&mut self, from: &str, to: &str, scope_depth: usize) {
        for records in self.active_borrows.values_mut() {
            let copies: Vec<BorrowRecord> = records
                .iter()
                .filter(|r| r.borrower_name.as_deref() == Some(from))
                .map(|r| BorrowRecord {
                    borrower_name: Some(to.to_string()),
                    scope_depth,
                    ..r.clone()
                })
                .collect();
            records.extend(copies);
        }
        for borrowers in self.borrowers_ever.values_mut() {
            if borrowers.contains(from) {
                borrowers.insert(to.to_string());
            }
        }
    }

    /// Every variable that has borrowed `base` in this function.
    pub(crate) fn borrowers_of(&self, base: &str) -> Vec<String> {
        self.borrowers_ever
            .get(base)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Start a function's list of borrowers, returning the enclosing function's.
    pub(crate) fn take_borrowers(&mut self) -> HashMap<String, HashSet<String>> {
        std::mem::take(&mut self.borrowers_ever)
    }

    pub(crate) fn restore_borrowers(&mut self, saved: HashMap<String, HashSet<String>>) {
        self.borrowers_ever = saved;
    }

    /// Record a new borrow of `base`.
    pub(crate) fn record(&mut self, base: &str, record: BorrowRecord) {
        if let Some(b) = &record.borrower_name {
            self.borrowers_ever
                .entry(base.to_string())
                .or_default()
                .insert(b.clone());
        }
        self.active_borrows
            .entry(base.to_string().into())
            .or_default()
            .push(record);
    }

    // --- scope + save/restore bookkeeping ---

    /// Lexical-lifetime cleanup at scope exit: drop every record borrowed in a scope at or below
    /// `depth` (the depth being popped).
    pub(crate) fn retain_scope(&mut self, depth: usize) {
        for borrows in self.active_borrows.values_mut() {
            borrows.retain(|b| b.scope_depth < depth);
        }
    }

    /// Clone the whole borrow table (a speculative/expression check restores it afterwards).
    pub(crate) fn snapshot(&self) -> HashMap<Symbol, Vec<BorrowRecord>> {
        self.active_borrows.clone()
    }

    /// Take (and clear) the whole table — isolates a callee's borrows from the caller's (#268), the
    /// caller restoring afterwards.
    pub(crate) fn take(&mut self) -> HashMap<Symbol, Vec<BorrowRecord>> {
        std::mem::take(&mut self.active_borrows)
    }

    /// Reinstate a table previously produced by [`Self::snapshot`] or [`Self::take`].
    pub(crate) fn restore(&mut self, table: HashMap<Symbol, Vec<BorrowRecord>>) {
        self.active_borrows = table;
    }

    /// Clone just `base`'s pre-call records — the function-call reborrow bookkeeping snapshots each
    /// distinct argument base so it can be selectively reverted after the call.
    pub(crate) fn snapshot_base(&self, base: &str) -> Option<Vec<BorrowRecord>> {
        self.active_borrows.get(base).cloned()
    }

    /// Selective revert for a non-deriving call argument: keep only the records present *before* the
    /// call (`prev`), dropping the ones this call added, and remove the entry entirely if now empty.
    /// Restoring the raw snapshot instead would resurrect borrows the argument loop legitimately
    /// NLL-released.
    pub(crate) fn retain_present(&mut self, base: &str, prev: &[BorrowRecord]) {
        if let Some(list) = self.active_borrows.get_mut(base) {
            list.retain(|r| prev.contains(r));
            if list.is_empty() {
                self.active_borrows.remove(base);
            }
        }
    }

    /// The records added since `before`, a [`Self::snapshot`].
    pub(crate) fn added_since(
        &self,
        before: &HashMap<Symbol, Vec<BorrowRecord>>,
    ) -> Vec<(Symbol, BorrowRecord)> {
        let mut added = Vec::new();
        for (base, list) in &self.active_borrows {
            let old = before.get(base).map(Vec::as_slice).unwrap_or(&[]);
            for r in list.iter().filter(|r| !old.contains(r)) {
                added.push((base.clone(), r.clone()));
            }
        }
        added
    }

    /// Drop each of `records`, as [`Self::added_since`] returned them.
    pub(crate) fn forget(&mut self, records: &[(Symbol, BorrowRecord)]) {
        for (base, record) in records {
            if let Some(list) = self.active_borrows.get_mut(base) {
                list.retain(|r| r != record);
                if list.is_empty() {
                    self.active_borrows.remove(base);
                }
            }
        }
    }
}
