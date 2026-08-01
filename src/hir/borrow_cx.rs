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

use crate::hir::env::BorrowRecord;
use crate::symbol::Symbol;
use std::collections::{HashMap, HashSet};

/// The active borrow table plus the NLL liveness that keeps it honest. See the module docs.
#[derive(Default)]
pub(crate) struct BorrowCx {
    /// `variable -> its borrow records`. **Private**: readable for a conflict check only via
    /// [`Self::live_borrows`], which sweeps dead borrows before returning.
    active_borrows: HashMap<Symbol, Vec<BorrowRecord>>,
    /// Per-block last-use index of each local (NLL). The innermost block is `.last()`.
    block_liveness: Vec<HashMap<Symbol, usize>>,
    /// The statement index currently being checked, one entry per open block. Innermost is `.last()`.
    current_stmt_idx: Vec<usize>,
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

    /// Record a new borrow of `base`.
    pub(crate) fn record(&mut self, base: &str, record: BorrowRecord) {
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
}
