//! Borrow-checking state whose coherence the frontend depends on: the active borrow records and the
//! per-block liveness that decides which of them are dead (NLL). The rest of the frontend reads
//! borrow records only through this module.
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
    /// The borrow table from before each expression statement around this point, innermost
    /// last. The end of each statement puts its table back.
    pub(crate) before_statements: Vec<HashMap<Symbol, Vec<BorrowRecord>>>,
    /// What each open block names and declares (NLL). The innermost block is `.last()`.
    block_liveness: Vec<BlockUses>,
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
    /// The names that a `match` on a reference bound to a part of the value behind it, one set
    /// per lexical scope, kept in step with `moved_vars`. They do not own their values.
    pub(crate) bound_behind_ref: Vec<HashSet<String>>,
    /// Provenance of each reference-typed *local* binding (#243): does it root in caller memory
    /// (`External`) or a function-local slot (`Local`)? Reset per function.
    pub(crate) ref_provenance: HashMap<Symbol, RefProvenance>,
    /// For each local that holds a `&mut` to something that can hold a reference: where a
    /// write through it lands. `Local(depth)` is the block of the variable it points at, and
    /// `External` is memory the caller owns. Reset per function.
    pub(crate) store_targets: HashMap<Symbol, RefProvenance>,
    /// For each local bound to `&mut v`: the variable `v`, and the block depth of the local.
    /// A store through the local is then a store into `v`. Reset per function.
    pub(crate) store_target_vars: HashMap<Symbol, (Symbol, usize)>,
    /// Parameters (name -> declared type) of the function currently being checked (#243). Reset per
    /// function; lets the return-escape analysis tell a caller-owned reference parameter apart from a
    /// local binding of the same reference type.
    pub(crate) current_params: HashMap<Symbol, Type>,
    /// The moves at each `break` and `continue` of the loops being checked, innermost last.
    pub(crate) loop_exits: Vec<LoopExits>,
    /// How many of `loop_exits` are around the closure being checked. A `break` or `continue`
    /// in the closure needs a loop inside the closure.
    pub(crate) loops_outside_the_closure: usize,
    /// Every borrower each variable has had in the function, kept after the borrow ends: a
    /// drop point is after the last use of all of them.
    borrowers_ever: HashMap<String, HashSet<String>>,
    /// The borrows of variables that a newer `let` of the same name hides: the depth of the
    /// new variable's scope, the name, and the records.
    shadowed: Vec<(usize, Symbol, Vec<BorrowRecord>)>,
    /// Where each owned tensor is dropped. See `check/drops.rs`.
    pub(crate) drops: crate::hir::check::drops::DropFrames,
    /// The borrows that an assignment stored in its holder during the statement being checked,
    /// `w.r = a.get_ref(0)` inside a call's argument. A call whose result goes to another part of
    /// that holder, `w.n = f(..)`, keeps them.
    stored: Vec<(Symbol, BorrowRecord)>,
    /// The tasks that have started and that nothing has joined yet. A task holds its borrows
    /// until then, also where no statement names it. See `check/tasks.rs`.
    running_tasks: HashSet<String>,
    /// What each view variable (`let r = q[i]`) borrows. Reset per function.
    pub(crate) views: HashMap<Symbol, crate::hir::check::views::View>,
    /// The tensors each view-making `let` may be part of, keyed by the `let`'s position. The
    /// traffic count reads it after the `let`'s scope has closed.
    pub(crate) view_owners_by_let: HashMap<crate::syntax::Span, Vec<String>>,
}

/// What the borrow checker needs to know about a block to decide when a borrow made in or
/// before it ends.
#[derive(Default)]
pub(crate) struct BlockUses {
    /// The index of the last statement in the block that names each variable.
    pub(crate) last_use: HashMap<Symbol, usize>,
    /// The variables the block declares. Nothing outside the block can name them.
    pub(crate) declared: HashSet<Symbol>,
    /// A loop body runs again after its last statement.
    pub(crate) is_loop: bool,
    /// For each statement, and the block's value after them: the variables it reads, and the
    /// variable it gives a new value without reading it (`r = &y`, `let r = ..`).
    pub(crate) reads: Vec<HashSet<Symbol>>,
    pub(crate) assigns: Vec<Option<Symbol>>,
}

/// What a loop's `break`s carry out of it and its `continue`s carry into its next pass: the
/// moved marks, joined over every such statement.
#[derive(Default)]
pub(crate) struct LoopExits {
    pub(crate) at_break: Option<Vec<HashSet<String>>>,
    pub(crate) at_continue: Option<Vec<HashSet<String>>>,
    /// What the function changed before the loop. While the loop is checked, `changed_vars`
    /// holds only what the loop changes.
    pub(crate) changed_before: HashSet<String>,
}

impl Default for BorrowCx {
    fn default() -> Self {
        Self {
            active_borrows: HashMap::new(),
            before_statements: Vec::new(),
            block_liveness: Vec::new(),
            current_stmt_idx: Vec::new(),
            skip_borrow_check: false,
            // One scope so `consume`'s `moved_vars.last_mut()` is always `Some` at top level.
            moved_vars: vec![HashSet::new()],
            mutable_vars: vec![HashSet::new()],
            bound_behind_ref: vec![HashSet::new()],
            ref_provenance: HashMap::new(),
            store_targets: HashMap::new(),
            store_target_vars: HashMap::new(),
            current_params: HashMap::new(),
            loop_exits: Vec::new(),
            loops_outside_the_closure: 0,
            views: HashMap::new(),
            view_owners_by_let: HashMap::new(),
            shadowed: Vec::new(),
            borrowers_ever: HashMap::new(),
            drops: Default::default(),
            running_tasks: HashSet::new(),
            stored: Vec::new(),
        }
    }
}

impl BorrowCx {
    // --- NLL block liveness ---

    /// Enter a block: install what it names and declares, and start at statement 0.
    pub(crate) fn enter_block(&mut self, uses: BlockUses) {
        self.block_liveness.push(uses);
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
            Some(block) => block.last_use.contains_key(&Symbol::from(name)),
            None => true,
        }
    }

    /// Whether `name` is still used after the statement being checked. The NLL predicate the
    /// dead-borrow sweep turns on.
    ///
    /// The blocks are searched from the innermost out. A block that names `name` in a later
    /// statement answers yes. A block that declares `name` ends the search, since no block
    /// around it can see that variable. A loop body that names a variable from outside it
    /// answers yes too, because its next pass names it again.
    pub(crate) fn is_variable_used_after(&self, name: &str) -> bool {
        if self.is_running_task(name) || self.drops.drop_reads_borrows(name) {
            return true;
        }
        self.used_after_in(name, self.block_liveness.len())
    }

    /// [`Self::is_variable_used_after`], asking only the outermost `blocks` blocks.
    fn used_after_in(&self, name: &str, blocks: usize) -> bool {
        let blocks = self.block_liveness[..blocks]
            .iter()
            .zip(&self.current_stmt_idx);
        for (block, &current_idx) in blocks.rev() {
            let last_use = block.last_use.get(name);
            if last_use.is_some_and(|&u| u > current_idx) {
                return true;
            }
            if block.declared.contains(name) {
                return false;
            }
            if block.is_loop && last_use.is_some() {
                return true;
            }
        }
        false
    }

    /// Whether the borrow that the previous pass of the loop with body block `root` left in
    /// `name` may still be read. It is, unless `name` gets a new value before any read: later
    /// in this pass, or in the next pass before the statement being checked. A statement that
    /// reads `name` anywhere counts as a read, so a branch is never missed.
    fn carried_borrow_live(&self, name: &str, root: usize) -> bool {
        if self.is_running_task(name) {
            return true;
        }
        let decide = |block: &BlockUses, from: usize, to: usize| -> Option<bool> {
            for i in from..to {
                if block.reads.get(i).is_some_and(|r| r.contains(name)) {
                    return Some(true);
                }
                if block
                    .assigns
                    .get(i)
                    .is_some_and(|a| a.as_deref() == Some(name))
                {
                    return Some(false);
                }
            }
            None
        };
        // This pass, from the statement being checked out to the end of the loop body.
        for b in (root..self.block_liveness.len()).rev() {
            let block = &self.block_liveness[b];
            let at = self.current_stmt_idx[b];
            if block.reads.is_empty() {
                // No statements of its own, such as the head of an `if`: only what it names.
                if block.last_use.get(name).is_some_and(|&u| u >= at) {
                    return true;
                }
            } else if block.reads.get(at).is_some_and(|r| r.contains(name)) {
                return true;
            } else if let Some(live) = decide(block, at + 1, block.reads.len()) {
                return live;
            }
            if b != root && block.declared.contains(name) {
                return false;
            }
        }
        // The loop may end here, and code after it read the borrow.
        if self.used_after_in(name, root) {
            return true;
        }
        // The next pass, up to the statement being checked.
        let at = self.current_stmt_idx[root];
        decide(&self.block_liveness[root], 0, at + 1).unwrap_or(false)
    }

    /// The number of blocks open now. The innermost is `self.block_count() - 1`.
    pub(crate) fn block_count(&self) -> usize {
        self.block_liveness.len()
    }

    /// Record `seeds`, the borrows that the previous pass of a loop may have left.
    pub(crate) fn seed_carried(&mut self, seeds: Vec<(Symbol, BorrowRecord)>) {
        for (base, record) in seeds {
            self.active_borrows.entry(base).or_default().push(record);
        }
    }

    /// Forget the borrows seeded for the loop with body block `root`, at the end of its body.
    pub(crate) fn forget_carried(&mut self, root: usize) {
        for list in self.active_borrows.values_mut() {
            list.retain(|r| r.carried_from != Some(root));
        }
        self.active_borrows.retain(|_, list| !list.is_empty());
    }

    /// A task started, and holds its borrows until [`Self::end_task`].
    pub(crate) fn start_task(&mut self, task: &str) {
        self.running_tasks.insert(task.to_string());
    }

    /// The task is joined, so its borrows end.
    pub(crate) fn end_task(&mut self, task: &str) {
        self.running_tasks.remove(task);
    }

    /// Whether `name` is a running task, or the hold of a running task on what it copied.
    pub(crate) fn is_running_task(&self, name: &str) -> bool {
        let task = name.split_once('/').map_or(name, |(task, _)| task);
        self.running_tasks.contains(task)
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
        let Some(borrows) = self.active_borrows.get(base) else {
            return;
        };
        let dead: Vec<bool> = borrows
            .iter()
            .map(|b| match (&b.borrower_name, b.carried_from) {
                (Some(borrower), Some(root)) => !self.carried_borrow_live(borrower, root),
                (Some(borrower), None) => !self.is_variable_used_after(borrower),
                (None, _) => false,
            })
            .collect();
        if !dead.contains(&true) {
            return;
        }
        let mut dead = dead.into_iter();
        if let Some(borrows) = self.active_borrows.get_mut(base) {
            borrows.retain(|_| !dead.next().unwrap_or(false));
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

    /// Ends every borrow that `holder` holds, also in the tables that the statements around
    /// this point put back when they end: the wait of a copy token, which ends the copy.
    pub(crate) fn forget_borrows_of(&mut self, holder: &str) {
        let held = |r: &BorrowRecord| r.borrower_name.as_deref() == Some(holder);
        for table in std::iter::once(&mut self.active_borrows).chain(&mut self.before_statements) {
            for list in table.values_mut() {
                list.retain(|r| !held(r));
            }
            table.retain(|_, list| !list.is_empty());
        }
    }

    /// Ends one borrow of `base` that no variable holds: the borrow of a closure literal whose
    /// call ended.
    pub(crate) fn forget_unheld_borrow(&mut self, base: &str, is_mut: bool) {
        let list = self.active_borrows.get_mut(base);
        let at = list.as_ref().and_then(|l| {
            l.iter().position(|r| {
                r.borrower_name.is_none() && r.carried_from.is_none() && r.is_mut == is_mut
            })
        });
        if let (Some(list), Some(at)) = (list, at) {
            list.remove(at);
        }
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

    /// `to` may point where anything that `from` reaches points: `to` holds every borrow that
    /// `from` holds, and every borrow that a variable `from` borrows holds, and so on. After
    /// `let x = it.next()`, `x` holds the borrow of the vector that `it` holds.
    pub(crate) fn copy_borrows_through(&mut self, from: &str, to: &str, scope_depth: usize) {
        let mut reached: Vec<String> = Vec::new();
        let mut pending = vec![from.to_string()];
        while let Some(var) = pending.pop() {
            if var == to || reached.contains(&var) {
                continue;
            }
            pending.extend(self.borrowed_by(&var).into_iter().map(|(base, _)| base));
            reached.push(var);
        }
        let held_by_reached = |r: &BorrowRecord| {
            r.borrower_name
                .as_deref()
                .is_some_and(|b| reached.iter().any(|v| v == b))
        };
        for records in self.active_borrows.values_mut() {
            // Two variables `from` reaches can hold one borrow. `to` gets one copy of it.
            let mut copies: Vec<BorrowRecord> = Vec::new();
            for r in records.iter().filter(|r| held_by_reached(r)) {
                let same = |c: &BorrowRecord| {
                    c.borrower_name.as_deref() == Some(to)
                        && c.is_mut == r.is_mut
                        && c.path == r.path
                };
                if !records.iter().any(same) && !copies.iter().any(same) {
                    copies.push(BorrowRecord {
                        borrower_name: Some(to.to_string()),
                        scope_depth,
                        ..r.clone()
                    });
                }
            }
            records.extend(copies);
        }
        for borrowers in self.borrowers_ever.values_mut() {
            if reached.iter().any(|v| borrowers.contains(v)) {
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

    pub(crate) fn has_borrowed(&self, base: &str, name: &str) -> bool {
        self.borrowers_ever
            .get(base)
            .is_some_and(|s| s.contains(name))
    }

    /// `name` was never a borrower of `base` after all, such as `s` in `let s = f(&a)` when `f`
    /// returns a number.
    pub(crate) fn forget_borrower(&mut self, base: &str, name: &str) {
        if let Some(s) = self.borrowers_ever.get_mut(base) {
            s.remove(name);
        }
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
    /// `depth` (the depth being popped), unless a variable still in scope holds it, such as
    /// `outer` after `if c { outer = W { r : &x }; }`. `holder_depth` gives the depth of such a
    /// variable's block, where the record then moves, to last as long as the variable does.
    pub(crate) fn retain_scope(
        &mut self,
        depth: usize,
        holder_depth: impl Fn(&str) -> Option<usize>,
    ) {
        for borrows in self.active_borrows.values_mut() {
            borrows.retain_mut(|b| {
                if b.scope_depth < depth {
                    return true;
                }
                match b.borrower_name.as_deref().and_then(&holder_depth) {
                    Some(held_at) => {
                        b.scope_depth = held_at;
                        true
                    }
                    None => false,
                }
            });
        }
    }

    /// Set the records of `name` aside while a new variable of that name, declared at `depth`,
    /// is in scope.
    pub(crate) fn shadow(&mut self, name: &str, depth: usize) {
        if let Some(records) = self.active_borrows.remove(name) {
            self.shadowed.push((depth, name.into(), records));
        }
    }

    /// The scope at `depth` ends: put back the records that its variables hid.
    pub(crate) fn unshadow(&mut self, depth: usize) {
        while self.shadowed.last().is_some_and(|(d, _, _)| *d >= depth) {
            let (_, name, records) = self.shadowed.pop().expect("checked above");
            self.active_borrows.entry(name).or_default().extend(records);
        }
    }

    pub(crate) fn take_shadowed(&mut self) -> Vec<(usize, Symbol, Vec<BorrowRecord>)> {
        std::mem::take(&mut self.shadowed)
    }

    pub(crate) fn restore_shadowed(&mut self, shadowed: Vec<(usize, Symbol, Vec<BorrowRecord>)>) {
        self.shadowed = shadowed;
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
    /// NLL-released. A record held by a variable other than `call_target`, the variable the call's
    /// result goes to, stays: an argument such as `if c { s = &x; 0 } else { 0 }` made it for `s`.
    pub(crate) fn retain_present(
        &mut self,
        base: &str,
        prev: &[BorrowRecord],
        call_target: Option<&str>,
    ) {
        // Each record an assignment stored stays once. The holder's block may have moved it
        // since, so its depth is not compared.
        let mut stored: Vec<&BorrowRecord> = self
            .stored
            .iter()
            .filter(|(b, _)| b.as_ref() == base)
            .map(|(_, s)| s)
            .collect();
        if let Some(list) = self.active_borrows.get_mut(base) {
            list.retain(|r| {
                if prev.contains(r)
                    || r.borrower_name.is_some() && r.borrower_name.as_deref() != call_target
                {
                    return true;
                }
                let same = |s: &&BorrowRecord| {
                    s.borrower_name == r.borrower_name && s.path == r.path && s.is_mut == r.is_mut
                };
                match stored.iter().position(same) {
                    Some(i) => {
                        stored.swap_remove(i);
                        true
                    }
                    None => false,
                }
            });
            if list.is_empty() {
                self.active_borrows.remove(base);
            }
        }
    }

    /// [`Self::added_since`], counting copies: a record that is now in the table twice, and was
    /// once before, was added once.
    pub(crate) fn added_copies_since(
        &self,
        before: &HashMap<Symbol, Vec<BorrowRecord>>,
    ) -> Vec<(Symbol, BorrowRecord)> {
        let mut added = Vec::new();
        for (base, list) in &self.active_borrows {
            let mut old: Vec<&BorrowRecord> = before.get(base).into_iter().flatten().collect();
            for r in list {
                match old.iter().position(|o| *o == r) {
                    Some(i) => {
                        old.swap_remove(i);
                    }
                    None => added.push((base.clone(), r.clone())),
                }
            }
        }
        added
    }

    /// Note `records` as stored by an assignment in the statement being checked.
    pub(crate) fn note_stored(&mut self, records: Vec<(Symbol, BorrowRecord)>) {
        self.stored.extend(records);
    }

    /// The statement ends, and with it what [`Self::note_stored`] noted.
    pub(crate) fn clear_stored(&mut self) {
        self.stored.clear();
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

    /// Start the next branch of an `if` or a `match` from `before`, the borrows before its first
    /// branch. Answers what the branch just checked left, for [`Self::add_back`] after the last
    /// branch: a borrow that one branch ended still holds on the path through another.
    pub(crate) fn next_branch(
        &mut self,
        before: &HashMap<Symbol, Vec<BorrowRecord>>,
    ) -> Vec<(Symbol, BorrowRecord)> {
        std::mem::replace(&mut self.active_borrows, before.clone())
            .into_iter()
            .flat_map(|(base, list)| list.into_iter().map(move |r| (base.clone(), r)))
            .collect()
    }

    /// Turn each of `records` into a shared borrow, and answer the shared ones: a mutable borrow
    /// that is only reserved for now, so the place can still be read but not borrowed mutably
    /// again (Rust's two-phase borrow).
    pub(crate) fn reserve(
        &mut self,
        records: &[(Symbol, BorrowRecord)],
    ) -> Vec<(Symbol, BorrowRecord)> {
        let mut shared = Vec::new();
        for (base, record) in records {
            if let Some(list) = self.active_borrows.get_mut(base) {
                for r in list.iter_mut().filter(|r| *r == record) {
                    r.is_mut = false;
                    shared.push((base.clone(), r.clone()));
                }
            }
        }
        shared
    }

    /// Add back each of `records`, as [`Self::added_since`] returned them, that is not there
    /// now: the borrows one branch made, joined with the other branches'.
    pub(crate) fn add_back(&mut self, records: Vec<(Symbol, BorrowRecord)>) {
        for (base, record) in records {
            let present = self
                .active_borrows
                .get(&base)
                .is_some_and(|list| list.contains(&record));
            if !present {
                self.record(&base, record);
            }
        }
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
