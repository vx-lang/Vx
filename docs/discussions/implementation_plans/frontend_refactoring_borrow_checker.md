# Frontend refactoring plan

**Status:** **R1 landed** (`522a2ae8` — `BorrowCx` encapsulates the borrow state; the NLL-sweep-before-read
invariant is now enforced by module privacy, not convention). R2–R5 open. Tracked as
[#279](https://github.com/hiraditya/Vx/issues/279).
**Motivation:** the borrow checker took ~8 rounds of fixes (#243, #268, #269, #275, #276, #277, #278) across
several months. The recurring cost was not that borrow checking is conceptually hard; it was that the
frontend has no *chokepoint* for the invariants those fixes maintain, so each round had to rediscover every
site that needed the same treatment. This plan targets that structural cause.

**Relates to:** [#279](https://github.com/hiraditya/Vx/issues/279) (this plan's tracking issue) ·
[#197](https://github.com/hiraditya/Vx/issues/197) (flat pipeline epic) ·
[#243](https://github.com/hiraditya/Vx/issues/243) (borrow-checker matrix) ·
[#276](https://github.com/hiraditya/Vx/issues/276) (the NLL-sweep-at-access bug this plan would prevent by
construction)

______________________________________________________________________

## 1. The evidence

Measured, not estimated.

| Signal | Value | Where |
|---|---|---|
| Largest frontend file | **5229 lines**, 63 functions | `src/hir/expr.rs` |
| Its commit count (following renames) | **236** — the most-churned file in the compiler | `sema.rs` → `sema/expr.rs` → `hir/expr.rs` |
| `TypeChecker` fields | **44**, spanning ~8 unrelated concerns | `src/hir/env.rs:203` |
| Largest single function | **660 lines** (`check_functioncall_expr`) | `src/hir/expr.rs:2249` |
| Next four | 322 / 320 / 245 / 243 lines | `check_methodcall_expr`, `lower_to_type_id`, `check_transfer_expr`, `check_identifier_expr` |
| `silent` / `consume` flag threading | 183 / 97 occurrences, 36 signatures | `src/hir/` |
| Borrow-state touch sites | 36 across 4 files | `active_borrows`, `skip_borrow_check`, `BorrowRecord` |

Churn and size agree on the same file. That is the strongest available signal that `hir/expr.rs` is where the
debt actually sits — it is both the biggest thing and the thing we keep having to change.

## 2. The structural defect, concretely

`sweep_dead_borrows` (`src/hir/expr.rs:4298`) implements the NLL liveness sweep: drop borrow records whose
borrower is dead at the current statement. **Correctness requires it to run before every read of
`active_borrows`.** Nothing enforces that.

There are four read sites. Three call the helper (`:562` identifier access, `:3264` member access, `:4346`
reference-arg reborrow). The fourth — `check_borrow_expr` at `:3998-4016` — contains a **hand-copied
duplicate** of the sweep rather than calling it:

```rust
let mut dead_borrowers = std::collections::HashSet::new();
if let Some(borrows) = self.active_borrows.get(&*name) {
    for b in borrows.iter() {
        if let Some(borrower) = &b.borrower_name {
            if !self.is_variable_used_after(borrower) {
                dead_borrowers.insert(borrower.clone());
            }
        }
    }
}
```

The copy has already drifted: it lacks the `if dead_borrowers.is_empty() { return; }` early-out the helper
has. Benign today, but drift is the point — two copies of an invariant diverge silently.

**This is exactly what #276 was.** The identifier-access path read `active_borrows` without sweeping, so a
sound program (`let r = &mut p.x; *r = 42; return p.x;`) was rejected. The fix was to add the sweep at one
more site. The *next* new access path will have the same bug, because "remember to sweep" is a convention
enforced by nothing.

The same shape produced #275's finding that the checker and lowerer had **two independent derivations** of
which memory an lvalue names. That one was fixed properly — `src/hir/places.rs` (46 lines, 2 pure functions,
7 call sites) is now the single source both call. **That extraction is the template this plan generalizes**:
it worked precisely because the extracted logic is pure and needs no `TypeChecker` state.

## 3. What `TypeChecker`'s 44 fields actually are

They partition cleanly into concerns that do not talk to each other:

| Concern | Fields |
|---|---|
| **Borrow checking** | `active_borrows`, `skip_borrow_check`, `moved_vars`, `ref_provenance`, `current_params`, `block_liveness`, `current_stmt_idx` |
| Type checking proper | `scopes`, `errors`, `next_id`, `expected_type`, `current_return_type` |
| Topology / memory | `active_topology`, `active_memory`, `transfer_cost_graph`, `memory_placements`, `placement_site`, `allow_cross_topology` |
| Seam verification (z3) | `seam_solver`, `seam_contracts`, `seam_checks`, `seam_check_time`, `solver_init_time`, `verify_seams`, `pending_transfer_relaxed` |
| Generics / monomorphization | `monomorphized_functions`, `pending_topo_vars`, `pending_topo_bindings`, `closure_signatures`, `generated_structs` |
| Comptime eval | `eval_env`, `constraints`, `return_constraints` |
| Closures | `closure_depths`, `closure_captures_stack` — both `#[allow(dead_code)]` |
| Warnings | `used_vars`, `declared_vars` |

The borrow cluster is **7 fields with no dependency on the other 37**. `is_variable_used_after`
(`src/hir/env.rs:384`) reads exactly two of them (`block_liveness`, `current_stmt_idx`). That is a cohesive,
separable unit — which is what makes R1 below tractable rather than a rewrite.

Note also two fields carrying `#[allow(dead_code)]`. Dead state on a hot struct is a maintenance tax with no
payer.

## 4. Constraints any refactor must respect

1. **The parallel architecture.** `TypeChecker` borrows `&'a mut LocalWorkerState` and is constructed
   per-worker. It must stay per-worker and non-shared; no extracted sub-context may introduce shared mutable
   state or a lock. This is the invariant `verify_phase_3_isolation` protects.
1. **The AST is the differential oracle.** The flat path is validated against the AST path. Frontend
   refactoring must not change diagnostics or inferred types — any behavioural delta invalidates the oracle
   for every in-flight flat milestone.
1. **`hir/flatten.rs` (4642 lines) is out of scope here.** It is the other churn hotspot, but it is under
   active development (#242, #278). Refactoring both at once would make every conflict ambiguous between
   "refactor broke it" and "feature broke it".

## 5. The safety net

Refactoring is only sane with a net, and this one is unusually good:

- **410** lib unit tests
- **171** integration tests, incl. the flat-vs-AST differential suite
- **309** `.vx` corpus files (`tests/backend/pass`, `tests/middle_end/{pass,fail}`, `tests/frontend`)
- **20** borrow-checker-specific corpus files, plus the #243 matrix

Crucially the corpus includes `fail/` cases, so a refactor that *loses* a diagnostic is caught, not just one
that crashes. Every phase below is a refactor with **zero intended behavioural change**, so the acceptance
criterion is uniformly: the whole suite passes unchanged, with no test edits.

## 6. Phased plan

Ordered by value-per-risk. Each phase is independently landable and independently valuable — if we stop
after R1 we have still removed the bug class that motivated this.

### R1 — Encapsulate borrow state so the NLL sweep cannot be forgotten — **LANDED (`522a2ae8`)**

Delivered in `src/hir/borrow_cx.rs`: `BorrowCx` owns `active_borrows` (**private**) plus the NLL liveness it
depends on (`block_liveness`, `current_stmt_idx` — the cohesive unit `is_variable_used_after` reads), exposed
only through `live_borrows()`, which sweeps dead borrows first. The hand-copied inline sweep duplicate in
`check_borrow_expr` and the standalone `sweep_dead_borrows` are gone (both fold into one private
`BorrowCx::sweep`); all four conflict-check readers route through `live_borrows`. Behaviour-preserving with
zero test edits. *Scope actually landed:* the three fields carrying the sweep invariant. The other four
fields §3 groups under "borrow checking" (`skip_borrow_check`, `moved_vars`, `ref_provenance`,
`current_params`) carry no sweep invariant, so folding them in is a cosmetic follow-up, tracked in #279.

**The highest-value change in this document.** Move the borrow cluster into a `BorrowCx` struct with
`active_borrows` **private**, exposed only through methods that sweep first:

```rust
impl BorrowCx {
    /// The only way to read borrow records. Sweeps dead borrows (NLL) before returning,
    /// so no caller can observe a stale record.
    fn live_borrows(&mut self, base: &str) -> &[BorrowRecord] { self.sweep(base); ... }
}
```

Then delete the inline duplicate at `:3998-4016` and route all four sites through it.

*Why this first:* it converts "remember to sweep" from a convention into a type-system guarantee. #276 and
its successors become **unrepresentable** rather than merely fixed. It is also small, local, and does not
move any files, so it will not conflict with in-flight work.

**Acceptance:** `active_borrows` has no direct reader outside `BorrowCx`; the duplicate sweep is gone; full
suite green with no test edits.

### R2 — Split `hir/expr.rs` along the dispatch seam

`check_expr_type_flag` (`:129`) is already a clean `match expr { … }` delegating one function per expression
kind. That is a natural, pre-existing seam: each arm's function moves to a submodule with **zero logic
change**.

Proposed split (`src/hir/check/`): `calls.rs` (function/method/indirect/generic instantiation — the largest
cluster), `places.rs`-adjacent `access.rs` (identifier/member/index/borrow/deref), `literals.rs`,
`control.rs` (if/match/closure/range), `transfer.rs` (transfer/spawn/topology/capacity/seam), `autodiff.rs`
(grad/vjp/jvp), `intrinsics.rs`.

*Why after R1:* this produces a large, mechanical diff. Landing it on top of a settled borrow encapsulation
means any post-split failure is attributable to the move, not to entangled semantics.

**Acceptance:** no file over ~800 lines; `git diff --stat` shows moves only (verify with
`git log --follow -M`); full suite green.

### R3 — Retire the `(consume, silent)` boolean pair

183 `silent` / 97 `consume` occurrences across 36 signatures. Two positional booleans at every call is the
classic unreadable-call-site smell (`check_expr_type_flag(e, false, true)` — which is which?).

`silent` is really "suppress diagnostics for a speculative check" — better modelled as a **diagnostics sink
swap** (check into a scratch sink, discard it) than as a flag every function must thread and honour. `consume`
is a move-semantics question that belongs to the borrow/move context R1 introduces.

*Why after R2:* the split makes it obvious which functions genuinely need each flag; several likely ignore
one (`_silent`, `_consume` already appear in signatures).

**Acceptance:** the boolean pair is gone from the dispatch signature; speculative checks provably emit no
diagnostics (a test that asserts the error count is unchanged after a speculative probe).

### R4 — Decompose the oversized functions

`check_functioncall_expr` at **660 lines** is the worst; then `check_methodcall_expr` (322) and
`lower_to_type_id` (320). These are where overload resolution, generic deduction, intrinsic dispatch,
provenance and reborrow tracking all interleave — the hardest code in the frontend to change confidently.

*Why last among the code phases:* R2 and R3 remove much of the incidental bulk (flag threading, unrelated
neighbours) and will make the real structure visible. Decomposing before that risks carving along the wrong
joints.

### R5 — Split the remaining `TypeChecker` concerns (optional)

Apply R1's pattern to the seam/z3, generics, and memory clusters. Lower value — those fields are touched by
far fewer sites and have not generated a recurring bug class — so this is genuinely optional. Also drop the
two `#[allow(dead_code)]` closure fields if they are still unused.

## 7. What this does *not* propose

- No change to the borrow checker's **algorithm**. The per-parameter provenance design
  (`borrow_checker_parameter_provenance.md`) is separate research; this plan only makes the existing
  algorithm's invariants enforceable.
- No `flatten.rs` work (§4.3).
- No new abstraction layers or traits for their own sake. Every phase either deletes duplication or moves
  code along a seam that already exists.

## 8. Suggested sequencing against feature work

R1 is small and conflict-free — land it next. R2 is the disruptive one; it wants a quiet window with no
in-flight frontend branch, so it should land immediately after a milestone closes rather than mid-feature.
R3–R5 are incremental and can interleave with normal work.
