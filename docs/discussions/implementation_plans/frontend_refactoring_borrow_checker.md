# Frontend refactoring plan

**Status:** **R1 + R2 + R3 + R4 landed; R5 optional.** R1 (`522a2ae8`, + follow-up): `BorrowCx` encapsulates the
borrow state; the NLL-sweep-before-read invariant is enforced by module privacy, not convention. The follow-up
folded the remaining four borrow-cluster fields (`skip_borrow_check`, `moved_vars`, `ref_provenance`,
`current_params`) in too, so all borrow/move state lives on `BorrowCx`. R2 (`c657a442`, `f8cb71de`,
`84dc1604`): `hir/expr.rs` split along the dispatch seam into seven `check/` submodules, **5165 → 545 lines**,
zero logic change. R3: the `silent` half retired — the plan's sink-swap was unsound (`silent` also changes
if/match result types and gates `consume`'s scope/move mutations), so `silent` became a `speculating` **field**
instead, removing the positional boolean from every checker signature with behavior preserved (§R3). `consume`
stays a parameter (a genuine positional signal, not a mode), as audited. R4: the three largest frontend
god-functions decomposed — `check_functioncall_expr` 652 → ~358, `check_methodcall_expr` 346 → ~195, and
`check_statement` 360 → 39 (eleven helpers extracted; `lower_to_type_id` was already ~33 lines, its plan count
stale) (§R4). R5 optional. Tracked as
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
zero test edits. *Scope landed in two steps:* first the three fields carrying the sweep invariant; then
(**follow-up, landed**) the other four fields §3 groups under "borrow checking" — `skip_borrow_check`,
`moved_vars`, `ref_provenance`, `current_params` — folded in too, so all borrow/move-checking state now lives
on `BorrowCx`. Those four carry no sweep invariant, so they are plain `pub(crate)` fields with direct access
(no new method surface); `BorrowCx` gains a manual `Default` so `moved_vars` keeps its one-scope initial value.
Cosmetic consolidation, behaviour-preserving, zero test edits.

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

### R2 — Split `hir/expr.rs` along the dispatch seam — **LANDED** (`c657a442`, `f8cb71de`, `84dc1604`)

`check_expr_type_flag` was already a clean `match expr { … }` delegating one function per expression kind —
a natural, pre-existing seam. Each arm's function moved to a `src/hir/check/` submodule as an additional
`impl TypeChecker` block, **zero logic change**, in three keep-green commits (autodiff pilot; then
operators/literals/control; then transfer/access/calls).

Delivered split: `autodiff.rs` (grad/vjp/jvp + differentiability), `operators.rs` (binary/relational/
logical/unary + `as`), `literals.rs` (number/array/struct-init/enum-variant/vec!), `control.rs` (if/match/
closure/range/unsafe & comptime blocks), `transfer.rs` (transfer/spawn + topology/memory/capacity/seam),
`access.rs` (identifier/member/index/borrow/deref + provenance/reborrow helpers), `calls.rs` (function/
method/indirect + generic instantiation + intrinsic resolution). Each submodule reaches the shared surface
via `use super::super::*`; methods a sibling or the dispatch calls became `pub(crate)`.

**`hir/expr.rs`: 5165 → 545 lines** — now the dispatch, `check_expr_type`/`check_expr_block`, and the shared
type helpers `is_assignable`/`lower_to_type_id`/`tensor_of`. Verified lib 410 + integration 172 green, no
test edits, clippy clean.

**Acceptance:** met except `calls.rs` (~1735 lines) — because `check_functioncall_expr` alone is ~655; a
single function can't drop below the ~800 target by *moving*. That is **R4**'s job (decompose the oversized
functions), not R2's (one family per file). The dispatch-seam split is complete.

### R3 — Retire the `(consume, silent)` boolean pair — **LANDED** (the `silent` half, as the `speculating` field)

183 `silent` / 97 `consume` occurrences across 36 signatures. Two positional booleans at every call is the
classic unreadable-call-site smell (`check_expr_type_flag(e, false, true)` — which is which?). The original
idea was to model `silent` as a **diagnostics-sink swap** (check into a scratch sink, discard) and to fold
`consume` into the borrow/move context. **A full audit of both flags (Aug 2026) shows that plan is unsound
and revises it.** Findings below; this is the guidance for whoever implements R3.

**1. `silent` is not a diagnostics flag — it gates four different things.** A sink-swap (or any post-hoc
"discard the errors" checkpoint) only covers the first:

- **(a) diagnostics** — `if !silent { self.errors.push / error_with_code / warn }`. The common case; a
  checkpoint could roll these back (`DiagnosticsVec.inner` is `pub`, appends in strict order with no dedup,
  so `inner.truncate(saved_len)` is exact; the 10-error cap doesn't break it).
- **(b) borrow-state side effects** — `!self.skip_borrow_check && !silent` gates the `live_borrows` sweep +
  conflict check in `check_identifier_expr` / `check_memberaccess_expr`; `calls.rs` gates the whole
  reborrow-tracking + `retain_present` block on `!silent`; `track_reference_arg_borrow` early-returns on
  `silent`. Covered by `BorrowCx::snapshot`/`restore` (R1) — but only `active_borrows`, see (d).
- **(c) it changes the RETURNED type** — `check_if_expr` and `check_match_expr` return the placeholder
  `Tensor(F32,[],None)` and skip checking their block when `silent`, instead of the real branch type. A
  post-hoc checkpoint **cannot** reproduce this: it's a *during-check* behavior, not a rollback. Running them
  non-silently yields a different (real) type, which can flow into the outer method-call's generic deduction.
- **(d) it gates `self.consume()`** in `check_identifier_expr` (`if consume && ty.is_linear() && !silent`)
  and `check_closure_expr`. `self.consume` mutates `self.scopes` **and** `self.moved_vars` — **neither is in
  `borrow.snapshot()`** — so a checkpoint that only restores errors+borrow leaks the move-mark, and a later
  real check then fires spurious E4001/E2001 and returns `Type::Unknown`.

**2. There is exactly ONE speculative (`silent = true`) call site:** `check_methodcall_expr`
(`src/hir/check/calls.rs`) re-checks a synthetic lowered `FunctionCall` to recover its return type without
duplicate diagnostics. It has **no** borrow/error isolation today — it relies entirely on `silent`. So the
"183 occurrences" are threading noise: `silent` propagates from the top-level `check_function` (which passes
`false`) and is flipped `true` only here.

**3. `consume` cannot be deleted the way the plan assumed.** It has a single *read* — `check_identifier_expr`,
`if consume && ty.is_linear() && !silent { self.consume(name) }` — and encodes "this is a by-value use, so
move a linear value." It is positionally overridden to `false` at receiver / index-base / borrow-inner /
assignment-LHS / builtin-ref-arg positions (so `x.f`, `&x`, `print(x)`, `lhs = ..` don't move `x`), and
`true` at value positions. Dropping it forces either always-consume (wrongly moves those) or never-consume
(loses all use-after-move / E4001). Removing it means reconstructing that positional signal another way
(derive value-vs-receiver from the parent expression, or a `Usage::{Value,Borrow}` enum) — a real
move-semantics change, not a flag deletion.

**Revised recommendation.** The clean, behavior-preserving win is to turn `silent` from a threaded
*parameter* into a `speculating` **field** on `TypeChecker` (per-worker, no shared state). The one
speculative site does `let prev = self.speculating; self.speculating = true; …; self.speculating = prev;`;
every `!silent` becomes `!self.speculating`. This removes the confusing positional boolean from all ~36
signatures **and preserves all four behaviors verbatim** — (c) and (d) included — because `!self.speculating`
substitutes for `!silent` in place, with nothing rolled back post-hoc. `consume` stays a parameter (it is a
genuine positional signal, not a mode); retiring it is separate, larger move-semantics work. If instead a
*true effect-free speculation* is ever wanted, the checkpoint must additionally snapshot `scopes` +
`moved_vars` and special-case if/match (c), and note that AST rewrites (`*expr = …`), `monomorphized_functions`,
`generated_structs`, `memory_placements`, `used_vars`, and `next_id` are mutated ungated today and would also
leak — they already do under the current single `silent=true` site, so it is not a regression, just not
"effect-free."

**Acceptance (field approach):** `silent` is gone from every signature; the sole speculative site
sets/restores `self.speculating`; full suite green with no test edits (the change is mechanical and
behavior-preserving). A regression test that a speculative probe leaves the error count, the active-borrow
table, **and** `scopes`/`moved_vars` unchanged documents the invariant.

**Implemented.** `speculating: bool` field on `TypeChecker` (per-worker; default `false`); the `silent`
parameter removed from every checker signature; each `!silent` → `!self.speculating`. One refinement the
audit above understated: `silent` had **three** literal setter sites, not just the one speculative flip. The
two "fresh check" entries — `check_expr_type` and `check_block` — passed a hard-coded `silent = false`, and
**both are reachable from *inside* the speculative probe**: a generic function instantiated during the probe
reaches `check_function → check_block`, and an `if`/`match` argument reaches `check_if_expr → check_expr_type(cond)`.
A naive field (flip on at the probe, never forced off) would therefore suppress diagnostics the parameter
emitted in those subtrees. The faithful conversion **forces `speculating = false` (save/restore) at those two
entries** and forces it `true` at the probe; in all non-probe execution both forces are no-ops, so the field's
dynamic scope reproduces the parameter's exactly. Guarded by
`hir::tests::r3_speculating_gates_diagnostics_and_the_fresh_entry_forces_it_off` (probe suppresses; callee
does not clobber the caller's flag; the fresh entry forces-off-then-restores). `consume` stays a parameter, as
audited.

### R4 — Decompose the oversized functions — **LANDED** (`check_functioncall_expr`, `check_methodcall_expr`, `check_statement`)

`check_functioncall_expr` at **660 lines** is the worst; then `check_methodcall_expr` (322) and
`lower_to_type_id` (320). These are where overload resolution, generic deduction, intrinsic dispatch,
provenance and reborrow tracking all interleave — the hardest code in the frontend to change confidently.
(*Correction from implementation:* `lower_to_type_id` is actually ~33 lines — a two-arm AST-type → `TypeId`
hash that does none of that interleaving; its "320" here was a stale/misnamed count. The genuinely largest
oversized function turned out to be `check_statement` at **360**, which was decomposed instead.)

*Why last among the code phases:* R2 and R3 remove much of the incidental bulk (flag threading, unrelated
neighbours) and will make the real structure visible. Decomposing before that risks carving along the wrong
joints.

**Landed — `check_functioncall_expr` 652 → ~358.** Its two biggest, most self-contained joints were pulled
out as pure moves: (1) the #243 reference-argument reborrow bookkeeping that straddled the argument loop —
`prepare_reference_arg_reborrows` (before the loop) + `commit_reference_arg_reborrows` (after), carried by a
`ReborrowPlan` struct, so the "snapshot before, revert after" invariant is a closed prepare → loop → commit
shape rather than logic interleaved with argument checking; and (2) the 144-line `Struct::method(...)`
static-call resolution → `check_static_method_call`. The remaining ~358 lines are the argument-processing
preamble plus the callee-resolution `if let … else if let` chain — long but flat and simple (each arm resolves
one name-kind and checks its arguments), so it reads cleanly as-is. Behavior-preserving, full suite green, no
test edits.

**Landed — `check_methodcall_expr` 346 → ~195.** Same treatment: (1) the impl-block walk that unifies the
receiver and fills the generic `mapping` (plus the #219 debug parity gate) → `resolve_method_in_impls`, a pure
resolver returning `Option<(Function, ImplBlock)>`; and (2) the generic-method instantiation + MethodCall →
FunctionCall rewrite → `instantiate_method_call_rewrite`, which returns `(return type, replacement node)` so
the caller does the `*expr = …` after the receiver/argument borrows are released (the same return-the-node
trick that keeps a whole-node rewrite out of a borrow conflict). Behavior-preserving, full suite green, no test
edits.

**Landed — `check_statement` 360 → 39.** The largest function in the frontend was a flat match on statement
kinds; its six substantial arms are now named helpers (`check_let_decl_stmt`, `check_for_loop_stmt`,
`check_loop_stmt`, `check_assign_stmt`, `check_return_stmt`, `check_assert_stmt`), each re-destructuring the
statement with the arm's own pattern so the bodies moved verbatim. The remaining 39 lines are a pure dispatcher.
Behavior-preserving, full suite green, no test edits.

**R4 done.** `lower_to_type_id` needs no work (see the correction above). The remaining large frontend
functions — `is_assignable` (286), `check_transfer_expr` (244), `check_identifier_expr` (238) — are either flat
decision cascades that read fine as-is (`is_assignable` is a type-pair rule table) or single-concern checkers,
not the multi-concern god-functions R4 targeted; and the closure-struct-call arm in `check_functioncall_expr`
still rewrites `*expr` inline (extractable with the same return-the-node trick). All are optional future cleanup,
not blocking work.

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
