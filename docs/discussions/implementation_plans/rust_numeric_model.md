# Adopt Rust's numeric model: no implicit conversions, literals infer to context (#240)

**Status:** Stage A **landed** (2026-07-26) — see §8 for what shipped vs. this plan; a first Stage-A
attempt was made and reverted before it (§5). Stage B not started.
**Date:** 2026-07-26. **Supersedes:** the coercion-materialization work (#236/#238) by removing its
reason to exist. **Umbrella:** #200.

## 1. Decision

A value of one scalar type never implicitly converts to another. The programmer writes an explicit
`as` cast; the compiler rejects a mismatch with a helpful error. The one thing kept is **literal
inference**: an *unsuffixed* numeric literal adopts the type of its context, so `let x: i64 = 0`,
`f(14)`, `n + 1` (with `n: i64`), `return 0`, `a[i] = 1.0`, `P { x: 0 }` all compile without suffixes.

Rationale: implicit coercions breed cryptic rules and silent bugs — Vx today even accepts silent
*lossy narrowing* (`let x: i32 = some_i64` truncates). This matches the repo's "crash rather than fail
silently" rule and deletes the whole coercion-materialization class.

## 2. The model (what "right" looks like)

- **Suffixed literal** (`80i32`, `1.5f64`): explicitly typed; never re-inferred.
- **Unsuffixed literal** (`80`, `1.5`): *untyped* until type-checking, then adopts the **expected
  type** of its position when that is a compatible scalar (integer literal → integer type, float
  literal → float type; `let x: f64 = 5` is an error — write `5.0`). With no expected type it falls
  back to a size-based default (`i32`/`i64`/`i128`, `f32`/`f64`).
- **`is_assignable` for scalars:** identical-only. A genuine typed-value mismatch is an error (point
  the message at `as`).

This is bidirectional checking: "checking positions" push an expected type down; a literal in a
checking position adopts it.

## 3. Where the expected type comes from (the checking positions)

`let` annotation · call argument (→ parameter type) · `return` (→ return type) · assignment RHS (→ LHS
type) · struct-field initializer (→ field type) · the other operand of a binary/relational op · a
tensor index (→ the index type). Each must set the expected type before checking the sub-expression
and restore it after (a `check_expr_expecting(expr, Some(ty))` helper), so an outer expectation
doesn't **leak** into an inner position (`let x: i64 = f(3)` — the `3` must take `f`'s parameter type,
not the `let`'s `i64`).

## 4. Sequencing

- **Stage A — literal inference (non-breaking *in principle*).** Parser emits `ty: None` for
  unsuffixed literals; the checker's `Number` arm infers from the expected type (mutating the node) or
  falls back to the default; expected-type propagation wired at every §3 position. `is_assignable`
  stays permissive, so no program is newly rejected — this stage only makes literals infer.
- **Stage B — tighten + migrate.** `is_assignable` scalar rule → identical-only, with a helpful
  error. Migrate corpus + stdlib (explicit `as` where a genuine typed-value conversion was relied on).
  Remove the now-dead `Lowerer::coerce_val` (#238) and `TypeChecker::coerce_to` (#236).

## 5. What the first attempt learned (why it's bigger than it looks)

An attempt at Stage A (parser `None` + a checker `Number` arm reading `expected_type`, wired only at
the `let` position) was **reverted** — it broke 15 tests, and the breakage exposed the real work:

1. **Eager argument checking is the structural blocker.** `check_functioncall_expr` checks *all*
   arguments up front (`for arg in args { arg_types.push(check(arg)) }`) **before** it resolves which
   callee/parameters apply. So a literal argument is already defaulted (to `i32`) by the time the
   parameter type is known — param-driven inference can't work in that order. Options: (a) re-type an
   unsuffixed literal argument to its parameter *after* resolution (the #236 shape, gated on
   "unsuffixed"); (b) restructure to resolve the callee first, then check args with the parameter as
   the expected type. (a) is smaller; (b) is cleaner and generalizes. This is the single biggest
   design choice.
1. **Two conflicting "defaults" must be unified.** The checker/parser default an unsuffixed int to
   `i32`; the flat lowerer's `infer_elem` defaults it to `i64`. As long as literals were eagerly typed
   by the parser this never showed; making them untyped exposes it, and checker-less unit tests
   (`flatten`/`flat` `tests` that don't run the checker) then disagree with real compilation. Pick one
   default (`i32`, Rust-style) and make `infer_elem` match — or make those unit tests run the checker.
1. **Expected-type leakage** (§3) is real and must be handled at every checking position, not just
   `let`.
1. **Pervasive test expectations.** Many parser unit tests assert `ty: Some(I32)` for `42`, and
   FileCheck backend/middle-end tests pin literal types in the emitted MLIR (`primitives_mlir.vx`,
   etc.). These encode today's eager typing and must be updated in lockstep — mechanical but broad.
1. **Migration blast radius is unmeasured.** It can only be measured by tightening `is_assignable`
   (Stage B) and counting the genuine typed-value mismatches across corpus + stdlib; expect a
   non-trivial set of explicit-`as` edits.

## 6. Recommendation

Do Stage A as a dedicated change, choosing option 5.1(b) (resolve-then-check) if feasible so inference
is principled rather than a re-typing patch, and unify the default (5.2) first. Land it behind the
still-permissive `is_assignable` and get the whole suite green (only test-expectation updates, no
program rejected). Then do Stage B as a second change, measuring and migrating. Both are journalled
and corpus-swept like the rest of the convergence work.

## 7. Non-goals

Non-numeric coercions Rust keeps (deref, autoref, unsize) — Vx doesn't have them. Full
Hindley–Milner inference — bidirectional "checking position adopts expected type" suffices for the
literal cases.

## 8. Stage A as landed (2026-07-26)

What shipped, and where it refined the plan above:

- **Parser (`infer_number_literal`) emits `None` for every unsuffixed literal** — the untyped-literal
  model. The spelling-based default moved to a shared `default_number_elem` (§5.2's unification),
  now the single source used by *both* the checker's fallback and the flat lowerer's `infer_elem`,
  so the two backends can't disagree on an un-annotated literal. `default_number_elem` reproduces the
  historical parser defaults (int → narrowest of `i32`/`i64`/`i128`; decimal → `f32`, `f64` on
  overflow), so an un-annotated literal keeps exactly the type it had — Stage A is non-breaking by
  construction.
- **Checker `Number` arm → `check_number_literal`**: an untyped literal adopts the expected scalar
  type when kind-compatible (`expected_numeric_elem`: int→int, float→float; a `let x: f64 = 5`
  mismatch is left to default and, under Stage B, an error), else the default. The inferred type is
  **written back** onto the node, so the flat lowerer and AST codegen both see a concrete type.
- **Expected-type wiring is unchanged from what already existed** — only `let` and `return` set it.
  Call arguments keep flowing through the existing `coerce_call_arg` (#236), which re-types the
  literal *after* callee resolution (the §5.1 eager-checking blocker is thus sidestepped, not solved
  — option (a), deferred to Stage B). Binary / assignment / struct-field / tensor operands are **not**
  newly wired: their literals default and are still materialized by the per-backend coercion
  (`coerce_val` on the flat path, `coerce_type` on the AST path). That is sufficient while
  `is_assignable` stays permissive; Stage B must wire these positions before it tightens, or those
  literals will fail to infer and be rejected.
- **Two refinements the plan didn't anticipate:**
  1. **Type-position dimensions stay typed.** A literal in a tensor shape or topology index
     (`Tensor<f32, 10>`, `Topology::NPU[0]`) is a compile-time integer that participates in
     *structural* type equality, not a runtime value inferred from context. Leaving it `None` made a
     parsed type unequal to a synthesized one (topology-match and return-type checks broke). Fixed by
     stamping dimension literals with their default type at parse time (`stamp_dim_literals`), keeping
     `None` only for value positions.
  1. **Two latent AST-codegen coercion gaps surfaced and were fixed.** With a `let mut i: i64`
     initializer now correctly typed `i64`, the mutable slot is genuinely `i64` (previously it
     degenerated to the initializer's default `i32`, masking the bug). The compound-assign and
     plain-assign store paths only special-cased `index`↔`i32`; they now use the general
     `coerce_type`, so a default-`i32` literal combined with / stored into a wider slot widens
     instead of emitting `arith.addi(i64, i32)` / a mismatched `memref.store`.
- **Result:** full suite green; corpus flat-vs-legacy sweep unchanged at flat-used **87**, three
  chronic pre-existing miscompiles, **zero new**. `is_assignable` still permissive — no program newly
  rejected.
