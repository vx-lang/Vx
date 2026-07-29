# Design: Per-Parameter Provenance in the Packed Region Encoding

**Status:** v1 landed — the precision fix (steps 1–5) is implemented; see [§8.1](#81-implementation-status--v1-landed-steps-15-precision-fix). Encoding (steps 6–7) deferred.
**Tracked in:** [#243](https://github.com/hiraditya/Vx/issues/243)
**Companions:** [`borrow_checker_architecture.md`](borrow_checker_architecture.md) (design of record) · [`borrow_checker_precision_analysis.md`](borrow_checker_precision_analysis.md) (the measurement that motivated this)

______________________________________________________________________

## 1. The claim this enables

The motivating fact is concrete and checkable. This signature —

```rust
fn pick(a : &Map, b : &Map) -> &i32 { return &b.slot; }
```

— is one **rustc cannot compile without a lifetime annotation**: elision fails (`expected named lifetime parameter`), and the programmer must write `fn pick<'a>(a: &Map, b: &'a Map) -> &'a i32` to
say "the result borrows from `b`, not `a`." Vx already accepts `pick` annotation-free; it just
answers the *use* of the result conservatively, treating `let r = pick(&x, &y)` as borrowing both
`x` and `y`. Closing that gap makes Vx accept the same programs annotated Rust accepts here —
**without the annotation** — at O(1) per call site.

Stated precisely, and only as far as it is true:

> **For the single-source-per-return class** — a reference-returning function whose result derives
> from exactly one parameter — per-parameter provenance gives Vx the precision of
> explicitly-annotated Rust with no annotation, checked in a single masked word comparison.

The scope qualifier is load-bearing, not a hedge. Rust lifetimes also express *relationships* this
summary deliberately does not model: outlives bounds (`'a: 'b`), or a return whose lifetime is a
fresh variable constrained by several inputs. Genuine multi-source returns (`if c { &a.f } else { &b.f }`) fall back to "borrows from all of them," which is exactly today's behaviour. The claim is a
precise statement about a common, well-defined class — the class rustc's elision rules were designed
for and then refuse — not a general superiority claim. Lead with `pick`, which is verifiable, over
any "as precise as annotated Rust" superlative, which is not true in the general case.

**Priority.** This is a design document first. The evaluation in §6 exists to *validate* the design,
not to drive it; where the two pull apart, the design principle wins. That principle is the one the
whole borrow checker is built on: the **common path is a single masked-word comparison**, and
less-frequent cases (multi-source returns, more than four reference parameters, deep nesting,
recursion through a cycle) take a conservative slow path or a fixed-width fallback — never a heavier
common path. No feature is worth slowing the case that runs on every call site.

______________________________________________________________________

## 2. Background: the two halves as they stand today

Vx splits borrow checking (see [`borrow_checker_architecture.md`](borrow_checker_architecture.md)):

**Local aliasing** lives in the type checker. `active_borrows: HashMap<Symbol, Vec<BorrowRecord>>` is keyed by base variable. Conflicts are raised by `check_borrow_expr` (`src/hir/expr.rs:check_borrow_expr`) for `&x` forms and by `track_reference_arg_borrow` (`src/hir/expr.rs:track_reference_arg_borrow`) for reborrows through reference parameters. Release is at **last use**, not scope end — `is_variable_used_after` (`src/hir/env.rs:is_variable_used_after`) backs an NLL-style dead-borrow cleanup that both entry points run before testing for a conflict.

**Global subtyping** lives in `verify_subtyping_bounds` (`src/borrow.rs:verify_subtyping_bounds`). Region and variance are bitpacked into word 2 of the 256-bit `TypeId` (`src/gid.rs:TypeId`, `words: [u64; 4]`), four 16-bit parameter slots:

```
Word 2:  [ Param 3 (16) | Param 2 (16) | Param 1 (16) | Param 0 (16) ]
Slot:    [ Variance (4) | Region (12) ]

VARIANCE_MASK = 0xF000     REGION_MASK = 0x0FFF     PARAM_MASK = 0xFFFF
```

Region ID is lexical scope depth; region 0 is `'static`; a *smaller* region outlives a *larger* one, so covariant subtyping is `region_a <= region_b`, invariant is `region_a == region_b`. The whole check is masked integer arithmetic over one word — no constraint graph, no side table, no per-call solving.

**Return-escape** (#243) added a third piece: every reference-typed binding carries a `RefProvenance` (`src/hir/env.rs:RefProvenance`), either `External` (roots in a reference parameter — safe to return) or `Local` (roots in a `let`, a by-value parameter, or a temporary — returning it is `E4005`). It is computed structurally by `ref_provenance_of` (`src/hir/expr.rs:ref_provenance_of`) and stored per binding in `ref_provenance` (`src/hir/env.rs`, field on the checker).

______________________________________________________________________

## 3. The finding: sound, but conservative at multi-parameter signatures

### 3.1 Reproduction

```rust
struct Map { slot : i32, present : i32 }
fn insert(m : &mut Map, v : i32) -> void { m.slot = v; m.present = 1; }

// The result derives from `b` only. Rust rejects this signature outright:
// ambiguous lifetime elision, `expected named lifetime parameter`.
fn pick(a : &Map, b : &Map) -> &i32 {
  return &b.slot;
}

fn main() -> i32 {
  let mut x = Map { slot : 1, present : 1 };
  let mut y = Map { slot : 2, present : 1 };
  let r = pick(&x, &y);
  insert(&mut y, 99);   // E4003 — CORRECT, r aliases y
  insert(&mut x, 99);   // E4003 — CONSERVATIVE, r does not alias x
  return *r;
}
```

Both mutations are rejected. Only the first should be.

### 3.2 Why

Two independent code paths lose the parameter identity, and both must be fixed.

**(a) The provenance lattice is a join, not a selection.** `join_arg_provenance_exprs` (`src/hir/expr.rs:join_arg_provenance_exprs`) folds over *all* arguments:

```rust
for a in args {
    if self.ref_provenance_of(a) == Some(RefProvenance::Local) {
        return Some(RefProvenance::Local);
    }
}
Some(RefProvenance::External)
```

`External` carries no payload, so "which parameter" is discarded at construction. The lattice is `Local ⊔ External`, two points.

**(b) The call site persists a record for every reference argument.** In the reborrow block of the `Expr::FunctionCall` arm (`src/hir/expr.rs`, the `resolve_callee_ref_signature` / `track_reference_arg_borrow` loop), `ret_is_ref` is computed **once for the whole call** and passed as `persist` to every argument:

```rust
let ret_is_ref = Self::is_ref_type(&ret_ty);
for (i, arg) in args.iter().enumerate() {
    ...
    self.track_reference_arg_borrow(&base, path, Self::is_mut_ref(param_ty), ret_is_ref, span, silent);
}
```

So "this call returns a reference" is treated as "this call's result borrows from every reference argument." For `&x`-literal arguments the same conservatism arrives via `check_borrow_expr`, which records a borrow for each `&` expression and keeps it alive as long as the binding that consumed the call result is used.

### 3.3 Why it is *sound*

Over-approximating the borrow set can only reject programs, never accept unsound ones. This is a precision defect, not a safety defect — which is why it did not surface in the nine-case matrix, all of whose cases are single-source.

______________________________________________________________________

## 4. Design

Three layers. (a) and (b) are the functional change; (c) is what makes it O(1) across module boundaries and is the part with paper-weight novelty.

### 4.1 (a) Give `External` a payload

```rust
pub enum RefProvenance {
    /// Roots in reference parameter `slot` of the enclosing function. Safe to return.
    External(ParamSlot),
    /// Roots in a local slot. Returning it is a dangling escape (E4005).
    Local,
}

pub struct ParamSlot(u8);        // 0..=3 inline; see §4.3 for overflow
```

`ref_provenance_of` (`src/hir/expr.rs:ref_provenance_of`) already walks the structure to decide `External` vs `Local`; the parameter index is available at the point where it matches a parameter type and is currently thrown away. The arms that change:

- **`Expr::Borrow`** — `&base` / `&base.field`: when `base` resolves to reference parameter *i*, yield `External(i)` instead of `External`.
- **`Expr::Identifier`** — look up the recorded provenance; it now carries the slot.
- **`Expr::FunctionCall`** — see (b).

### 4.2 (b) Turn the join into a selection

The callee's return provenance is a **function summary**: which parameter slot (if any) the returned reference roots in. Compute it once per function when its body is checked, alongside the existing `E4005` return check, and store it on the resolved signature that `resolve_callee_ref_signature` (`src/hir/expr.rs:resolve_callee_ref_signature`) already returns:

```rust
enum ReturnProvenance {
    NotAReference,
    FromParam(ParamSlot),
    FromAnyOf(ParamSlotSet),   // genuine multi-source: `if c { &a.f } else { &b.f }`
    Local,                     // already an E4005 in the callee
}
```

At the call site, `persist` stops being one bool for the whole call and becomes per-argument:

```rust
let ret_prov = summary.return_provenance;
for (i, arg) in args.iter().enumerate() {
    let persists = ret_prov.includes(i);   // was: ret_is_ref
    self.track_reference_arg_borrow(&base, path, Self::is_mut_ref(param_ty), persists, span, silent);
}
```

Non-deriving arguments keep the call-duration borrow (correct — the callee may read through them) but do not persist past the call.

`FromAnyOf` preserves today's behaviour exactly for genuinely multi-source returns, so the change is a strict precision improvement with no soundness delta in that case.

**Recursion and cycles.** A function whose return provenance depends on itself needs a fixpoint. Initialise every summary to `FromAnyOf(all reference params)` (the conservative top) and iterate to a fixpoint over each SCC of the call graph. Monotone and finite-height (the lattice is the powerset of at most 4 slots), so it terminates. For the common acyclic case this degenerates to a single post-order pass.

### 4.3 (c) The encoding

The functional change above works with an ordinary side table. The reason to put it in the word is that `verify_subtyping_bounds` is the cross-module/cross-crate path, where a side table means either serialising it into the module interface (`#220`'s `.vxlib`) or re-analysing the callee. Encoding it inline keeps the O(1) property and keeps the summary in the type's identity, where it is already being compared.

Budget: the return's provenance needs `⌈log2(4 params + 1 sentinel)⌉ = 3 bits`. Region is currently 12 bits = 4096 lexical nesting levels, which is far beyond any real program (Rust's own recursion limit for type nesting is ~128).

**Proposal — steal 3 bits from the region field of slot 0**, which is the return's slot:

```
Slot 0 (return):   [ Variance (4) | Prov (3) | Region (9) ]
Slots 1..3 (params): [ Variance (4) | Region (12) ]        (unchanged)

VARIANCE_MASK = 0xF000
PROV_MASK     = 0x0E00        // slot 0 only
REGION_MASK   = 0x0FFF        // slots 1..3
REGION_MASK_0 = 0x01FF        // slot 0, 9 bits = 512 nesting levels
```

`Prov` values: `0` = not a reference / no provenance; `1..=4` = derives from parameter slot 0..3; `7` = `FromAnyOf` (conservative top, today's behaviour). Values `5`–`6` reserved.

Consequences:

- `verify_subtyping_bounds` gains one masked compare on slot 0 and stays branch-light. The existing `bits_a == bits_b` fast-out is unaffected.
- A function with more than 4 reference parameters, or nesting deeper than 512, sets `Prov = 7` and falls back to today's conservative behaviour. **Degradation is graceful and local**, never unsound — this is the property to emphasise in the paper, and it is what makes an inline fixed-width encoding defensible at all.
- Slot 0's narrower region field only binds the *return*, whose region is by construction one of the parameters' or `'static`, so it never needs deep nesting.

**Alternative considered — a fifth word.** `TypeId` is `[u64; 4]`; widening to `[u64; 5]` removes all pressure but changes the size of the most-copied structure in the compiler and breaks the 256-bit register story that motivates the design. Rejected, but worth measuring in the evaluation (§6, E5) so the choice is defended with numbers rather than aesthetics.

______________________________________________________________________

## 5. Soundness argument (sketch)

The property to preserve: *if a returned reference `r` may alias storage reachable from argument `a_i`, then a conflicting access to `a_i` while `r` is live must be rejected.*

Today's rule over-approximates the alias set to all reference arguments. The proposed rule narrows it to `ret_prov`. The obligation is therefore:

> **O1.** If the callee's returned reference may alias storage reachable from parameter *i*, then `i ∈ ret_prov`.

`ret_prov` is computed from the callee's body by the same structural walk that already decides `E4005`. The cases:

1. **Return of `&p.path` where `p` is parameter *i*.** ⇒ `FromParam(i)`. Direct.
1. **Return of an identifier** bound to a reference. ⇒ its recorded provenance, transitively.
1. **Return of a call `g(...)`.** ⇒ map `g`'s summary through the argument positions; union if `g`'s summary is `FromAnyOf`.
1. **Return on multiple paths.** ⇒ union over paths ⇒ `FromAnyOf`.
1. **Return of a `Local`.** ⇒ already `E4005`; the caller never sees it.

The risk cases that must be covered by tests (§6, E2) rather than assumed:

- **Interior mutability / raw pointers.** Vx has `*mut T` and `unsafe`. A callee that stashes a parameter-derived pointer into a struct field reachable from another parameter defeats a structural provenance walk. Today's conservative rule accidentally covers some of this. **Recommendation: `ret_prov` must be `FromAnyOf` for any function whose body contains an `unsafe` block or a raw-pointer store**, until an explicit aliasing story exists (relates to #187).
- **Nested references** (`&&T`, references in struct fields) — the walk must not stop at the outer reference.
- **Trait/dyn dispatch and closures.** `resolve_callee_ref_signature` already returns `None` for these, so they must default to `FromAnyOf`, not to "no record."
- **Externs.** A `extern "C"` function's body is unavailable ⇒ `FromAnyOf`.

The safe default at every unknown is `FromAnyOf`, which is exactly today's behaviour — so an incomplete implementation is a precision regression, never a soundness one. **This is the single most important structural property of the design and should be stated as such in any paper.**

______________________________________________________________________

## 6. Evaluation strategy

This section *validates* the design; it does not shape it (see the priority note in §1). Its purpose is to make the claim in §1 falsifiable and to pre-empt the reviewer questions a compiler-conference submission on borrow checking would attract — but none of it is a reason to alter the common path, which stays a single masked-word comparison regardless of what the evaluation would prefer.

### 6.1 Research questions

- **RQ1 (precision).** On signatures where Rust requires an explicit lifetime annotation, does Vx with per-parameter provenance accept the same programs as annotated Rust?
- **RQ2 (annotation burden).** What fraction of reference-returning functions in a realistic corpus would require an explicit lifetime in Rust but need none in Vx?
- **RQ3 (cost).** What is the per-call-site and whole-compilation cost of the check, and how does it scale with parameter count and program size?
- **RQ4 (encoding headroom).** What fraction of real functions fit the inline budget (≤4 reference parameters, ≤512 nesting) versus falling back to `FromAnyOf`?
- **RQ5 (soundness).** Does the narrowed rule reject every program that the conservative rule rejected *for aliasing reasons*, on a corpus that includes adversarial cases?

RQ4 is the one that decides whether the inline encoding is a contribution or a curiosity. If 95%+ of functions fit inline, the story is "constant-time in practice." If it is 60%, the honest paper is a negative result about fixed-width inline encodings — still publishable, but a different paper.

### 6.2 The corpus problem

This is the weakest link in any evaluation of a young language's analysis, and reviewers will go straight at it. `tests/` is not a corpus — self-evaluation on one's own regression suite is the standard reject reason. Three viable options, in descending order of credibility:

1. **Port a Rust corpus.** Take *N* crates that are heavy on reference-returning APIs (collections, parsers, arena allocators) and mechanically translate the subset Vx supports. Expensive, and the translation is itself a threat to validity, but it gives a defensible Rust baseline for RQ1/RQ2 *on the same programs*.
1. **Generate.** A grammar-directed generator over reference-returning signatures, parameter counts, and aliasing patterns. Cheap, scales to millions of call sites for RQ3/RQ4, weak for RQ2 (generated code is not representative code).
1. **Use the real Vx workloads.** `benchmarks/llama2_*.vx`, the attention corpus, `tests/backend/pass/flash_attention_v4.vx`, `stdlib/std/*`. Small, but *real*, and it is the only place where "what does Vx code actually look like" is answerable.

**Recommendation: all three, for different RQs.** (3) for RQ2's headline number with an honest sample-size caveat; (1) for RQ1's comparison; (2) for RQ3/RQ4's scaling curves. State plainly which number came from which.

### 6.3 Baselines

- **B1 — Vx today (conservative).** The precision floor. Every accepted-program delta over B1 is the contribution.
- **B2 — annotated Rust.** The precision target for RQ1. On ported programs, add the minimal lifetime annotations rustc demands and record both the verdict and the annotation count.
- **B3 — rustc NLL, timing.** For RQ3. **Do not report wall-clock rustc-vs-vxc as a headline** — the compilers do different total work and reviewers will say so. Report **cost per call site / per region check / per loan**, which is the honest comparable, and report total borrow-check phase time as secondary with the caveat attached.
- **B4 — Polonius.** Precision only, on the nine-case matrix plus the multi-parameter cases. Not a timing baseline; the comparison of interest is "same verdicts, different cost class."

### 6.4 Metrics

| Metric | RQ | Definition |
|---|---|---|
| Accept-rate delta vs B1 | RQ1 | programs accepted that today's conservative rule rejects |
| Agreement with B2 | RQ1 | verdict-identical on the ported corpus; disagreements enumerated individually |
| Annotations elided | RQ2 | count of `'a` that rustc requires and Vx does not |
| ns / call site | RQ3 | borrow-check time ÷ reference-passing call sites |
| Instructions / region check | RQ3 | `perf stat` or `cachegrind` on `verify_subtyping_bounds` |
| Borrow-check phase % | RQ3 | fraction of total front-end time |
| Inline-fit rate | RQ4 | functions with `Prov != 7` ÷ all reference-returning functions |
| Fallback locality | RQ4 | when `Prov == 7`, how many *call sites* lose precision |
| Adversarial rejects | RQ5 | must be 100% on the E2 suite |

### 6.5 Experiments

**E1 — Precision matrix.** Extend the nine cases in [`borrow_checker_precision_analysis.md`](borrow_checker_precision_analysis.md) with a multi-parameter family: *k* reference parameters, return derived from parameter *j*, conflicting access on parameter *i*, for all (*i*, *j*, *k*≤6). Expected: accept iff *i* = *j* (or *k* > 4, where `FromAnyOf` makes it reject). Table columns: Vx-today, Vx-proposed, Rust-annotated, Polonius. This table is the paper's centrepiece — it is the same shape as the NLL/Polonius discriminator table the field already recognises, with two columns the field has not seen.

**E2 — Adversarial soundness.** Hand-written attacks on O1, each expected to reject: parameter-derived pointer stashed through `unsafe`; return through a nested reference; return through a trait object; return through a closure capture; mutual recursion where the summary fixpoint could wrongly narrow; `extern` callee. A single false accept here sinks the paper, so this suite should be written *before* the implementation, adversarially, ideally by someone other than the implementer.

**E3 — Annotation burden.** For every reference-returning function in the corpus, mechanically determine whether rustc's elision rules suffice. Report the fraction where they do not. This is RQ2's number and it is the most *communicable* result in the paper — "*N*% of reference-returning functions need a lifetime annotation in Rust and none in Vx" is a sentence a program committee remembers.

**E4 — Cost.** Microbenchmark `verify_subtyping_bounds` in isolation (instructions/check, cache behaviour) and end-to-end on the corpus (phase time, scaling with call-site count). Sweep parameter count 1..8 to show where the inline path ends and what the cliff costs.

**E5 — Encoding ablation.** Three variants: (i) proposed 3-bits-from-slot-0; (ii) side table keyed by `TypeId`; (iii) a fifth `u64`. Measure `TypeId` copy cost, cache pressure, and total compile time. This defends §4.3's choice with data and pre-empts "why not just use a side table," which *will* be asked.

**E6 — Scaling.** Synthetic programs at 10³/10⁴/10⁵/10⁶ call sites. Establish the cost curve is flat per call site — that is the claim, and it should be a figure, not a sentence.

### 6.6 Threats to validity

State these explicitly; a reviewer who finds them unstated assumes they were missed.

- **Corpus representativeness.** Vx has no ecosystem. Any RQ2 number is a claim about *this* corpus. Do not generalise to "real-world code."
- **Translation bias.** Rust→Vx porting may systematically simplify away hard cases. Mitigate by reporting what was dropped and why.
- **The comparison is not apples-to-apples.** rustc does trait solving and monomorphisation the Vx front end does not. Per-check metrics mitigate; wall-clock does not.
- **Soundness is argued, not proved.** §5 is a sketch. Either mechanise it (a Coq/Lean model of the summary lattice and O1 would materially strengthen the paper) or state plainly that it is a tested argument, not a theorem. Do not imply otherwise.
- **Fixed-width encodings degrade.** The inline budget is 4 parameters; a corpus dominated by wider signatures would erase the benefit. RQ4 measures this honestly rather than assuming it.

### 6.7 What this explicitly does *not* claim

Worth a short subsection in the paper itself.

- **Not location sensitivity.** bc8 stays rejected. This design does not implement Polonius; it reaches one of Polonius's motivating cases (bc2 / NLL Problem Case #3) by a different decomposition, and the paper must be precise about that or it invites a correctness objection in review.
- **Not faster than rustc overall.** Only the borrow-check phase, per unit of work.
- **Not "as precise as annotated Rust" in general — only for the single-source-per-return class (§1).** Multi-source returns, outlives bounds (`'a: 'b`), and lifetime relationships between parameters are out of scope and take the conservative `FromAnyOf` default. The verifiable claim is the `pick` case; the superlative is not, and stating it invites a correctness objection.
- **Not a full region inference.** No inference of region *relationships* beyond which parameter a return derives from.

______________________________________________________________________

## 7. Related work positioning

Honest framing, since two nearby claims will be checked:

- **rustc NLL** already represents region values as interval sets / sparse bit matrices over program points, and its dataflow borrow checker uses bitsets. *Verify this directly in `rustc_borrowck` before writing the related-work section.* The novelty here is therefore **not** "bitvectors for liveness" — that framing will not survive review. It is the **inline, fixed-width, comparison-in-place summary** and the **provenance decomposition**.
- **LLVM** uses interval representations for live ranges in register allocation — adjacent technique, different problem, and not a competing claim.
- **Polonius** reformulates as loan-liveness to gain location sensitivity. This design gains one of its motivating cases without that reformulation. That is the interesting comparison and should be stated as a *different point in the design space*, not as "we beat Polonius."
- **Region-based type systems** (Tofte–Talpin, Cyclone) put regions in types. The distinguishing feature here is that the region is a *totally ordered scalar* (lexical scope depth), which is what makes subtyping a single `<=` — and which is also precisely why location sensitivity is out of reach (§6 of the precision analysis). That trade should be presented as the paper's central design tension, because a reviewer will find it anyway.

______________________________________________________________________

## 8. Implementation plan

| Step | Change | Files / functions |
|---|---|---|
| 1 | `RefProvenance::External(ParamSlot)` | `src/hir/env.rs:RefProvenance` |
| 2 | Thread the slot through the structural walk | `src/hir/expr.rs:ref_provenance_of` |
| 3 | Compute the per-function `ReturnProvenance` summary; SCC fixpoint | new, alongside `src/hir/expr.rs:resolve_callee_ref_signature` |
| 4 | Join → selection at the call site (`persist` per argument) | `src/hir/expr.rs`, the `track_reference_arg_borrow` loop in the `Expr::FunctionCall` arm |
| 5 | Conservative defaults for unsafe / raw pointers / dyn / closures / externs | same |
| 6 | Bit layout + `PROV_MASK`; slot-0 compare | `src/borrow.rs` (masks near `VARIANCE_MASK`), `src/borrow.rs:verify_subtyping_bounds` |
| 7 | Serialise the summary into the module interface | `#220` `.vxlib` — coordinate with the stdlib-decoupling epic `#224` |
| 8 | Test suites E1 + E2 | `tests/middle_end/pass/`, `tests/middle_end/fail/` |

Steps 1–5 are a self-contained precision fix that can land independently of the encoding; steps 6–7 are what make it O(1) across modules. **Land 1–5 first, measure, then do 6–7** — that way the precision result and the encoding result are separately attributable, which the paper needs anyway.

### 8.1 Implementation status — v1 landed (steps 1–5, precision fix)

Steps 1–5 are implemented. The realised design differs from the literal table above where the
**parallel architecture** required it — the checker builds a fresh `TypeChecker` per function via
`par_iter_mut`, and the shared env is signature-only, so an on-demand/side-table summary would have
meant a lock or a missing body. What actually landed:

- **`ReturnProvenance`** (`FromParams(bitset) | AnyParam | Local | NotAReference`) is a *separate*,
  precomputed summary in `src/hir/provenance.rs`, **not** a payload on `RefProvenance` (the escape
  analysis kept its two-point lattice). `compute_return_provenance` is a pure structural walk of the
  body — no `TypeChecker`, no shared state.
- **Frozen into the immutable env** (`GlobalAstEnv.return_provenances`), filled from present bodies in
  `build` and refilled from the full pre-strip modules at the production entry points
  (`annotate_return_provenances`). The per-function parallel checkers only *read* it — the lock-free
  `type_check_phase` is untouched.
- **Call site is a per-argument persist decision applied by selective revert** (in the
  `Expr::FunctionCall` arm), covering *both* the `&x`-literal (`check_borrow_expr`) and bare-reference
  (`track_reference_arg_borrow`) paths — so the `pick(&x, &y)` repro (§3.1) is actually fixed, which
  patching only the reborrow loop (step 4 as written) would not have done. A borrow persists past the
  call iff the callee returns a reference **and** its summary includes that argument's slot; the
  revert keeps only pre-call records (never resurrecting NLL-released borrows), and a non-deriving
  argument records nothing so `f(x, x)` in-place reborrows do not self-conflict.
- **No SCC fixpoint (yet).** `return callee(...)` → `AnyParam` (conservative). Recursion and
  cross-function derivation are the deferred fixpoint (would live in the precompute, off the hot
  path). `unsafe` bodies, >32 params, and every unknown → `AnyParam`.
- **`verify_subtyping_bounds` untouched**; the inline 3-bit `TypeId` encoding (steps 6–7) is deferred
  — the env map gives full intra-compilation precision, and the encoding is `.vxlib`-gated (#220/#224)
  and must be minted deterministically in the GID-mint pre-pass (the 1-thread==8-thread invariant).

Tests: `src/hir/provenance.rs` unit tests (the summary lattice) + E1/E2 fixtures under
`tests/middle_end/{pass,fail}/borrow_multiparam_*`, `borrow_multisource_*`, `borrow_unsafe_*`. Full
suite green.

______________________________________________________________________

## 9. References

- [#243](https://github.com/hiraditya/Vx/issues/243) — borrow checker soundness + precision (this design amends it)
- [#187](https://github.com/hiraditya/Vx/issues/187) — memory-model: use-site scope-crossing enforcement (adjacent escape analysis)
- [#220](https://github.com/hiraditya/Vx/issues/220) / [#224](https://github.com/hiraditya/Vx/issues/224) — module-interface serialisation, needed for step 7
- [`borrow_checker_architecture.md`](borrow_checker_architecture.md) — design of record
- [`borrow_checker_precision_analysis.md`](borrow_checker_precision_analysis.md) — the nine-case matrix; E1 extends it
- `tests/middle_end/fail/borrow_mut_imm.vx`, `borrow_mut_twice.vx`, `borrow_use_after_mut.vx` — existing negative cases
- `tests/middle_end/pass/borrow_lexical.vx`, `borrow_split_fields.vx` — existing positive cases
