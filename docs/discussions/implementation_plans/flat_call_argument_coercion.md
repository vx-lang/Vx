# Flat call-argument coercion to callee parameter types (#236)

**Status:** design / not started. **Date:** 2026-07-25. **Umbrella:** flat-codegen
convergence (#200/#201). **Related:** #231/#235 (which surfaced this), #234 (return
coercion), #232 (tensor-store coercion), #220 (`.vxlib` bodies + loader).

______________________________________________________________________

## 1. The gap

The flat path does **not** coerce a call argument to the callee's declared
*parameter* type. It types each `func.call` operand from the *argument value's*
tracked type, then emits the call — so if a caller passes a value whose lowered
type differs from the parameter type, the emitted `func.call` disagrees with the
callee's `func.func` signature and MLIR verification fails.

Concrete failure (a defined callee):

```vx
fn take(p: *const u8, n: i64) -> i64 { return n; }
fn main() -> i32 { let msg = "hi"; return take(msg, 14) as i32; }
```

`14` is a default-`i32` literal; `take`'s second parameter is `i64`. The flat
emitter produces:

```
func.func @take(%arg0: !llvm.ptr, %arg1: i64) -> i64 { ... }   // definition
%r = func.call @take(%msg, %c14) : (!llvm.ptr, i32) -> i64      // call — i32 arg
```

→ `'func.call' op operand type mismatch: expected 'i64', but provided 'i32'`.

**Why the FFI corpus dodged it (#231/#235).** An `extern`'s private declaration is
emitted *from* its call site (`emit_module_mlir` builds `func.func private @f(...)`
out of the observed argument types), so the declaration always matches the call by
construction. But that means the flat path can call an extern with a *half-width*
argument that disagrees with the symbol's real C ABI — e.g. `vx_stdout_write(msg, 14)`
passes `14` as `i32` where the C `len` is `i64`; it prints correctly only by ABI
luck (the high 32 bits happen to be zero). The corpus's *defined*-callee calls all
pass matching scalar widths, so nothing verified-fails today — but the gap is real
for both correctness (extern ABI) and coverage (defined wrappers).

## 2. What the AST oracle does

`src/codegen/lower/expr.rs` (`FunctionCallExpr`, ~line 2042) coerces every argument
to the callee's MLIR parameter type, in two layers:

1. **`expected_type` hint** — before lowering each argument it sets
   `gen.expected_type = Some(param_ty)`, so an untyped literal adopts the parameter
   type as it is lowered (`14` becomes `i64` directly).
1. **`coerce_type` fallback** — if the lowered argument type still differs, it casts:
   `memref.cast` for memref→memref, otherwise `coerce_type` (`arith.extsi/trunci`,
   `extf/truncf`, `sitofp/fptosi`, the scalar→tensor `linalg.fill` broadcast, …).

`coerce_type` (`generator.rs:92`) is the single source of truth for a legal implicit
conversion; the flat path already mirrors a subset of it via `cast_op` in the `Ret`
(#234) and `TensorStore` (#232) arms.

## 3. Why the flat path emits the mismatch

By the time the flat lowerer runs, the frontend has already type-checked the call
(arity + `is_assignable` per argument) — so `flatten::lower_call` trusts a well-typed
AST and re-checks nothing; it resolves the callee **by name** (`fn_sigs`), pulls only
the *return* type from `FnSig { gid, ret_ty }`, and lowers each argument to whatever
type the checker annotated on it. The checker validated *whether* the argument fits the
parameter but never recorded the implicit coercion its assignability check implied, so
`14` stays `i32`, and the emitted `func.call` disagrees with the callee's `i64`
parameter.

The fix is therefore about *where the coercion the checker already computed gets
recorded* — the design question of §4. The naïve reading ("carry the parameter types
into `FnSig` so a later phase can re-derive it") is one answer (§4.1), but not the
lightest: the checker is already holding the parameter types at the moment it decides
assignability, so it can record the coercion in place (§4.2, Option C) without any
later phase re-plumbing the signature. That is where the "wider consequences" analysis
below applies — and, under Option C, where most of it turns out to be *avoidable*.

## 4. Design

### 4.0 Recommended: the type checker records the coercion (Option C)

The primary design is §4.2, Option C — the frontend inserts the coercion into the
typed AST at the call it already validates, so both backends consume a
coercion-explicit tree and nothing downstream needs the parameter types. §4.1 and
§5.1 below (`FnSig.params` + the `.vxlib` bump) are **the deferred cross-module
fallback**, not part of the near-term fix; they are kept here as the completed design
for the day a cross-`.vxlib` call needs coercion.

### 4.1 Data model — `FnSig.params` *(deferred; cross-module fallback only)*

```rust
pub struct FnSig {
    pub gid: TypeId,
    pub params: Vec<crate::syntax::Type>,   // NEW — only if cross-module coercion is driven
    pub ret_ty: crate::syntax::Type,
}
```

`FnBody` already carries `params: Vec<Type>` and serializes it (count-prefixed
`write_type` loop, `metadata.rs::write_fn_body`), so the pattern and the
type-encoder machinery already exist — this is not new ground, only a new field on
the *signature*.

Population sites (`pipeline.rs::build_frozen_registry`):

- the **function** loop (~422): `f.params.iter().map(|(_, t)| t.clone())`
  (`Function.params: Vec<(Symbol, Type)>`).
- the **extern** loop (~441): `ext.params.iter().map(|(_, t)| t.clone())`
  (`ExternDecl.params: Vec<(Symbol, Type)>` — same shape).
- the **method** loop (~480): the method's explicit parameters (the receiver is
  implicit — decide whether `self` is index 0; see §5.3).

### 4.2 Where to coerce — the frontend, at type-check time (recommended)

There are three places this coercion could live; they are not equal.

**Option C — the type checker inserts the coercion (recommended).** The frontend
*already* validates every call against the full signature: `check_functioncall_expr`
(`src/hir/expr.rs`, and the fn-pointer/closure arms) has `args: &mut Vec<Expr>`,
computes `param_types` from the callee, and runs `is_assignable(param_ty, arg_ty)` per
argument. It already stands exactly where the coercion is decided — it checks *whether*
each argument fits but doesn't record *how*. So when `param_ty != arg_ty` but the two
are assignable, rewrite the argument in place: re-annotate a literal to the parameter
type (a `NumberExpr` born as `i64` — the flat `number_elem` and the AST codegen both
read `n.ty`, so no cast op is needed at all), or wrap a non-literal in an explicit
coercion node (`Expr::AsCast { expr, target_ty: param_ty }`), which **both** backends
already lower (flat → a `Cast` opcode; AST → its existing coerce path).

**Option A — in `flatten::lower_call` (insert `Cast`).** Carry `params` in `FnSig` and
have the flat lowerer re-derive the coercion the checker already computed, emitting a
`Cast` before the `Arg`.

**Option B — in the emitter's `Call` arm (`cast_op`).** Leave the HIR mismatched and fix
it while emitting text, as the `Ret` (#234) / `TensorStore` (#232) arms do.

**Recommendation: Option C.** It is the *elaboration inserts coercions* pattern, and it
wins on every axis this change is measured by:

- **Single source of truth.** The coercion is computed once, in the one phase that
  already knows the parameter types and already runs `is_assignable`. Options A/B make
  a *later* phase re-derive a decision the checker already made.
- **Both backends benefit, neither gets callsite logic.** The mutated (coercion-explicit)
  tree flows to the AST codegen *and* the flat lowerer; the flat `Cast` falls out of the
  ordinary `AsCast` lowering, and the AST path's own call-site `coerce_type`
  (`expr.rs:2061`) becomes redundant (it can stay as defense-in-depth or be removed after
  verifying no divergence).
- **Status quo elsewhere — lighter and faster.** `FnSig` stays `{ gid, ret_ty }`; **no
  `.vxlib` format bump** (§5.1 becomes a *deferred fallback*, not a required step), no
  per-signature `Vec<Type>` in the registry, no re-comparison at every callsite. This is
  the "keep the status quo, faster rewrite, less memory" property.
- **Still represented in the IR.** The coercion lands in the AST *and* the flat HIR (as
  the re-typed literal or the `Cast` from the `AsCast`), so the earlier "put it in the
  IR, not the emitter" principle holds — the difference from Option A is only *who*
  decides it (the checker, once) versus *where it is re-derived* (the lowerer, again).

**Cross-module is the one caveat that keeps A alive.** The checker can only insert the
coercion where it can see the callee's parameter types. For an *in-module* call it has
them (`GlobalAstEnv` / `lookup`) — that covers every current corpus driver. For a call
*across a `.vxlib` boundary*, the imported interface today carries only `ret_ty` (see
`program_links_a_function_body_from_a_vxlib_artifact`), so the checker can't see the
imported callee's params to coerce against — and arguably can't fully arity/type-check
that call either. So: **Option C fixes the in-module case now with zero registry/format
change; the `FnSig.params` + `.vxlib` v4 work (Option A / §4.1 / §5.1) is deferred and
becomes the documented fallback for cross-module coercion, if and when a driver needs
it.** The two are complementary, not competing — C is the near-term fix, A the
cross-module completion.

### 4.3 Coercion taxonomy (what to handle vs decline)

| param vs arg | action |
|---|---|
| identical `LoweredTy` | no-op (pass through — no `Cast`) |
| scalar↔scalar (width/int↔float/sign) | emit `Cast` in the HIR (already modelled end to end) |
| pointer↔pointer (`!llvm.ptr`) | no-op (opaque) |
| aggregate↔aggregate | require exact GID match, else decline |
| memref↔memref (shape/layout differ) | **decline** until a tensor/shape cast opcode lands (then represent it in the HIR, not the emitter) |
| scalar→tensor (broadcast) | **decline** (the AST's `linalg.fill`; rare at a call) |
| scalar→pointer (`inttoptr`) | **decline** (rare; the AST does `inttoptr` only via `AsCast`) |
| any unmodelled conversion | **decline** (AST stays the oracle) |

Declining keeps the module-level keep-green atomicity: an unmodelled coercion drops
the whole program to the AST path rather than emitting a wrong or invalid call. The
memref row declines because the flat lowerer has no tensor/shape cast yet (an `AsCast`
to a memref type isn't lowered); a memref coercion waits for that opcode rather than
being special-cased. The #236 drivers are all scalar-width mismatches, which the
existing scalar path already covers.

## 5. Wider consequences (the reason for this doc)

> **Scope note.** §5.1 and the `FnSig.params` change it describes are the **deferred
> cross-module fallback** (see §4.0/§4.2). The recommended near-term fix (Option C —
> the checker records the coercion) needs *none* of this: `FnSig` and the `.vxlib`
> format are untouched. §5.2–§5.6 remain relevant either way (they describe the coercion
> *semantics* and non-consequences, which are the same wherever the coercion is recorded).

### 5.1 `.vxlib` format version bump — v3 → v4 *(only if cross-module coercion is driven)*

`FnSig` is serialized in the import-oracle interface (`metadata.rs`, the `fn_sigs`
and `methods` sections). Adding `params` changes the on-disk layout, so:

- **Bump `VXLIB_FORMAT_TAG`** `"vxlib-interface-v3"` → `"v4"`. The tag is hashed into
  the header stamp; a stale v3 artifact is then *detected* (version mismatch on load,
  `deserialize_registry_interface`) rather than misread — the existing staleness guard
  does the right thing once the tag moves.
- **Encoder** (`serialize_registry_interface`): in the `fn_sigs` / `methods` loops,
  after `typeid(gid)`, write `u64(params.len())` then `write_type` each — mirroring
  `write_fn_body`. Keep the *skip-on-unencodable* policy: if any param type isn't
  encodable, skip the whole signature (as the return-type skip already does), so a
  partially-encodable signature is never half-written.
- **Decoder** (`deserialize_registry_interface`): read the param count + types before
  the return type; construct `FnSig { gid, params, ret_ty }`.
- **Round-trip test** (`metadata` tests): the existing interface round-trip must cover
  a signature with params (add a param-carrying fn to the fixture).

There is **no cross-version compatibility to preserve** — `.vxlib` artifacts are
build outputs, regenerated from source; the version stamp exists precisely so a stale
one is rejected, not migrated. So the bump is safe, but it does mean **every `.vxlib`
in a build tree is invalidated** and must be rebuilt. Document that in the changelog.

### 5.2 Cross-module linking

Once param types are in the serialized `FnSig`, a downstream compile that imports a
module (folds its interface via `merge_from`) gets the imported callee's param types
for free — so a flat-path call *across* a `.vxlib` boundary coerces its arguments the
same as an in-module call. Without this, cross-module calls would be a second, silent
instance of the same bug. (The `program_links_a_function_body_from_a_vxlib_artifact`
differential test is the natural place to add a width-mismatched cross-module call.)

### 5.3 Methods (`methods` map is also `FnSig`)

`registry.methods` is keyed by `(receiver GID, name)` and holds `FnSig` too, so it
inherits `params`. The flat path currently declines almost all method calls
(`flatten` only special-cases `with_memory`), so method-arg coercion isn't exercised
yet — but the data model must stay consistent, and when method calls land (a later
convergence step) they'll want the same coercion. Decision to record now: **does the
method `FnSig.params` include the implicit `self`?** Recommend **excluding** `self`
(params = the *explicit* parameters), and coercing the receiver separately when
method calls are implemented — matching how the AST threads the receiver.

### 5.4 The `expected_type` hint is *not* required

The AST's `expected_type` is a *threaded context* that influences how any expression
lowers. Option C needs no such machinery: the checker makes a single, local rewrite at
the one place it already knows the parameter type (the call's `is_assignable` loop) —
re-annotate the literal, or wrap the argument in an `AsCast`. That targeted edit is
**sufficient** for the
width/kind mismatches at issue (it produces exactly what the AST's `coerce_type`
fallback would). We do *not* need to replicate the `expected_type` machinery — one
rewrite at the call check suffices, which keeps the change small. (The one case it
can't recover is an argument whose *checked* type is already wrong for a non-assignable
reason — but that is a type error the checker already rejects, not a coercion.)

### 5.5 Generic templates in `fn_sigs`

A generic function's `FnSig` would carry *generic* param types (`Type::Scalar(Generic)`
/ `Type::Generic`). The flat path only lowers *monomorphized* (concrete) functions, and
a call resolves to the monomorph's `FnSig` (concrete params). Coercion against a
generic param type must be a no-op/decline (never a cast). Guard: if a param type is
generic/unmodelled, skip coercion for that argument (pass through) — the monomorph's
concrete signature is what actually gets called.

### 5.6 Non-consequences (things that stay put)

- **GID stability / determinism.** GIDs are content hashes of *names*, not of `FnSig`
  contents, so adding `params` changes no GID and no Phase-6/7 routing. The frozen
  registry stays deterministic (serialization is already sorted by name / receiver).
- **Ambiguity policy.** A name defined in >1 module (distinct GIDs) is still dropped
  from `fn_sigs`; params don't change the drop decision. (The extern-dedup fix from
  #231/#235 is orthogonal.)
- **Argument evaluation order / side effects.** Coercion appends ops *after* an
  argument is evaluated, so evaluation order is unchanged.

## 6. Implementation plan (ordered, each step green) — Option C

1. **Coercion insertion in the checker.** In `check_functioncall_expr` (and the
   fn-pointer / closure arms) in `src/hir/expr.rs`, at the per-argument
   `is_assignable(param_ty, arg_ty)` loop: when the two differ but are assignable,
   rewrite `args[i]` — re-annotate a `NumberExpr`/literal to `param_ty` in place, else
   wrap the argument in `Expr::AsCast { expr, target_ty: param_ty, .. }`. Only on the
   *committing* pass (guard on `!silent`, so speculative type resolution doesn't mutate
   the tree). Do it *after* generic monomorphization has fixed concrete param types.
1. **Debug assertion.** Right after the rewrite, `debug_assert!` the (re-checked)
   argument type equals `param_ty` — a cheap, in-place invariant that the frontend did
   its job, active only in debug builds. (MLIR verification remains the runtime backstop:
   a missed coercion → verify failure → the flat path declines to the AST oracle, i.e.
   keep-green rather than a miscompile.)
1. **Prune the now-redundant AST-path coercion (optional).** The AST codegen's call-site
   `coerce_type` (`expr.rs:2061`) becomes a no-op once arguments arrive at the parameter
   type. Leave it as defense-in-depth initially; remove it only after the differential
   suite confirms no divergence.
1. **Differential tests** — the `take(msg, 14)` case (now lowers, JIT parity); a
   float-width call (`f32` arg → `f64` param); a non-literal coerced arg
   (`let x: i32 = …; wants_i64(x)`); an aggregate/pointer arg (unchanged). Re-run the
   extern drivers to confirm the ABI is now *correct*, not lucky.
1. **Corpus sweep** — flat-vs-legacy parity; expect flat-used to hold or rise, zero new
   miscompiles. Watch the whole AST-codegen suite too (the mutated tree feeds it).
1. Commit (`Fixes: #236`), journal, close.

**Deferred (cross-module fallback, only if driven):** `FnSig.params` (§4.1) + `.vxlib`
v4 (§5.1), so the checker can coerce arguments to an *imported* callee's parameters.
Not needed for the in-module fix above.

## 7. Testing

- **Unit / differential:** the exact scalar-width failure (`take(msg, 14)`), a float
  widen/narrow at a call, a non-literal coerced arg, an aggregate arg (must still match
  exactly), a pointer arg (`!llvm.ptr` no-op). Each `assert_parity` vs the AST oracle.
- **Whole AST-codegen suite:** the checker mutates a tree both backends consume, so the
  existing backend tests must stay green (no divergence from the inserted `AsCast`s).
- **Corpus:** full flat-vs-legacy sweep, zero new miscompiles.
- **`.vxlib` round-trip** *(fallback only)*: if `FnSig.params` lands, serialize→
  deserialize a param-carrying signature; assert params survive and a v3 stamp is
  rejected.

## 8. Risks & mitigations

- **Blast radius of mutating the typed AST *(the main risk of Option C)*.** The checker
  feeds one tree to the AST codegen, the borrow checker, monomorphization, *and* the
  flat lowerer; an inserted `AsCast` must not confuse any of them (e.g. a borrow/linear
  argument wrapped in a cast, or double-coercion with the AST path's own `coerce_type`).
  Mitigation: insert only on assignable-but-differing edges; run the full backend +
  differential + corpus suites; keep the AST-path `coerce_type` until parity is proven.
- **Speculative (`silent`) passes.** The checker runs in `silent` mode for trial
  resolutions; mutating the tree there would corrupt it. Mitigation: gate the rewrite on
  `!silent` (the committing pass only).
- **Generic ordering.** A generic call's concrete parameter types exist only after
  monomorphization. Mitigation: insert the coercion after monomorphization has fixed the
  instance's params (or per-instance), never against a generic parameter.
- **Over-coercing (masking a real type error).** Mitigation: only coerce along
  `is_assignable`/`coerce_type`-legal edges; a genuine mismatch is still an error.
- **Method `self` indexing ambiguity** *(fallback only)*. If `FnSig.params` lands, fix
  the convention (params exclude `self`) and document it on the field.
- **Format bump invalidates `.vxlib` artifacts** *(fallback only)*. The stamp guard
  rejects stale artifacts cleanly; document the required rebuild.

## 9. Non-goals

- `&x`/`*p` scalar pointers and void externs — that is #235's remainder, independent
  of coercion.
- Replicating the AST's threaded `expected_type` machinery — a single rewrite at the
  call check is enough (§5.4).
- Cross-`.vxlib` argument coercion — deferred to the `FnSig.params` + format-bump
  fallback (§4.1/§5.1), landed only when a driver needs it.
- Removing the AST codegen's own call-site `coerce_type` — optional cleanup after
  parity is proven, not part of the fix.
