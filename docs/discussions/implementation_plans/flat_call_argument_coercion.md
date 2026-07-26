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

## 3. Why the flat path can't do it yet

`registry::FnSig` carries only `{ gid, ret_ty }` — **not the parameter types.** So
neither `flatten::lower_call` (no param types to coerce against) nor the emitter's
`Call` arm (its `Callee` is built from `FnSig`) can know what to coerce to. Fixing
#236 therefore requires **adding parameter types to `FnSig`**, which is where the
"wider consequences" live.

## 4. Design

### 4.1 Data model — `FnSig.params`

```rust
pub struct FnSig {
    pub gid: TypeId,
    pub params: Vec<crate::syntax::Type>,   // NEW
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

### 4.2 Where to coerce — two options

**Option A — in `flatten::lower_call` (insert `Cast`).** After lowering each argument
`Val`, compare its `LoweredTy` to `lowered_ty(param)`; if they differ and the pair is
a modelled scalar conversion, emit a `Cast` opcode (target = param scalar) before the
`Arg`. Pros: the coercion lives in the HIR, so a serialized body (`.vxlib`) carries it;
the emitter stays simple. Cons: `Cast` today is scalar-only; a memref/tensor arg
coercion (`memref.cast`) has no HIR opcode yet.

**Option B — in the emitter's `Call` arm (`cast_op`).** Thread the callee's param
types into `Callee`, and coerce each argument's SSA value to its param type with the
existing `cast_op` (scalars) / a new `memref.cast` (tensors), exactly as the `Ret`
and `TensorStore` arms do. Pros: reuses the emitter coercion already proven for #232
/#234; handles memref→memref uniformly. Cons: the coercion is re-derived at emit time
rather than recorded in the HIR (fine — codegen is deterministic from the stream +
registry).

**Recommendation: Option B**, for consistency with the existing return/store coercion
and because it handles the non-scalar (`memref.cast`) case without a new opcode. Add
`params: Vec<Type>` (or a pre-lowered `Vec<ParamKind>`) to `Callee`; in the `Call`
arm, for each `pending_args` entry coerce its value to the param type before building
the operand list, declining (return `None`, keep-green) on an unmodelled conversion.

### 4.3 Coercion taxonomy (what to handle vs decline)

| param vs arg | action |
|---|---|
| identical MLIR type | no-op (pass through) |
| scalar↔scalar (width/int↔float/sign) | `cast_op` (already modelled) |
| memref↔memref (shape/layout differ) | `memref.cast` (mirror the AST) |
| pointer↔pointer (`!llvm.ptr`) | no-op (opaque) |
| aggregate↔aggregate | require exact GID match, else decline |
| scalar→tensor (broadcast) | **decline** (the AST's `linalg.fill`; rare at a call — defer) |
| scalar→pointer (`inttoptr`) | **decline** (rare; the AST does `inttoptr` only via `AsCast`) |
| anything `cast_op` returns `None` for | **decline** (AST stays the oracle) |

Declining keeps the module-level keep-green atomicity: an unmodelled coercion drops
the whole program to the AST path rather than emitting a wrong or invalid call.

## 5. Wider consequences (the reason for this doc)

### 5.1 `.vxlib` format version bump — v3 → v4 *(the biggest one)*

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

The AST's `expected_type` makes a literal adopt the param type *during* lowering; the
flat path instead lowers the argument to whatever type the checker annotated, then
coerces post-hoc. Post-hoc `cast_op` coercion is **sufficient** for the width/kind
mismatches at issue (it is exactly what the AST's `coerce_type` fallback does). We do
*not* need to replicate the `expected_type` machinery — one coercion point suffices,
which keeps the change small. (The one case post-hoc coercion can't recover is a
literal whose *checked* type is already wrong for a non-cast reason; those are already
declined by `cast_op` returning `None`.)

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

## 6. Implementation plan (ordered, each step green)

1. **`FnSig.params` field** + populate the three `build_frozen_registry` loops
   (fn/extern/method). Fix all `FnSig { .. }` literals (registry defaults, tests).
   *No behavior change yet — the field is unused.* Build + full test suite green.
1. **`.vxlib` v4** — bump `VXLIB_FORMAT_TAG`; encode/decode params in `fn_sigs` /
   `methods`; extend the interface round-trip test with a param-carrying signature.
1. **Emitter coercion (Option B)** — carry param types into `Callee`; coerce each
   argument in the `Call` arm via `cast_op` (scalars) + `memref.cast` (memrefs),
   declining unmodelled conversions.
1. **Differential tests** — the `take(msg, 14)` case (now lowers, JIT parity); a
   float-width call (`f32` arg → `f64` param); a cross-`.vxlib` width-mismatched call
   (extend the artifact-linking test). Re-run the extern drivers to confirm robust
   (not lucky) ABI.
1. **Corpus sweep** — flat-vs-legacy parity; expect flat-used to hold or rise, zero
   new miscompiles.
1. Commit (`Fixes: #236`), journal, close.

## 7. Testing

- **Unit / differential:** the exact scalar-width failure (`take(msg, 14)`), a float
  widen/narrow at a call, an aggregate arg (must still match exactly), a pointer arg
  (`!llvm.ptr` no-op). Each `assert_parity` vs the AST oracle.
- **`.vxlib` round-trip:** serialize→deserialize a registry whose `fn_sigs` includes a
  param-carrying signature; assert params survive; assert a v3 stamp is rejected.
- **Corpus:** full flat-vs-legacy sweep, zero new miscompiles.

## 8. Risks & mitigations

- **Format bump invalidates existing `.vxlib` artifacts.** Mitigation: the stamp guard
  already rejects stale artifacts cleanly; document the required rebuild.
- **Over-coercing (masking a real type error).** Mitigation: only coerce along
  `coerce_type`-legal edges; decline everything else, so a genuine mismatch drops to
  the AST path rather than being silently bridged.
- **Method `self` indexing ambiguity.** Mitigation: fix the convention now (params
  exclude `self`); document it on the field.

## 9. Non-goals

- `&x`/`*p` scalar pointers and void externs — that is #235's remainder, independent
  of coercion.
- Replicating the AST's `expected_type` literal-typing pass (post-hoc coercion is
  enough — §5.4).
- Method-call lowering itself (a separate convergence step); this only makes the
  *data model* ready for it.
