# Convergence finish: flat-codegen coverage (externs, methods) + the `--flat-codegen` production path

The two things standing between the flat pipeline and being able to compile a real stdlib program
(`import std::math; x.exp()`) end to end — and thus consume the `.vxlib` artifacts from #220:

1. **Coverage** — the flat HIR + emitter must lower **extern calls** and **method calls** (a stdlib
   math body is `impl Math for f32 { fn exp(self) { return expf(self); } }` — a method whose body calls
   an extern). Today both decline.
1. **A production path (C3)** — the flat codegen currently lives *only* in the differential test harness;
   `compile_pipeline`'s codegen phase emits no MLIR. `vxc` must be able to drive the flat path (behind
   `--flat-codegen`) and run it, then flip the default after corpus parity.

Grounded in a full read of `src/hir/flatten.rs`, `src/codegen/flat.rs`, `src/pipeline.rs`,
`src/driver.rs`, `src/jit.rs`. See also [`flat_pipeline_convergence.md`](./flat_pipeline_convergence.md)
(C0–C3), [`c2_flat_codegen.md`](./c2_flat_codegen.md) (the emitter bricks),
[`vxlib_bodies_and_loader.md`](./vxlib_bodies_and_loader.md) (#220 artifacts).

## 0. TL;DR / order of work

1. **Externs in the flat path** — register `module.externs` in `fn_sigs`; the emitter declares any
   *called-but-undefined* callee as `func.func private`, synthesizing its signature from the call's own
   operand types (no `FnSig.params` needed). Differential-tested with a libm extern (`sqrtf`).
1. **Method calls** — the type checker already rewrites `x.exp()` → `f32$exp(x)` before lowering; the
   remaining gap is that the mangled method fn isn't in the *frozen* `fn_sigs`/emit set. Land method
   bodies as emittable functions (#217).
1. **`--flat-codegen` production path (C3)** — a driver branch that lowers every function via
   `lower_function_to_hir` + `emit_module_mlir` and feeds the existing `execute_mlir` / object path;
   per-function fallback to the AST path so one declining function doesn't sink the program. Flip the
   default after the corpus is at parity.

Each step keeps the suite green and is differential-tested against the AST oracle.

## 1. Externs (the foundation)

**The gaps** (from the map):

- `build_frozen_registry` builds `fn_sigs` from `module.functions` only (`pipeline.rs:396-419`); it
  never touches `module.externs` (`ExternDecl { name, is_safe, params, return_type }`,
  `syntax/decl.rs:86-92`). So an extern is invisible to the flat path.
- `lower_call` resolves the callee via `registry.fn_sigs.get(name)?` (`flatten.rs:655`) → `None` for an
  extern → the whole function declines to lower.
- The emitter resolves the callee via `ctx.callees` (built from `fn_sigs`, `flat.rs:195-209`) at
  `flat.rs:682`; and `emit_module_mlir` only prepends `func.func private` decls for a hardcoded print
  list (`flat.rs:317-330`) — a called extern would emit `func.call @sqrtf` with no declaration and fail
  MLIR verification.

**The fix:**

1. **Register externs in `fn_sigs`.** In `build_frozen_registry`, after the `module.functions` loop,
   iterate `module.externs` and insert `FnSig { gid, ret_ty }` (GID minted from the extern's name like a
   function's). Reuse the cross-module ambiguity drop. This alone unblocks `lower_call` (`flatten.rs:655`)
   and the emitter callee lookup (`flat.rs:682`) for **scalar-returning** externs (libm math is all
   `f32→f32` / `f64→f64`).
1. **Declare called-but-undefined callees in `emit_module_mlir`.** Compute the set of *defined* function
   names (the `funcs` the module emits). Thread a `&mut` collector into `emit_function_mlir`; in the
   `Call` arm, when the callee name is not in the defined set, record `(name, operand MLIR types, ret MLIR type)` — the operand types come from the emitter's existing `etypes`/arg handling, so the
   synthesized `func.func private @name(<argtys>) -> <ret>` matches the emitted `func.call` by
   construction. Prepend the deduped private decls (alongside the print helpers).
1. **Void / pointer returns are out of scope** for this step (libm math is scalar). `lowered_ty`
   (`flatten.rs:914-931`) and the emitter `Call` arm (`callee.ret.clone()?`, `flat.rs:683`) keep
   declining them; a later step models `void`/`ptr`.

**Test.** A self-contained program declaring a real libm extern and calling it, compared flat-vs-AST via
the differential harness — e.g. `extern fn sqrtf(x: f32) -> f32; fn main() -> i32 { print(sqrtf(16.0)); return 0; }` (stdout parity; `sqrtf(16)=4`). `-lm` is already linked by `execute_mlir` (`jit.rs:197`),
so the symbol resolves at JIT time for both paths.

## 2. Method calls (#217)

The type checker's `check_methodcall_expr` rewrites `x.exp()` in place into a `FunctionCall`
`f32$exp(x)` (`codegen/lower/expr.rs`), *and* monomorphizes the method body into
`monomorphized_functions` under that mangled name. So after type-checking, the flat lowerer sees a plain
`FunctionCall` — the missing piece is that `f32$exp` is a *monomorphized* function that isn't in the
frozen `fn_sigs` (the registry froze before type-checking) and isn't in the module's original
`functions` when the emitter builds its callee map.

**The fix (two coordinated pieces):**

- The monomorphized method functions (from `check_results`) are already routed into `module.functions`
  by `codegen_and_metadata_phase` (`pipeline.rs:783+`). So in the **production flat path** (step 3), the
  emitter's `funcs` list *will* include `f32$exp` — its body emits as a `func.func @f32$exp`, and the
  caller's `f32$exp(x)` resolves against it. The remaining need is that the *callee GID* the `Call`
  carries matches — i.e. the monomorph's GID is in the callee map. Extend `build_callee_map` (or the
  emit-time name set) to include the emitted functions' own names, not only `fn_sigs`.
- For the differential harness (single-module, no import), a struct/scalar method whose `impl` is in the
  same source already type-checks + monomorphizes; the test lowers the rewritten `FunctionCall` + the
  monomorph body together (as the multi-function harness already does for free-function calls).

**Test.** `impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } } fn main() -> i32 { let r = 2.0.sq(); print(r); return 0; }` flat-vs-AST. Then a method whose body calls an extern
(`fn exp(self) { return expf(self); }`) — combines step 1 + step 2, the stdlib shape.

## 3. `--flat-codegen` production path (C3)

**Gap:** no non-test caller of `emit_module_mlir` exists; `compile_pipeline` emits no MLIR
(`pipeline.rs:783+` only writes metadata). The AST production path is `driver.rs::run_codegen` →
`MeliorGenerator` → `execute_mlir`/object (`driver.rs:471-583`, `jit.rs:30`).

**The fix:** add a flat branch reusing the existing backend. The cleanest first cut is in the **driver**
(not `compile_pipeline`, which is the parallel *frontend* and separate), so `--flat-codegen` runs after
the same load/resolve/type-check the AST path uses, then:

1. build the frozen registry over the resolved modules;
1. lower every (monomorphized) function via `lower_function_to_hir`, collecting `local_tensor_types`
   (mirror `flat.rs::emit_module_and_verify`, `flat.rs:1013-1076`);
1. `emit_module_mlir(&funcs, &registry, &tensor_types)` → module text;
1. feed it to the *same* `lower_to_llvm` + `execute_mlir` / `EmitObj` the AST path uses
   (`driver.rs:571-583`).

**Keep-green:** `emit_module_mlir` is whole-module atomic — one declining function returns `None`.
Until coverage is total, either (a) `--flat-codegen` falls back to the AST path for a module that
declines (so it never regresses), or (b) it errors clearly ("function `f` outside the flat subset").
Start with (a) — the flag is opt-in and differential; the AST path stays the oracle.

**Flip (later):** once the corpus compiles + runs at parity under `--flat-codegen`, make it the default
and move the AST path behind `--legacy-codegen`, per [`flat_pipeline_convergence.md`](./flat_pipeline_convergence.md) §C3.
This is the point at which a downstream compile can consume a precompiled `std.vxlib` (#220) — the flat
path links the artifact bodies; the AST path never could.

## 4. Non-goals / deferred

- `void`/`ptr` extern returns; varargs (`printf`) — modelled later, as the corpus demands.
- `Cast`/`Neg`/`Not`/`Matmul` emitter arms (#214) and struct returns (#215) — needed for *total* corpus
  coverage (the flip), tracked separately; the `--flat-codegen` fallback (3a) tolerates them meanwhile.
- Removing the AST back end — only after a soak at parity.

## 5. Work breakdown (each a green, differential-tested commit)

- **E1.** Externs → `fn_sigs`; emit private decls for called-undefined callees; `sqrtf` differential.
- **E2.** Method-call emit (callee map includes emitted monomorphs); scalar/struct method differential;
  then method-calling-extern (stdlib shape).
- **E3.** `--flat-codegen` driver branch (lower-all + `emit_module_mlir` + reuse `execute_mlir`), with
  per-module AST fallback; golden/differential over a slice of the corpus.
- **E4.** Widen the emitter (casts/neg/not, struct returns) until a target corpus subset is total under
  `--flat-codegen`; then the `import std::math` end-to-end (source *and* `std.vxlib`) parity.
