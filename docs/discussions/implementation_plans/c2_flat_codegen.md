# Implementation Plan: C2 — Flat Codegen (`local_hir_stream` → MLIR)

> Epic [#197](https://github.com/hiraditya/Vx/issues/197); issue
> [#200](https://github.com/hiraditya/Vx/issues/200). Depends on C1 (#198, done). Journal:
> [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md) Entries 12, 19, 20.

## Goal

A backend that lowers a function's flat streams — `local_hir_stream` (bytecode) + `local_type_stream`
(GIDs) + the frozen registry — to MLIR, driven by the **instruction array** instead of an AST walk.
Reuse the existing melior emission at the leaves conceptually, but the driver is the flat array
("Phase 7: O(1) array codegen"). The AST path (`driver.rs::run_codegen` → `MeliorGenerator`) stays the
**oracle** until parity; only then does `vxc` flip (C3).

## Current state

`src/codegen/flat.rs` emits `func.func`s as **text**, driven by a per-register `etypes[]` type
recovery. `emit_function_mlir(func, hir, types, callees)` does **one** function; `emit_module_mlir( funcs, registry)` does a **whole program** (needed for calls — the callee's `func.func` must be
present). Handled subset: the C2.0 scalar core (`Load` → block arg, `Const`, `Add`/`Sub`/`Mul`/`Div`,
`Ret`); **brick 1 — control flow** (`Cmp` → `arith.cmpi/cmpf`; `BlockStart`/`Br`/`CondBr` → `cf`;
`Alloca`/`Store`/`SlotLoad` → rank-0 `memref`); **brick 2 — fixed-arity scalar calls** (`Arg`/`Call`
→ `func.call @name(...)`, callee GID→name via `build_callee_map` over the registry `fn_sigs`); and
**brick 3a — all-scalar-field structs** (aggregate `Alloca` → `llvm.alloca`; `FieldLoad`/`FieldStore`
→ `getelementptr` + `llvm.load`/`store`, layout via `build_agg_map`); and **brick 3b — tensor alloc +
scalar element access** (`TensorAlloc` → static `memref.alloc`; scalar `TensorIndex`/`TensorStore` →
`index_cast` + `memref.load`/`store`, tensor shapes via the side table). All bundled into `EmitCtx`.
Everything else returns `None` (the AST path stays the oracle). String-based: the text is wrapped in
`module { … }`, parsed by melior, run through `lower_to_llvm`, then JIT'd.

The **differential harness** (`tests/integration_test/flat_codegen_differential.rs`) is the acceptance
gate: for a `main` the flat path lowers, `flat_exit_code == ast_exit_code == expected` (process exit
code via `execute_mlir`). Every C2 brick below extends this harness with cases that were previously
declined.

## The parity contract

"Matches the AST path" means **same JIT result**, not identical text. SSA names and block structure
can differ; what must agree is the observable behavior (exit code, and later `print` output). So each
brick emits MLIR that is *semantically equivalent* to what `MeliorGenerator` emits for the same
construct — study the AST lowering in `src/codegen/lower/` and match its op choices (same dialects, so
`lower_to_llvm` + JIT behave identically).

## Cross-cutting work

- **Per-register types.** ✅ *scalar done* (Entry 21). `flat.rs` tracks a parallel `etypes[idx]`
  (recovered from each producing instruction's `type_idx` → GID via `elem_of_gid`), so consumers
  (`Cmp`, `Store`, `Call` args) can print operand types. **Brick 3 must extend it** to tensor (invert
  `tensor_gid`) and aggregate (registry `layouts` GID → `!llvm.struct`/memref) recovery.
- **Module-level emitter.** ✅ *done* (Entry 22). `emit_module_mlir(funcs, registry)` emits each
  function and concatenates, declining the whole module if any function is outside the subset
  (keep-green atomicity at the module level). `build_frozen_registry` is now `pub` so external callers
  (the differential harness, later C3) can build the registry that resolves callees.

## Bricks (in order)

### Brick 1 — Control flow ✅ done (Entry 21)

**Landed.** `if`/`for`/`loop` `main`s now JIT-match the AST oracle through the differential harness.
`Cmp` (needed for conditions) joined the subset; terminators are emitted inline; block 0 is the func's
implicit entry (no label). Per-register `etypes[]` recovery feeds `Cmp`/`Store` their operand/slot
types. The original brick sketch (kept for reference):

C1.2 opcodes: `BlockStart(imm=block id)`, `Br(imm=target)`, `CondBr(operand1=cond, imm=then|else<<32)`, `Alloca(type_idx=elem, → slot)`, `Store(operand1=slot, operand2=val)`,
`SlotLoad(operand1=slot)`.

MLIR (match the AST codegen's `cf` + memref/alloca locals — see `MeliorGenerator::generate_function`
and `codegen/lower/control_flow.rs`):

- `BlockStart b` → open MLIR block `^bbb:` (block 0 is the entry; the flat entry `BlockStart 0` maps to
  the func's entry block). Track a name per block id.
- `Br t` → `cf.br ^bbt`. `CondBr` → `cf.cond_br %cond, ^bbthen, ^bbelse`.
- `Alloca` → `%s = memref.alloca() : memref<T>` (element `T` from `type_idx`). `Store` →
  `memref.store %v, %s[] : memref<T>`. `SlotLoad` → `%r = memref.load %s[] : memref<T>` (rank-0
  memref; confirm against how the AST path allocs scalar locals — it may use a 1-elem memref or
  `llvm.alloca`, match it).

Gotcha: the flat stream interleaves value instructions and block markers; emit into the *current*
block, switching on `BlockStart`. Terminators (`Br`/`CondBr`/`Ret`) close a block. The differential
test's `flat_declines_control_flow_leaving_ast_the_oracle` becomes a *parity* case once this lands.

### Brick 2 — Calls ✅ done (Entry 22)

**Landed.** Fixed-arity scalar calls JIT-match the AST oracle. `build_callee_map(registry)` inverts
`fn_sigs` to `GID → Callee { name, ret }`; `Arg`s push value regs onto a `pending_args` stack and the
`Call` consumes its `imm` trailing entries (nested calls nest cleanly — each call's args are the
tail); emits `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret` (arg types from `etypes[]`, `Tret` from
the callee's `fn_sig`). Scalar-returning only; void/non-scalar return declines (#198, brick 3). The
original brick sketch (kept for reference):

Opcodes: `Arg(operand1=arg reg)` (N before a `Call`), `Call(type_idx=callee GID, imm=arg count)`.

- **Callee name.** `Call.type_idx` is the callee's *GID*. To emit `func.call @name`, build the reverse
  of registry `fn_sigs` (GID → name) — pass the registry (or a `HashMap<TypeId, (name, ret_ty)>`) into
  the emitter. The AST callee is a `FlatSymbolRefAttribute` name (`codegen/lower/expr.rs` ~2068).
- **Args + signature.** Gather the `operand1`s of the `imm` `Arg`s immediately preceding the `Call`;
  their types come from the per-register `types[]` (cross-cutting work). Emit
  `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret` where `Tret` = callee's return type (registry
  `fn_sigs[gid].ret_ty`).

### Brick 3 — Non-scalar (structs + tensors)

The biggest brick, split into: **3a structs ✅ done**, then tensor read, then tensor write. Match the
AST lowering in `codegen/lower/{expr,tensors,mod}.rs`; the flat opcodes were designed to mirror it.

**3a — Structs ✅ done (Entry 23).** All-scalar-field struct construction + field access JIT-match the
AST oracle. `build_agg_map(registry)` → `GID → AggLayout { struct_ty, offsets }` (bundled with the
callee map into `EmitCtx`); an aggregate `Alloca` → `llvm.alloca` of `!llvm.struct<(...)>` (slot
pointer tracked in `agg_of[reg]`); `FieldLoad`/`FieldStore` → `llvm.getelementptr %slot[0, idx]`
(field index recovered by matching the byte offset against the layout) + `llvm.load`/`store`. The
differential harness + unit-test helper now type-check first (for the `StructInit` GID annotation).
Deferred: struct params/returns/copy (#215), nested-aggregate/pointer fields (#212).

**3b — Tensor alloc + scalar element access ✅ done (Entry 25).** Backed by the tensor-type side table
(below): `TensorAlloc` → `memref.alloc()` of a static `memref<NxT>` (shape recovered by GID);
scalar-element `TensorIndex` → `arith.index_cast` + `memref.load` (read) or a recorded element place
(`imm = 1`); scalar-element `TensorStore` → `memref.store`. First tensor program JIT-matches the AST.

**3c — Tensors (remaining).** `TensorIndex` sub-views (rows) → `memref.subview`/`reinterpret_cast`
(slice-ops S1, `slice_operators.md`). `Reduce` → `vector.load` (+ `arith.mulf` for `dot`) +
`vector.reduction<add|maximumf|minimumf>` (S2). Tensor elementwise (`Mul`/`Add`/… with a tensor result
type) → `vector.load` + `arith.*` + `vector.store` (S3). Row `TensorStore` → `vector.store`. `Transfer`
→ `vx.transfer` (`target_topology` = the imm dispatch id). Tensor params → a `memref` in the func
signature. The attention corpus (`tests/backend/pass/*_attention.vx`) is the eventual differential
target.

#### Design decision — tensor-type recovery needs a side table (not GID inversion)

The plan's original cross-cutting note said to recover an operand's tensor type by "inverting
`tensor_gid`". **That isn't possible.** `tensor_gid(elem, shape)` is a content **hash**; there is no
inverse. Contrast the two type families that *are* recoverable:

- **Scalars** — `elem_of_gid` brute-forces the finite set of scalar variants (`scalar_gid(e) == gid`).
- **Structs** — the frozen registry holds `layouts: GID → TypeDefinition`, so `build_agg_map` recovers
  the `!llvm.struct` shape by GID lookup.

Tensors have neither: the set of `(elem, shape)` is unbounded, and tensor types are *structural*, so
they never enter the nominal registry. Yet the emitter must reconstruct a memref type — most acutely
for **`TensorAlloc`**, which introduces a fresh tensor whose shape exists *only* as the hash in the
type stream (it can't be derived by forward-propagation from other tracked values the way
`TensorIndex`/elementwise/`Transfer` results can).

**Decision:** carry a **tensor-type side table** — `GID → (elem, shape)` — recorded by the lowerer
(`hir/flatten.rs`) as it emits each tensor-typed value, stored on `LocalWorkerState`
(`local_tensor_types`), merged across functions (the hash is globally consistent), and threaded to the
emitter via `EmitCtx.tensors`. This is the tensor analogue of struct `layouts`. It keeps the flat type
stream unchanged (still GIDs) while making tensor shapes recoverable at codegen. (The alternative —
encoding the full `(elem, shape)` inline in the type stream instead of a hash — is a larger change to
the stream format and is not pursued.)

Two related constraints found while matching the AST oracle:

- **Reductions are f32-only in the AST** (`lower_slice_reduction` hardcodes `vector<Dxf32>` → `f32`).
  So a reduction can't produce an i32 exit code, and casts are declined (#214) — early tensor
  differential tests reduce to a scalar element read (`return q[k]`) rather than a `sum`.
- **The flat path may use static memrefs** (`memref<4xi32>`) where the AST uses dynamic
  (`memref<?xi32>` + size operands). Parity is the JIT result, not the text, so either is fine; static
  is simpler (no dynamic-size operands). Indices are `arith.index_cast`'d to `index` for
  `memref.load`/`store`, matching the AST.

## Then C3 (#201)

Add a `--flat-codegen` driver flag (opt into the flat path per function, AST fallback), run the corpus
differentially under it, and once parity holds, flip `vxc` from `driver.rs::run_codegen` to the flat
codegen. Retire the AST middle/back-end after a soak.

## Key files

- Emitter: `src/codegen/flat.rs`. Streams/opcodes: `src/hir/{flatten.rs,bytecode.rs}`. Registry:
  `src/registry.rs` (`layouts`, `fn_sigs`), `src/pipeline.rs::build_frozen_registry`.
- AST oracle to match: `src/codegen/lower/{expr,tensors,control_flow,mod}.rs`,
  `src/codegen/generator.rs`. Lowering pipeline: `src/codegen/mod.rs::lower_to_llvm`; JIT:
  `src/jit.rs::execute_mlir`.
- Harness to extend: `tests/integration_test/flat_codegen_differential.rs`.
