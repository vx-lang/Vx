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

`src/codegen/flat.rs::emit_function_mlir(func, hir, types) -> Option<String>` emits a **single**
`func.func` as **text**, handling the C2.0 scalar subset (`Load` → block arg, `Const`,
`Add`/`Sub`/`Mul`/`Div`, `Ret`) **plus brick 1 — intra-function control flow** (`Cmp` →
`arith.cmpi/cmpf`; `BlockStart`/`Br`/`CondBr` → `cf`; `Alloca`/`Store`/`SlotLoad` → rank-0
`memref`), driven by a per-register `etypes[]` type recovery. Everything else returns `None` (the AST
path stays the oracle). It is string-based: the emitted text is wrapped in `module { … }`, parsed by
melior, run through `lower_to_llvm`, then JIT'd.

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

## Cross-cutting work (needed before/with brick 2)

- **Per-register types.** Emitting most ops needs the *type* of each operand register (e.g.
  `func.call @f(%a) : (i32) -> i32`, `memref.store %v, %s[...]`). `flat.rs` currently tracks only SSA
  *names* (`names[idx]`). Add a parallel `types[idx]` (recovered from each producing instruction's
  `type_idx` → GID → element/aggregate/tensor), so any consumer can print operand types. `elem_of_gid`
  already inverts scalar GIDs; extend with tensor (invert `tensor_gid`) and aggregate (registry
  `layouts` GID → `!llvm.struct`/memref) recovery.
- **Module-level emitter.** `emit_function_mlir` does one function; calls need *all* functions in one
  module. Add `emit_module_mlir(functions, per-fn streams, registry) -> Option<String>` that emits
  each function and concatenates, declining the whole module if any function is outside the subset
  (keep-green atomicity at the module level).

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

### Brick 2 — Calls

Opcodes: `Arg(operand1=arg reg)` (N before a `Call`), `Call(type_idx=callee GID, imm=arg count)`.

- **Callee name.** `Call.type_idx` is the callee's *GID*. To emit `func.call @name`, build the reverse
  of registry `fn_sigs` (GID → name) — pass the registry (or a `HashMap<TypeId, (name, ret_ty)>`) into
  the emitter. The AST callee is a `FlatSymbolRefAttribute` name (`codegen/lower/expr.rs` ~2068).
- **Args + signature.** Gather the `operand1`s of the `imm` `Arg`s immediately preceding the `Call`;
  their types come from the per-register `types[]` (cross-cutting work). Emit
  `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret` where `Tret` = callee's return type (registry
  `fn_sigs[gid].ret_ty`).
- Needs the **module-level emitter** so the callee's `func.func` is present. Extend the differential
  harness with a `main` that calls a scalar helper (`fn add(a,b){a+b} fn main(){ add(3,4) }`).

### Brick 3 — Non-scalar (structs + tensors)

Match the AST lowering in `codegen/lower/{expr,tensors,mod}.rs`; the flat opcodes were designed to
mirror it:

- `Alloca` of an aggregate (imm = size) / `TensorAlloc` (imm = byte size) → `memref.alloc` /
  `llvm.alloca` of the struct/tensor type. `FieldLoad`/`FieldStore` → GEP + `llvm.load`/`store` at the
  layout offset (registry `layouts[gid].fields`). `TensorIndex` → `memref.subview`/`reinterpret_cast`
  (row) or `memref.load` (element) — see the slice-ops S1 lowering (`slice_operators.md`). `Reduce` →
  `vector.load` (+ `arith.mulf` for `dot`) + `vector.reduction<add|maximumf|minimumf>` (S2). Tensor
  elementwise (`Mul`/`Add`/… with a tensor result type) → `vector.load` + `arith.*` + `vector.store`
  (S3). `TensorStore` → `vector.store`. `Transfer` → `vx.transfer` (`target_topology` = the imm
  dispatch id).
- This is the biggest brick; split it (structs first, then tensor read, then tensor write). The
  attention corpus (`tests/backend/pass/*_attention.vx`) is the eventual differential target — once
  these emit, a corpus program can be JIT-compared flat-vs-AST.

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
