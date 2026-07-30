# Implementation Plan: HIR Lowering — AST → flat bytecode stream (C1, #197)

> Milestone **C1** of the flat-pipeline convergence
> ([`flat_pipeline_convergence.md`](./flat_pipeline_convergence.md)). Populates
> `LocalWorkerState::local_hir_stream` (`Vec<HirInstruction>`) by lowering type-checked function
> bodies to flat bytecode — the "instruction selection" pass the architecture calls Phase 7's input.
> Journal: [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md).

## Why

Today the flat pipeline harvests only function *signature* type references into `local_type_stream`
(`pipeline.rs::emit_function_type_gids`); `local_hir_stream` is defined (`hir/bytecode.rs`) but never
populated, so codegen has nothing flat to consume and still walks the AST. C1 is the long pole: an
instruction-selection pass from the AST body to a flat `Vec<HirInstruction>`.

Note: `hir/lower_ast.rs` lowers the AST to a *different* HIR — an arena-of-nodes **tree**
(`HirArena`/`ExprId`). C1 targets the **flat bytecode** `local_hir_stream`, which is what Phase 7
codegen (C2) will loop over with O(1) type-stream lookups. They are distinct representations.

## The instruction & SSA model

`HirInstruction { opcode, operand1: Register, operand2: Register, type_idx: TypeIdx, imm: u64 }`
(`hir/bytecode.rs`). Conventions this pass establishes (and C2 must honor):

- **SSA by position:** the instruction at index `i` in `local_hir_stream` *defines* SSA value
  `Register(i)`. Operands name earlier instructions by their index. No separate destination field.
- **`type_idx`** indexes `local_type_stream` (the shared `Vec<TypeId>` pool, appended to by both
  signature harvesting and this pass). It is the GID of the value the instruction produces.
- **Params** are materialized as `Load` instructions at the top of the body (`imm` = param index),
  giving each parameter an SSA register; the name→register map seeds from them.
- **Locals:** `let x = e` binds `x` to `e`'s result register (pure SSA alias, no store) for the
  immutable subset; mutation (`Assign`) is deferred to C1.2 with `Store`/reload.

> [!IMPORTANT]
> **Register by default, demote on address-taken.** Because locals start as SSA registers
> (`Binding::Reg`) rather than allocas, this pass is the *inverse* of LLVM's `mem2reg`: clang allocas
> everything and promotes what is never addressed; Vx registers everything and must **demote** what
> is. Anything needing an address — an aggregate, a value crossing a block boundary, or an
> `&x` — has to be moved to a slot (`Binding::Slot`) rather than being refused for lacking one.
>
> Today that choice is a **function-global** flag (memory mode: on if the function has control flow
> or any aggregate local), which both over-allocates and blocks `&x` on a scalar. Making it per-local
> is the fix, and it is the load-bearing idea behind
> [`scalar_references_flat.md`](scalar_references_flat.md) — read that before touching the
> `Reg`/`Slot` decision.

## All-or-nothing per function (keep-green)

Lowering a function is **atomic**: an unsupported construct aborts and the partial output is
discarded, so `local_hir_stream` is *either a complete, correct lowering or empty* — never partial or
wrong. This lets the corpus grow safely: unsupported functions are simply un-lowered until their
constructs land, and C2 differential testing (later) can trust every non-empty stream.

## Subsets (grown by corpus)

- **C1.1 — scalar core.** Params (scalar), integer/float/bool literals (`Const`), locals
  (`let` + identifier reads), arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Matmul`), `return` (`Ret`), and
  expression statements. Scalar-typed only; any struct/generic/call/control-flow aborts. Verified
  structurally.
- **C1.2 — value ops & control flow.** Comparisons (`Cmp`), `as` casts (`Cast`), unary (`Neg`/`Not`);
  `if`/`else` and loops (`loop`, `for`-range) → basic blocks (`BlockStart`) + branches
  (`Br`/`CondBr`) with `break`/`continue`; mutable/loop-carried locals via the **memory model**
  (`Alloca`/`Store`/`SlotLoad`), matching the AST codegen's `cf`+`alloca` lowering. Straight-line
  functions stay pure-SSA. Function/method calls are *not* here — they need function-symbol
  resolution into the flat path (a separate C0.2-style step) and a variadic-arg representation.
- **C1.3 — memory & the `vx` surface.** Struct/field access, tensor/slice ops, `spawn`/`transfer` —
  the dialect ops codegen must ultimately emit.
  - `spawn on (<topology>) { body }` → a `Spawn`/`SpawnEnd` region carrying the topology dispatch
    id (done; statement-form, straight-line body). Codegen rebuilds `vx.spawn` from the marker pair.
  - **Blocked (needs new modelling):** `transfer` (its source is a tensor/`Ref`, so it needs
    non-scalar values in the stream), tensor/slice ops (need tensor types in the type stream), and
    struct/field access (need field offsets — `ImmutableGlobalRegistry` layouts are currently
    `size_bytes = 0`; layout computation is a prerequisite). These are the next sub-project: extend
    the type/value model beyond scalars (aggregate + tensor GIDs, `Alloca` sizing, field/index
    opcodes), which also unblocks `transfer`.

## Verification

A debug hook (`verify_hir_stream`) asserts, per worker: every `type_idx` is in-bounds of
`local_type_stream`; every operand an opcode actually reads references a strictly-earlier instruction
(SSA dominance for the straight-line subset); and the stream is well-formed. Where feasible, later
subsets add re-execution/parity checks against the AST path (C2).

## Code pointers

- `src/hir/flatten.rs` (new) — the pass: `lower_function_to_hir(func, worker) -> bool`.
- `src/pipeline.rs::type_check_phase` — calls it per function next to `emit_function_type_gids`.
- `src/hir/bytecode.rs` — `HirInstruction`/`Opcode`; `src/session.rs` — the streams.

## Status

- **C1.1 — done** (`1af30aa`): scalar core.
- **C1.2 — done**: value ops (`0c687a7`), `if`/`else` (`dee88b4`), loops + break/continue (`7d792d6`).
  Calls deferred (need function-symbol resolution + variadic args).
- **C1.3 — in progress**: `spawn` region done (`77f66d3`). `transfer`/tensor/struct **blocked on
  non-scalar type + layout modelling** (registry layouts are size 0) — the next sub-project.
