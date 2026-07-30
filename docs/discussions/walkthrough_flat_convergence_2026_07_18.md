# Walkthrough — Flat-pipeline convergence push (2026-07-17 → 07-18)

A session-handoff record: what landed, the tracker state, and where to pick up. The convergence
detail lives in the journal ([`parallel_pipeline_convergence.md`](./parallel_pipeline_convergence.md),
Entries 13–20); the C2 plan is [`implementation_plans/c2_flat_codegen.md`](./implementation_plans/c2_flat_codegen.md).
This doc is the index over the whole session, including the non-convergence work.

## Headline

The flat pipeline went from **"structs and tensors are fully blocked in the flat HIR"** to
**C1 complete** — the flat HIR now lowers the entire language surface a function body uses (scalars,
control flow, structs, tensors read+write, and fixed-arity calls), and **C2 is started** (a working
flat-vs-AST differential harness for the scalar subset). The FlashAttention inner loop lowers to the
flat HIR in both directions.

## What landed (grouped)

### 1. Bug fixes (start of session)

- **#185** — scalar `print(x)` (call form) failed codegen; now routes to the `print_*` FFI helpers
  (added `print_i64`). `87dde50`.
- **#186** — a loop index used in a float expr left an unreconciled `i64→index` cast; `coerce_type`
  now routes `index↔float` through i64 (`index_cast` + `sitofp`/`fptosi`). `4d01f3d`.

### 2. Backlog hygiene (verified-done issues closed)

Several "open" issues were already implemented on `main` (commits used `(#N)`, not `Fixes:`), so they
never auto-closed. Verified against code + passing acceptance tests, then closed: **#193/#194/#195**
(GID word-2 codec / cross-module resolution / stable FNV hash) and the entire slice-ops epic
**#188–#192** (S1 indexing … S4 FA-4). No code change — tracker reconciliation only.

### 3. #199 — the non-scalar flat HIR (C1.3), now closed

The bulk of the session. Journal Entries 13–18.

- Freeze-time **struct/enum layouts** (size/align/field offsets) — new `src/layout.rs`,
  `TypeDefinition.fields`. `3a19fa0`.
- **Aggregate values**: `LoweredTy::{Scalar,Aggregate,Tensor}`; struct params bind to registry-sized
  `Alloca`; `FieldLoad` (scalar field read); `StructInit` GID annotated by the type checker
  (`GlobalAstEnv::struct_gids`) → struct construction via `Alloca` + `FieldStore`. `f55466d`,
  `a90828a`, `fe4b414`, `9f19d01`.
- **Tensors**: stable `tensor_gid` (element + shape) in the flat type stream (signature + body);
  `LoweredTy::Tensor{elem,shape}`; rank-reducing `TensorIndex` (`q[i]`/`q[i][j]`); `Reduce`
  (`dot`/`sum`/`max`/`min`); elementwise (arith ops with a tensor result type); `TensorAlloc` (byte-
  sized storage), `TensorStore` (`o[i]=slice`), `Transfer`. `f0aa1c2`, `0ac1bf5`, `51f7b6f`,
  `6c77f29`, `97626bf`. Capstones: `flashattention_score_expression_composes`,
  `flashattention_write_path_composes`.

### 4. Attention corpus (#200/#197)

Five hand-checkable, JIT-verified attention variants under `tests/backend/pass/` — the differential
target for C2 and standalone AST-oracle coverage now: `full_softmax_attention.vx`,
`multi_query_attention.vx`, `grouped_query_attention.vx`, `linear_attention.vx`,
`sparse_local_attention.vx`. `9398c4d`.

### 5. #200 — C2 start: differential harness

`tests/integration_test/flat_codegen_differential.rs` — flat-vs-AST **JIT exit-code parity** for the
scalar subset, through the real `lower_to_llvm` + `execute_mlir`. `bfe5156`. Journal Entry 20.

### 6. #198 — C1 Calls (C1 now complete)

Fixed-arity calls: registry `fn_sigs` (name → GID + return type) for callee resolution, the `Arg`
opcode for N-ary args, `Call` (callee GID in `type_idx`, arg count in `imm`). `08c8d3a`. Journal
Entry 19.

## Tracker state after the session

- **Closed:** #185, #186 (via `Fixes:` trailers, on push), #188–#195, #199.
- **#198** — C1 complete (C1.1–C1.3 + Calls); left open only for void/non-scalar-return calls.
- **#200** — differential harness landed; the flat-emitter growth (C2) is the open bulk. Roadmap in
  `implementation_plans/c2_flat_codegen.md`.
- **#201** — C3 (flip `vxc`), untouched.
- **Filed this session:** #211 (parallel-test struct/tensor coverage), #212 (declined flat-HIR edge
  cases: nested/pointer field access, scalar-element tensor store), #213 (evaluate C-style varargs —
  *not* needed for ordinary calls).

Everything is committed locally; **nothing pushed** (standing rule). The `Fixes:` trailers on the
#185/#186 commits will auto-close those on push.

## Where to pick up (fresh session)

**C2** (`implementation_plans/c2_flat_codegen.md`). The flat emitter `src/codegen/flat.rs` handles only
straight-line scalar arithmetic; grow it opcode-family by family, extending the differential harness at
each step: **(1) control flow** (`cf` + `memref.alloca`), **(2) calls** (needs a module-level emitter +
callee GID→name), **(3) the non-scalar surface** (memref/vector, so the attention corpus runs through
the flat path). Then **C3**: `--flat-codegen` flag → flip `vxc`.

Read first: this doc → journal "Status (2026-07-18)" + Entry 20 → `c2_flat_codegen.md` → `flat.rs` and
its tests.
