# Proposal: Slice-Level Operators → Vector Dialect

> Status: proposal / design. Driving use case: FlashAttention
> ([`flash_attention.md`](./flash_attention.md)) — the inner `for d { acc += q[i][d]*k[j][d] }`
> and the row-wise softmax reductions. Design decision taken (with the user): expose
> **slice-level operators** (slice indexing + `dot`/`sum`/`max`/elementwise) that lower directly
> to the MLIR **`vector`** dialect, so the code is vectorized *by construction* rather than by
> hoping LLVM's autovectorizer recovers it from a scalar loop.

## 1. Motivation

A scalar reduction loop throws away the programmer's intent:

```rust
// FlashAttention score: a dot product over the head dim, written as scalars.
let mut acc : f32 = 0.0;
for d in 0..4 { acc += q[i][d] * k[j][d]; }
```

The backend then has to *rediscover* that this is a reducible, contiguous, vectorizable loop —
which the LLVM/MLIR autovectorizers do slowly and fragilely (loop-carried dependence analysis,
indvar reasoning, alias checks). Far better to let the programmer say *"reduce this row"* and
emit an op that is vectorizable by construction:

```rust
let acc = dot(q[i], k[j]);          // one op, obvious intent
```

which lowers to `vector.load ×2 → vector.fma → vector.reduction<add>` and thence to real SIMD.

## 2. Current state (grounded)

| Fact | Evidence |
|---|---|
| `@` (matmul) and `.map()` already lower to **`linalg`** | `linalg.generic`/`linalg.fill` in `codegen/generator.rs`, `codegen/lower/mod.rs` |
| The pipeline lowers `linalg` → **scalar loops** (no vectorization pass) | `convert-linalg-to-loops` in `codegen/mod.rs:125`; no `-linalg-vectorize` |
| The pipeline **does** have `convert-vector-to-llvm` | `codegen/mod.rs:125` |
| So anything emitted as `vector.*` becomes SIMD directly | **de-risked**: `mlir-opt --convert-vector-to-llvm | mlir-translate` turns `vector.load`/`arith.mulf`/`vector.reduction<add>` into `load <4 x float>` / `fmul <4 x float>` / `@llvm.vector.reduce.fadd.v4f32` |
| There is **no** slice-level surface | only `q[i][j]` (one element) or `q @ k` (whole matmul); no `q[i]` row, no dot/reduce over a slice |

So the machinery exists (tensor-op → linalg → LLVM), and the vector path is proven; the two
gaps are (a) a slice surface, and (b) emitting `vector` ops from it.

## 2.5 Where this lives: standard MLIR, not a `vx` dialect op (decision)

The `vx` dialect is reserved for semantics MLIR **lacks** — Vx's heterogeneity concepts
(`vx.spawn`, `vx.transfer`, topology, seams). A row dot / reduction / sub-view is **not** novel:
MLIR already has `vector.reduction`, `linalg.dot`, `memref.subview`. A `vx.dot` op would only
re-skin `linalg.dot` and add a C++ conversion pass to maintain. And the codebase already sets
the precedent: `@` and `.map()` emit **`linalg` directly** (not `vx.matmul`), `IndexAccess`
emits `memref.load` directly.

**Decision: emit standard dialects directly from the Vx codegen** (`src/codegen/lower/`, via
melior `OperationBuilder`) — `memref.subview` for slicing, `vector.load`/`vector.reduction` for
reductions. **Not `linalg`**: the pipeline does `convert-linalg-to-loops` with *no* vectorization
pass, so `linalg` would lower to scalar loops; `vector` lowers to SIMD immediately through the
existing `convert-vector-to-llvm` (de-risked in §2). The ops live as AST/HIR nodes (the frontend
must type-check `dot(q[i], k[j])`) and lower straight to MLIR — HIR holds the intent, standard
dialects are the target.

A `vx.*` op (or, better, `linalg` + a target-aware vectorize pass) would only be justified later
if we want a *per-hardware lowering strategy* preserved into a pass (e.g. B200 TMEM+MMA vs CPU
SIMD). Even then the retargetable layer is `linalg`, not a bespoke `vx.dot`. Out of scope now.

## 3. Design

### 3.1 Slice indexing

`q[i]` on a `Tensor<f32, [N, D]>` yields a **rank-1 view** of row `i` — a `Tensor<f32, [D]>`
whose data aliases `q` (a `memref.subview`, zero-copy). Read-only to start (the operators below
consume it), which keeps it out of the way of the linear/borrow system for the first cut.

### 3.2 Slice operators (the surface)

Two families, both requiring **statically-known slice length** (true for FA tiles):

- **Reductions → scalar**: `dot(a, b)`, `sum(a)`, `max(a)`, `min(a)`.
- **Elementwise → slice**: `a * b`, `a + b`, `a - b`, `a * scalar`, `a - scalar`, and mapped
  math (`exp(a)`, `abs(a)`, …).

### 3.3 Lowering (the payoff)

| Slice op | MLIR emitted | LLVM |
|---|---|---|
| `dot(a, b)` | `vector.load` a,b → `arith.mulf` → `vector.reduction<add>` | `fmul <Nxf32>` + `@llvm.vector.reduce.fadd` |
| `sum(a)` / `max(a)` | `vector.load` → `vector.reduction<add\|maximumf>` | `@llvm.vector.reduce.*` |
| `a * b`, `a ± b` | `vector.load` → `arith.mulf`/`addf`/`subf` → `vector.store` | `fmul/fadd <Nxf32>` |
| `a * scalar` | `vector.broadcast` scalar → `arith.mulf` | `fmul <Nxf32>` |
| `exp(a)` | `vector.load` → `math.exp` (vector) → `vector.store` | vectorized `math.exp` / libm calls |

All flow through the existing `convert-vector-to-llvm`. (Elementwise-that-returns-a-slice needs
a destination; for the first cut, elementwise ops can be *fused into* the reduction/store site,
or write back into a named slice.)

## 4. Milestones

- **S1 — Slice indexing. ✅ Done.** `q[i]` : `Tensor<f32,[D]>` view, lowered to a rank-reduced
  `memref.reinterpret_cast` (strided row view) at the flat offset. Codegen recovers the static
  tile dims (the memref is `?x?`, but the Vx type carries them: the `Tensor<T>([…])` constructor
  and `transfer(…)` are taught to `infer_ast_type`; the index type rule peels a `Pinned`/`Ref`
  device wrapper). Test: `backend/pass/slice_indexing.vx`.
- **S2 — Slice reductions. ✅ Done.** `dot(a, b)`, `sum(a)`, `max(a)`/`min(a)` → scalar, lowering
  to `vector.load` (+ `arith.mulf` for dot) + `vector.reduction<add|maximumf|minimumf>`. Reads its
  operands (not consumed). Verified: `dot` == the scalar-loop oracle; MLIR has `vector.reduction`;
  `--emit-llvm` has `@llvm.intr.vector.reduce.*`. Tests: `backend/pass/slice_reductions.vx`,
  `frontend/fail/slice_reduction_non_slice.vx`.
- **S3 — Slice elementwise + store. ✅ Done.** `a*scalar`, `scalar*a`, `a±b`, `a/scalar` →
  `vector.broadcast`/`vector.load` + `arith.{mulf,addf,subf,divf}` → `vector<Dxf32>`; the
  assignment `o[i] = <slice>` becomes a `vector.store` into the S1 row view. Gated on a genuine
  slice operand (strided view or vector) so whole-tensor loops (scf-to-cf / unroll tests) are
  untouched. Test: `backend/pass/slice_elementwise.vx`. *(Vectorized `exp(a)` via `math.exp` is
  deferred — FA uses its own software-emulated `exp_poly`, which stays scalar.)*
- **S4 — FA-4 flagship, vectorized. ✅ Done.** `backend/pass/flash_attention_v4_slice.vx` is
  `flash_attention_v4.vx` with every head-dim loop collapsed: the score `for d`-loop → `dot(q[i], k[j])` (a `vector.reduction`), and the O rescale / accumulate / normalize `for d`-loops →
  `o[i] = o[i]*corr`, `o[i] = o[i] + p*v[j]`, `o[i] = o[i]/l` (`vector.store`s). Produces the
  identical `2.8448 / 1.5` as the scalar oracle; the score reduction is a `vector.reduction`, not
  a scalar loop.

### Full-MLIR regression lock

Beyond the behavioural (`EXPECT`) tests, the exact post-codegen MLIR of each slice op is pinned
line-by-line so the lowering cannot silently regress —
`tests/optimizations/pass/slice_{indexing,reductions,elementwise}_mlir.vx`. They use minimal
static-shape kernels (`fn f(q : Tensor<f32,[2,4]>) …`) so the golden MLIR is compact, and are
run by the FileCheck harness (`run_optimization_test`). `CHECK-NOT: scf.for`/`affine.for` locks
the core invariant: the ops stay vectorized, never a scalar loop. Regenerate after an intentional
lowering change with `cargo run --bin update_mlir_test_checks -- <file>` (idempotent; preserves
the `CHECK-NOT` guards).

## 5. What FlashAttention becomes

```rust
// Phase-2/V4 inner loop, vectorized at the source:
for jj in 0..Bc {
  ts[jj] = dot(q[i], k[j]) * scale;          // S2: vector.reduction
}
let m_new = max(m, max(ts));                 // S2
let p = exp(ts - m_new);                     // S3: vectorized map
l = l * corr + sum(p);                       // S2
o[i] = o[i] * corr + p_dot_v;                // S3
```

## 6. Open questions

1. **View vs copy / aliasing.** `q[i]` as a zero-copy view is right for performance but aliases
   `q`; interaction with the linear/borrow checker needs care. First cut: read-only views feeding
   reductions (no aliasing hazard). A writable slice (`o[i] = …`) is S3+ and needs a store path.
1. **Static vs dynamic length.** `vector<Nxf32>` needs a static `N`. FA tiles are static, so S1–S4
   assume it; a dynamic-length slice would fall back to a loop (or a masked/strip-mined vector
   form) — later.
1. **Operator surface / spelling.** Free functions (`dot(a,b)`, `sum(a)`) vs methods
   (`a.dot(b)`, `a.sum()`) vs operators (`a · b`). Reductions as free functions read best; pick in
   S2 and keep consistent.
1. **`@` for the matmul-shaped parts.** The full `S = QKᵀ` is a matmul; a `transpose` + tile-`@`
   (already `linalg`) plus a `-linalg-vectorize` pass is a complementary path for those, orthogonal
   to slice reductions (which also cover softmax). Out of scope here, noted for later.
