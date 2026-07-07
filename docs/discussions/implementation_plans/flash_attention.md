# Implementation Plan: FlashAttention in Vx

> Source: *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric
> Hardware Scaling* (Zadouri et al., arXiv:2603.05451v1). This plan adapts the **algorithm**
> to Vx; it deliberately does **not** attempt to reproduce the paper's Blackwell kernel
> co-design. See "Scope" below for why.

## 1. What "FlashAttention in Vx" means (scope)

The FA-4 paper is two things layered together:

1. **An algorithm** — tiled attention with *online softmax*: never materialize the full
   `N×N` score matrix; stream over K/V blocks while maintaining a running max `m`, running
   normalizer `ℓ`, and a rescaled output accumulator `O`. FA-4 adds two algorithm-level
   refinements: *conditional/skipped rescaling* (a threshold `τ`) and a *software-emulated
   exponential*.
1. **A Blackwell kernel co-design** — tensor-memory (TMEM) partitioning, the 2-CTA MMA mode,
   warp-specialized producer/consumer pipelines, async MMA overlap, and MUFU/FMA scheduling.
   This is written in CuTe-DSL and is specific to B200/GB200 SM internals.

**Vx can express (1); Vx should not try to express (2).** In Vx you write the math and the
*placement* (`spawn on(Topology::GPU)`, `transfer(...)`); turning that into a warp-specialized,
TMEM-resident, 2-CTA-pipelined Blackwell kernel is the *compiler/backend's* job. Vx's GPU path
currently emits target-independent LLVM IR (see `docs/lang/` and the `--emit-llvm --target nvptx64` path), not hand-tuned SASS. So:

- **In scope:** the FlashAttention *algorithm* — forward (and, as a stretch, backward) tiled
  online softmax, including FA-4's conditional-rescaling threshold — expressed with Vx tensors,
  loops, `@`, `.exp()`, and the topology/`transfer` model.
- **In scope (co-goal): evolving the language.** FlashAttention is the *forcing function* for a
  real Vx memory model. The paper's whole argument is asymmetric bandwidth across a memory
  hierarchy (HBM → L2 → SMEM → TMEM → RMEM), and today Vx cannot *describe* those spaces —
  `Memory::TMEM` is an opaque name with no capacity, bandwidth, or nesting. We will make memory
  spaces & sub-spaces first-class (declarable, with properties and a hierarchy, capacity checks,
  and derived costs). That work has its own proposal:
  [`first_class_memory_spaces.md`](./first_class_memory_spaces.md); it converges with this plan
  at Phase 4.
- **Out of scope (by design):** the backend *scheduling* that exploits the hierarchy —
  TMEM/2-CTA/warp-specialization/async-MMA pipelining and the MUFU polynomial-exp
  micro-emulation. *Describing* the hierarchy is language work (above); *scheduling kernels over
  it* is a compiler project. The paper's hardware roofline (§3.1.1) becomes **computable** once
  spaces carry bandwidth — that is exactly what the memory-model proposal's derived-cost
  milestone delivers, re-cast as Vx's `TransferCostGraph` / seam model.

The deliverable is two intertwined tracks: a correct, readable Vx implementation of the
algorithm, and the memory-model language features it pulls into existence — staged so each phase
is independently valuable and verifiable.

## 2. The algorithm (from the paper)

Notation: `Q,K,V ∈ ℝ^{N×d}`, output `O ∈ ℝ^{N×d}`, scale `α = 1/√d`.

Non-flash reference (§2.1): `S = αQKᵀ`, `P = softmax(S)` (row-wise, minus row-max for
stability), `O = PV`.

Flash forward — tile Q into blocks of `B_r` rows, K/V into blocks of `B_c` rows. For each Q
tile `Q_i`, initialize `m = -∞`, `ℓ = 0`, `O_acc = 0`, then stream over K/V tiles `j` (§3.1.4):

```
S_ij   = α · Q_i · K_jᵀ                    # (B_r × B_c)
m_new  = max(m, rowmax(S_ij))
P_ij   = exp(S_ij - m_new)                 # (B_r × B_c)
ℓ      = e^{m - m_new} · ℓ + rowsum(P_ij)
O_acc  = e^{m - m_new} · O_acc + P_ij · V_j
m      = m_new
# after all j:
O_i    = O_acc / ℓ
```

FA-4 refinements:

- **Conditional rescaling (§3.1.4, eq. 6).** The `e^{m-m_new}` rescale of `O_acc`/`ℓ` is only
  needed when the running max actually grows. FA-4 skips it unless `m_new - m > τ`, with
  `τ = log2(256) = 8` (a rescale factor of 256). Correctness is preserved because everything is
  renormalized by the true `m_final`/`ℓ_final` at the end. This cuts the number of vector
  rescales sharply. Directly expressible as an `if` in Vx.
- **Software-emulated `exp` (§3.1.3).** Range-reduce `2^x = 2^⌊x⌋ · 2^{x-⌊x⌋}`, bit-manipulate
  the integer part into the IEEE-754 exponent, and approximate the fractional part with a
  degree-3 Horner polynomial on FMA units. This is a *MUFU-throughput* optimization specific to
  GPUs where the exponential unit is the bottleneck. **Vx already has `.exp()`** (`stdlib/std/ math.vx`, lowering to `expf`), so this is optional and only interesting as a self-contained
  numeric exercise (Phase 3, optional).

Backward (§2.1, §3.2): `dV = Pᵀ·dO`, `dP = dO·Vᵀ`, `dS = dsoftmax(dP)`, `dQ = α·dS·K`,
`dK = α·dSᵀ·Q`, with `ds = (diag(p) - p pᵀ)·dp`. Recomputes `S`/`P` from the saved `m`/`ℓ`.

## 3. What Vx expresses today (grounded)

Every primitive the forward algorithm needs already exists and is exercised by tests:

| Need | Vx today | Evidence |
|---|---|---|
| `Tensor<f32,[N,d]>`, element index/mutate | ✅ `q[i][j]`, `s[i][j] += …` | `tests/backend/pass/ane_attention.vx` |
| Shape-generic functions | ✅ `fn f<N, D>(x: Tensor<f32,[N,D]>)` | `tests/frontend/fail/formal_verification/smt_generic_mismatch.vx` |
| Matmul | ✅ `a @ b` (lowers + runs) and explicit triple-loops | `tests/backend/pass/llama2_v2.vx:157`, `matmul_execution.vx` |
| `exp` | ✅ `x.exp()` (f32/f64) | `stdlib/std/math.vx`, `math_exp_f32.vx` |
| row-max / row-sum | ✅ for-loop reductions | reference softmax `tests/backend/pass/llama2.vx:365` |
| online-softmax state (`m`,`ℓ`,`O`) | ✅ `mut` scalars/tensors across loops | `ane_attention.vx` |
| conditional rescale (`τ`) | ✅ `if` | language core |
| device placement | ✅ `spawn on(Topology::GPU/NPU[…])` | `llama2.vx:392`, `ane_attention.vx` |
| host↔device movement | ✅ `transfer(x, Memory::NPU_HBM)` | `ane_attention.vx` |
| pin/verify result | ✅ `Pinned<T,Topo>`, `Verified<T>` | `ane_attention.vx` return type |

There is even an existing **non-flash** attention (`tests/backend/pass/ane_attention.vx`) — but
its softmax is *stubbed as the constant `0.25`*; it materializes the full score matrix and never
computes a real `exp`/max/sum. That file is the natural starting oracle to replace.

## 4. Gaps / risks to de-risk first (Phase 0)

These are the unknowns that actually shape the code. Each has a fallback so the plan does not
stall on any one:

1. **No tensor-slice syntax.** There is no `Q[i0..i1]` sub-tensor slicing (grep finds only
   `NPU[0..4]` topology ranges). **⇒ Tile by index-offset loops**: a tile is a base offset plus
   local indices (`q[tile_r*B_r + a][…]`), exactly like the reference matmul/softmax loops.
   (Adding real slicing to the frontend is a *separate* enhancement, not a prerequisite.)
1. **`@` and transpose.** `@` computes a full matmul of 2-D tensors. `Kᵀ` is needed for `QKᵀ`.
   The existing attention pre-transposes K on the host into `k_t`. **⇒ Either pre-transpose K,
   or write `S_ij` as an explicit `Q_i·K_jᵀ` triple-loop** (dot over `d`), which sidesteps both
   the transpose and any `@`-on-tile shape questions. Start with explicit loops; adopt `@` on
   tiles once verified.
1. **`exp` inside a device `spawn`.** Existing kernels do matmul on-device but softmax on the
   **host** (`.exp()` is a libc `expf` call; device dispatch may not link libm). True flash
   needs `exp` *inside* the tile loop. **⇒ Do the whole forward on CPU first (Phase 1–3);
   attempt on-device in Phase 4 and treat host-libm-in-kernel as an explicit finding.** If
   device `exp` is unavailable, the optional polynomial `2^x` (Phase 3) becomes the on-device
   path, which is the paper's own reason for emulating it.
1. **State threading across the tile loop.** `m`,`ℓ` are per-Q-row scalars; `O_acc` is a
   `B_r×d` tile. Confirm `mut` tensors/scalars survive across nested `for` iterations (they do
   in `ane_attention.vx`) and that a `spawn` block can carry them (Phase 4).

**Phase 0 deliverable:** four ≤20-line probe `.vx` files under a scratch dir that each isolate
one risk (offset-tile matmul; `Q·Kᵀ` without transpose; `exp` in a `spawn`; `mut` tile carried
across a loop), compiled with `--emit-mlir` and, where possible, JIT-run. Record results inline
in this doc. Nothing else proceeds until (1) and (2) have a chosen approach.

## 5. Phased plan

Each phase lands as its own commit with tests, and each is independently useful.

### Phase 1 — Reference (non-flash) attention, CPU *(correctness oracle)*

- `fn attention_ref<N, D>(q, k, v, scale) -> Tensor<f32,[N,D]>`: materialize `S = αQKᵀ`, real
  row-wise softmax (max → `exp(·−max)` → sum → divide, modeled on `llama2.vx:365`), then `O=PV`.
- Small fixed sizes first (e.g. `N=8, D=4`) with a hand-checkable input, then shape-generic.
- **Validates:** the math end-to-end; replaces the stubbed `0.25`. Becomes the bit-close oracle
  for every later phase.

### Phase 2 — FlashAttention forward (tiled online softmax), CPU *(the core)*

- `fn flash_attention_fwd<N, D>(q, k, v, scale) -> Tensor<f32,[N,D]>` with compile-time tile
  sizes `B_r`, `B_c` (start `B_r=B_c=4`). Outer loop over Q tiles; inner loop over K/V tiles;
  per step compute `S_ij`, `m_new`, `P_ij`, update `ℓ` and `O_acc` with the `e^{m−m_new}`
  rescale; final `O_i = O_acc/ℓ`.
- **Validates:** output equals Phase 1 within f32 tolerance across a few `N,D,B` configs
  (including `N` not divisible by `B` — masked/ragged last tile).

### Phase 3 — FA-4 numerical refinements *(algorithmic delta from FA-2/3)*

- **Conditional rescaling** (eq. 6): skip the `O_acc`/`ℓ` rescale unless `m_new − m > τ`
  (`τ = 8`); still normalize by the true final `m`/`ℓ`. Assert identical result to Phase 2 plus
  a counter proving rescales were actually skipped on a workload with a slowly-growing max.
- **(Optional) polynomial `2^x`**: a self-contained `fn exp2_poly(x)` (Cody-Waite range
  reduction + degree-3 Horner). Compare against `.exp()`; document the accuracy (paper Table 2:
  degree-3 matches hardware within ~1 BF16 ULP). Only pursue if it unlocks Phase 4's
  on-device path.

### Phase 4 — Heterogeneous placement + memory model *(Vx's actual value-add; the convergence)*

This is where the algorithm meets the language work. It pulls in the memory-model milestones
M1–M4 from [`first_class_memory_spaces.md`](./first_class_memory_spaces.md):

- **Declare the hierarchy** (M1–M2): `Memory HBM/SMEM/TMEM { within:, capacity:, bandwidth:, … }`
  so the tiles have real spaces to live in.
- Wrap the compute in `spawn on(Topology::GPU)` (and/or `NPU[0]`); `transfer` Q/K/V into device
  memory and the result back; **place** the Q/K/V tiles in `Memory::SMEM` and the `O_acc`/score
  in `Memory::TMEM`; return `Pinned<Tensor<f32,[N,D]>, Topology::GPU>` / `Verified<…>`.
- **Capacity-check** the tile placement (M3): the chosen `B_r×B_c` must fit SMEM/TMEM, now a
  compile-time check with real byte numbers.
- Inspect the emitted `vx.spawn` / `vx.transfer` and the **seam obligations** (`--verify-seams`),
  now with **derived, bandwidth-parameterized costs** (M4) — the paper's data-movement roofline,
  computed rather than analogized. Resolve the exp-on-device finding from Phase 0 (host-fallback
  softmax between device matmuls if libm is unavailable, i.e. a "tiled but not fully fused"
  variant, documented as such).
- **Validates:** `--emit-mlir` FileCheck for the spawn/transfer/placement structure; a rejected
  over-capacity tile; JIT-run on CPU fallback; `--emit-llvm --target nvptx64` produces IR (no
  execution claim).

### Phase 5 — Backward pass *(stretch)*

- `flash_attention_bwd`: recompute `S`/`P` from saved `m`/`ℓ`, then `dV, dP, dS, dQ, dK`
  (§2.1/§3.2). Larger and numerically fussier (the `dsoftmax` Jacobian). Gate behind Phases 1–4
  landing and a finite-difference gradient check as the oracle.

### Phase 6 — Tests + writeup

- Consolidated tests: `tests/backend/pass/flash_attention.vx` (execution + FileCheck) and a
  frontend/middle-end MLIR-structure test. Reuse the `expect_matches` / `// CHECK` harness.
- A short `docs/` note mapping paper sections → Vx constructs (the table in §8) and stating the
  out-of-scope boundary, so the artifact is not mistaken for a Blackwell kernel.

## 6. Recommended order & sizing

`0 → 1 → 2 → 3(conditional-rescale) → 4 → 6`, with `3(poly-exp)` and `5(backward)` as stretch.
Phases 0–2 are the bulk of the intellectual work and the highest value (a correct flash forward
with a real online softmax). Phase 4 is what makes it *a Vx program* rather than a generic C
loop nest. Keep tile sizes tiny and hand-verifiable until Phase 2 passes its oracle check.

## 7. Validation strategy

- **Oracle:** Phase 1 reference vs. Phase 2/3 flash, bit-close within f32 tolerance; Phase 5 vs.
  finite-difference gradients.
- **Structure:** `--emit-mlir` FileCheck on the tiled loop nest and the `vx.spawn`/`vx.transfer`
  ops; `--verify-seams` for Phase 4.
- **Determinism:** the reference and flash paths must agree regardless of tile size (FA-4's
  determinism motivation, §1) — a good invariant to assert.
- Every phase passes the existing pre-commit gate (fmt, clippy, full suite, doctests).

## 8. Paper → Vx mapping (and the honest boundary)

| Paper (FA-4) | Vx expression | Status |
|---|---|---|
| Online softmax `m`,`ℓ`,`O` recurrence (§3.1.4) | scalars + `mut` tile, nested `for` | **write it** (Phase 2) |
| Conditional rescale, `τ=8` (eq. 6) | `if m_new − m > τ { … }` | **write it** (Phase 3) |
| `αQKᵀ`, `PV` matmul | `@` or explicit dot-loops on tiles | **write it** (Phase 1–2) |
| Row-wise softmax stability (`−rowmax`) | reduction loops + `.exp()` | **write it** (Phase 1) |
| Placement on the accelerator | `spawn on(Topology::GPU)` | **write it** (Phase 4) |
| The memory hierarchy itself (HBM/SMEM/TMEM, §2.2) | `Memory X { within:, capacity:, bandwidth:, … }` | **build it** — [`first_class_memory_spaces.md`](./first_class_memory_spaces.md) M1–M2 |
| Shared-mem vs MMA vs exp roofline (§3.1.1) | derived, bandwidth-parameterized `TransferCostGraph` | **build it** — memory-model M4 (was "Vx analog", now source-level) |
| Software `2^x` on FMA (§3.1.3) | optional `fn exp2_poly` | optional (Phase 3) |
| TMEM partitioning, 2-CTA MMA, warp specialization, async-MMA pipeline (§3.1.2, §3.2.2, §4) | — | **out of scope** (backend/compiler concern) |

The last row is the key honesty: the FA-4 *speedups* come mostly from that last row, which is
kernel/hardware engineering below Vx's abstraction. What Vx contributes instead is a *checked*
account of the algorithm's **data movement and placement** — the same thing the paper's roofline
reasons about informally — via the topology/`transfer`/seam machinery. A realistic success
criterion is: "a correct, tiled, online-softmax attention that type-checks its cross-topology
data movement," not "1613 TFLOP/s on a B200."
