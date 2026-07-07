# FlashAttention in Vx — a walkthrough

> Companion to the plan
> ([`implementation_plans/flash_attention.md`](../implementation_plans/flash_attention.md)) and
> the memory-model proposal
> ([`implementation_plans/first_class_memory_spaces.md`](../implementation_plans/first_class_memory_spaces.md)).
> This is the retrospective: what was built, how the paper maps onto Vx, and where the
> boundary is. Source: *FlashAttention-4* (Zadouri et al., arXiv:2603.05451v1). Tracking epic:
> issue #182.

## 1. What was built

FlashAttention was implemented as a driving use case for evolving Vx's memory model. The
forward pass, its FA-4 refinement, the backward pass, and — critically — the *heterogeneous
placement* all work and are verified:

| Phase | Artifact | What it proves |
|---|---|---|
| 0 | (probes) | The algorithm is expressible today: index-offset tiling, `Q·Kᵀ` via a dot-loop (no transpose), `exp` in a loop, and `mut` online-softmax state threaded across a loop all compile + JIT-run. |
| 1 | `tests/backend/pass/attention_reference.vx` | The **oracle**: full-materialized `S`, a *real* row-wise softmax (replacing the `0.25` stub in `ane_attention.vx`), `O = PV`. Discriminating inputs: `2.84482` (non-uniform query) and `1.5` (uniform). |
| 2 | `tests/backend/pass/flash_attention.vx` | Tiled forward with **online softmax** (running `m`, `ℓ`, rescaled `O_acc`; never materializes `S`). Output **identical** to the oracle. |
| 3 | `tests/backend/pass/flash_attention_conditional.vx` | **FA-4 conditional rescaling** (eq. 6): skip the rescale unless the max grows past `τ = log₂256 = 8`. Exact (same output) + a counter proving the `τ` branch fires (2 skips). |
| 4 | `tests/backend/pass/flash_attention_placed.vx` (+ `frontend/fail/flash_attention_tile_too_big.vx`) | **Heterogeneous placement + memory model.** See §3. |
| 5 | `tests/backend/pass/flash_attention_backward.vx` | The **backward pass** (`dV, dP, dS, dQ, dK` with the softmax Jacobian), verified against an independent reference computation. |
| ★ | `tests/backend/pass/flash_attention_v4.vx` | **The FA-4 flagship**: online softmax + conditional rescaling + software-emulated `exp` + placement, fused into one device kernel. See §7. |

The correctness spine is a chain of bit-identical results: *full-materialized softmax ==
tiled online softmax == FA-4 conditional rescaling*, each JIT-checked. The inputs are chosen
so a real softmax gives `2.845` where the old stub (or a uniform mean) gives `1.5` — the test
would catch a fake.

## 2. The algorithm, briefly

Forward (per query tile, streaming K/V tiles `j`): `S_ij = α·Q_i·K_jᵀ`;
`m_new = max(m, rowmax(S_ij))`; `P_ij = exp(S_ij − m_new)`;
`ℓ = e^{m−m_new}ℓ + rowsum(P_ij)`; `O_acc = e^{m−m_new}O_acc + P_ij·V_j`; finally `O = O_acc/ℓ`.

FA-4's *conditional rescaling* only applies the `e^{m−m_new}` factor when `m_new − m > τ`. It
is exact because the running-max reference cancels in the final `O_acc/ℓ` ratio and `e^{S−m}`
stays bounded by `e^τ = 256` (no overflow) — so it trades vector rescales for nothing.

Backward (paper §2.1): `dV = Pᵀ·dO`, `dP = dO·Vᵀ`,
`dS_ij = P_ij(dP_ij − Σ_k P_ik dP_ik)`, `dQ = α·dS·K`, `dK = α·dSᵀ·Q`.

## 3. The convergence: FlashAttention *is* a heterogeneous program (Phase 4)

This is the point of doing it in Vx rather than C. Phase 4 pulls in the first-class memory
model (milestones M1–M5, epic #181):

```rust
Memory CPU_DRAM {}
Memory GPU_HBM { within: Memory::CPU_DRAM, capacity: 40 GB, bandwidth: 3 TB/s }

// ... Q/K/V/O transfer(...)ed into GPU_HBM, then:
spawn on(Topology::GPU) { /* the whole online-softmax loop */ }
```

- The **hierarchy is declared** (`Memory { within:, capacity:, bandwidth: }`, M1/M2).
- Each `transfer` into `GPU_HBM` is **capacity-checked** (M3) and stamped with a
  **bandwidth-derived cost** on `vx.transfer` (M4) — the paper's roofline, *computed* from
  declarations rather than analogized.
- The **exp-on-device** question from Phase 0 is resolved: the runtime dispatcher falls back
  to CPU libffi when no GPU backend is present, so the fully-fused matmul+exp+rescale kernel
  JIT-runs to the same `2.845 / 1.5`.
- **Capacity as a static check**: a 128×64 f32 tile (32 KiB) staged into a 16 KiB `Local_SRAM`
  scratchpad is rejected at compile time — `E6009: … needs 32768 bytes but memory space 'Local_SRAM' has capacity 16384 bytes`. That is FlashAttention's on-chip tile constraint,
  made static.

`--emit-llvm --target nvptx64` emits valid IR (`target triple = "nvptx64-nvidia-cuda"`), with
no claim of an optimized SASS kernel.

## 4. Paper → Vx mapping

| FA-4 (paper) | Vx | Status |
|---|---|---|
| Online softmax `m`,`ℓ`,`O` recurrence (§3.1.4) | scalars + `mut` tile, nested `for` | **built** (Phase 2) |
| Conditional rescale, `τ=8` (eq. 6) | `if m_new − m > τ { … }` | **built** (Phase 3) |
| `αQKᵀ`, `PV` | explicit dot-loops (no transpose) | **built** (Phase 1–2) |
| Row-wise softmax stability | reductions + `.exp()` | **built** (Phase 1) |
| Backward `dV/dP/dS/dQ/dK` (§2.1) | explicit loops + softmax Jacobian | **built** (Phase 5) |
| The memory hierarchy (HBM/SMEM/TMEM, §2.2) | `Memory X { within:, capacity:, bandwidth: }` | **built** (M1–M2) |
| Placement on the accelerator | `spawn on(Topology::GPU)` + `transfer` | **built** (Phase 4) |
| Shared-mem vs MMA vs exp roofline (§3.1.1) | derived, bandwidth-parameterized `vx.transfer` cost | **built** (M4) |
| Software-emulated `2^x` on FMA (§3.1.3) | polynomial `exp` (see the V4 flagship) | **built** (algorithmic essence) |
| TMEM partitioning, 2-CTA MMA, warp specialization, async-MMA pipeline (§3.1.2, §3.2.2, §4) | — | **out of scope** (backend scheduling) |

## 5. The boundary (what Vx is and isn't doing)

FA-4 is two layers. Vx expresses the **algorithm** and the **memory hierarchy it moves data
through** — including a *checked* account of placement, capacity, and movement cost, which is
exactly the paper's roofline reasoning made mechanical. Vx does **not** express the Blackwell
**kernel scheduling** (TMEM partitioning, 2-CTA MMA, warp-specialized producer/consumer
pipelines, async-MMA overlap) — that is a compiler/backend concern below the language, and it
is where most of the paper's raw TFLOP/s come from. The honest success criterion here is *"a
correct, tiled, online-softmax attention whose cross-device data movement type-checks and
whose tile placement is capacity-checked,"* not *"1613 TFLOP/s on a B200."*

## 6. Gaps found along the way

- **#185** — `print()` of a *scalar* `f32` fails in codegen (`Unsupported MLIR element type`);
  tensor `print` is fine. Worked around by writing scalars into a 1×1 tensor.
- Shapes are fixed (small, hand-verifiable) rather than shape-generic; a mixed int×float
  expression (`1.0 * loop_var`) hits an `index`↔`i64` cast, so test data uses float literals.
- A loop index in a float expression (`i as f32`) and a computed loop bound (`for _ in 0..(expr)`)
  both hit an unreconciled `i64 → index` cast at MLIR finalization. The flagship `exp_poly`
  (§7) sidesteps both — scalar `as` casts and *fixed-bound* conditional loops do work.
- These are orthogonal to the attention work and do not affect the results above.

## 7. FlashAttention-4 proper — the flagship

`tests/backend/pass/flash_attention_v4.vx` fuses every algorithm-level FA-4 innovation into one
placed device kernel:

1. **Online softmax** (§3.1.4) — running `m`/`ℓ`/`O_acc`, streamed K/V tiles.
1. **Conditional rescaling** (eq. 6) — skip the rescale below the `τ = 8` growth threshold.
1. **Software-emulated `exp`** (§3.1.3) — `exp_poly` computes `exp` with *no libm*:
   range-reduce `exp(x) = 2^{x·log₂e} = 2ⁿ·2^f`, a degree-4 Horner polynomial for the fractional
   `2^f`, and `2ⁿ` by scaling. Max relative error vs libm over the softmax range is `< 1e-3`
   (below BF16 precision), reproducing the paper's Table 2 result. In FA-4 this offloads the
   MUFU exponential onto FMA units; here it removes the dependency on a device libm.
1. **Placement** — declared `GPU_HBM`, `transfer` of operands (capacity-checked + bandwidth-
   costed), the whole kernel in `spawn on(Topology::GPU)`.

Two Vx features come together here. `exp_poly` is declared **`on Topology::GPU`** — a
*device-bound* function — so it is legal to call from inside the GPU `spawn` (a plain CPU
function is rejected with `E6001` at the boundary). And the fused matmul+exp+rescale kernel
JIT-runs (CPU-fallback dispatch) to `2.84484`/`1.5` — the reference `2.84482`, off only by the
polynomial's `~2e-5`. That is the whole thing: the FA-4 *algorithm* — including its emulated
exponential — running as a placement-checked, cost-annotated heterogeneous Vx program.
