# M4 on-die seams: what is predicted, written before the instrument exists

Dated 2026-08-12, Vx `461b6e72`. Committed **before** `measure_m4.mm` is written, for the reason
the freeze exists: a prediction recorded after the measurement is not a prediction.

`fleet/m4-uma.vx` is one of the seven held-out SKUs. Its 30 frozen cells were taken at
`memalg-freeze-2026-08-07` and have never been scored. Only its host seam has been measured at all
(the Tier-0 shakedown, `results/shakedown-20260807T173942Z`), and that measurement turns out to be
defective — see below. The GPU-side seams are untouched.

## P1 — The composition law on Apple silicon should be SUM, not bottleneck

This is the real reason to do this on a laptop rather than wait for a pod.

`compose.py` scored two parameter-free composition laws against the H100 powerset and found they
split by **route kind**, not by machine:

| route | law that fit | error |
|---|---|---|
| `HBM->L2->SMEM` (hardware streams a fill through L2) | bottleneck | −13.9% |
| `L1->REG->SMEM` (a load into a register, then a store out) | sum | −9.3% |

The mechanism claimed there is that a hardware-streamed route is limited by its narrowest leg,
while an instruction-sequenced route genuinely adds because the instructions genuinely happen one
after the other. #16 says the next thing that claim needs is a **second architecture**.

The Apple GPU is that second architecture, and it discriminates cleanly, because filling
threadgroup memory on Apple family 9 has no asynchronous-copy engine equivalent to
`cp.async` — it is an ordinary load followed by an ordinary threadgroup store. Every hop of
`DRAM -> threadgroup` is instruction-sequenced.

**So the prediction is: `DRAM -> threadgroup` on the M4 fits SUM and not bottleneck.**

- If sum fits and bottleneck does not, the route-kind mechanism survives a change of vendor, ISA,
  and memory architecture, and the model change it implies is a property on the edge saying which
  kind of crossing it is.
- **If bottleneck fits here too, the mechanism is wrong** and the H100's `L1->REG->SMEM` result
  needs another explanation. That is the outcome that falsifies it, and it is a real possibility:
  Apple's compiler may coalesce the load/store pair, or the threadgroup store may retire
  asynchronously.
- If neither fits, the composition law is machine-specific, which is the worst case for the
  portability argument in #22 and worth knowing early.

## P2 — The host seam is nearly free, and the model over-predicts its cost enormously

`CPU_DRAM -> HBM` is declared at 120 GB/s on this SKU, but the two spaces are the same physical
memory: the GPU can read a buffer the CPU wrote with no copy at all (`hasUnifiedMemory` is true and
`MTLStorageModeShared` is a pointer, not a transfer).

**Predicted: a shared-storage buffer handed to a kernel costs no measurable time**, so the model's
prediction for that seam is wrong by however large the transfer is — unboundedly, not by a
percentage. The interesting question is not the error but whether the algebra can express "these
two declared spaces are one", which today it cannot: they are separate `Memory` declarations joined
by a `transfer` edge with a rate.

## P3 — `Memory::L2` on this SKU is describing the wrong cache

`m4-uma.vx` cites `hw.perflevel0.l2cachesize` = 16 MiB. That is the **CPU P-core cluster L2**. The
topology that declares it is `arch: applegpu`, and the Apple GPU does not use the CPU cluster's L2
— it has its own cache, behind a System Level Cache shared between them.

**Predicted: a GPU streaming read shows a working-set knee at a size that is NOT 16 MiB.** If the
knee is at 16 MiB the declaration is accidentally right; anywhere else and a tile the model places
in `Memory::L2` for a GPU kernel is priced against a cache that kernel never touches.

## P4 — The `L2 -> SMEM` cell cannot be scored on this machine at all

It is declared in `B/cyc` with an explicitly UNVERIFIED `clock: 1.4 GHz` placeholder. Metal exposes
no cycle counter to a kernel, so the seam can only be timed in wall-clock. Protocol decision 5 in
PREDICTIONS.md forbids converting between the two with an invented clock.

**Predicted: this cell is unscorable**, and stays unscorable until the GPU clock is measured
independently. Recorded so that reporting it as "no measurement" later is visibly the expected
outcome rather than a gap someone papers over with 1.4 GHz.

## What the instrument must not repeat

Six of the ten points in the existing shakedown report memory traffic **above the M4's 120 GB/s
hardware peak** — 1.22× to 1.79× — and they are exactly the points whose working set (2× the size,
source plus destination) fits in the 16 MiB L2. Those rows measure cache, not DRAM. It is the same
defect as the H100's `HBM->L2` cell, the fifth of this shape in the campaign, and
the first found on the CPU side.

Every buffer in `measure_m4.mm` must exceed the largest cache it could be served from by a margin,
and the instrument must report the ratio against declared peak so that a figure above 1.0× is
visible in the output rather than needing to be noticed.

---

# Addendum, dated 2026-08-12, Vx `a6ab362c`

P1, P3 and P4 have now been measured (see `results/m4-*`). P2 was only half done. These two are
for experiments not yet written, and are recorded before the code exists for the same reason as
above.

## P5 — contention: the aggregate is already saturated, so sharing is a straight division

M3 is the most expensive item on the fleet list and has no predicted column: the
model has no contention term, so it says each of K concurrent transfers gets the whole edge. This
machine can measure the sharing law for nothing.

A single streaming read already reaches 104 GB/s against a 120 GB/s declared peak — 87%, which is
about as close to saturated as a real workload gets. So:

**Predicted: the aggregate across K concurrent flows stays roughly constant at ~104 GB/s, and each
flow's own rate falls as ~1/K. The falloff is smooth, not a cliff**, because DRAM bandwidth is a
continuously shared resource rather than an allocated one.

The falsifier, and it is a real possibility: **if the aggregate RISES with K**, then one flow was
not saturating after all and there is headroom. That would mean `bandwidth / users` is the wrong
patch at small K — the model would need a saturation term (min(K × per-flow-max, edge-max)) rather
than a division, which is a different shape of fix.

Secondary, and the thing a `bandwidth / users` model cannot produce at all: **the spread between
the fastest and slowest of the K flows.** If the GPU shares fairly, all K finish at about the same
time and the spread is small. If it serialises them, flow 1 finishes at T and flow K at K×T, and
the spread is the whole range. Both give the same aggregate, so the aggregate alone cannot tell
them apart — and a placement decision needs to know which, because "when does my tensor arrive" is
answered by the slowest flow, not the average.

## P6 — the host seam is free, and the model charges 120 GB/s for it

`CPU_DRAM -> HBM` is declared at 120 GB/s, but `hasUnifiedMemory` is true: a `MTLStorageModeShared`
buffer the CPU wrote is readable by a kernel with no copy and no API call at all.

**Predicted: a kernel reading a buffer the CPU just wrote runs at the same rate as one reading a
buffer the GPU already had — the "transfer" costs no measurable time.** The model's prediction for
that seam is therefore not wrong by a percentage; it is wrong by the entire quantity.

Measured against a `MTLBlitCommandEncoder` copy, which is what a naive port of the discrete path
would do. **Predicted: the blit is real work and runs at a rate well under 120 GB/s**, because a
copy both reads and writes and so moves two bytes of traffic per byte copied — the same argument
`m4-uma.vx` already makes about `memcpy`. If the blit comes out at or above 120 GB/s it was served
from cache and the buffer was too small, which is the defect this instrument exists not to repeat.
