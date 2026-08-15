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
defect as the H100's `HBM->L2` cell (vx-review#22), the fifth of this shape in the campaign, and
the first found on the CPU side.

Every buffer in `measure_m4.mm` must exceed the largest cache it could be served from by a margin,
and the instrument must report the ratio against declared peak so that a figure above 1.0× is
visible in the output rather than needing to be noticed.
