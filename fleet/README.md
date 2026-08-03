# `fleet/` — SKU machine models

Machine-model files for `vxc --machine <file> program.vx` (#281). Each file declares one SKU's
memory hierarchy and its transfer topology, and nothing else: no functions, no program logic. The
admission program is written once and admitted against each SKU by swapping the flag, which is the
claim the MLSys fleet-admission paper rests on.

## The shared vocabulary

Every SKU declares the **same space names**, so one program text can reference them:

| Space  | What it is                                | Scope    |
|--------|-------------------------------------------|----------|
| `HBM`  | device-global memory (HBM2e/HBM3/HBM3e)   | `device` |
| `L2`   | last-level on-die cache / Infinity Cache  | `device` |
| `SMEM` | per-SM/CU shared memory (scratchpad)      | `sm`     |

A program that places a tile in `Memory::HBM` therefore means "this SKU's device memory", and the
capacity check resolves that to whichever number the selected machine file declares. Adding a SKU
means adding a file with these three spaces, not touching any program.

`node-8gpu.vx` is different in kind: it models one **8-GPU node**, adding the interconnect spaces
and edges (NVLink, PCIe, InfiniBand) that a multi-GPU placement crosses. It is composed with a SKU
file rather than replacing it — but note that the `--machine` flag currently takes a single file
(#281), so the node file re-declares the SKU spaces it needs. Declaration import (#224) would let
these compose properly; see the note in that file.

## Provenance of the numbers

**Every capacity, bandwidth, and link figure carries a `spec:` comment naming its source.** The
paper's capacity numbers are only as credible as their provenance, so a number without a citation
is a bug, not a detail.

> **⚠ Verify before submission.** The figures here were transcribed from vendor specifications and
> have *not* been re-checked against the datasheet PDFs. They are close enough to develop and test
> the admission machinery against, and several are rounded (see per-file notes). Before any
> published result, each `spec:` line must be confirmed against the cited document, and this
> warning removed.

Known roundings and judgment calls:

- Per-SM shared memory is the *configurable maximum* per SM, not the default carve-out. Vx models
  a single SM's budget, since that is what a tile placement has to fit.
- `L2` on NVIDIA parts and `Infinity Cache` on MI300X are modelled under one name (`L2`) because
  they play the same role in a placement decision, despite differing in architecture.
- Interconnect bandwidths are quoted **unidirectional** where vendors publish a bidirectional
  aggregate; halved figures are marked in the comment.
- Transfer costs (the `: N` on an edge) are relative latencies for path selection, not measured
  numbers. The bandwidth-derived roofline (`bandwidth:`) is what carries physical meaning.
