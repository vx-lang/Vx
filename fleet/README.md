# `fleet/` — SKU machine models

Machine-model files for `vxc --machine <file> program.vx` (#281). Each file declares one SKU's
memory hierarchy and its transfer topology, and nothing else: no functions, no program logic. The
admission program is written once and admitted against each SKU by swapping the flag, which is the
claim the MLSys fleet-admission paper rests on.

## The shared vocabulary

Every SKU declares the **same space names**, so one program text can reference them:

| Space | What it is | Scope |
|--------|-------------------------------------------|----------|
| `HBM` | device-global memory (HBM2e/HBM3/HBM3e) | `device` |
| `L2` | last-level on-die cache / Infinity Cache | `device` |
| `SMEM` | per-SM/CU shared memory (scratchpad) | `sm` |

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

### Verification status

| Figure class | Status |
|---|---|
| `HBM` capacity and bandwidth (all six SKUs) | **verified 2026-08-03** — quoted from per-GPU vendor spec tables |
| `L2` capacity and bandwidth | **unverified** — transcribed from architecture whitepapers from memory |
| `SMEM` capacity | **unverified** — per-SM/CU configurable maximum |
| Interconnect figures in `node-8gpu.vx` | **unverified** |
| Transfer costs (`: N` on an edge) | not physical — relative latencies for path selection |

The first verification pass found **four wrong figures**, all in the direction of understating the
machine, which is worth recording as evidence that transcribing from memory is not adequate here:

- H100 bandwidth was written as `3 TB/s`; the spec is **3.35 TB/s** (10% low).

- H200 bandwidth was written as `4 TB/s`; the spec is **4.8 TB/s** (17% low).

- MI300X bandwidth was written as `5 TB/s`; the spec is **5.3 TB/s**.

- B200 was written as `192 GB` at `8 TB/s`. HGX B200 — the part a neocloud rents — is
  **180 GB at 7.7 TB/s**. `192 GB` is a pre-release figure still repeated by secondary sources;
  `8 TB/s` belongs to the 186 GB GB200 NVL part. Modelling 192 GB would *admit configurations that
  do not fit a rented B200* (a false accept, the class this campaign exists to catch), and pairing
  180 GB with 8 TB/s would understate every transfer cost on that SKU.

  This one took three passes to get right, which is the useful lesson: the first pass used memory
  (192 GB), the second divided a DGX system aggregate (180 GB, 8 TB/s — capacity right, bandwidth
  wrong, and both derived rather than quoted), and only the third found a per-GPU spec table. A
  figure that is *plausible and self-consistent* can still be a mix of two different SKUs. The
  variant table is enumerated in `b200.vx` so the next reader does not repeat it.

A rounded bandwidth is not cosmetic: `bandwidth:` drives the derived roofline cost, so an
understated figure inflates every transfer cost on that SKU.

**Units are binary.** Vx parses `GB` as 2^30 and `TB` as 2^40, so `capacity: 180 GB` means
180 GiB. Vendors are inconsistent about whether their published "GB" is decimal or binary, and for
memory capacity it is conventionally binary — but this has not been confirmed per figure, and a
decimal reading would make each capacity ~7% smaller than modelled. That is smaller than the
margins in the current matrix but large enough to flip a marginal cell, so it belongs on the
verification list above rather than in a footnote.

> **⚠ Still to verify before submission.** The unverified rows above. Each `spec:` line must be
> confirmed against the cited document, and this notice narrowed as rows are cleared.

Known roundings and judgment calls:

- Per-SM shared memory is the *configurable maximum* per SM, not the default carve-out. Vx models
  a single SM's budget, since that is what a tile placement has to fit.
- `L2` on NVIDIA parts and `Infinity Cache` on MI300X are modelled under one name (`L2`) because
  they play the same role in a placement decision, despite differing in architecture.
- Interconnect bandwidths are quoted **unidirectional** where vendors publish a bidirectional
  aggregate; halved figures are marked in the comment.
- Transfer costs (the `: N` on an edge) are relative latencies for path selection, not measured
  numbers. The bandwidth-derived roofline (`bandwidth:`) is what carries physical meaning.
