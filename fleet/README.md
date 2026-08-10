# `fleet/` — SKU machine models

Machine-model files for `vxc --machine <file> program.vx` (#281). Each file declares one SKU's
memory hierarchy and its transfer topology, and nothing else: no functions, no program logic. The
admission program is written once and admitted against each SKU by swapping the flag, which is the
claim the MLSys fleet-admission paper rests on.

## Machines and hosts

The files here are **machine** models: accelerators, passed with `--machine`. A **host** — the
machine an accelerator hangs off — is a different thing, passed with `--host`, and declares
`Memory::CPU_DRAM` and nothing else (`host-x86-e5-2666v3.vx`). Neither is assumed, and a program
staging through host memory against a machine model must name a host or be refused (`E6014`).

A host declares no capacity: host memory is virtual, so a hard limit would reject programs that
page rather than fail. See [`docs/lang/hosts_and_machines.md`](../docs/lang/hosts_and_machines.md).

## The shared vocabulary

Every SKU declares the **same space names**, so one program text can reference them:

| Space | What it is | Scope |
|--------|-------------------------------------------|----------|
| `HBM` | device-global memory (HBM2e/HBM3/HBM3e) | `device` |
| `L2` | last-level on-die cache / Infinity Cache | `device` |
| `SMEM` | per-SM/CU shared memory (scratchpad) | `sm` |

A program that places a tile in `Memory::HBM` therefore means "this SKU's device memory", and the
capacity check resolves that to whichever number the selected machine file declares. Adding a SKU
means adding a file with these spaces, not touching any program.

**A SKU declares a space only if the hardware has one.** The vocabulary is shared so that programs
are portable, not so that every file is the same length, and a machine file is a description of
hardware before it is a row in a table. `xeon-e5-2666v3.vx` declares no `SMEM`, because x86 has no
software-managed scratchpad: the nearest thing is a per-core cache, and a tile is resident there
only while the hardware has no better use for the lines.

Declaring it anyway would make a program that places a tile in `Memory::SMEM` *admitted* against
that machine, on a capacity number that does not mean what it means on every other SKU. Omitting it
makes the compiler say what is true:

```
Error: Cannot transfer from CPUDRAM to Custom("SMEM"): no hardware path exists
```

A program that needs a scratchpad cannot run on a machine without one, and finds that out at
compile time. That is the claim this directory exists to support, and it is not worth weakening to
keep a table rectangular.

To turn an admission verdict into an engine launch command, see
[`utils/vllm/map_admission.py`](../utils/vllm/). It reads the `--diagnostics-json` record, not these
files — the mapping lives outside `fleet/` because it changes with vLLM's flag surface, not with
SKU data.

`node-8gpu.vx` is different in kind: it models one **8-GPU node**, adding the interconnect spaces
and edges (NVLink, PCIe, InfiniBand) that a multi-GPU placement crosses. It is composed with a SKU
file rather than replacing it — but note that the `--machine` flag currently takes a single file
(#281), so the node file re-declares the SKU spaces it needs. Declaration import (#224) would let
these compose properly; see the note in that file.

## Provenance of the numbers

**Every capacity, bandwidth, and link figure carries a `spec:` comment naming its source.** The
paper's capacity numbers are only as credible as their provenance, so a number without a citation
is a bug, not a detail.

### Units: SI is decimal, IEC is binary, and the file must say which

`GB` is 10^9; `GiB` is 2^30. Both spellings parse (`src/units.rs`), conversion is exact integer
arithmetic, and a figure that is not a whole number of bytes is rejected rather than rounded.

The rule exists because breaking it cost 9%. `TB` was previously read as 2^40 while every `spec:`
line here cites a vendor figure in **decimal** — NVIDIA's "3.35 TB/s" is 3.35e12 B/s, from a
5120-bit bus at 3.2 Gbps. The compiler therefore treated every link as ~10% faster than its own
citation claimed and understated every predicted transfer time by **9.05%**. In a calibration study
that residual would have been charged to the hardware.

Which to write:

| Field | Convention | Why |
|---|---|---|
| `bandwidth:` | **SI** (`3.35 TB/s`, `2039 GB/s`) | vendors quote memory bandwidth decimal |
| `capacity:`, `granule:` | **IEC** (`80 GiB`, `228 KiB`, `1 KiB`) | HBM stacks, caches and SMEM are genuinely powers of two |

Copy the digits from the spec sheet and pick the spelling that matches what the sheet meant — do
not pre-convert, since a hand-converted figure no longer matches its citation.

Note that whether a *quoted capacity* is decimal or binary is itself contested per SKU (the
192-vs-180 GiB gap below), and settling it by measurement is experiment **M4** in
`memory-algebra-paper/EXPERIMENTS.md`. The spellings here record current belief, not a verified
fact; `spec:` says where the belief came from.

### Verification status

| Figure class | Status |
|---|---|
| `HBM` capacity and bandwidth (A100 40/80, H100, H200, MI300X) | **verified 2026-08-03** — quoted from per-GPU vendor spec tables |
| `HBM` capacity (**B200**) | **measured on hardware** — 192 GB reported by the device. Vendor materials say 180 GB, most likely a usable-after-reservation figure |
| `HBM` bandwidth (**B200**) | **contested** — 7.7 TB/s (Lenovo per-GPU table) vs 8.0 TB/s (NVIDIA DGX aggregate ÷ 8). 7.7 used; see `b200.vx` |
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

  This one took four passes, which is the useful lesson. Memory said 192 GB. Dividing a DGX system
  aggregate gave 180 GB @ 8 TB/s — capacity right, but derived rather than quoted. A per-GPU spec
  table gave 180 GB @ 7.7 TB/s. Checking a secondary source that claimed 192 GB @ 8 TB/s then
  showed the bandwidth is *genuinely contested* between two reputable sources, not settled as the
  third pass had asserted.

  Two distinct lessons. First: a figure that is *plausible and self-consistent* can still be a mix
  of two SKUs — nothing about `180 GB @ 8 TB/s` looks wrong. Second, and easier to miss: the
  correction can overclaim too. Pass three explained the discrepancy with a tidy story ("8 TB/s is
  the GB200 NVL part") that the evidence did not support. **A resolved-sounding narrative is not
  the same as a resolved question**, and the write-up should say which one it has.

A rounded bandwidth is not cosmetic: `bandwidth:` drives the derived roofline cost, so an
understated figure inflates every transfer cost on that SKU.

**Units are binary.** Vx parses `GB` as 2^30 and `TB` as 2^40, so `capacity: 192 GB` means
192 GiB. Vendors are inconsistent about whether their published "GB" is decimal or binary, and for
memory capacity it is conventionally binary — but this has not been confirmed per figure, and a
decimal reading would make each capacity ~7% smaller than modelled. That is smaller than the
margins in the current matrix but large enough to flip a marginal cell, so it belongs on the
verification list above rather than in a footnote.

### Declared capacity is the device, not the deployment budget

**These files declare what the hardware has. They do not declare how much of it a deployment may
use, and those are different numbers — by more than the errors this document has been tracking.**

A serving engine cannot allocate 100% of device memory: framework overhead, fragmentation, and the
CUDA context all take a share. vLLM's `gpu_memory_utilization` defaults to `0.9`. So on a 192 GiB
B200 the usable budget is roughly 173 GiB, and a configuration in the ~173–192 GiB band is
**admitted by the current capacity check and will still OOM in practice**.

This is a false accept of the same class as the B200 capacity error, but it is not a mistake in any
figure — every number can be right and the verdict still wrong, because the check is comparing
against the wrong bound. The correct bound is `capacity x utilization`, and nothing in the model
expresses the second factor yet.

Two consequences worth stating before results are collected:

- A verdict near a SKU's ceiling means "fits the device", not "fits a serving deployment on the
  device". Cells close to the boundary should be treated as unresolved until this is modelled.
- This is what makes #285's `gpu_memory_utilization` field load-bearing rather than a convenience:
  it is the factor that turns a capacity verdict into an admission verdict. Deriving it is not
  optional polish.

The alternative — baking the headroom into `capacity:` — is rejected deliberately. It would
make the machine files describe a policy rather than a machine, hide the assumption where no
reviewer would find it, and silently change every verdict if a deployment tuned the knob.

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
