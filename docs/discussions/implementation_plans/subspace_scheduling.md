# Proposal: Scheduling Memory into Sub-spaces (TMEM/SMEM/RMEM)

> Status: proposal / design. Follows
> [`first_class_memory_spaces.md`](./first_class_memory_spaces.md) (which declared + *checked*
> spaces) and the FlashAttention line ([`flash_attention.md`](./flash_attention.md),
> [`slice_operators.md`](./slice_operators.md)). Driving case: the FA-4 tile dance — Q/K/V in
> SMEM, the score/probability tiles in TMEM, softmax statistics in registers — assigned to
> concrete positions *inside* each sub-space, granule-rounded and capacity-bounded.

## 1. Why this isn't done yet

Two reasons — one deliberate, one mechanical.

**(a) It was an explicit scope boundary.** `first_class_memory_spaces.md` §1 draws the line: that
work is about *describing and reasoning over* the hierarchy (declaring spaces; checking placement,
capacity, coherence, movement cost) and **not** the backend *scheduling* that exploits those
spaces. So we built the checking half — `check_capacity` / `check_cumulative_capacity`
(`src/hir/expr.rs`, `E6009`/`E6010`/`W1028`), coherence (`MemoryHierarchy::coherence_issues`), and
`derived_transfer_cost` — but never the *placement* half that assigns a tile to an offset inside a
sub-space.

**(b) The sub-space identity is erased before the optimizer sees it.** This is the real blocker.
`transfer(x, Memory::TMEM)` lowers (`src/codegen/lower/tensors.rs`) to a `vx.transfer` carrying
only:

- `target_topology` — a coarse dispatch id (`arch::memory_space_dispatch_id`), and
- `cost` — the bandwidth-derived roofline number.

And the result memref's address space collapses **every** custom sub-space to one number —
`arch::memory_space_address_space`: `MemorySpace::Custom(_) => 4`. So by the time the vx-dialect
passes (`src/dialect/VxLowering.cpp`) run, `Memory::TMEM` and `Memory::SMEM` are
**indistinguishable**: no `granule`, no `scope`, no `TMEM ⊂ SMEM ⊂ HBM` containment survives into
the IR. A scheduling pass literally cannot see the sub-spaces it would schedule into.

So the prerequisite for *any* later pass — a scheduler here, or a real device backend later — is to
**preserve the sub-space and its descriptor in the MLIR as metadata**.

## 2. Design

Two slices. Slice A is the enabling metadata; Slice B is the scheduling that consumes it.

### 2.1 Slice A — preserve the sub-space descriptor in the IR (the metadata)

Attach the declared descriptor to each `vx.transfer` as attributes, sourced from the `MemoryDecl`
(already on `Program.memories`, indexed in `GlobalAstEnv`; the queries live on
`MemoryHierarchy`). All values are already normalized to bytes in the `MemoryDecl`:

| Attribute | Type | Source | Why a later pass needs it |
|---|---|---|---|
| `space` | string | `MemorySpace::name()` | the identity — group placements by sub-space |
| `within` | string | `MemoryDecl.parent` name | containment (TMEM ⊂ SMEM ⊂ HBM) |
| `granule` | i64 (bytes) | `MemoryDecl.granule` | allocation rounding (TMEM's 16 KB granules) |
| `capacity` | i64 (bytes) | `MemoryDecl.capacity` | the bump-allocator bound |
| `scope` | string | `MemoryDecl.scope` (`device`/`sm`/`cta`/`thread`) | locality (an `sm`-scoped 256 KB is *per-SM*) |

`space` is always emitted (even for a bare `Memory::X` with no body — the name still identifies it);
the rest are emitted only when declared. These are **discardable metadata** on the op: they do not
change the memref type or the lowering, so `run-jit` (which flattens everything onto the CPU) is
untouched. They are visible in `emit-mlir` at the vx-dialect level — exactly where a `vx.*` pass
runs, before `convert-vx-to-standard` lowers `vx.transfer` away.

**Why attributes, not the memref memory-space / address-space (for now).** MLIR *can* carry a
memory space in the type (`memref<…, "TMEM">`, or a distinct integer address space). That is the
"right" long-term home, but it breaks the CPU-fallback JIT today: `finalize-memref-to-llvm` needs an
integer/absent memory space to compute addresses, and a per-sub-space address space would have to be
mapped to real NVPTX spaces (shared = 3, …) that the CPU path can't honor. Discardable op-attributes
give later passes the full descriptor now, with zero lowering risk; promoting the identity into the
memref type (or a `#vx.memspace<…>` attribute) is a follow-up gated on a device backend that acts on
it (§4, open question 1).

### 2.2 Slice B — the `vx-schedule-memory` scheduler (consumes the metadata)

A pass that groups placements by `space`, runs a **granule-rounded bump allocator** per sub-space,
and assigns each placed tile a concrete `offset` (+ slot count) within its sub-space, respecting
`capacity`. It annotates each `vx.transfer` (or the allocation) with its assignment and can raise a
sharper diagnostic than `E6010` — one that *shows the computed layout* when granule-rounded
sub-allocations overflow.

**Recommended first form: a Rust pre-pass, not C++.** The scheduler needs only the placements (sema
already collects `memory_placements` per space in the `TypeChecker`) and the granule/capacity from
the `MemoryHierarchy` — all Rust-side. It computes offsets and emits them as attributes (`offset`,
`slots`) on the transfers during/after codegen, and is fully exercised through the existing
golden-MLIR harness (`update_mlir_test_checks` + `run_optimization_test`). A C++ `VxLowering` pattern
that *acts* on the offsets belongs with the device backend and can come later.

## 3. Milestones

- **SS1 — Sub-space metadata on `vx.transfer` (Slice A). ✅ Done.** `Program.memories` is plumbed
  into the codegen generator; `vx.transfer` carries `space`/`within`/`granule`/`capacity`/`scope`
  from the `MemoryDecl` as discardable attributes (no type/lowering change — the CPU JIT is
  untouched). A sub-space is also now reachable via its enclosing space (the transfer resolves the
  target to the nearest reachable `within:` ancestor), so `transfer(t, Memory::SMEM)` type-checks
  instead of failing "no hardware path". Test: `middle_end/pass/subspace_metadata.vx`.
  *Foundational — the metadata every later pass reads.*
- **SS2 — Granule-rounded per-sub-space offsets (Slice B, analysis). ✅ Done.** A per-function bump
  allocator assigns each tile transferred into a granule'd sub-space an `offset` (bytes) and `slots`
  (granule count), emitted on `vx.transfer`. The tile size is the source tensor's static shape
  (`infer_ast_type` + `static_tensor_bytes`); the offset is the next free byte, granule-rounded, and
  the cursor advances (reset per function, since a sub-space is reused across kernels). Scoped to
  granule'd sub-spaces, so HBM/topology transfers are untouched. Test:
  `middle_end/pass/subspace_schedule.vx` (two 64 KB tiles → `0` and `65536`, 4 slots each).
- **SS3 — Granule-aware overflow diagnostic. ✅ Done.** The cumulative-capacity check (`E6010`/
  `W1028`) rounds each tile up to the granule before summing — a tile smaller than a granule still
  occupies a whole one — catching over-subscription the raw sum misses. The message names the
  granule and the rounded total. Test: `frontend/fail/memory_granule_overflow.vx` (96 KB raw fits,
  192 KB granule-rounded does not).
- **SS4 — (Later) promote identity into the type / a device pattern.** Once a device backend exists,
  move the sub-space onto the memref (`#vx.memspace<…>` or a real address space) and add a
  `VxLowering` pattern that consumes the SS2 offsets. Gated on hardware; out of scope now.

## 4. Open questions

1. **Where the identity ultimately lives.** Op-attributes now (zero JIT risk); memref memory-space /
   `#vx.memspace<…>` once a backend consumes it. The attributes are forward-compatible — SS4 can read
   them to synthesize the type.
1. **What carries the offset.** `vx.transfer` is convenient but is lowered away by
   `convert-vx-to-standard`; a scheduler pass runs before that and can see it. If offsets must
   survive lowering, they move onto the resulting `memref.alloc` / view (SS4).
1. **Interaction with the cumulative-capacity check.** SS3's layout-aware overflow is a strict
   refinement of `E6010` (which sums raw sizes); the `overcommit` escape hatch (`W1028`) should apply
   to both. Keep one budget notion, two granularities.
1. **Multiplicity / scope.** An `sm`-scoped sub-space is *per-SM*; a schedule is per-instance, not
   global. SS2 assumes a single instance (correct for one kernel); multi-SM replication is a later
   concern, tracked with `scope`.
