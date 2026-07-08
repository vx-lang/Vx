# Proposal: First-Class, Programmer-Definable Memory Spaces & Sub-Spaces

> Status: proposal / design. Driving use case: FlashAttention
> ([`flash_attention.md`](./flash_attention.md)) and, more broadly, any kernel whose
> performance is governed by an asymmetric memory hierarchy (HBM → L2 → SMEM → TMEM → RMEM).
> Design decisions taken: **standalone space tree** (spaces nest via `within:`, topologies
> untouched) with **derived transfer costs** (cost computed from bandwidth along the
> hierarchy; explicit edges override).

## 1. Motivation

Vx's reason to exist is heterogeneity: making the *placement* and *movement* of data across a
machine's memories a first-class, checkable part of the language. Today Vx does this for
whole *topologies* (`Topology GPU`, `Topology NPU[i]`, user-declared `Topology Acme { … }`),
but the *memories* those topologies contain are second-class:

```rust
// Today: `Memory::TMEM` parses, but it is just an opaque name.
let acc: Ref<Tensor<f32, [128, 128]>, Memory::TMEM> = ...;
```

`Memory::TMEM` lexes to `MemorySpace::Custom("TMEM")` — a bare symbol with **no properties and
no structure**. The compiler cannot answer any of the questions a heterogeneous programmer
actually has:

- *Does this tile fit?* TMEM is 256 KB per SM; SMEM ~228 KB. There is no `capacity` to check.
- *How expensive is this move?* The FlashAttention-4 roofline is entirely about bytes ÷
  bandwidth per space (`T_smem = 3MNd/8192` cycles, etc.). There is no `bandwidth` to reason with.
- *Where does it live?* TMEM and SMEM are *inside* an SM; HBM is *outside*. There is no notion
  of containment / sub-space.
- *Must I move it explicitly?* SMEM/TMEM are programmer-managed; L2 is hardware-cached. There is
  no `managed` distinction to drive whether a `transfer` is required.

FlashAttention makes this concrete and unavoidable: the whole algorithm is a dance of *which
memory a tile lives in and when it moves* (Q/K/V tiles staged in SMEM, the score/accumulator in
TMEM, softmax statistics in registers). We cannot express — let alone check — that dance until
memory spaces carry real semantics. This proposal makes them first-class.

**Scope boundary.** This proposal is about *describing and reasoning over* the memory hierarchy
(declaring spaces, their capacity/bandwidth/nesting; checking placement and movement). It is
**not** about the backend *scheduling* that exploits those spaces (warp specialization, async
MMA pipelining, 2-CTA mode) — that remains a compiler concern below the language, exactly as in
the FlashAttention plan.

## 2. Current state (what exists to build on)

| Concept | Where | Shape |
|---|---|---|
| Memory space (value) | `src/syntax/types.rs` | `enum MemorySpace { CPUDRAM, NPUHBM, GpuHbm, LocalSRAM, NicRam, RemoteHbm, Custom(Symbol) }` |
| Surface syntax | `src/parser/types.rs::parse_memory_space` | `Memory::CPU_DRAM … Memory::<any-ident>` → `Custom` |
| Topology descriptor | `src/arch.rs` | `TopologyDescriptor { default_space, visibility, transfers }` |
| Transfer edge | `src/arch.rs` | `TransferEdge { from, to, cost: u32, sync: bool }` |
| Cost graph | `src/arch.rs` | `TransferCostGraph` (all-pairs shortest paths over edges) |
| Topology declaration | `src/parser/decl.rs::parse_topology_decl` | `Topology X { memory:, visible:, transfer … }` |
| Registry + reset | `src/arch.rs` | `TOPOLOGY_REGISTRY`, `register_topology`, `reset_topology_registry` (per-compilation) |
| Coherence checks | `src/arch.rs::descriptor_coherence` | default-space visible, host-reachable |

Key observation: **memory spaces have no descriptor.** There is a rich `TopologyDescriptor` but
no `MemoryDescriptor`. This proposal adds the missing half, mirroring the topology machinery
(which was recently made per-compilation-safe; see brittleness item #9).

## 3. Design

### 3.1 The `Memory` declaration

A standalone top-level declaration, parallel to `Topology X { … }`:

```rust
Memory HBM {
    capacity:  192 GB,
    bandwidth: 8 TB/s,
    managed:   cached,          // hardware-cached (default); no explicit transfer required
}

Memory SMEM {
    within:    HBM,             // hierarchy: SMEM is a sub-space "closer" than HBM
    capacity:  228 KB,
    bandwidth: 128 B/cyc,       // paper's SMEM read throughput
    managed:   explicit,        // programmer-managed; movement needs an explicit `transfer`
}

Memory TMEM {
    within:    SMEM,            // (or `within: HBM` — a sibling; see 3.2)
    capacity:  256 KB,
    granule:   16 KB,           // TMEM is allocated in 32-column / 16 KB granules
    managed:   explicit,
}
```

All fields are optional except the name; a space with no `within:` is a hierarchy root. Every
field is a property the compiler can *use* (capacity → fit checks, bandwidth → cost, `within` →
containment, `managed` → transfer obligation, `granule` → allocation rounding).

Grammar (extends `docs/lang/grammar.md`):

```ebnf
memory_decl  ::= "Memory" identifier "{" ( memory_field ","? )* "}"
memory_field ::=
    | "within"    ":" memory_space
    | "capacity"  ":" size_literal
    | "bandwidth" ":" rate_literal
    | "granule"   ":" size_literal
    | "managed"   ":" ( "explicit" | "cached" )

size_literal ::= number ( "B" | "KB" | "MB" | "GB" | "TB" )
rate_literal ::= number ( "B" | "KB" | "MB" | "GB" | "TB" ) "/" ( "s" | "cyc" )
```

`Memory::<Name>` in a type (`Ref<T, Memory::SMEM>`) resolves to the declared space. An
undeclared `Memory::Foo` stays a bare `Custom("Foo")` (unchanged, so nothing regresses) — a
declaration is what *upgrades* a name to a described space.

### 3.2 Hierarchy — the standalone space tree

`within:` forms a tree (a space has at most one parent). Containment is the transitive closure:
`TMEM within SMEM within HBM` ⇒ TMEM ⊂ SMEM ⊂ HBM. Spaces sharing a parent are **siblings**
(SMEM and TMEM may both sit `within: HBM` if modeled as peers inside the device rather than
nested). The tree is intentionally independent of the *execution* hierarchy (device/SM/thread) —
that keeps topologies untouched and lets a space be reasoned about on its own. (If we later want
"per-SM" semantics, a space can gain an optional `scope:` tag without disturbing the tree.)

The tree gives us, for free: nearest-common-ancestor (the level two tiles must meet at to share
data), depth (distance from compute), and a canonical move path between any two spaces.

### 3.3 Derived transfer costs

The current cost graph uses fixed integer `TransferEdge.cost`. We make it
**bandwidth-parameterized** so the paper's roofline is *computable*, not hand-tuned:

- A move between adjacent spaces on the tree costs `ceil(bytes / bandwidth)` (matching the
  paper's `T = bytes / (B/cyc)` cycle formulas).
- A move between non-adjacent spaces is the path through their nearest common ancestor, summing
  per-hop costs.
- An explicit `transfer Memory::A -> Memory::B : <cost> [relaxed|sync]` (existing topology-decl
  syntax) remains an **override** for edges that are not a simple bandwidth division.

Concretely, `TransferEdge.cost: u32` becomes `cost: TransferCost` where
`TransferCost = Fixed(u32) | PerByte(bandwidth)`, and the cost-graph query takes a transfer size.
Fixed costs preserve today's behavior exactly; `PerByte` is what a bandwidth-annotated `Memory`
contributes. This is the one change that ripples into `TransferCostGraph` and the seam engine.

### 3.4 Capacity checking & allocation-into-space

With `capacity` known, the compiler can reject a placement that cannot fit:

```rust
// SMEM is 228 KB; a 256×256 f32 tile is 256 KB → compile error with the numbers.
let tile: Ref<Tensor<f32, [256, 256]>, Memory::SMEM> = ...;   // E: 256 KB exceeds SMEM capacity 228 KB
```

The check fires wherever a statically-shaped tensor is bound to a `Ref<_, Memory::X>` /
`Pinned` / allocated into `X`, using `size_of(element) · Π(shape)` rounded up to `granule`.
Allocation-into-space needs a surface form; the existing (currently unimplemented) `with`
syntax in `docs/lang/types.md` is the natural home:

```rust
let tile = Tensor::new([128, 64]) in Memory::SMEM;   // spelling TBD: `in` vs `with`
```

### 3.5 Management model & the transfer obligation

`managed: explicit` (SMEM, TMEM, HBM device memory) means data movement into/out of the space
requires an explicit `transfer(...)` — the compiler flags an implicit cross-space use, exactly
as it does today for topology-specific memory. `managed: cached` (L2, host-cached) means moves
may be implicit. This generalizes the current hard-coded "you must `transfer`" rule into a
per-space property, and feeds the seam obligations (a `relaxed` move into an `explicit` space is
where a coherence seam must be discharged).

## 4. Integration with existing machinery

**Where descriptors live (decision): on the AST, indexed by `GlobalAstEnv` — *not* a second
process-global registry.** Topology descriptors today live in a process-global
`TOPOLOGY_REGISTRY` that the parser mutates by side effect; the `TypeChecker` reads it via
`seed_from_topology_registry()`, and `Program.topologies` carries only the *names*. That split
is exactly the leakage that forced `reset_topology_registry()` (brittleness #9). Memory
descriptors are a chance to do it right and avoid repeating it:

- **AST.** `Memory X { … }` parses into a `MemoryDecl` node collected on `Program.memories`
  (like `struct`/`enum` decls), *not* registered by a parse-time global side effect.
- **Sema.** `GlobalAstEnv` (`src/hir/env.rs`) — the per-compilation symbol table the
  `TypeChecker` already uses for `structs`/`enums`/`functions` — gains
  `memories: HashMap<Symbol, &MemoryDecl>`, populated in `build()`. `Memory::X` resolution and
  capacity checks then flow through the same lookup the checker trusts for every other symbol.
  No global ⇒ nothing to leak across in-process compilations ⇒ no reset to remember. (This also
  charts the migration path for topologies — the deferred remainder of #9.)
- **Codegen.** The generator is sema-agnostic (it never receives `GlobalAstEnv`) but *does*
  receive the `Program`, so it reads descriptors from `Program.memories` — mirroring how it
  already builds its `structs`/`enums` maps from the program. Descriptors it needs:
  allocation-into-space, memref address space, granule rounding.
- **`MemoryDescriptor`** itself: `{ parent: Option<MemorySpace>, capacity, bandwidth, managed, granule }`, stored on the `MemoryDecl` (normalized units).

Rest of the wiring:

- **Parser.** `parse_memory_decl` mirrors `parse_topology_decl`; dispatch on the `Memory`
  keyword at top level. (`Memory` is already `TokenType::Memory`, so `Memory X { … }` reads
  naturally, just like `Topology X { … }`.)
- **Cost graph / seam.** `TransferCostGraph` learns `PerByte` edges and a size-aware query,
  seeded per-compilation via a new `seed_from_memory_env(env)` alongside the existing
  `seed_from_topology_registry()` call in `TypeChecker::new` — same seeding point, but the source
  is the per-compilation env, not a global. The seam engine (`src/hir/seam.rs`) already reasons
  about coherence across a transfer and needs only the `managed` bit to decide when a seam
  obligation applies.
- **Coherence.** A memory-coherence pass (analogous to `check_topology_coherence(&ast.topologies)`)
  runs over `Program.memories`: `within:` is acyclic; a child's `capacity ≤` parent's;
  `bandwidth`/`granule` are positive; units parse. Emit diagnostics (new `E60xx` codes), not panics.
- **Docs.** Fold the grammar into `docs/lang/grammar.md`, and add a "Memory spaces" section to
  `docs/lang/types.md` / a new `docs/lang/memory.md`, cross-linking `hardware_monad.md` (spaces
  are the objects of the memory-space category; this proposal gives those objects structure).

## 5. Milestones (each lands with tests, mirroring the topology work)

Pulled by FlashAttention where noted; FA phases 0–3 (CPU reference + flash forward) proceed in
parallel and do not block on these.

- **M1 — `Memory` declarations, AST-carried + env-indexed (no hierarchy yet).** Parse
  `Memory X { capacity, bandwidth, managed, granule }` into a `MemoryDecl` on `Program.memories`;
  index it in `GlobalAstEnv.memories`; `Ref<T, Memory::X>` resolves through the env; codegen reads
  it from the `Program`. No process-global registry, no reset (see §4). Unit tests for parse +
  env indexing + `Program`-clone round-trip. *Unblocks: naming TMEM/SMEM with real semantics.*
- **M2 — Hierarchy (`within:`) + coherence. ✅ Landed.** `src/hir/memory.rs`'s
  `MemoryHierarchy` (built per-compilation from `Program.memories`, keyed by
  `MemorySpace::from_name`) provides `parent` / `ancestors` / `contains` /
  `nearest_common_ancestor` (all cycle-safe) and `coherence_issues`. `check_memory_coherence`
  on the `TypeChecker` emits `E6006` (a `within:` cycle), `E6007` (a sub-space larger than its
  parent), and `E6008` (a non-positive `capacity`/`bandwidth`/`granule`); it reads the
  per-compilation env, so it needs no scoping list (unlike topologies). Wired into the driver
  and the frontend test harness. Tests: 7 unit + 3 `.vx` (cycle, capacity-exceeds-parent,
  well-formed sibling hierarchy).
- **M3 — Capacity checking. ✅ Landed (allocation sugar deferred to #184).**
  `hir::memory::static_tensor_bytes` computes `ceil(element_bits × Π(dims) / 8)` (sub-byte types
  like `i4` pack; dynamic/generic shapes are skipped). `TypeChecker::check_capacity` rounds up to
  the space's `granule` and compares to its `capacity`, emitting **E6009** with the byte numbers.
  Hooked at two syntax-free sites: `transfer(tensor, Memory::X)` (the FlashAttention path) and
  `Ref`/`Pinned` `let`-annotations (`check_type_placement`). Tests: 2 unit (dense + sub-byte +
  dynamic/generic) and 3 `.vx` (over-capacity, granule tips it over, fitting). The dedicated
  `in Memory::X` **allocation** surface form is a separate additive change (#184) — capacity
  checking works today via `Ref`/`transfer`. *Pulled by FlashAttention Phase 4.*
- **M4 — Derived, bandwidth-parameterized costs. ✅ Landed (additive, not a TransferEdge
  rewrite).** `MemoryHierarchy::derived_transfer_cost(src, dst, bytes)` computes the paper's
  roofline — `Σ ceil(bytes / bandwidth)` over each non-NCA space on the tree path (so a tile read
  into SMEM at 128 B/cyc costs `bytes/128` cycles). Sema computes it from the transferred tensor's
  static byte size and stamps it on the (previously vestigial) `TransferExpr.cost`; codegen emits
  it as a `cost` attribute on `vx.transfer`. **Design choice:** rather than change
  `TransferEdge.cost: u32` into a `Fixed | PerByte` enum (which would ripple into
  reachability/seam — the flagged risk), the derived cost is an *additive, size-aware query* over
  the memory hierarchy; the fixed cost graph and `transfer_path` are untouched, so reachability
  and seam behavior are byte-for-byte unchanged. A `cost` attribute appears **only** when a
  bandwidth makes it derivable (bandwidth-less transfers emit nothing, as before). Explicit
  topology `transfer … : C` edges remain the fixed reachability cost (a full override-precedence
  is a later refinement). Tests: 3 unit (single-hop roofline, two-hop sum, the non-derivable
  cases incl. mixed units) + 1 `.vx` (`cost = 128` on `vx.transfer`).
  *Pulled by FlashAttention Phase 4 (the roofline / seam story).*
- **M5 — Management-driven transfer obligation. ✅ Landed.** `managed: cached` (hardware-coherent)
  vs `explicit`/undeclared (programmer-managed) now drives two checks, both via one helper
  (`space_is_cached`, reading the per-compilation env). **Access facet (always-on):** the
  cross-space access error (`NeedsSeam`) is suppressed when the value's memory space — taken from
  its *type* (`Ref`'s space / `Pinned`'s topology default), not its binding scope — is `cached`;
  so declaring `Memory NPU_HBM { managed: cached }` legalizes an otherwise-rejected cross-topology
  use. **Seam facet (`--verify-seams`):** a relaxed transfer into a `cached` space skips the
  coherence obligation (E6004), while an `explicit` space enforces it — the proposal's "relaxed
  move into an explicit space is where a seam is discharged." Undeclared spaces stay strict, so
  nothing regresses. Tests: 3 `.vx` (cached allows cross-access; relaxed→explicit rejected;
  relaxed→cached fine) — the seam ones via a new `// VERIFY-SEAMS` harness directive.

### Mapping to the FlashAttention plan

| FA phase | Memory-model milestone it exercises |
|---|---|
| 0–3 (CPU reference + flash forward) | none — pure algorithm on host tensors |
| 4 (heterogeneous placement) | M1–M4: declare `HBM/SMEM/TMEM`, place Q/K/V tiles + accumulator, check capacity, derive move costs, read them off in the seam report |
| 5 (backward, stretch) | reuses the same spaces |

The convergence point is FA Phase 4: that is where "FlashAttention in Vx" stops being a C-style
loop nest and becomes a program whose *memory hierarchy is declared and checked* — the actual
point of the language.

## 6. Open questions / risks

1. **Size-aware cost graph.** Making `TransferCostGraph` size-parameterized touches the seam
   engine and any current consumer that assumes a scalar cost. M4 must keep `Fixed` costs
   behaving exactly as today (regression-tested) and confine the change behind the query API.
1. **Units in the lexer.** `256 KB`, `8 TB/s`, `128 B/cyc` need literal-with-unit parsing.
   Simplest: parse `number` then a unit identifier in `parse_memory_decl` (no lexer change);
   store normalized to bytes and bytes/cycle.
1. **Siblings vs nesting for TMEM/SMEM. Resolved: siblings.** TMEM (256 KB) and SMEM (~228 KB)
   are physically peers per-SM — TMEM is not *inside* SMEM — so they are modeled as siblings,
   both `within: HBM` (the device space). Nesting them would (correctly) trip the
   capacity-monotonic check (`E6007`). The `memory_hierarchy.vx` test encodes this.
1. **Relationship to `Custom`.** A declared `Memory` and an undeclared `Memory::Foo` both map to
   `MemorySpace::Custom(name)` at the value level; the *descriptor* is what differs. Keep the
   value representation unchanged so existing custom-memory code (and the topology `memory:` /
   `transfer` clauses that already accept custom names) keeps working.
1. **Execution scope. ✅ Landed (the `scope:` tag).** `Memory X { scope: device | sm | cta | thread }` labels the execution level a space is private to, so `capacity` reads as a per-scope
   budget (an `sm`-scoped 256 KB is one SM's TMEM). Coherence enforces that locality *narrows*
   down `within:` (a broader-scoped child is `E6011`). Driven by mapping FA-4 tiles to the B200
   sub-spaces (TMEM/SMEM are per-SM). Full *use-site* scope-crossing enforcement (a value can't
   escape its thread/CTA) still wants a finer execution model and remains future work — tracked
   in issue #187.

## Working-set (cumulative) budget check + `overcommit`

Beyond M3's *per-tile* capacity check, the compiler now checks the **cumulative** working set:
the sum of the tiles a function places in a space must fit its `capacity` (`E6010`), naming the
space, the total, and the tile count. This catches the collective overflow the per-tile check
misses — the exact FA-4 constraint of fitting Q/K/V in SMEM or S/P/O in TMEM. Because the sum is
conservative (it assumes all placed tiles are simultaneously live), a space declared
`overcommit` downgrades the error to a warning (`W1028`) — the programmer asserts the tiles do
not all coexist, so the check informs without blocking. The precise per-tile check (a single
tile larger than the whole space, always impossible) stays a hard `E6009`.

1. **Deliberate divergence from topologies.** Memory descriptors will be AST-carried + env-indexed
   while topology descriptors remain in the process-global `TOPOLOGY_REGISTRY` (§4). This is an
   intentional inconsistency: memories set the better precedent and topologies should migrate to
   match (the open remainder of brittleness #9). Until then, two subsystems answer "what is this
   space/topology?" differently — acceptable, but worth a tracking issue so it converges rather
   than calcifies.
