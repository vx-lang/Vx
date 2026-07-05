# The Hardware Monad: Open, User-Definable Topologies

> Status: **design record / brainstorming** (not yet implemented). Captures a
> design discussion on replacing Vx's closed `Topology`/`MemorySpace` enums with
> an open, user-extensible model, and the type-level encoding that keeps it sound.
> Companions: [`hardware_topology_plan.md`](../../hardware_topology_plan.md) (the
> original closed design), [`scalable_plugin_system.md`](../../scalable_plugin_system.md)
> (the execution-side plugin trait), and the per-seam obligation engine in
> `src/hir/seam.rs`.

## Problem

Today a topology is a closed enum baked into the language:

- `Topology` — `src/syntax/types.rs` (CPU, NPU, GPU, ANE, AMX, AccCore, …).
- `MemorySpace` — CPUDRAM, NPUHBM, GpuHbm, LocalSRAM, NicRam, RemoteHbm.
- The wiring that assumes both: `TransferCostGraph` in `src/arch.rs`
  (`default_memory_for`, and the hardcoded special-cases in `is_type_accessible`,
  `arch.rs:170-178`), plus `topology_to_i32` magic numbers in codegen.

A user cannot add a topology without editing the language and every pass that
matches the enum. We want: **users define their own topologies, admitted as long
as they satisfy certain criteria** — an *open* set with a *closed* interface.

There are already two half-abstractions to build on:

- `VxHardwarePlugin` (`src/plugin/hardware_trait.rs`) — the *execution* side
  (layout, alignment, `is_op_supported`, `lower_to_binary`). Extensible, but only
  covers "how to run," not "how memory/transfers behave."
- `TransferCostGraph` (`src/arch.rs`) — the *memory algebra* (edges, costs,
  visibility), but populated by a hardcoded `Default`.

## The categorical core

Memory spaces form a **category**: objects = spaces, morphisms = transfers,
composition = path concatenation, cost = a grading monoid. That category *is* the
`TransferCostGraph`; `transfer_path` (Dijkstra) finds a morphism `P → Q`.

A value "living on" a space is `Located<P, A>` — already present as
`Pinned<A, Topology>` / `Ref<A, MemorySpace>`.

The "hardware monad" is a **graded, indexed (parameterised) monad** over that
category — indexed because the *location changes*, graded because each step
carries a cost + consistency effect:

```
M[P, Q, w, A]   -- a computation from location P to location Q, grade w, yielding A

return   : A                -> M[P, P, 0, A]              -- a pure value doesn't move
bind     : M[P,Q,w,A] -> (A -> M[Q,R,v,B]) -> M[P,R,w·v,B] -- indices meet at Q; grades multiply
transfer : Located<P,A>     -> M[P, Q, cost(P→Q), Located<Q,A>]  -- the graded generator = a seam
```

Every piece already exists in Vx, informally:

| monad concept | Vx today |
| ---------------------------- | ------------------------------------------ |
| objects (spaces) | `MemorySpace` |
| morphism `P→Q` | a `transfer` / seam hop |
| composition of morphisms | `transfer_path` multi-hop routing |
| grade `w` (cost monoid) | the `u32` cost (`+` is the monoid) |
| **coherence law on `bind`** | **the per-seam obligation in `hir/seam.rs`**|
| index `Q` (current location) | `active_topology` / `active_memory` |
| "indices meet at Q" | the access check at a use site |

So `spawn on(D){…}` is a **Kleisli arrow**, and the seam obligation engine is the
monad's coherence condition. Chaining transfers = associativity of `bind`; the
reason a multi-hop path discharges a per-hop obligation is monad associativity on
a graded index. This is operational content, not analogy.

(Consistency — `Sync` vs `Relaxed` in `seam.rs` — is a *second grade component*: a
lattice of "how much visibility survives the hop." The read-back / observation
side is the comonadic dual, and is where the POPL companion's Galois connection
lives; not needed for composition soundness, so out of scope for v1.)

## The judgment

Write `Γ ; q ⊢ e : A @ p` — "in context Γ, running on topology `q`, expression `e`
has type `A` located on `p`." The load-bearing rules:

**Var** — location is recorded, not yet checked:

```
Γ, x : A @ p ;  q ⊢ x : A @ p
```

**Use (read)** — to read a `p`-located value while running on `q`, the location
must be *directly visible* from `q`. Visibility is a cost-0, no-copy relation
declared by the topology (unified memory). The value stays `@p` (no move):

```
Γ ; q ⊢ e : A @ p       Visible(q, p) ∈ Registry
────────────────────────────────────────────────    (USE-DIRECT)
Γ ; q ⊢ read(e) : A @ p
```

If `p` is *not* visible from `q`, USE-DIRECT does not apply. Under the
**explicit-seam policy** (see Decisions) the checker does **not** auto-insert a
transfer — it errors with the fix:

```
Γ ; q ⊢ e : A @ p    ¬Visible(q,p)    Reachable(p,q)
────────────────────────────────────────────────────  (USE-NEEDS-SEAM → ERROR)
error: `e` is @p, not visible from q; insert `transfer(e, q)`  (cost = …)
```

**Spawn** — index switch, and the result is *located*:

```
Γ ; d ⊢ body : B @ d
────────────────────────────────────────────
Γ ; q ⊢ spawn on(d){ body } : Pinned<B, d>
```

Note the conclusion is `Pinned<B, d>`, not `B`. Today `check_spawnon_expr`
(`src/hir/expr.rs:1389`) returns the *unwrapped* `ret_ty` — a soundness gap: the
result of a device kernel is silently treated as host-accessible. Returning
`Pinned<B, d>` makes reading it on `q` re-trigger USE (→ visibility or an explicit
transfer). This single change makes the monad honest.

**Transfer** — the graded generator (a Kleisli arrow / a seam):

```
Γ ; q ⊢ x : A @ p     Transfer<p, p'> ∈ scope     ⊢ seam_obligation(p → p')
──────────────────────────────────────────────────────────────────────────
Γ ; q ⊢ transfer(x, p') : A @ p'
```

`check_transfer_expr` + the seam engine already implement the obligation premise.

## Decisions

### 1. Explicit seams, pluggable via a `Transfer<From, To>` trait

Crossing a non-visible boundary is **always explicit** (`transfer(x, D)` in
source; no implicit coercion). But the *morphism itself* is a **trait impl** the
user provides, so the set of legal seams is open:

```rust
/// A morphism P -> Q in the memory-space category: the pluggable seam.
/// The set of in-scope impls IS the registry's edge set; `transfer(x, To)`
/// resolves to one (or to a compiler-composed path over several).
trait Transfer<From: Topology, To: Topology> {
    fn cost() -> Cost;                    // the grading (a monoid)
    fn consistency() -> Consistency;      // Sync | Relaxed | …  (feeds hir::seam)
    fn seam<A>(x: Located<From, A>) -> Located<To, A>;   // how to actually move it
}
```

Consequences, and why this is clean:

- **`Reachable<S, D>` ≡ "an impl `Transfer<S, D>` exists"** (directly, or as a
  composite the compiler builds via `transfer_path` over available impls). The
  type-level "indices must meet" constraint *is* trait resolution.
- The `TransferCostGraph` becomes the **closure of in-scope `Transfer` impls under
  composition**; Dijkstra finds the cheapest composite.
- The per-seam obligation is discharged at each `Transfer` step using its declared
  `consistency()` — the existing `check_seam` call, now fed from the trait instead
  of a hardcoded `relaxed` flag.
- Explicit call site keeps *cost visible* (this is a performance language); the
  type system still knows the coercion exists, so it can produce the precise
  USE-NEEDS-SEAM error above.

Three cleanly separated traits result:

| trait | role (category) | supplies |
| --------------------------- | ----------------------- | ------------------------------------------ |
| `Topology` | an **object** | spaces, `default_space`, `visibility`, id |
| `Transfer<From, To>` | a **morphism** | `cost`, `consistency`, `seam` lowering |
| `VxHardwarePlugin` (exists) | **functor to hardware** | `is_op_supported`, layout, `lower_to_binary`|

`Topology` is the "certain criteria": a user topology is admitted iff it supplies
these and passes a coherence check (`default_space ∈ spaces`; every claimed-
supported op has a lowering; consistency values form a lattice). Those coherence
conditions are the monad/functor laws made checkable — same z3 engine as the seams.

## Topology polymorphism (what makes user topologies usable in libraries)

Replace the closed `Topology` *in the type* with an abstract identity that is
either a **concrete registered topology** (`MyTPU`, an interned `TopologyId`) or a
**topology variable** bound by the trait:

```
fn normalize<D: Topology>(x: Pinned<Tensor<f32>, D>) on D -> Pinned<Tensor<f32>, D> {
    spawn on(D) { /* uses x @D; USE needs Visible(D,D) = identity, OK */ }
}

fn stage<S: Topology, D: Topology>(x: Pinned<T, S>) -> Pinned<T, D>
    where Transfer<S, D>            // the morphism must exist (Reachable<S,D>)
{ transfer(x, D) }
```

`normalize`/`stage` work for *any* topology a user later defines: extension =
satisfying a bound, not editing a variant. Concrete `Topology::GPU` becomes just
one instance — no privileged code path. The `where Transfer<S,D>` bound is threaded
like any trait bound and discharged at instantiation (graph lookup for ground
types; carried as an assumption for polymorphic code).

## The hard part: dependent topologies

`NPU[i]` where `i` is a runtime value (`Topology::NPU(Box<Expr>)`) is not pure
type-level identity — it is a topology *indexed by a runtime int* (a dependent
type). `NPU[i] ≡ NPU[j]` iff `i == j`, which is undecidable syntactically and is
already special-cased in `check_spawnon_expr` (`expr.rs:1360-1367`). Clean
encoding: **topology identity = (registered-kind, index-term)**, with index
equality decided by the const-evaluator / prover. Vx already has z3 wired, so
`NPU[i] ≡ NPU[j]` becomes a small `i == j` obligation — the *same* engine that
discharges seam and topology-coherence obligations. One prover, three uses.

## What changes in the checker

| today | becomes |
| -------------------------------------------- | -------------------------------------------------------------- |
| `active_topology: Topology` (enum) | `active_topology: TopologyId` — interned; may be a rigid var |
| `is_type_accessible(q,p) -> bool` | `reachable(q,p) -> Visible \| NeedsSeam(path) \| None` |
| special-cases `arch.rs:170-178` | deleted — they are ordinary registry (visibility) edges |
| `check_spawnon` returns `ret_ty` | returns `Pinned<ret_ty, d>` (closes the soundness gap) |
| `func.topology` (one fixed topology) | may bind a topology parameter `<D: Topology>` |
| hardcoded `relaxed` flag at the seam | `Transfer::consistency()` for the resolved impl |
| — | `Transfer<P,Q>` resolution = morphism existence (`Reachable`) |

None of this needs a new backend; it is type-checker + a registry the current enum
seeds into.

## Migration path (incremental, enum-as-seed)

1. **[LANDED]** **Registry behind the enum.** `TopologyDescriptor` +
   `topology_descriptor()` / `register_topology()` in `arch.rs`, seeded from
   `builtin_descriptors()`.
1. **[LANDED]** **Route hardcoded logic through it.** `default_memory_for` and
   `is_type_accessible` read the registry; the NPU/AccCore/GPU special-cases and the
   `visibility_edges` field are gone. (`topology_to_i32` still matches, plus a hashed
   id for `Custom`.)
1. **[LANDED, identity only]** **Open identity.** `Topology::Custom(Symbol)` +
   `TopologyKind::Custom(Symbol)`; the parser resolves any non-built-in
   `Topology::<Name>` to `Custom`. A user topology now flows parse→typecheck→MLIR;
   its memory model is supplied via `register_topology` (plugin API). *Not yet:* a
   source-level `topology { … }` declaration (below), and the `Transfer<From,To>` /
   `Topology` traits.
1. **[SUBSTANTIALLY LANDED, as data]** **`Transfer` + `Topology` "traits".** The
   object (`TopologyDescriptor`) and the morphism (`TransferEdge { from, to, cost, sync }`) exist as data with a language surface (`Topology <Name> { memory / visible / transfer ... }`); the cost graph *is* their closure (`seed_from_topology_registry`
   - Dijkstra); the consistency grade (`sync`/`relaxed`) is discharged via the seam
     engine in coherence checking. *Not done:* exposing these as first-class Vx `trait`s
     you `impl` per user type (`impl Transfer<A,B> for ...`) — largely redundant with the
     declaration surface, so deprioritized.
1. **[LANDED]** **User declarations + coherence check.** `Topology <Name> { … }`
   registers descriptors in-language; admitted iff coherence obligations discharge
   (E6005 / W1026 / W1027, the last via `hir::seam`). Typo-safety: W1025 for an
   undeclared/unregistered custom topology. Coherence is scoped per-program (no
   cross-file leakage). *Still open:* a full "unknown-unless-declared" hard error
   (kept a warning so plugin-registered topologies still work).
1. **[NOT STARTED — the big remaining one]** **Topology polymorphism.** `<D: Topology>`
   params and `where Transfer<S,D>` constraint solving. Needs a new generic-param kind
   (today `GenericParam` is only `Type`/`Const`; `Function.topology` is a single fixed
   `Topology`), topology-variable substitution in monomorphization, an `on D` binding,
   and constraint solving over topology variables at instantiation. A dedicated,
   design-first effort — not a tail-end slice.

## Highest-leverage first slice

Two small changes convert the *informal* index tracking into the *real* indexed
monad, independent of the full refactor. **Both landed.**

1. **[LANDED]** `spawn on(D)` returns `Pinned<B, d>` for a bare result `B` (a result
   already `Pinned<..>` is passed through, not double-wrapped; a void spawn is
   unchanged) — `check_spawnon_expr`.
1. **[LANDED]** `is_type_accessible` → `reachable` returning a morphism (visibility =
   cost-0 case); USE-DIRECT vs USE-NEEDS-SEAM is an explicit diagnostic.

## Open questions

- Composition of `Transfer` impls: auto-compose via Dijkstra (convenient) vs
  require the user to declare each composite (explicit)? Leaning auto-compose with
  the cost graph as the closure.
- Consistency lattice depth: start with `Sync`/`Relaxed`; leave room for the
  paper's scoped-RC11 scopes as richer grades.
- Inference limits with topology variables — likely require annotations at
  kernel/function boundaries (mostly already true).

## Use cases for topology polymorphism

Concrete scenarios that motivate `<D: Topology>` and pin down what the feature must
support. Each says what breaks *without* polymorphism and which existing machinery it
leans on, so the use cases double as design constraints.

### 1. Write-once library kernels

A library author writes an accelerator kernel *once*, generic over the device:

```
fn layernorm<D: Topology>(x: Pinned<Tensor<f32>, D>) on D -> Pinned<Tensor<f32>, D> {
    spawn on(D) { /* uses x @D; USE needs Visible(D,D) = identity, OK */ }
}
```

`layernorm` runs on `GPU`, `ANE`, or a vendor's `Topology MyTPU { … }` with no edit to
the library. **Without polymorphism** the author either duplicates the kernel per device
or hardcodes a closed `Topology` set — exactly the coupling this whole effort removes.
**Leans on:** the open topology identity (already landed) + the `on D` binding; the body
type-checks because a value `@D` used on `D` is the cost-0 identity morphism.
**Demands:** a topology generic-param kind, `on D`, and monomorphization over the
concrete `D` at each call.

### 2. Portable models across accelerators (the practical payoff)

Write a transformer layer — or the whole llama2 model in `benchmarks/` — with its
compute parameterized by `<D: Topology>`, then instantiate it on GPU, ANE, or `MyTPU`
by supplying the topology. One model, N backends, no per-backend fork. This is the
end-to-end version of use case 1 and the concrete answer to "why bother": a vendor ships
a `Topology` descriptor and every generic model runs on their silicon. **Without it:**
`matmul_ane` vs `matmul` forks (as in the current benchmark) multiply per device.
**Demands:** the same as (1), at whole-program scale.

### 3. Generic data staging with a *proven* path

A data-movement library (prefetch, double-buffer, tiling) written once for every device
pair:

```
fn stage<S: Topology, D: Topology>(x: Pinned<T, S>) -> Pinned<T, D>
    where Transfer<S, D>          // a morphism S ⇝ D must exist
{ transfer(x, D) }
```

`stage` works for host→GPU, GPU→NPU, NPU→remote — the `where Transfer<S, D>` is the
evidence a path exists, discharged against the cost graph (`transfer_path`; multi-hop
allowed, the grade is the join of the hops). **Without it:** one `stage` per ordered
device pair. **Leans on:** the cost graph as the morphism closure + the seam consistency
grade (both landed). **Demands:** `where`-clause constraint solving over topology
variables — carried as an assumption for polymorphic code, discharged at instantiation
(like a trait bound). This is the hard part.

### 4. Cost-directed device selection

Let the type system *and the cost monoid* pick the device:

```
fn run<D: Topology>(x: Tensor<T>) -> Pinned<T, D>  where Reachable<Host, D>
```

A scheduler instantiates `run` on whichever reachable `D` minimizes transfer cost —
`TransferCostGraph` already finds the cheapest morphism via Dijkstra, so "choose `D`" is
"minimize the grade." **Without it:** placement is hand-coded. **Demands:** everything in
(3) plus a policy that *chooses* the instantiation rather than taking it as given — the
most speculative use case, but it falls straight out of the graded-monad structure.

### 5. Topology-agnostic seam certificates

Ties to `--emit-seam-certs`. A guarded kernel written once, for all devices:

```
fn masked_block<D: Topology>(kblk_start: i32, qblk_end: i32, x: Pinned<T, D>) on D {
    assert(kblk_start > qblk_end);       // host-proven relation
    spawn on(D) { if kblk_start > qblk_end { /* skip */ } else { expensive(x) } }
}
```

The host-proven relation is transported into the `spawn on(D)` body as an
`llvm.intr.assume` regardless of `D` — the causal block-skip (the CGO `cert_fold`
result) written once and specialized per accelerator. **Leans on:** the seam-certificate
emission (landed) + polymorphism. **Demands:** cert emission already keys off the spawn
seam, so this mostly needs `on D` to exist; a nice proof that the two features compose.

### 6. Mock topologies for hardware-free testing

Declare a host-backed stand-in and instantiate a generic kernel over it:

```
Topology MockNPU { memory: Memory::CPU_DRAM }          // host-backed
// test: layernorm::<MockNPU>(x) runs on CI with no accelerator
```

Generic (`<D: Topology>`) accelerator code becomes testable on CI without the hardware,
and the coherence check still validates the mock's declaration. **Leans on:**
user-definable topologies + coherence (landed) + polymorphism. **Demands:** only that a
generic kernel can be instantiated at a user-declared `D` — the cheapest use case to
reach once (1) works, and a strong argument to build (1) first.

### What the use cases tell the design

- (1),(2),(5),(6) need only the **monomorphic core**: a topology generic-param kind,
  `on D` / `Pinned<T, D>` over a variable, and instantiation over a concrete `D`. That is
  a self-contained first slice and unlocks four of the six.
- (3),(4) need **constraint solving** over topology variables (`where Transfer<S,D>` /
  `Reachable<Host,D>`) — a second slice, and (4) additionally a *selection* policy.
- Build order therefore: **monomorphic `<D: Topology>` first** (use cases 1/2/5/6), then
  `where`-constraints (3), then cost-directed selection (4).
