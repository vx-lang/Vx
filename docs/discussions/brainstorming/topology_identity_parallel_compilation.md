# Topology identity under parallel compilation

Status: brainstorm, 2026-08-18. Follows #355 (topology equality by value) and the two
questions it raised: why hashing the index AST breaks maps, and why `GPU[i]`/`GPU[j]`
stay unequal. Both turned out to be corners of one disease, and the parallel pipeline
is where the disease stops being latent.

## The invariant parallel compilation imposes

Any identity that crosses a worker boundary — a key in a shared map, a mangled symbol
name, a dedup key, an emitted id — must be four things at once:

1. **Canonical.** One representation per semantic device, independent of which worker,
   which phase, or which construction path minted it. Two spellings of device 0 must be
   the *same key*, not merely `==`.
1. **Context-free.** Meaningful without a scope or an eval environment. A worker that
   receives the key cannot ask the binder that produced it what it meant.
1. **Deterministic across schedules.** The same program at any thread count produces the
   same identity, byte for byte. The tree already tests this
   (`pipeline_emits_byte_identical_mlir_across_thread_counts`,
   `compile_pipeline_gid_stream_is_deterministic`).
1. **Totally ordered.** Anything emitted from a keyed collection needs a stable order
   (the `BTreeMap` discipline `region_traffic.rs` already follows). To be explicit,
   because it read otherwise on review: this is a NON-SEMANTIC sort key, nothing more.
   It does not order memories or topologies by size, speed, or hierarchy — the
   intra-topology hierarchy is the `within:` containment tree, cross-topology
   relationships are the transfer GRAPH (edges, not an order), and neither needs or
   gets a cross-topology ordering. This bullet exists only so that emission order
   cannot depend on thread scheduling.

An AST fragment fails 1 and 2. A variable name fails 2. A kind without its index fails
canonicality's inverse — it maps two devices to one key.

## Five identities today, none agreeing

The compiler derives "which topology is this" independently at five consumers, with
different fidelity each time:

| consumer | key used | index fidelity |
|---|---|---|
| type equality — `PartialEq` (src/syntax/types.rs) | kind + index **value** for literals; structure (name) for variables | full, for literals |
| monomorph mangle — `instantiate_function` (src/hir/env.rs, `${:?}", top.kind()`) | `TopologyKind` only | **dropped** |
| symbol dedup — codegen phase, first-wins by name (src/pipeline.rs) | the mangled name | inherits the mangle |
| dispatch id — `topology_dispatch_id` (src/arch.rs:319) | kind + index, via `topology_index` = `const_topology_index(e).unwrap_or(0)` | silent device-0 fallback (#345) |
| descriptor map (src/arch.rs:68) and plugin registry (`TopologyID = u32`, src/plugin/hardware_trait.rs:31) | `TopologyKind` / the dispatch id | dropped / inherits the fallback |

A fix at one consumer does not propagate: #355 repaired the first row and left the
other four, and the second row promptly bit.

## The failure modes, most severe first

### H1 — REPRODUCED: the mangle drops the device, and dedup collapses devices

```vx
fn on_dev<D: Topology>(x: Pinned<i32, Topology::D>) -> i32 {
  spawn on(Topology::D) { let y = x; }
  return 0;
}
// main: a placed on GPU[0], b placed on GPU[1]
let r1 = on_dev(a);
let r2 = on_dev(b);
```

Both instantiations mangle to `on_dev$GPU`, the name-level first-wins dedup keeps one
body, and the emitted module contains ONE `func.func @on_dev$GPU` whose spawn is
`topology(500)` — device 0. Both call sites call it. **The GPU[1] kernel runs on
device 0, silently and deterministically.**

Three notes on this one:

- It is #302's symptom ("every shard ran on device 0") resurrected one layer above the
  fix. #302 repaired index substitution *into* the instantiated body; the kind-only
  mangle then merges the two correctly-substituted bodies back into one.
- **#355's fix is what made it reachable.** Before it, a placed value could not be
  passed to any annotated function, so two same-kind different-index instantiations
  were unconstructible. Fixing one identity consumer exposed the next one down — which
  is the signature of scattered identity, not of a bad fix.
- Determinism makes it *worse* to diagnose: module-order first-wins means it always
  reproduces and always looks like the program's bug, not the compiler's.

### H2 — name-based variable equality is context-dependent

`GPU[i] == GPU[i]` compares the *names*. Two workers checking unrelated functions each
have their own `i`; any session-global structure that compares or keys on a
variable-indexed topology unifies unrelated binders (false sharing) or, with `i` and
`j` holding the same value, splits equal ones. The unsoundness direction — same name,
different binder, compares equal — is latent today only because comparisons happen
within one substitution context; a parallel cache is exactly what would break that
assumption.

### H3 — the Hash pressure, and why the absence of `Hash` is load-bearing

The open multi-device work (#331 dispatch must name a device, #346 the CUDA plugin
keeps one device's worth of state, #348 location transparency) all want per-device
state, and the natural spelling is `HashMap<Topology, DeviceState>` behind a lock or
sharded per worker. Rust's contract is `a == b ⟹ hash(a) == hash(b)`; a derived
`Hash` walks the index AST (`ty`, `span`), so two `==`-equal keys land in different
buckets. Lookups miss; double-inserts of "the same" device succeed; **which entry a
lookup finds depends on which worker's spelling minted the key first** — a
thread-interleaving heisenbug in exactly the code that is hardest to reproduce.
Today the compiler refuses `HashMap<Topology, _>` because `Topology` implements
neither `Hash` nor `Eq`. That refusal is a feature. The fix is not to implement
`Hash` carefully; it is to give the map a different key (below).

### H4 — per-kind maps conflate devices

`descriptors: HashMap<TopologyKind, TopologyDescriptor>` is correct while descriptors
are genuinely per-kind. The moment anything per-*device* lands in a kind-keyed map (a
cuBLAS handle, an allocation table, a stream), every GPU shares one slot. #346 is this
exact bug already filed against the runtime plugin.

## Solution space

### S1 — a single resolved `DeviceId`, minted once (recommended core)

```rust
struct DeviceId { kind: TopologyKind, index: IndexTerm }
enum IndexTerm { Const(i64), Var(BinderId) }   // derive Eq + Hash + Ord
```

- **Minted in sema, once**, where the const-eval environment exists: literals and
  const-generic indices fold to `Const`; a bound topology parameter becomes
  `Var(binder id)` — identity by *binding site*, not by name, which kills H2 in both
  directions. A genuinely runtime index gets **no id**: constructing one is a
  diagnostic, never a fallback, which retires #345's silent `unwrap_or(0)` as a side
  effect.
- **Attached to the checked AST** the way struct GIDs already are (#199: resolve once,
  carry the id, downstream consumers never re-derive the name). That pattern is this
  tree's own precedent for exactly this problem.
- **Content-derived, not interned.** An interning arena hands out ids in insertion
  order, and insertion order is scheduling — the id itself would become
  nondeterministic across thread counts. `(kind, index)` needs no arena, the same way
  GIDs are module-hash-derived rather than sequence numbers.
- **Consumers migrate to it**, deleting a bespoke derivation each time: the mangle
  (`on_dev$GPU_0` / `on_dev$GPU_1` — closes H1), the dedup key (inherits the fix),
  `topology_dispatch_id` (derive the i32 *from* DeviceId, keep the banded scheme and
  E6016's collision check), the descriptor/plugin maps where per-device data appears
  (closes H4), and any future `HashMap<DeviceId, DeviceState>` (closes H3 lawfully —
  the derives are correct because the type contains no AST).

### S2 — normalize the index AST at construction

Fold every index literal to a canonical form (value kept; `ty`/`span` excluded from
identity) behind smart constructors. Cheap, and it restores lawful derives for the
literal case — but Rust enum variants cannot be private, so the funnel is convention
only, and convention is precisely what produced the two-spellings problem (`ty: None`
in one file, `Some(I32)` in another). Useful hygiene *under* S1; insufficient alone,
and it does nothing for H2.

### S3 — context-carrying equality plus the prover tier

For the cases S1 refuses: `GPU[i]` vs `GPU[j]` under `assert(i == j)` needs the
const-evaluator or z3 (already wired for seam obligations; per-check cost is tracked
as M1). Two hard constraints shape this:

- It **cannot be `PartialEq`** — `eq(&self, &other)` has no slot for an environment.
  It must be a checker method (`topo_eq(a, b)` resolving both sides to `IndexTerm`s,
  then Const-by-value, Var-by-binder, else an obligation).
- It must **never be a container key**. Proof-dependent identity in a map is a
  contradiction; keys stay `DeviceId`, and the prover runs only at seam/spawn checks.

This is the design the monad doc already names: "topology identity =
(registered-kind, index-term), with index equality decided by the const-evaluator /
prover". S1 is that sentence's data structure; S3 is its decision procedure.

### S4 — serialize the topology-touching phases

Rejected. The architecture's stated point is per-compilation state with no global
registry "so the parallel pipeline needs no lock"
(`concurrent_compilations_have_isolated_topologies` pins it). Locking identity back
into a shared mutable table reintroduces the coupling the pipeline was built to
remove, and does nothing for determinism of emitted names.

## Phasing

1. **Mangle carries the index** (`$GPU_0`), plus a two-device generic fixture asserting
   two symbols and two distinct spawn ids — H1 is a reproduced wrong-device miscompile
   and should not wait for the full design. Existing `$GPU` symbol names in test
   expectations churn; that churn is the fix working.
1. **`DeviceId` minted in sema**, dispatch id derived from it (fixes #345's silent
   fallback), descriptor/plugin migration where per-device data appears.
1. **`Var(BinderId)`** for topology parameters — retires name-based cross-scope
   equality, the H2 unsoundness.
1. **Prover tier** when the multi-device campaign (#331/#345/#348) actually needs
   dependent-index equality, not before.

Guardrails to carry forward: `Topology` stays `Hash`-less and `Eq`-less on purpose
(the doc comment on its `PartialEq` says so; `DeviceId` is the hashable identity);
reflexivity-style unit tests for `DeviceId` mirroring
`every_topology_variant_is_equal_to_itself`; the two-device fixture joins the
byte-identical-MLIR corpus so schedule-dependence of identity can never return
silently.

## The symbolic tier: const-compared without const-evaluated

**Status: PARKED design note.** None of this is scheduled work. The minimal core that
fixes everything actually reproduced is three small changes — the mangle carries the
const index, the dispatch fallback becomes a diagnostic and consults the const
evaluator (so `let i = 1` places on device 1), and tests pin both — which is phases 1–2
above. The tiers below exist so that when multi-device work (#331/#345/#348) needs
symbolic indices, the design questions are already answered and red-teamed, not so
that anyone builds them now.

*(Added after review: what happens to `GPU[i]` vs `GPU[i]`, and `GPU[i]` vs `GPU[j]`
under `i == j` — the indices no const-evaluator can fold, whose COMPARISON is still
decidable. Grounded by six probes against the built compiler and an adversarial pass
over the design; the attack names below refer to that review.)*

### What the compiler does today, probed

The current state is worse than "unequal": it is wrong in both directions at once.

- `let i = 1; spawn on(Topology::GPU[i])` dispatches to **device 0**. Not "conflated
  with GPU[j]" — *misplaced*: the programmer named device 1 and got device 0, with only
  W1030 to say so. `const_topology_index` folds literal arithmetic only and never
  consults the checker's own const-evaluator, so even a comptime-known `let` is
  "runtime" (P1/P4). #345 is filed against runtime indices; the misplacement covers
  comptime-known ones too.
- `for i in 0..2 { spawn on(Topology::GPU[i]) }` emits ONE spawn op with the static id
  500 — the fleet fan-out idiom is unwritable today, because `vx.spawn topology(N)` is
  a static attribute with no runtime device operand (P3).
- `GPU[i]` is not even **reflexive**: ascribing `Pinned<i32, Topology::GPU[i]>` to a
  value spawned on `Topology::GPU[i]` — same variable, same scope — is a type error,
  because the non-literal fallback compares the index as an AST node and the two
  occurrences differ in *span* (P6d/P6f: the E3003 payload shows two `IdentifierExpr`
  differing only in line number).
- Meanwhile the comptime evaluator can already DECIDE `i == j`: a false
  `assert(i == j)` over comptime-known values fails at compile time as E8002 (P6a).
  The evaluator, the dispatch id, and type equality are three machines that never
  consult each other.

So the user-visible answer today is: `GPU[i]` equals nothing, including itself, and
runs on device 0 regardless of `i`. Every piece of the fix below replaces a probed
wrongness, not a hypothetical.

### The tier ladder

`IndexTerm` grows one inhabitant beyond the original S1 sketch, and equality becomes
three-valued (`Same | Different | Unknown`) with a uniform safe default: **Unknown
degrades to "not the same device" at every consumer** — a spurious explicit transfer or
a rejection at a safety check, duplicated code at dedup, a legal self-copy at the
no-op-transfer elision. Unknown never degrades to Same; that is the direction that
runs a kernel on the wrong device.

```
IndexTerm ::= Const(i64)      -- folded by the const-evaluator (which must consult
                                 eval_env, fixing P1/P4's misplacement en route)
            | Var(BinderId)   -- an in-scope immutable binding, identity by BINDING
                                 SITE, valid only within the binder's region
            | SigVar(k)       -- in signatures only: "the k-th value parameter of
                                 this function", de Bruijn-style, alpha-invariant
            | (none)          -- everything else: no identity, and constructing one
                                 is a DIAGNOSTIC, never a fallback (retires W1030's
                                 warn-and-use-0 in the same change; attack L)
```

**T2 — `GPU[i]` vs `GPU[i]`, same binder: Same, by construction, free.** No prover, no
values: two mentions of one immutable binding denote one device by the meaning of
"binding". Three conditions make it sound, each earned by an attack:

- *Immutable* means never reassigned AND never mutably borrowed (attack A) — a `let mut` index gets no `Var` at all. Cheap and sound; SSA versioning is the general fix
  if it ever pinches.
- *Within the binder's region* (attack B, the rank-1 unsoundness): a loop induction
  binder is a fresh region per iteration, so a `Pinned<_, GPU[Var(i)]>` escaping the
  loop (pushed into an outer Vec, returned) would let iteration 0's handle and
  iteration 1's handle type-check as one device. A type mentioning `Var(b)` may not
  flow out of `b`'s region; at the escape point it degrades to the no-identity state.
  Control-flow joins degrade the same way (attack N): `if c { x } else { y }` with
  different device ids yields a value with no identity, diagnosed at the first use
  that *demands* one, not at the join.
- Capture is fine as-is (attack D): a spawn body capturing an immutable `i` reaches
  the original binding site; immutability is exactly what makes deferred execution
  irrelevant.

**T3 — `GPU[i]` vs `GPU[j]`: Different-unless-shown, with three rungs.**

- *T3a, copy propagation at mint:* `let j = i;` resolves `j`'s index uses to `i`'s
  binder — but only through chains whose every source is itself a valid IndexTerm
  (attack K: `let j = i; i = 1; let k = i;` must NOT unify j and k, so propagation
  stops at the first mutable source and mints the copy's own binder).
- *T3b, entry-dominating asserts, canonicalized before minting:* `assert(i == j)`
  merges the two binders' classes in a union-find, and both mint the class
  representative. Soundness needs dominance the existing prescan cannot supply —
  `collect_assert_contracts` (transfer.rs:73) deliberately flattens both if-arms and
  loop bodies into one map, which is correct for consumer *obligations* and unsound
  for assumed *facts* (attack F). The identity collector is therefore its own pass:
  straight-line function-entry asserts only, facts in program order in a Vec (never
  driven off HashMap iteration), representative = Ord-minimum BinderId of the class so
  the outcome is a function of the class and not of union history (attack Q).
- *T3c, path facts at the comparison site:* inside `if i == j { ... }` the checker's
  `topo_eq` may consult the const-evaluator or z3 (the seam Solver, same M1 cost
  accounting, same `--verify-seams`-style gating) with the path condition. A path fact
  never rewrites a minted id — ids outlive branches; only the local verdict changes.

**The load-bearing assert (attack J).** Vx asserts are comptime-checked when foldable
(E8002) and otherwise emit **no runtime check** (`flatten.rs:2403` says so in as many
words). An assert whose fact T3b consumed is different in kind: compile-time identity
was built on it, so if it is false at runtime two "same-device" values live on
different silicon and no transfer was inserted — a miscompile with a signed
confession. Rule: an identity-consumed assert is load-bearing — it lowers to a runtime
trap (a scalar compare, noise next to a device launch), is never strippable, and its
standing is recorded as *asserted-not-proven*, the exact epistemics `raw::` bounds
(E6018) and unverified `spec:` figures already carry.

**Interprocedural: `SigVar`, and the one-body monomorph (attacks E, I).** A free
`Var(BinderId)` is meaningless at a call boundary — the binder is not in the caller's
scope. Signatures close over their indices positionally:
`fn f(i: i64, x: Pinned<T, GPU[i]>, y: Pinned<T, GPU[i]>)` types both tensors as
`GPU[SigVar(0)]` — alpha-invariant, context-free, and *inside* `f` the two are Same by
T2. At the call site the checker substitutes the argument's IndexTerm and decides
there. The payoff lands at monomorphization: `on_dev<D>` instantiated with
`D := GPU[SigVar(0)]` from two different callers is ONE index-polymorphic body whose
device index is a hidden runtime parameter — which requires the same IR change the
probes showed is missing anyway (`vx.spawn` has no runtime device operand, P2/P3), and
which is the same operand #331's dynamic dispatch needs. Three campaigns converge on
one op change.

**BinderId itself (attacks G, H).** Content-derived or nothing: a global counter is
scheduling-visible state and breaks byte-identical-across-thread-counts the moment any
artifact renders an id. The tree already owns the right shape:
`DefPath::Anonymous { parent_hash, structural_hash }` (hash.rs) identifies closures "by
structural layout or position within a specific parent item, rather than file line
number". `BinderId = (owner function/instantiation GID, ordinal of the binding site in a canonical pre-order walk)` — deterministic at any thread count, totally ordered, and
the embedded owner makes cross-function fact leakage a debug-assertable bug rather
than a silent one (identity fact tables live and die inside one `check_function`, like
`memory_placements` already does).

### The two questions, answered

**`GPU[i]` vs `GPU[i]`.** Same immutable binder: **Same**, decided structurally, no
prover, no value needed — this is precisely "const-comparable without
const-evaluable", and it is the common case (a function indexing all its work off one
device parameter). Different binders that happen to spell `i` the same way: Different,
which is the safe and correct reading. Today's answer, for calibration: *never equal,
not even to itself, and it runs on device 0.*

**`GPU[i]` vs `GPU[j]` where `i == j`.** Depends on where the fact lives, in
decreasing order of comfort: `let j = i` through an immutable chain — Same at mint,
free. A function-entry `assert(i == j)` — Same via T3b, with the assert now
load-bearing and trapped at runtime. A branch condition — Same inside the branch via
T3c, ids untouched. True at runtime but never stated — **Different**, on purpose, with
an actionable diagnostic ("cannot prove these name the same device; `assert(i == j)`
before this use if they do"), and the cost of the programmer declining is an explicit
transfer or a legal self-copy — never a wrong device. The compiler must not divine
value equality it cannot see; it must make stating it cheap, checked, and recorded.
