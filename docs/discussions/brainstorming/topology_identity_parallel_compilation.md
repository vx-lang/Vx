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
2. **Context-free.** Meaningful without a scope or an eval environment. A worker that
   receives the key cannot ask the binder that produced it what it meant.
3. **Deterministic across schedules.** The same program at any thread count produces the
   same identity, byte for byte. The tree already tests this
   (`pipeline_emits_byte_identical_mlir_across_thread_counts`,
   `compile_pipeline_gid_stream_is_deterministic`).
4. **Totally ordered.** Anything emitted from a keyed collection needs a stable order
   (the `BTreeMap` discipline `region_traffic.rs` already follows).

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
2. **`DeviceId` minted in sema**, dispatch id derived from it (fixes #345's silent
   fallback), descriptor/plugin migration where per-device data appears.
3. **`Var(BinderId)`** for topology parameters — retires name-based cross-scope
   equality, the H2 unsoundness.
4. **Prover tier** when the multi-device campaign (#331/#345/#348) actually needs
   dependent-index equality, not before.

Guardrails to carry forward: `Topology` stays `Hash`-less and `Eq`-less on purpose
(the doc comment on its `PartialEq` says so; `DeviceId` is the hashable identity);
reflexivity-style unit tests for `DeviceId` mirroring
`every_topology_variant_is_equal_to_itself`; the two-device fixture joins the
byte-identical-MLIR corpus so schedule-dependence of identity can never return
silently.
