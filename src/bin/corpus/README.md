# The parallel-frontend corpus generator (#296)

Shared by `parallel_demo` and `intern_bench` via `#[path = "corpus/mod.rs"]`, so a demo and a
benchmark cannot quietly measure different programs.

```
cargo run --release --bin intern_bench -- \
    --modules 8,64,512 --fns 16,128 --density 0,1 --threads 1,2,4,8 --reps 10
```

Set `VX_PIPELINE_QUIET=1` for any timed run. See "Why quiet matters" below — it is not cosmetic.

## Knobs

| flag | meaning |
|---|---|
| `--modules N` | modules (plan sweeps 8 / 64 / 512) |
| `--fns M` | functions per module (plan sweeps 16 / 128) |
| `--density D` | fraction of parameter slots holding a generic instantiation; `0` = plain, `1` = instantiation-dense |
| `--arity A` | type parameters per carrier; key space is `vocabulary^A` |
| `--shared-frac S` | fraction of generic arguments drawn from scalars rather than module-local structs |
| `--params-per-fn P` | parameter slots per function (constant across density settings) |
| `--locals-per-module L` | module-local structs: the local half of the argument vocabulary |
| `--seed`, `--out` | reproducibility |

`intern_bench` accepts comma-separated lists for `--modules`, `--fns`, `--density` and `--threads`
and sweeps the grid. It also takes `--emit`:

| flag | meaning |
|---|---|
| `--emit mlir` (default) | compile all the way to MLIR text (`compile_pipeline_mlir`, #311) |
| `--emit none` | stop after the SIMD patch, the old frontend-only measurement |

Default `mlir`, because a run that stops at the SIMD patch times a *frontend*, and its number cannot
be quoted as a compile-time speedup however careful the rest of the harness is. Every cell reports
how many bytes of MLIR it produced; `0 bytes` plus a `NOTE` means the flat emitter declined and the
cell is not a compile measurement. Each phase's share, codegen included, is on the same line — on a
small corpus codegen is already about half of measured phase time, which is the ratio a frontend-only
sweep silently assumed away.

## The carrier is pointer-backed on purpose

A generic carrier is emitted as `struct G<T0> { f0: *mut T0 }`, never `{ f0: T0 }`. `lowered_ty`
resolves a generic instance through its base layout, which is instance-independent only when every
type parameter sits behind a pointer — a by-value `f0: T0` leaves the base a 0/0 stub and the whole
function drops out of the flat subset. With a by-value carrier the density-1 arm compiles its
frontend and then emits nothing, so the arm that exists to *show* interning pressure would be the one
arm that never reaches codegen.

Nothing the density knob controls changes: a parameter slot still holds either a settled nominal GID
or an interned instantiation GID, and the key space is still `vocabulary^arity`. What a carrier's
field looks like never reaches the interner.

## The design point

**Density changes the type in a parameter slot, never the number of slots.** A plain slot takes a
module-local struct (settled GID); a dense slot takes `G<...>` (interned GID). So the two variants
have matched function counts, matched signature widths, and matched type-stream lengths — the only
difference is whether a slot's GID needs interning. Without that control, a density comparison also
varies how much work the type-checker does, and the interning effect cannot be separated from it.

`--shared-frac` is the other control that matters. Scalars have module-independent GIDs, so a shared
argument is a key two threads can collide on; a module-local struct is a key only one thread ever
mints. That is the difference between measuring lock *contention* and lock *overhead*.

## What actually creates interning pressure

A generic GID is minted in exactly one place: `emit_type_gid`, reached from
`emit_function_type_gids`, which walks a function's **parameter and return types**.

1. **Instantiations in function bodies mint nothing.** `let x = wrap<i32>(1);` is checked and
   monomorphized, but monomorphized functions are appended to modules in
   `codegen_and_metadata_phase` — after the type stream is extracted, and after
   `compile_pipeline_type_stream` has returned. A corpus with its generics in bodies exercises the
   type-checker and leaves the interner idle.
1. **The key is the argument list, not the instantiation.** `intern_generic` and
   `deduplication_phase` both key on `Vec<TypeId>` of the arguments; the base lives in words 0–1. So
   `Pair<i32>` and `Box<i32>` share one arena entry, and distinct keys come from distinct *argument
   lists*. Hence `--arity`.
1. **Every argument counts, now.** `nominal_gid` used to answer `None` for a bare type parameter or
   a nested `GenericInstance` and the call site dropped those silently, so a nested argument interned
   the empty key and measured nothing. That was also a correctness bug — `Foo<Bar<i32>>` and
   `Foo<Bar<f64>>` collided on one arena entry — and it is fixed (#305/#309, `5d8116e6`):
   `nominal_gid` is total, and an argument list that cannot be resolved fails rather than shrinking.
   The generator still emits non-nested concrete arguments, because that keeps the key count
   predictable, but it is no longer forced to.

Every run prints the **actual** generic-slot and distinct-key counts, and `intern_bench` cross-checks
the predicted key count against what the interner really ends up holding, printing a NOTE if the
two disagree. A corpus that claims pressure it does not produce is the failure mode this generator
exists to eliminate.

## Why quiet matters

`parse_phase` opens its per-module closure with `println!("Parsing file: ...")` — inside a rayon
parallel-for. `println!` takes the global stdout mutex, so every worker serialises on it once per
module: 512 lock acquisitions per rep at the plan's largest cell, inside the region whose scaling is
being measured. `perf lock` on such a run reports contention on stdout, not on the interner.

`VX_PIPELINE_QUIET=1` suppresses the progress chatter (diagnostics still print, so a corpus that
stops compiling stays visible). The gate is eval-branch scaffolding; the underlying issue — a
`println!` per module inside the parallel front-end — is not eval-only and is filed as #306.

## Reproducibility

A corpus is written to a directory named by a digest of its parameters **and** `GENERATOR_VERSION`,
with a `manifest.json`, and is reused when the id matches. Bump `GENERATOR_VERSION` whenever the
emitted text changes for the same parameters: without it, changing the generator leaves every cached
corpus looking current and the next run measures the old programs while reporting the new flags.
That is not hypothetical — it happened during development, and the version constant is the fix.

Module `m`'s text depends only on `(seed, m)` and the shape parameters, so module 3 is identical
whether the corpus has 8 modules or 512. Growing `N` extends the corpus rather than reshuffling it,
which is what makes an N-sweep vary only N.

## Measurements: previously published numbers are void

Earlier revisions of this file carried a table of scaling results for this corpus. **Those numbers
are withdrawn.** They were taken against a frontend that built a `TransferCostGraph` per *function*,
running an all-pairs shortest-path precompute twice per function — which a profile later put at the
top of the whole compiler by self time. Removing it (`d300b6ee`) cut wall clock by 2.35x at one
thread, so every ratio measured before it was a ratio over work that no longer exists.

The correction is not only that the numbers were too slow. Deleting that work made apparent parallel
scaling *worse* — 2.9x to 2.67x across 8 threads — because the deleted work sat inside the parallel
region and scaled well. **A scaling curve measured over avoidable work overstates how parallel a
compiler is**, which is the methodological point worth keeping from the episode.

To regenerate, on hardware where the result would mean something:

```
VX_PIPELINE_QUIET=1 cargo run --release --bin intern_bench -- \
    --modules 64 --fns 16 --density 0,1 --threads 1,2,4,8 --reps 7
```

Two conditions before any of it is quotable:

- **Homogeneous cores.** The development machine is an Apple M4 — 4 performance cores and 6
  efficiency cores — so at 8 threads half the workers sit on much slower cores, and 4→8 threads
  gaining ~31% is what adding four weak cores looks like, not what scaling looks like. Ratios
  between modes on one machine survive this; absolute scaling curves do not.
- **A corpus above the measurement floor.** After the frontend fixes, a profile of this generator at
  6,000 modules is dominated by corpus file I/O, idle worker threads and process startup rather than
  by compilation. Resolving further improvements needs more work *per file*, not more files.

## Where the ceiling is

Also to be re-established. The serial-fraction and Amdahl figures previously quoted here were
computed from the same pre-fix phase timings and are withdrawn with the rest.

What is durable: the phases are instrumented (`crate::intern_mode::phases_csv`, #297), so the
accounting is a measurement rather than an estimate, and the phases that do not parallelise —
registry freeze, env build, the reconciliation barrier — are the ones to account for.
