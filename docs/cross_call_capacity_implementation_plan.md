# Cross-call capacity: a whole-program fold that keeps the frontend parallel

How the compiler budgets memory across function boundaries, and why a whole-program analysis
costs the parallel frontend nothing. This closes the admission gap of Vx#444 and doubles as the
existence proof for a more general claim: a frontend can do cross-module reasoning without
serializing per-function work, by exporting per-function summaries and folding them after the
parallel phase.

## The gap

The per-function working-set check (`check_cumulative_capacity`,
`docs/working_set_peak_implementation_plan.md`) budgets each function against the full capacity
of every space it touches. A tile the caller still holds while a callee runs is invisible to the
callee's budget:

```
fn g() -> i32 {                     // peak 3 MiB: fits a 4 MiB space
  let y  = Tensor<f32, [1024, 768]>::uninit();
  let _s = transfer(y, Memory::W);
  return 1;
}
fn f() -> i32 {                     // peak 3 MiB: fits
  let x  = Tensor<f32, [1024, 768]>::uninit();
  let sx = transfer(x, Memory::W);
  let r  = g();                     // sx is read below, so it is resident here
  let _b = transfer(sx, Memory::CPU_DRAM);
  return r;
}
```

The true peak in `W` is 6 MiB against 4. Written as one function, the same bytes are refused
(E6010); split across a call, they were admitted. Recursion is the unbounded form of the same
gap: one tile per activation, depth known only at run time.

## The quantity

The peak over the call tree: the maximum over call paths of what is resident along the path.
Sequential calls compose by **max** (a returned callee's tiles are gone before the next call's
arrive); a tile held across a call composes by **+**. Summing every function's working set would
refuse `let a = f(); let b = g();` — a false refusal, since those tiles never coexist. Computed
as a frame rule, per space `s`:

```
total(f, s) = max( self_peak(f, s),
                   max over call sites c in f:  live(f, c, s) + total(callee(c), s) )
```

This is the recurrence of worst-case stack-usage analysis (GCC `-fstack-usage` + link-time fold,
ELF `.stack_sizes`, SPARK stack contracts), applied per memory space instead of to one stack.

## The architecture: summaries in the parallel phase, one fold after it

**Phase 1 — per function, parallel, unchanged in shape.** Each function's check already walks
its placements to compute the peak. It now also exports a summary
(`FnCapacitySummary`, `src/hir/check/capacity_fold.rs`):

- `self_peak: space -> bytes` — the peak the walk already computed, kept instead of dropped;
- `calls: [(callee, live: space -> bytes, span)]` — one edge per resolved call site, where
  `live` is every tile placed earlier in program order whose block is still open at the call.
  The same prefix-of-scope-chain test the peak walk uses, evaluated at the call.

Call edges are recorded where the checker resolves calls (`record_call_edge`,
`src/hir/check/transfer.rs`): direct calls, monomorphized generics (the edge names the
instance), static and instance methods through their rewrites, module functions, closures
resolved to their `_call` instances, and registry-imported callees. Recording mirrors
`check_capacity`'s discipline — span-keyed dedup, no speculation guard — so a node the checker
visits twice lands on one record.

**Phase 2 — whole-program, once, after every body.** `fold_cross_call_capacity` builds the call
graph from the summaries, collapses strongly-connected components (iterative Tarjan, emitted in
reverse topological order), and evaluates the recurrence bottom-up. Overflows are reported at
the smallest function that exhibits them; callers whose peak path runs through an
already-reported node stay silent, so one overflow is one diagnostic. The refusal names the
path and each hop's contribution:

```
E6027: the working set along call path 'f -> g' in memory space 'W' peaks at 6291456 bytes,
over its 4194304 byte capacity: 'f' holds 3145728 bytes across its call, 'g' itself peaks
at 3145728 bytes
```

Both frontends run the same fold: the sequential driver after its check loops
(`src/driver.rs`), the parallel pipeline after `type_check_phase` (`src/pipeline.rs`), reading
the summaries each rayon worker exported in its `FunctionCheck`.

## Why this costs the parallel frontend nothing

The claim this design demonstrates: cross-module analysis at the frontend is compatible with an
embarrassingly parallel per-function phase, provided the whole-program step reads only what the
parallel phase exports.

1. **The fold reads summaries only.** No function body, AST, or type stream is touched after
   the parallel phase. The fold is O(functions + call edges) integer arithmetic over a few
   dozen bytes per function — unmeasurable against a parse.
1. **The parallel phase is unchanged in shape.** Summaries are computed inside the walk the
   per-function check already does, on the same rayon fan-out, with no shared state: each
   worker owns its checker, and the summaries merge by collection into the already-collected
   `FunctionCheck` results.
1. **It only adds refusals.** Nothing the per-function checks admitted is re-derived or
   re-admitted, so per-function results remain sound in isolation and the fold composes with
   any subset of them.
1. **No compile-to-compile dependency.** Under separate compilation
   (`docs/cross_module_compilation_plan.md`), each module compiles in isolation and exports its
   summary table; the fold runs once at link time over the union. Module A's compile never
   reads module B. The precedent is exactly `.su` files / `.stack_sizes` sections: per-unit
   summaries in the artifact, one cheap link-time pass.
1. **Incrementality survives.** Editing a body invalidates one summary; re-running the fold is
   the only whole-program cost, and it is microseconds.

A worthwhile refinement when modules land: print each function's summary into the emitted MLIR
as a `vx.max_memory`-style attribute, making the artifact self-describing the way GPU kernel
descriptors carry their SMEM usage.

## Edge semantics: the operator belongs to the edge

Call edges compose sequentially (max across a function's calls, + for residency held across
one). `spawn on(...)` is today synchronous — the lowering splits the parent block and inlines
the region in place (`src/dialect/VxLowering.cpp`), so the continuation can never outrun the
region. A spawn is therefore scope structure inside one function's summary rather than a
call-graph edge, and the max rule holds with the join at region exit. If an asynchronous spawn ever lands,
it becomes a second edge kind whose operator is **+ across concurrent siblings** for as long as
the task may be outstanding — the fold's structure anticipates that; the operator is read off
each edge. (The same +-versus-max split as the transfer cost law:
store-and-forward sums, streamed takes the max.)

## Recursion

A cycle in the call graph has no compile-time depth, so any residency per activation makes the
true peak unbounded. The policy, per space with a declared capacity:

- A cycle placing **nothing** in the space folds through silently and still conducts: what it
  reaches outside itself joins what its callers hold. `fact` needs no annotation.
- A cycle placing **anything** is refused conservatively (E6028), naming the cycle and the
  per-activation bytes. `overcommit` downgrades to W1028, as everywhere.

The principled lift is the planned `requires{...}` capacity contract (Vx#444): a recursive
group carrying a declared bound checks as `own footprint + recursive requires <= declared requires`, forcing constant-footprint recursion or a depth-parameterized bound (the SPARK
precedent). An annotated function is then a **cut-point**: the fold checks its body against the
contract once and callers compose against the declaration, which buys incrementality and
per-function verification conditions a proof assistant can discharge. `requires` already exists
on functions as a prover constraint; the capacity atom extends it.

## A prerequisite fix: summary isolation for nested checks

A generic instantiation is type-checked from inside its caller's body (`check_function` is
re-entered). The placement map was shared, so the caller's tiles placed before the call landed
in the callee's cumulative check — mis-attributed to the callee's resident set and lost to the
caller's. `check_function` now swaps the per-function traffic state (placements, call sites,
scope chain) out on entry and back on exit, the same isolation the borrow context already had.
Every function's summary now describes that function alone, on both frontends.

## Capacity omission stays "unconstrained"

A space that omits `capacity:` is unconstrained and takes no part in the fold; a space that
declares one is folded. Silence means the file does not know — honest for a pageable host,
opt-in for a cgroup-limited pod that does know its wall. This bounds the fold's work by
construction: on the current fleet only device spaces are folded.

## Limitations, stated

- **Function values.** A call through a variable holding a function
  (`let h = f; h();`) resolves no callee and records no edge. Closures called through their
  struct do resolve (the checker rewrites them to their `_call` instance). The honest treatment
  for true function pointers is a conservative refusal when any address-taken function has
  residency; deferred until the corpus has such programs.
- **Span-keyed dedup.** Two synthesized calls to the same callee carrying no source position
  collapse to one edge, keeping the first visit's program order. Live bytes at the kept site
  may understate the dropped one's.
- **Dynamic shapes.** A placement whose shape is not a compile-time literal records no bytes
  (W1029 already reports this); its residency is invisible to both the per-function check and
  the fold.
- **Liveness fidelity.** `live` at a call inherits the checker's scope model — release at end
  of releasing block, never-read tiles freed at the placement. The known gaps in the use-walker
  (Vx#447) bound its precision.

## Tests

- `tests/frontend/fail/callee_tiles_join_the_callers_working_set.vx` — the motivating admit,
  refused with the full path (E6027).
- `tests/frontend/pass/sequential_calls_do_not_sum.vx` — max composition, admitted.
- `tests/frontend/pass/a_closed_block_releases_before_the_call.vx` — live-at-call honors block
  release.
- `tests/frontend/fail/generic_callee_joins_the_working_set.vx` — the fold over the
  monomorphized graph; the path names the instance.
- `tests/frontend/fail/recursion_with_residency_is_refused.vx` /
  `tests/frontend/pass/recursion_without_residency_is_admitted.vx` — the recursion policy,
  both directions.
- `tests/integration_test/cross_call_capacity_test.rs` — the same refusal and admission
  through the parallel frontend's rayon fan-out.
- Unit tests in `src/hir/check/capacity_fold.rs` — max/+ composition, path naming,
  one-report-per-overflow, overcommit downgrade, cycle policy, mutual recursion,
  residency-free cycles conducting, unknown callees, unbudgeted spaces.

Verified by sabotage in both directions: dropping `live` re-admits the motivating program;
dropping the scope test refuses the correct one.
