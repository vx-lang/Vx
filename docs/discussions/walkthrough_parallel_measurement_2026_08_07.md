# Walkthrough — Codegen on the parallel pipeline, and learning to measure it (2026-08-06 → 08-07)

A session record kept deliberately symmetric: the wrong turns are written down at the same weight as
the results, because three of the four most useful things learned here were failures, and two of them
were *published* before being caught.

Tracking: #300 (CGO'27). Issues opened this session: #311–#318. Closed: #304, #305, #308, #309,
#311, #312 (retracted), #314, #315, #316.

______________________________________________________________________

## Headline

The parallel frontend went from **"produces a GID stream and stops"** to **a complete compile whose
scaling is measured against a rayon-free baseline on 48 homogeneous cores, and whose race-freedom is
verified rather than asserted.**

| | before | after |
|---|---|---|
| artifact | none — no MLIR emitted | complete `module { … }` text |
| baseline | pipeline at 1 thread | the same compiler with rayon off the path |
| corpus | 64 mutually oblivious modules | 1,000 modules, 221k lines, 2,715 import edges |
| hardware | Apple M4 (4P + 6E) | bare-metal 48-core Xeon |
| race-freedom | a CI grep for lock keywords | ThreadSanitizer, with a positive control |
| speedup | unquotable | **5.89× at 48 threads**, knee at ~32 |

______________________________________________________________________

## Part 1 — The measurement failures

These are first because they are the transferable part. Every one produced a number that looked
publishable.

### 1.1 A false finding, reproducible across three runs, that I published

**Claim made:** "Deferred interning gains nothing from a second core (1.01×) while content-addressed
gains 1.6×." Filed as #312. Reported to the user. Quoted in a comment on #300.

**It was an artifact.** rayon fixes thread placement when a pool is built, and the development
machine is an Apple M4: 4 performance cores and 6 efficiency cores. A two-thread pool either lands on
two P-cores or it does not, and the difference is **1.7×** on that corpus. The harness ran each
interning mode's whole ladder end to end, so the two modes built their pools separately and drew
placement independently. Whichever mode drew the efficiency cores looked like it could not use a
second core.

**Why repetition did not catch it.** The bad draw is not noise — it is a *constant for the life of
the pool*. Medians over 15 reps did nothing to it. The cell came out with a 2% IQR: tight,
reproducible across reps, reproducible across whole runs, and completely false.

| process | t=1 | t=2 | t=4 |
|---|---|---|---|
| A | 44.12 | **30.13** | 20.06 |
| B | 45.40 | **50.03** | 21.24 |
| C | 46.03 | **50.66** | 20.23 |

Same binary, same corpus, same mode, three processes. That table is what finally showed it, and the
thing that prompted running it was a detail that never fitted the story: **the entire regression sat
in `codegen`, a phase that does not touch the interner**, while the barrier it was supposedly caused
by measured 0.3 ms against a 35 ms gap. The narrative was plausible enough ("deferred has to
reconcile") that it survived longer than the arithmetic should have allowed.

**Fixed structurally, not by care.** One pool per cell; every mode runs inside it, alternating rep by
rep. Whatever placement the pool drew, both modes drew it. With pairing the two modes agree to within
1% at every thread count, which is what they had been saying all along. Two guards added: a row
slower than a row with fewer threads is flagged `SUSPECT`, and `--modes` runs one mode per process so
*mode versus position in the run* is decidable rather than assumed.

**The lesson worth keeping:** a low variance is evidence about the measurement's precision and says
nothing about its accuracy. A nuisance variable that is constant within a run is invisible to every
statistic computed within that run.

### 1.2 Reading speedup against thread count instead of against the machine

"4 threads gave 2.6×" reads as *35% of the compiler fails to parallelise*. It is not, because 4× was
never available. A pure-compute parallel-for — no allocation, working set in registers — reaches
**3.47×** on 4 threads of the M4 and **3.98×** on the Xeon. Clock scaling alone costs 13% on the
first.

Without that control, hardware behaviour is silently charged to the compiler's serial fraction. The
harness now runs a calibration ladder before every sweep, so each number arrives with the machine's
own ceiling attached rather than depending on someone remembering to reason about it.

### 1.3 A phase table that hid its largest serial section inside a parallel phase

The total said 2.37× on 4 threads; Amdahl on that implies ~26% serial; the phases *known* to be
serial summed to 1.3%. The gap was not distributed — it was concentrated in two places the
instrumentation could not see:

- `codegen` was reported as one parallel-for. Splitting it showed a **serial prologue that did not
  move with thread count** (2.2 ms, `EmitCtx::from_registry` plus a walk over all 1,600 signatures)
  plus a serial reassembly, against an emit that scaled at 4.5×.
- ~4.5 ms sat in `unaccounted` and *grew with thread count*. A remainder that grows when you add
  workers is not rounding error. It was **teardown** — freeing the compile.

**Lesson:** an unattributed remainder is a place for serial work to hide from an Amdahl estimate, and
the phase whose share is largest deserves to be attributed at finer grain than its own name.

### 1.4 A fix measured where it could not possibly show

Parallelising the registry folds (`build_agg_map`, `build_callee_map`) bought **0.1 ms** on the
100-module corpus — noise. I was one sentence from reporting it as a win.

These folds are O(program size). A 100-module corpus is structurally incapable of telling you whether
they matter. Measured at 1,000 modules: **24.9 ms → 8.4 ms (2.96×)**. The commit message says both
numbers, because "we made it 3× faster" and "on a corpus a third the size it was unmeasurable" are
the same fact.

### 1.5 A hypothesis that was simply wrong

Reassembly was 111 ms of a 315 ms compile at 48 threads. I diagnosed remote frees — 16,000
per-function `String`s allocated on workers, dropped on one thread — which was a *good* guess,
because the same mechanism had just been confirmed for teardown (§2.5).

Par-dropping them moved it **110.9 → 108.7 ms**. Nothing.

The actual cause was that the module text was **copied three times** before anything could parse it:
bodies into `out`, then `globals + decls + out` into a result, then a `format!` at each call site to
wrap it in `module { … }`. 51 MB × 3, and the cost is not memcpy — it is faulting in 150 MB of fresh
pages. Fixing that: 111 → 56 ms.

**Lesson:** a mechanism that was true last time is a hypothesis, not a diagnosis. The measurement
that discriminated took two minutes and I should have run it before the change, not after.

### 1.6 A sanitizer run that would have reported a false clean

The first ThreadSanitizer run printed:

```
WARNING: ThreadSanitizer: memory layout is incompatible, possibly due to high-entropy ASLR
```

TSan maps shadow memory at fixed addresses and this kernel can place the program on top of it. It is
**intermittent** — same binary, same command, most runs fine. It surfaced only because I happened to
pass `verbosity=1` while checking something else. A clean report from a run that warned about its own
shadow mapping is not a clean report.

Compounding it: a linked sanitizer runtime is not a working one. Instrumentation can be dropped, a
flag mis-set, the shadow map wrong — and every one of those failure modes produces *zero races*,
which is exactly the answer being hoped for.

Both are now structural. The script builds a program whose only purpose is to race, with identical
flags, and **refuses to proceed unless TSan reports it**. The layout warning is fatal. The sweep runs
under `setarch -R`.

### 1.7 Smaller ones, recorded for completeness

- **Sloppy grep, self-caught.** Checking whether any TSan report implicated Vx code, I grepped the
  whole log and matched *test names* from the output stream rather than frames in the race blocks.
  Re-extracted only the `WARNING …/SUMMARY` blocks before concluding.
- **Instance sizing advice reversed.** I had recommended a 192-vCPU metal box. After the serial
  fraction was measured at ~10%, the Amdahl arithmetic said 48→192 cores buys 14% while halving the
  serial fraction raises the ceiling from 10× to 20×. Recommendation changed to 48 physical cores
  plus fixing the serial work. The confirmation ladder later showed the knee at ~32 cores, which
  vindicates the revision.

______________________________________________________________________

## Part 2 — What landed

### 2.1 #311 — codegen on the parallel pipeline

The pipeline ran the frontend and stopped. No artifact, so no speedup it produced could be called a
compile-time speedup, and there was nothing to check for correctness.

- `emit_module_mlir`'s per-function loop became a `par_iter`. The two module-wide counters that
  forced it sequential (`str_base` for string globals, `distinct_ctr` for alias scopes) are prefix
  sums, so the output is **byte-identical**, not merely equivalent.
- `compile_pipeline_mlir` composes the frontend with codegen. Monomorphs needed handling that was not
  in the issue's scope: they have no HIR, because they *are* the check phase's output.
- `FunctionCheck.lowered` was required for correctness, not tidiness. `lower_function_to_hir` is
  atomic, so a declined function leaves an **empty** HIR stream — indistinguishable at codegen from a
  function with nothing to do. Without the flag it would have emitted a silently empty `func.func`.

Two tests, because the claims are now about output: MLIR compared byte-for-byte across thread counts,
and a two-file program driven through the pipeline → parse → LLVM → JIT, checked against the AST
oracle.

### 2.2 A module could not be named by an `import`

`parse_phase` stored each module's full filesystem path as its `module_path`. `build_symbol_map` keys
the cross-module symbol table on exactly that string, so every module was filed under a name **no
`import` statement can spell**. A cross-module *type* reference through the pipeline could not
resolve, ever.

Verified rather than argued: reverting the one line makes the new test fail with two semantic errors.

The stem is not an invention — `ModuleLoader`, the loader `vxc` ships, already files an imported
module under its import path. The pipeline was the odd one out. Two files with the same stem are now
an error rather than a silent merge into one module hash.

### 2.3 A corpus whose modules depend on each other

Every corpus until now was N modules that had never heard of each other, which flatters a parallel
frontend: the phases that cannot parallelise are the ones that reconcile modules.

`--files-per-layer` arranges modules into a DAG. Each dependency emits **two** edges, both
load-bearing: an imported *type* (symbol map → `resolve_nominal` → frozen registry, the serial spine)
and a cross-module *call* (`GlobalAstEnv` must know the callee before any worker can check a caller).

The generic carrier had to become `*mut T`. `lowered_ty` resolves a generic instance through its base
layout, which is instance-independent only when every type parameter sits behind a pointer — so with
a by-value carrier the **density-1 arm compiled its frontend and generated nothing**. The one arm
that exists to show interning pressure was the one arm that never reached codegen.

Reference corpus: 1,000 modules, 221,145 lines, 16,000 functions, 2,715 import edges, ~1.5 s
sequential.

### 2.4 The sequential baseline is a flag, not a build

`Schedule::{Parallel, Sequential}`, chosen at run time. Sequential is **not** rayon-with-one-thread —
it is `iter()` where the parallel form is `par_iter()`, with rayon nowhere on the path, and the bench
runs that arm outside the thread pool.

Why it matters: with the 1-thread column as baseline, the parallel machinery's cost sits on both
sides of every ratio and cancels. The sweep could then only answer "do more threads help a design
that already pays for parallelism", never "is parallelising this worth it at all".

Measured answer: the parallel structure costs **1.00×** when unused. That is what licenses reading
the scaling column as speedup, and it had been assumed until it was measured.

A test asserts both schedules emit byte-identical MLIR — a baseline that is not the same compiler
measures something else.

### 2.5 The serial-fraction work

Each of these was invisible below ~16 threads and dominant at 48.

| what | before | after |
|---|---|---|
| `codegen:setup` — signature walk over every function | 2.2 ms flat (0.96×) | 2.26× |
| `teardown` — freeing a compile | **0.72×**, anti-scaling | ~1.5× |
| 10 × `out.contains()` over the whole 5 MB module | 50 MB of scanning, serial | per-function, parallel |
| module text copied three times | 51 MB × 3 | ×1, wrapper included |
| `sig_clone` — deep clone per module | 1.20× | **20.4×** |
| registry folds | serial | 2.96× (at 1,000 modules) |

**Teardown deserves its own note.** It was the only phase in the compiler that got *worse* with more
threads. ~16,000 `LocalWorkerState`s plus every module's AST are allocated across the workers and
freed on one — and a cross-thread free is the expensive path for essentially every allocator. Handing
the collections back to the pool with `into_par_iter().for_each(drop)` fixed the direction.

The finding under all of it: **the two largest serial costs in this frontend were freeing memory and
concatenating strings** — not the reconciliation barrier the design is named after, which measures
0.0 ms.

### 2.6 Correctness, before instrumenting

Cleared first so a sanitizer report would be news:

- **#305/#309** — the fix was in `nominal_gid`, but **the tests pinning it had been lost in a branch
  restructure**. For several commits the property was held up by nothing. Restored, asserting on the
  *argument list* — that list is literally the interner's key, so it is the equality that decides
  collisions before any mode's arena index or digest enters in. Writing them immediately caught that
  `nominal_gid` returns `None` for an unresolved nominal (correct and fail-closed, but it means a
  parse-only module tests the wrong thing).
- **#308 was half-fixed** — `verify_phase_3_isolation` still hand-decoded `words[2] & INDEX_MASK`.
- **#304 would have corrupted the sanitizer run itself** — two tests wrote to a fixed path inside the
  repo, and an instrumented build runs alongside a normal one.

Still open by choice: **#310**, type-parameter identity is name-based. It *over*-distinguishes, so it
cannot collide and cannot miscompile.

### 2.7 ThreadSanitizer

**Zero data races in Vx code.** Sweep of 300 modules at 1/8/32/48 threads, both interning modes; 430
library tests pass instrumented.

The test suite produced three reports in one run and none in the next. All inside
`crossbeam_epoch`'s reclamation and `crossbeam_deque::Stealer::steal`; Vx symbols appear only as the
frame that *enters* rayon, never in the racing access. crossbeam synchronises with `fence(SeqCst)`
against relaxed loads and TSan does not fully model `atomic_thread_fence`. Suppressed, scoped to
crossbeam — which cannot mask a compiler race, since Vx never uses crossbeam directly.

The uninstrumented libMLIR link is not a gap: `compile_pipeline_mlir` produces MLIR *text* in pure
Rust and never calls into MLIR, so no uninstrumented code runs on the measured path.

### 2.8 The measurement box

`utils/cgo/push.sh` carries the working tree over the EC2 keypair, so the instance never holds a git
credential — it is rented, shared-tenant hardware that gets terminated, and a key that has been on it
should be treated as disclosed. What crosses is exactly `git ls-files` minus `config.local`.

Four provisioning bugs that only a real box would surface: rustup installing into `/root` under
sudo (leaving the login user without cargo, and a root-owned `target/`); missing `clang-22` (the
driver, distinct from `libclang-22-dev`, and `build.rs` shells out to `clang++`); missing LLVM link
dependencies (`-lzstd` et al, named by nothing); and a thread ladder of powers of two that skipped
the machine's own core count.

______________________________________________________________________

## Part 3 — The numbers

Bare-metal Intel Xeon Platinum 8488C, 48 physical cores, 1 NUMA node, 188 GB. 1,000 modules /
221k lines / 16k functions, both modes, 9 reps. `utils/cgo/results/final-20260807T055232Z/`.

| threads | deferred (ms) | content (ms) | speedup | machine ceiling |
|---|---|---|---|---|
| seq (no rayon) | 1563.6 | 1512.0 | 1.00 | 1.00 |
| 1 | 1567.7 | 1510.2 | 1.00 | 0.99 |
| 2 | 908.9 | 885.9 | 1.71 | 1.96 |
| 4 | 645.3 | 620.5 | 2.43 | 3.98 |
| 8 | 428.3 | 419.5 | 3.63 | 7.92 |
| 16 | 320.3 | 312.0 | 4.86 | 15.16 |
| 32 | 268.7 | 261.7 | 5.80 | 27.10 |
| 48 | 266.2 | 256.0 | 5.89 | 34.21 |

Three readings:

1. **The knee is at ~32 cores.** 32→48 buys 1.5% against hardware that goes 27× → 34×. That is the
   ~10% serial fraction made visible, and it is the argument against reaching for a bigger machine:
   the ceiling is ours.
1. **The two interning modes are indistinguishable**, within 1% at every point. Content-addressed
   identity's case is the structural one #307 makes — no barrier, no patch, determinism across any
   scheduling or process boundary. **It never needed a speed claim and does not have one.** The
   earlier claim that it did was §1.1.
1. **The parallel phases are done.** `codegen:emit` reaches 39.6×, `parse` 29.6×, `type_check` 20.4×.
   What remains is serial: reassembly 56 ms, teardown 52 ms, unaccounted 36 ms, `registry_freeze`
   27 ms, `env_build` 20 ms.

______________________________________________________________________

## Part 4 — Where to pick up

**Open, ordered by size at 48 cores:**

- Reassembly (~56 ms) — one 51 MB copy remains, plus the declaration loop. Parallel copy into a
  preallocated buffer via prefix-summed offsets is the obvious next step.
- Teardown (~52 ms, #315) — par-drop fixed the *direction*; an arena would fix the cost, and would
  cut allocation during the compile too.
- `registry_freeze` (~27 ms) and `env_build` (~20 ms) — genuinely serial algorithms. Real work, not
  mechanical wins.
- #313 — adopt the pipeline in `vxc`. Deliberately postponed: keeping "does the frontend scale"
  separate from "does the shipping compiler get faster" keeps both answerable.
- #317 — run TSan on a schedule. A race-freedom claim decays as parallel phases are added.
- #318 — revisit the crossbeam suppression. A suppression nobody rechecks is where a real race hides.

**Two habits worth carrying forward, both of which exist because they failed here first:**

1. Pair the comparison inside the shared nuisance variable, and flag any row that is slower than a
   row with fewer threads. Tight variance is not accuracy (§1.1).
1. Make a measurement prove it can detect what it is looking for before believing its silence —
   the positive control in `tsan.sh` is the general form (§1.6).
