# RFC: Bounded extents

**Status:** Proposed
**Target:** After the first public tag. Additive — no existing spelling changes.
**Prerequisites:** the tensor-spelling RFC (`Tensor<T, [?, ?]>`, static rank, memory space in the flat identity, gates); dimension-expression folding on the flat path; top-level `const`.
**Builds on:** `rfc-unified-tensor-vx-review.md` (findings against `main` at `85342711`). File references are to that commit and that review.

______________________________________________________________________

## 0. For the implementing agent

This RFC implements **bounded dynamism (Vx#245)**, which is open and unimplemented. It adds one new dimension state, `?<=B`, to the tensor type introduced by the tensor-spelling RFC. Everything here is additive on the surface; the one behavior change — promoting W1029 to an error for unbounded tensors on capacity-declaring spaces — is staged and gated on the stdlib migration.

Section 4 states the frontend constraints. Each one names the regression that motivated it. Do not weaken any of them for convenience.

Gates that must hold at every checkpoint: everything the tensor-spelling RFC established (Section 6 there), plus the corpus and determinism additions in Section 9 here.

______________________________________________________________________

## 1. Summary

A dimension may be **static** (`4096`), **bounded** (`?<=B`, a runtime extent with a comptime upper bound), or **unbounded** (`?`). Bounds are comptime integer constants from top-level `const`s (and, later, machine-file figures). Widening is implicit structural subtyping across all three states. Narrowing is explicit, proved by the QF_LIA prover or checked at runtime on the host; device regions require proof. Allocation of a bounded tensor is itself a narrowing of its extent scalars. The memory algebra computes working sets from bounds, so admission stays byte-precise for tensors whose extents are runtime values. The stdlib becomes bounded-generic using the `const` generic mechanism that already exists for static dims.

______________________________________________________________________

## 2. Problem

### 2.1 Runtime extents are invisible to admission

A runtime dimension skips the capacity check with W1029 (`src/hir/check/transfer.rs:278`); `static_extent_of_dims` (`src/hir/check/raw.rs:318`) sizes literal dims only. Every KV cache, every variable-batch tensor, is unchecked. The admission story is exact for static shapes and silent for the shapes inference actually has.

### 2.2 The stdlib is written against unbounded shapes

`stdlib/std/tensor.vx` allocates results with extents from scalar arguments — `from_ptr_2d(ptr, d1, d2)`, `slice_2d(self, row, d1, d2)`, `uninit([a.shape[0], b.shape[1]])`. No bound flows anywhere. Every device matmul goes through these. A capacity rule for runtime extents cannot be enforced until the stdlib carries bounds, which means the stdlib's signatures are on this RFC's critical path, not after it.

### 2.3 Bounds in signatures move footprint information toward the frozen past

The frontend cannot see past a call edge; peak-over-call-tree admission (Vx#444) is open. A callee whose placed tensors have bounds derived from its parameter types has a footprint computable from its signature, which is in the frozen registry. This RFC does not solve Vx#444; it moves footprint information in the only direction the freeze allows.

______________________________________________________________________

## 3. Goals and non-goals

**Goals**

- `?<=B` as a per-dimension state; bounds from top-level `const`.
- Implicit widening across the three-state lattice; `as` as an optional explicit spelling.
- `narrow` (proved) and `narrow_checked` (runtime, host-only); allocation as a narrowing.
- Working sets from bounds; unbounded-on-capacity-space becomes an error, staged.
- Bounded const generics; stdlib migrated.
- `.shape` removed.
- Every new construct on the flat path with oracle parity, in the corpus, under the determinism gate.

**Non-goals**

- Dynamic rank.
- Grow-in-place. Extent is fixed at allocation.
- Call-site-observed specialization on bounds (Section 4.5).
- Scalar `where` bounds in signatures (`fn f(n: i32) where n <= 512`) — the general mechanism for proving allocation extents. Noted as the follow-up; Section 5.6 works without it for the common case.
- A `config` block. `const` first; `config` is sugar over it later.
- Ownership changes. `Tensor` is linear with `NEEDS_DROP` (`src/syntax/types.rs:608-618`, `src/gid.rs:51`).
- Struct-field layout; FFI descriptor ABI.

______________________________________________________________________

## 4. Frontend constraints this RFC must preserve

### 4.1 Identity is a pure function of content

A tensor type's GID is a content hash of a formatted string (`tensor_gid`, `src/hir/flatten.rs:48`, extended by the spelling RFC). A bounded dimension contributes a tag plus its **evaluated** bound. `[32]`, `[?<=32]`, and `[?]` hash differently. A bound must be a fully evaluated integer before minting; bound evaluation is integer folding over frozen declarations. The solver is never involved in minting a type. Provenance (which `const`, which file) is kept for diagnostics and is **not** hashed.

### 4.2 Frozen past, thread-local present

Every source of a bound is a declaration merged before the registry freeze. A bound may not depend on anything computed in a body. A tensor type minted inside a body (a `narrow` target, or an instantiation not already in the registry) goes through the existing deferred-identity path. No second reconciliation mechanism.

Machine-derived bounds (later) make type identity depend on `--machine`. That is correct; it means the machine file is config, config lives on the frozen session, and one compilation's machine file must be invisible to another's. Regression test required when machine-derived bounds land.

### 4.3 A function is a closed world

Facts available to a `narrow` proof: the enclosing body, the function's own signature, frozen declarations. Never another body. "The caller checked it" is inadmissible.

### 4.4 Implicit conversions

Widening across the extent lattice is structural subtyping decided from the two types alone, changes no bits, and is already how the tree treats static → dynamic (`is_assignable`, `src/hir/expr.rs:496-530`). It stays implicit. Method receivers depend on it (`a.fill(x)` on a shaped `a` against a `[?, ?]` impl). The flattener mints the `memref.cast` deterministically, as today.

### 4.5 No specialization after the fact

Bounded const generics are ordinary declaration-driven instantiation — the mechanism that already handles `impl<T: Float, const N: i32, const M: i32> Tensor<T, [N, M]>` (`stdlib/std/tensor.vx:219`). Every instantiation the program names is minted; none is inferred from observed call sites. The compiler makes no scheduling-dependent choice about which variant to emit.

### 4.6 No comptime that mints declarations

Bound evaluation yields a constant. `narrow` yields a type instantiation (the existing single exception). Nothing here mints a function, struct, or module during body checking.

### 4.7 No process-global state

The lint (`.github/workflows/ci.yml:38`) forbids locks and process-global atomics; the rule behind it forbids global registries. **Per-worker state is fine.** The persistent `hir::seam::Solver` held per checker (`check_state.rs:91`, spawned lazily at `src/hir/check/transfer.rs:747`) is per-worker and lock-free and stays as it is. What this RFC must not add: any process-global cache of solver availability, solver results, evaluated bounds, or instantiated types. (Three `OnceLock`s probing z3 were the last regression of this kind; the fix was deletion.)

Solver queries are yes/no. Never consume a model to choose a value.

### 4.8 The flat path is the shipping default, with the oracle behind it

New opcodes are implemented on the flat path with a differential test each (`flat_codegen_differential.rs`). `KNOWN_DECLINES` (`flat_corpus_sweep.rs`) may not grow. The `HirInstruction` layout (`src/bytecode.rs:307`) does not change; targets ride in `type_idx`.

### 4.9 The corpus is part of the compiler

The generator (`src/bin/corpus/mod.rs:106`, `--memalg`) emits one static tensor and nothing dynamic. This RFC adds a generator flag that emits bounded tensors, widenings, both narrowings, and bounded allocations, with a content test (the pattern at line 577). A construct the corpus does not exercise has unmeasured scaling by construction.

### 4.10 Determinism of emitted MLIR

The bound attribute (Section 5.9) is emitted in a canonical form derived from content.

______________________________________________________________________

## 5. Design

### 5.1 The extent lattice

| State | Notation | Meaning |
| --- | --- | --- |
| Static | `4096` | comptime constant |
| Bounded | `?<=B` | runtime extent; comptime constant `B` is a proven upper bound |
| Unbounded | `?` | runtime extent, no bound |

Per dimension:

```
Static(n)    ⊑  Bounded(b)   iff n <= b
Bounded(b1)  ⊑  Bounded(b2)  iff b1 <= b2
Bounded(b)   ⊑  Unbounded
Static(n)    ⊑  Unbounded
```

Whole-type `⊑`: same rank, element, memory space; every dimension `⊑`.

### 5.2 Syntax

Element first, dims second, placement third.

```
Tensor<f16, [?<=MAX_CTX, 4096], Memory::HBM>
Tensor<f16, [?<=MAX_BATCH, ?<=MAX_CTX, 8, 128], Memory::HBM>
```

`?<=` is lexed as `?` followed by `<=`; the parser accepts it only in a dims list.

### 5.3 Bound sources

**Prerequisite:** top-level `const` does not exist (only `const N` generic parameters do). Add it first:

```
const MAX_CTX: i32 = 8192;
const MAX_BATCH: i32 = 32;
```

A bound is a comptime-evaluable integer expression over top-level `const`s and literals, with `+ - * / ceil_div`. Machine-file figures (`--machine`, `src/driver.rs:93-99`) are merged as declarations before the program's own but are not reachable from type-level expressions today; making them reachable is a follow-up that needs the Section 4.2 regression test. A `config` block is sugar over `const` and is a follow-up.

### 5.4 Widening

Implicit, per Section 4.4. Lowering: `memref.cast` (already emitted). `as` between tensor types — `Expr::AsCast` (`src/syntax/expr.rs:194`) has no tensor-to-tensor arm — may be added as an explicit spelling of the same widening; it is optional and never required.

Widening rejected when a static extent exceeds the target bound (E-A).

### 5.5 Narrowing

Two forms. Names are proposals.

**`narrow<Tensor<T, [?<=B, ...], Memory::X>>(t)` — proved.** Discharged by `hir::prover::SmtProver` (QF_LIA, fresh process per proof, driven today by `comptime` checks at `src/hir/stmt.rs:581`). It is per-worker; if narrow sites become numerous, a persistent per-worker QF_LIA solver with `(push)`/`(pop)` — the `seam::Solver` model — is the optimization, not a cache.

**Fact sources, first version.** The checker is an AST walk with no dominator or path-sensitive fact collection (`prover.rs` takes explicit `comptime` assertions only). Tensors are linear, so a value cannot be reassigned between a guard and a use. Therefore, without any new flow analysis:

- a `narrow` of `t` is permitted lexically inside the then-branch of an `if` whose condition is a conjunction of `t.extent(i) <= C` comparisons (with `C` comptime-evaluable), and the proof obligation is discharged from those comparisons plus
- the bounds in the enclosing function's own signature, plus
- top-level `const` facts.

Loop-bound facts (via the mechanism in `fold_raw_extent`, `src/hir/check/raw.rs:432`) and dominating `narrow_checked` facts are follow-ups.

Unprovable → E-B, naming the bound and the facts the prover had.

**`narrow_checked<...>(t)` — runtime.** `TensorDim` / compare / trap, then `memref.cast`.

**Placement rule.** In a region placed on a non-host space, `narrow_checked` is refused (E-C). Device code does not get runtime surprises.

### 5.6 Allocation is a narrowing

`Tensor<T, [?<=B0, ?<=B1]>::uninit([e0, e1])` must establish `e0 <= B0` and `e1 <= B1`. The same rule as Section 5.5 applies: proved, or checked on the host, or refused on the device.

Provable sources for an extent scalar, first version:

- `.extent(i)` of a tensor whose type carries a bound on that dimension (the fact `extent <= bound` is in the type) — this covers the KV cache, whose extents come from the token tensor;
- a literal;
- a `const`.

A bare scalar parameter (`fn from_ptr_2d(ptr, d1, d2)`) has no fact; on the host it gets a runtime check, on the device it is refused. The general fix is scalar `where` bounds in signatures (the Vx#245 shape, `where` parses only for `impl Transfer<A, B>` today, `src/parser/decl.rs:126`) — a follow-up. The stdlib migration in Section 5.8 avoids needing it by taking bounds from tensor arguments.

### 5.7 Allocation size, extent, strides

- **Allocation is sized by the bound**: `∏ bound(i) × sizeof(elem)`, rounded to the machine file's granule. This makes admission the actual footprint and makes the disaggregation transfer size match the compile-time figure structurally.
- **This has a cost, on the host too.** `Tensor<f16, [?<=8192, 4096]>` holding extent 10 allocates 64 MiB. Bounds must be tight; that is why stdlib bounds are derived from arguments (Section 5.8), never "the widest bound."
- **Extent is fixed at allocation.** No growth.
- **Strides are the standard contiguous strides of the extent.** Kernels, cuBLAS, and the oracle see an ordinary dense memref. The allocation's tail beyond the extent is unused.
- An unbounded dimension has no bound; allocation takes an explicit runtime size and is permitted only where Section 5.10 allows unbounded tensors to live.

### 5.8 Bounded const generics and the stdlib

The mechanism exists for static dims. Extend `const` parameters to bound positions:

```
impl<T: Float, const B0: i32, const B1: i32> Tensor<T, [?<=B0, ?<=B1]> { ... }

fn matmul<T: Float, const M: i32, const K: i32, const N: i32>(
    a: Tensor<T, [?<=M, ?<=K], Memory::HBM>,
    b: Tensor<T, [?<=K, ?<=N], Memory::HBM>,
) -> Tensor<T, [?<=M, ?<=N], Memory::HBM> {
    let c = Tensor<T, [?<=M, ?<=N], Memory::HBM>::uninit([a.extent(0), b.extent(1)]);  // proved: extents carry bounds
    ...
}
```

Instantiation is declaration-driven (Section 4.5). A call with `a: Tensor<f16, [512, 4096]>` widens `512 ⊑ ?<=M` with `M` inferred as 512 — or as any bound the caller states; inference rules are Phase 0.

**Stdlib migration list** (from the review): `from_ptr_2d`, `slice_2d`, `uninit`, `fill` / `fill_static` (the duplication in `tensor.vx` this RFC removes), and every helper on the device path. Each gets bounds from its tensor arguments where it has them. `from_ptr_2d(ptr, d1, d2)` has no tensor argument; it stays host-only until scalar `where` bounds exist, and is documented as such.

### 5.9 Lowering

| State | memref dim | Extra |
| --- | --- | --- |
| Static(n) | `n` | — |
| Bounded(b) | `?` | canonical `vx.bound` attribute (no bound-like attribute exists today) |
| Unbounded | `?` | — |

The bound survives lowering for kernel generation (tiling ceilings, unroll factors), for the admission and transfer-size computations, and for diagnostics.

New opcodes: `TensorNarrow` and `TensorNarrowChecked` (`type_idx` = target). Widening reuses the cast already emitted. Bounded allocation reuses the alloc path with size operands from the bound and extents written to the descriptor.

**Prerequisite:** the flat path must fold dimension expressions. `tensor_dim_string` (`src/hir/flatten.rs:162`) admits a literal or bare identifier and declines everything else; the three `const_generics*.vx` fixtures are in `KNOWN_DECLINES` for this reason. The checker folds (`src/hir/check/calls.rs:1241`); the flattener must too. This is independent work and unblocks those fixtures on its own.

### 5.10 Memory algebra and admission, staged

Working set for a placed allocation: `∏ bound(i) × sizeof(elem)`, rounded to the granule. `static_extent_of_dims` extends from literal dims to bounded dims.

Unbounded-on-capacity-space (E-D) is introduced in two steps:

1. **Warning.** A new warning alongside W1029: *"dimension {i} is unbounded; working set not computable against declared capacity."* Lands with the type work.
1. **Error.** Promoted after the stdlib migration (Section 5.8) and after the 18 in-tree files that combine `DynTensor` with `transfer` or `spawn` (including `examples/llama.vx`) have bounds. Each of those files is a latent admission hole; each needs a real bound, not a wide one.

The 90-cell matrix must reproduce 67 / 23 / 0 at both steps.

### 5.11 `.shape`

Removed. `extent(i)` (spelling RFC) is the replacement. Sites still using `.shape` are part of the migration.

### 5.12 Generic bounds across calls (what the caller gets)

A callee's signature now states bounds. At a call, the caller knows the callee's placed footprint for every tensor whose bound derives from a parameter, without a call graph. This is not a change to admission accounting in this RFC; it is what makes a future per-function footprint summary (Vx#444) representable in the frozen registry.

______________________________________________________________________

## 6. Worked examples

### 6.1 KV cache

```
const MAX_BATCH: i32 = 32;
const MAX_CTX: i32 = 8192;
const N_KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;

type KCache = Tensor<f16, [?<=MAX_BATCH, ?<=MAX_CTX, N_KV_HEADS, HEAD_DIM], Memory::HBM>;
```

Working set: `32 × 8192 × 8 × 128 × 2 B = 536,870,912 B` per layer, before granule rounding. Admission compares this against the declared HBM. Runtime extent is irrelevant to admission.

### 6.2 Prefill produces a bounded KV; decode consumes it

```
fn prefill(tokens: Tensor<u32, [?<=MAX_BATCH, ?<=MAX_CTX], Memory::HBM>) -> (KCache, VCache) {
    spawn on(Topology::GPU[0]) {
        let k = KCache::uninit([tokens.extent(0), tokens.extent(1)]);   // proved from tokens' bounds
        ...
    }
}
```

`transfer(k, ...)` moves the bound-sized allocation; the transfer size equals the compile-time figure.

### 6.3 Widening at a call

```
fn rmsnorm(x: Tensor<f16, [?<=MAX_CTX, 4096], Memory::HBM>) -> ... { ... }

let a: Tensor<f16, [512, 4096], Memory::HBM> = ...;
rmsnorm(a);                    // implicit: 512 ⊑ ?<=8192
let c: Tensor<f16, [16384, 4096], Memory::HBM> = ...;
rmsnorm(c);                    // error[E-A]: dimension 0: static extent 16384 exceeds bound 8192 (MAX_CTX, consts.vx:2)
```

### 6.4 Narrowing from a guard

```
fn f(t: Tensor<f16, [?<=8192, 4096], Memory::HBM>) {
    if t.extent(0) <= 2048 {
        let u = narrow<Tensor<f16, [?<=2048, 4096], Memory::HBM>>(t);   // proved from the guard
        small_kernel(u);
    } else {
        big_kernel(t);
    }
}
```

### 6.5 Host check, device refusal

```
fn host_side(t: Tensor<f32, [?, 4096], Memory::CPU_DRAM>) {
    let u = narrow_checked<Tensor<f32, [?<=8192, 4096], Memory::CPU_DRAM>>(t);   // runtime check + trap
}

fn device_side(t: Tensor<f16, [?<=8192, 4096], Memory::HBM>) {
    spawn on(Topology::GPU[0]) {
        let u = narrow_checked<Tensor<f16, [?<=1024, 4096], Memory::HBM>>(t);
        // error[E-C]: runtime-checked narrowing is not permitted in a device region (Topology::GPU[0])
    }
}
```

### 6.6 Allocation refused on device without a provable extent

```
fn stage(n: i32) {
    spawn on(Topology::GPU[0]) {
        let s = Tensor<f16, [?<=1024], Memory::HBM>::uninit([n]);
        // error[E-B]: cannot prove n <= 1024; known facts: (none for `n`)
        //   help: derive `n` from a bounded tensor's extent(i), or use a literal
    }
}
```

### 6.7 Unbounded on device (after promotion)

```
let raw: Tensor<f32, [?, 3], Memory::CPU_DRAM> = read_points(path);
let on_gpu = transfer(raw, Memory::HBM);
// error[E-D]: dimension 0 is unbounded; working set not computable against declared capacity 80 GiB (fleet/a100-80.vx:4)
//   help: narrow_checked to a bounded extent on the host first
```

______________________________________________________________________

## 7. Diagnostics

Codes continue after **E6018** (`src/diagnostic.rs:270-340`). Placeholders:

| Code | Trigger | Message shape |
| --- | --- | --- |
| E-A | widening where a static extent exceeds the target bound, or bounds not ordered | `dimension {i}: extent {n} exceeds bound {b} ({provenance})` |
| E-B | `narrow` or bounded allocation the prover cannot discharge | `cannot prove {obligation}; known facts: {list}` |
| E-C | `narrow_checked` in a device region | `runtime-checked narrowing is not permitted in a device region ({topology})` |
| E-D | unbounded dimension placed in a capacity-declaring space (after promotion) | `dimension {i} is unbounded; working set not computable against declared capacity {cap} ({machine file})` |
| E-F | bound expression not evaluable from declarations | `bound must be a comptime constant; `{expr}` depends on {body-local thing}` |
| W-D | E-D's warning form, before promotion | as E-D, as a warning |

Every message mentioning a bound prints its provenance.

______________________________________________________________________

## 8. Phases

### Phase 0 — remaining questions

1. Bound inference at calls into bounded const generics: is `M` inferred from the argument's static extent, from its bound, or stated by the caller? (Follow the existing `const N` inference.)
1. Does `SmtProver` have a way to pass facts, or only assertions? Shape of the change to accept a fact set plus one obligation.
1. How the persistent `seam::Solver` and `SmtProver` share (or don't share) the z3 binary path — confirm nothing process-global.
1. Confirm `static_extent_of_dims` can be extended to bounds without touching the raw-traffic accounting.
1. Cost of "sized by bound" on the current host-side demos: list each allocation whose bound is much larger than its typical extent.

### Phase 1 — prerequisites

Top-level `const`; dimension folding on the flat path (the three `const_generics*.vx` fixtures leave `KNOWN_DECLINES`).

### Phase 2 — type and identity

`?<=B` state; bound evaluation; digest tag; `⊑`; E-A, E-F.

### Phase 3 — narrowing and allocation

`narrow` (lexical-guard fact sources, per 5.5), `narrow_checked`, placement rule, allocation as narrowing; `SmtProver` fact-set interface; E-B, E-C.

### Phase 4 — lowering

`vx.bound`; `TensorNarrow` / `TensorNarrowChecked` with differential tests; bounded alloc.

### Phase 5 — bounded const generics and stdlib

Section 5.8. `fill` / `fill_static` become one. `.shape` removed.

### Phase 6 — admission

W-D warning; working set from bounds; matrix at 67 / 23 / 0.

### Phase 7 — corpus, gates

Generator flag; content test; determinism over the flagged corpus.

### Phase 8 — migration of the 18 files, then E-D promotion

Each file gets real bounds. Matrix at 67 / 23 / 0 after promotion.

### Phase 9 — docs

The lattice, bound sources, both narrowings, allocation rule, the host cost of sized-by-bound, the placement rule.

______________________________________________________________________

## 9. Tests

- Every row of 5.1, positive and negative, implicit at each site.
- Widening where a static extent exceeds the bound → E-A.
- `narrow` from a lexical guard (proved); `narrow` outside any guard (E-B); `narrow` where the guard is on a different value (E-B); a `narrow` whose only justification is in another function must fail.
- `narrow_checked` on host (pass, trap); in a device region (E-C).
- Allocation: extents from a bounded tensor (proved); literal (proved); bare scalar on host (checked); bare scalar on device (E-B).
- Sized by bound: allocation bytes equal the bound-derived figure; extent stored; strides contiguous in extent.
- Bounded const generics: instantiation at two different bounds yields two GIDs; the same bound spelled via two `const` paths yields one GID.
- Differential (flat vs oracle) for both new opcodes and for bounded allocation.
- Corpus: flag on emits the constructs; flag off emits none; determinism byte-identical over the flagged corpus.
- Admission: 67 / 23 / 0 before and after promotion; W-D fires on an unbounded device tensor before promotion; E-D after.
- Regression: 64/64; 442,368 B; `KNOWN_DECLINES` did not grow.

______________________________________________________________________

## 10. Open questions for the author

1. Names for `narrow` / `narrow_checked`.
1. Bound inference at calls into bounded const generics (Phase 0, item 1) — a language decision once the existing `const N` behavior is known.
1. Should widening between two bounds (`?<=2048` → `?<=8192`) lint when a much tighter bound is discarded at a site that will then allocate at the wider one?
