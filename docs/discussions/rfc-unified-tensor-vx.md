# RFC: Unified `Tensor` with per-dimension extent states (retire `DynTensor`)

**Status:** Proposed
**Target:** Pre-open-source. This changes the public type surface; acceptable now, not after launch.
**Supersedes:** `DynTensor`

## 0. How to use this document

Written to be handed to an implementing agent working in the `vxc` tree.

1. **Investigate first.** Section 9, Phase 0 lists what to find. Report findings before writing code. Assumptions are marked as such; confirm or correct each.
1. **Syntax is illustrative.** `Tensor<[?<=N, H, D], f16, Memory::HBM>` is notation for the concept. Map it onto the existing grammar. If a construct has no counterpart, propose the minimal addition and flag it. Do not silently invent surface syntax. Note that the CppCon deck shows `Tensor<f16>([8, 64])` and `Ref<T, Memory::NPU_HBM>`; Phase 0 must establish which of these is current.
1. **Invariants that must hold at every checkpoint** (Section 8 lists them in full): all tests pass; MLIR byte-identical across thread counts and with rayon off the path; TSan zero races; zero locks and no new atomics in `src/`; the admission matrix reproduces **67 / 23 / 0**; the demos do not decline to the AST oracle.
1. **Any conflict between this RFC and the tree: stop and report.** Do not pick one.

______________________________________________________________________

## 1. Summary

Vx has two tensor types: `Tensor` (static dimensions) and `DynTensor` (dynamic extents, dynamically allocated). This RFC replaces both with one `Tensor` type in which **rank is always static** and **each dimension independently has one of three extent states**: static, bounded, or unbounded.

- Widening (static → bounded → unbounded) is **explicit**, spelled with `as`, and is a runtime no-op.
- Narrowing is explicit and comes in two forms: proved at comptime by the existing solver, or checked at runtime on the host. Device regions require proof.
- Bounds are comptime constants drawn from declarations that exist before the registry freeze, so tensor-type identity remains a pure function of content.
- The memory algebra computes working sets from bounds; an unbounded extent cannot be placed in a capacity-declaring space.
- `DynTensor` is removed; its uses become `Tensor` with `?` dims.

The bounded state is the important one. It is Vx's existing "bounded dynamism," made per-dimension, and it lets admission remain byte-precise for tensors whose extents are runtime values (KV caches, variable batch, variable context).

______________________________________________________________________

## 2. Problem

### 2.1 Two nominal types cannot share code

A function over `Tensor` cannot accept a `DynTensor` and vice versa. Every shape-generic operation is written twice or once. "Hard to carry the dimensions across" is not a property of dynamic shapes; it is a property of two types that do not know about each other.

### 2.2 `DynTensor` conflates three orthogonal axes

| Axis | Governs | Should be |
| --- | --- | --- |
| **Rank** | number of dimensions | always static |
| **Extent** (per dim) | size along the dimension | static, bounded, or unbounded — independently per dim |
| **Allocation / ownership** | when the buffer exists, who frees it | independent of shape |

A fully static tensor can be heap-allocated at runtime; a bounded tensor can be arena-allocated because its maximum footprint is known at compile time. `DynTensor` fuses "dynamic extents" with "allocated at runtime," and those are unrelated.

### 2.3 Fully-dynamic tensors have no place on a capacity-walled device, and nothing says so

Device HBM is a wall: the machine file declares a capacity and a refusal is a prediction that comes true. A tensor with an unbounded extent has no computable working set and cannot be admitted. Today `DynTensor` exists as if it were placeable anywhere. The correct behavior is a diagnostic.

### 2.4 The lowering target already models this

`memref<32x128xf16>` and `memref<?x128xf16>` lower to the same descriptor (`{ptr, ptr, i64, [N x i64], [N x i64]}`). MLIR made dynamism per-dimension for this reason; `memref.cast` converts between them. The two-type design discards that unification and rebuilds a worse one in the frontend.

### 2.5 Bounds in signatures move footprint information toward the frozen past

The frontend's closed-world model cannot see past a call edge, which is why peak-over-call-tree admission (Vx#444) is open. A callee whose placed tensors have bounds derived from its **parameter types** has a footprint computable from its *signature* — which lives in the frozen registry, readable by any caller without a call graph. This RFC does not solve Vx#444, but it moves footprint information in the only direction compatible with the freeze.

______________________________________________________________________

## 3. Goals and non-goals

**Goals**

- One `Tensor` type constructor; static rank; per-dimension extent state.
- Explicit widening via `as`; explicit narrowing with proof-or-check and a device-region proof requirement.
- Bounds are comptime constants from pre-freeze declarations; tensor-type identity stays content-hashed.
- Memory algebra computes working sets from bounds; unbounded-on-device is refused.
- `DynTensor` removed; migration mechanical.
- Every new construct implemented on the flat path with oracle parity, exercised by the benchmark corpus, and covered by the determinism gate.

**Non-goals**

- Dynamic rank. Never.
- Grow-in-place. Extent is fixed at allocation.
- Any specialization on bounds beyond ordinary declaration-driven generic instantiation. Deferred to a later RFC.
- Redesigning ownership (Section 5.11 states the assumption and recommended direction).
- Struct-embedded tensor layout (this RFC makes it simpler — one descriptor shape per rank — but does not implement it).
- An FFI descriptor ABI.

______________________________________________________________________

## 4. Frontend constraints this RFC must preserve

These are the properties of the `vxc` frontend (per the CppCon 2026 deck and the pipeline design docs) that the implementation must not weaken. They are stated up front because every one of them has been broken before by an innocent feature.

### 4.1 Identity is a pure function of content

A GID is 256 bits, minted by hashing what a thing is. A generic instantiation's identity is base + digest of its arguments' identities. Therefore:

- A tensor type is a generic instantiation whose arguments include its element type, memory space, and **each dimension's extent state**. The three states must be **tag-distinguished in the digest**: `[32]` and `[?<=32]` are different types and must hash differently. `?` needs a stable encoding.
- Dimension encoding must go through the full 128-bit identity `(w0, w1)`. Never through any narrower side identifier. (Two memory spaces once collided because an ID was only a thousand values wide; the determinism gate caught it only at corpus scale.)
- A bound that appears in a type must be a **fully evaluated integer** at minting time. Bound evaluation is pure integer folding over frozen facts. The solver is never involved in minting a type.

### 4.2 Frozen past, thread-local present

The `GlobalSession` (registry, arenas, config) is sealed before the first worker exists; workers read it by `&` and write only private streams. Therefore:

- Every source of a bound — config declarations, `const`s, machine-file figures — must be a declaration merged **before the freeze**. A bound may not depend on anything computed inside a function body.
- A tensor type minted inside a body (the result of a `narrow`, or any instantiation not already in the registry) goes through the **existing deferred-identity path**: local arena, IOU bit, reconciled at the single barrier, patched in the linear sweep. This RFC adds no second reconciliation mechanism.
- Machine-derived bounds make type identity **depend on the compile's configuration** (same program, different `--machine`, different GIDs). That is correct, and it means the machine file is config, and config is state on the frozen session — never a process global, never a `thread_local`. A regression test must show one compilation's machine file is invisible to another's (same test shape as the topology-name test and the `intern_mode` fix).

### 4.3 A function is a closed world

Bodies depend on signatures, never on other bodies. Therefore:

- The fact set available to a `narrow` proof is: **facts in the enclosing body** (guards, loop bounds, dominating checked narrowings), **facts in the function's own signature** (parameter bounds), and **frozen declarations** (config invariants). **Never facts from another function's body.** "The caller already checked it" is not admissible; it would require a call graph.
- Working-set accounting stays per-function, as today. Section 2.5 notes what bounds-in-signatures give the caller for free; nothing else changes.

### 4.4 No implicit conversions; assignment requires identical types

Every conversion is an explicit `as`. Literals get their type from context; nothing else does. Implicit coercion once miscompiled exactly the "same value, two types" shape. Therefore:

- **Widening is explicit.** The flattener never mints a cast the user did not write. A tensor of one type assigned to a slot of another type is a type error whose help line suggests the `as`.
- This is the same principle as mandatory `transfer()` on a free hardware boundary: provable from the source text.

### 4.5 No specialization after the fact; a generic's meaning closes at its definition

Therefore:

- No call-site-observed specialization on bounds. If shape-generic functions ever specialize, it is because the declaration is generic over a shape parameter and each instantiation is minted like `Vec<i32>`. Deferred.
- The compiler makes **no** scheduling-dependent choice about which variant to emit. Byte-identical output at every thread count is non-negotiable.

### 4.6 No comptime that mints declarations

Bound evaluation produces a constant, not a declaration. A `narrow` produces a type instantiation (the existing single exception, handled by the deferred path). Nothing in this RFC mints a function, a struct, or a module during body checking.

### 4.7 Zero locks, no process-global atomics, no global registries

CI greps for the primitives; exemptions are inline and justified; the rule behind the lint is "no global registries of any flavor." Therefore:

- **No cache of any kind for solver availability, solver results, evaluated bounds, or instantiated tensor types.** Evaluated bounds live on the frozen session (declarations) or in the worker's private streams (body-local). Solver invocation is per query, per worker, holding no shared state. (Three `OnceLock`s probing "is z3 installed?" were the last regression of this kind; the fix was deletion.)
- Solver queries are **yes/no** (unsat = proved). Never consume a model to choose a value; models are not deterministic across solver versions and would make output depend on the solver build.

### 4.8 The flat path is the shipping default, with the AST oracle behind it

Per-function lowering is atomic; a construct outside the flat subset is a decline, and the oracle compiles that function. Therefore:

- New opcodes (Section 5.7) must be implemented on the **flat path**, with differential tests against the oracle for every one.
- **CI must gate that `llama-cpp.vx` and `flashattention.vx` do not decline.** Otherwise the demos silently run through the oracle path and the frontend's scaling number does not apply to the code being shown.
- The 24-byte `HirInstruction` does not change: the cast target rides in `type_idx`; `narrow_checked` needs no extra operand.

### 4.9 The corpus is part of the compiler

A phase the benchmark corpus does not exercise has unmeasured scaling by construction (env_build ran serial at 51% for two weeks because the generator never emitted a `Memory` block). Therefore:

- The benchmark generator must emit bounded tensors, `as` widenings, and both narrowing forms, behind a flag (the `--memalg`-style pattern), with a **corpus-content test** asserting the constructs are present when the flag is on and absent when off.
- The byte-identical gate runs over that corpus at 1, 8, 32, and 48 threads and with rayon off the path.
- Any new phase or pass introduced by this RFC must be reachable by the `Schedule` (no bare `par_iter` outside it) and must appear in the timed-phase table.

### 4.10 Determinism of emitted MLIR

The `vx.bound` attribute (Section 5.8) is emitted in a canonical textual form derived from content — never from an interning order or an arrival order.

______________________________________________________________________

## 5. Design

### 5.1 The extent lattice

Each dimension has exactly one state:

| State | Notation (illustrative) | Meaning |
| --- | --- | --- |
| **Static** | `32` | extent is the comptime constant 32 |
| **Bounded** | `?<=B` | extent is a runtime value; comptime constant `B` is a proven upper bound |
| **Unbounded** | `?` | extent is a runtime value with no comptime bound |

Per-dimension ordering (`⊑` = "can widen to"):

```
Static(n)    ⊑  Bounded(b)    iff n <= b
Bounded(b1)  ⊑  Bounded(b2)   iff b1 <= b2
Bounded(b)   ⊑  Unbounded
Static(n)    ⊑  Unbounded
```

`T1 ⊑ T2` iff same rank, same element type, same memory space, and every dimension of `T1 ⊑` the corresponding dimension of `T2`. This relation decides whether an `as` widening is legal. It does **not** make `T1` assignable to a `T2` slot without the `as`.

**Rank is static.** There is no unranked tensor.

### 5.2 Type syntax (illustrative — map to the real grammar)

```
Tensor<[32, 8, 128], f16, Memory::HBM>
Tensor<[?<=Serving.max_ctx, 8, 128], f16, Memory::HBM>
Tensor<[?<=Serving.max_batch, ?<=Serving.max_ctx, 4096], f16, Memory::HBM>
Tensor<[?, 3], f32, Memory::CPU_DRAM>
```

Assumption to confirm (Phase 0): memory space is already part of the tensor type. If it lives elsewhere, leave it there; this RFC requires only that it be part of the type's identity.

### 5.3 Bound expressions

A bound `B` in `?<=B` is a comptime-evaluable, non-negative integer expression over **pre-freeze declarations only** (Section 4.2). Acceptable sources, most preferred first:

1. A config declaration field (`Serving.max_ctx`). Preferred: admission checks and tensor bounds then derive from one declaration and cannot drift.
1. A machine-file figure (a declared capacity or granule, used in derived bounds).
1. A `const` in scope.
1. A literal.

Integer arithmetic over these (`+ - * / ceil_div`) is permitted; it is the expression language comptime already folds. The evaluated constant is what participates in identity and subtyping. The **provenance** (which declaration, which file) is retained for diagnostics, not for identity.

### 5.4 Widening: explicit, via `as`

Widening is a runtime no-op and is always spelled:

```
let a: Tensor<[512, 4096], f16, Memory::HBM> = ...;
rmsnorm(a as Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>);
```

Rules:

- `t as T2` is legal iff `typeof(t) ⊑ T2` per Section 5.1. Otherwise E-A (Section 7).
- Without the `as`, passing `a` to `rmsnorm` is the ordinary type-mismatch error, with a help line suggesting the cast.
- Lowering: `memref.cast` from the more-static memref type to the less-static one.
- The `as` is never inserted by the compiler.

Open question 2 asks whether `as _` may infer its target from the expected type at the site (the literal-typing mechanism), so the conversion stays visible without restating the type.

### 5.5 Narrowing: explicit, two forms

Narrowing asserts a tighter extent state than the type carries. It is never implicit and is **not** spelled with `as`, because `as` in Vx is total and narrowing can fail to compile or trap. Names are proposals (open question 1).

**`narrow<T2>(t)` — proved at comptime.** The compiler must discharge the narrowed bounds from the closed-world fact set (Section 4.3):

- an enclosing guard: `if t.extent(0) <= 2048 { let u = narrow<...>(t); }`
- a loop bound whose induction variable feeds the extent
- a bound in the enclosing function's own signature
- a frozen config invariant (`Serving.max_ctx <= 8192` declared once)
- a dominating `narrow_checked` on the same value

Unprovable → E-B, naming the bound and listing the facts the solver had. Lowering: `memref.cast` only.

**`narrow_checked<T2>(t)` — checked at runtime.** Emits an extent comparison and a trap on failure, then the cast. Lowering: `memref.dim` / `arith.cmpi` / `cf.assert` (or the existing trap), then `memref.cast`.

**Placement rule.** Inside a region placed on a non-host space (a `spawn on(Topology::GPU[...])` body, or code the placement analysis assigns to a device):

- `narrow` is permitted.
- `narrow_checked` is **refused** (E-C). Device code does not get runtime surprises; an unprovable bound means the program does not compile.

Both forms are permitted on host regions.

Narrowing to `Static(n)` (proving `extent == n`) is a narrowing, useful before handing a tensor to a fully static kernel.

### 5.6 Allocation, extent, and the absence of growth

A bounded dimension has two numbers:

- **bound** — comptime, in the type, `t.bound(i)` is a comptime expression.
- **extent** — runtime, in the descriptor's `sizes[i]`, `t.extent(i)` is a runtime value, with the comptime fact `t.extent(i) <= t.bound(i)` available to the solver.

**Allocation is sized by the bound**: `∏ bound(i) × sizeof(elem)`, rounded to the machine file's granule. This is what makes admission the *actual* footprint rather than an upper bound, and what makes the disaggregation transfer size match the compile-time figure byte-exactly.

**Extent is set at allocation and never changes.** There is no grow-in-place. A bounded tensor received as a parameter has whatever extent its producer gave it.

**Strides are the standard contiguous strides of the extent** — the layout a dense `memref<?x?xf16>` has, so kernels, cuBLAS, and the oracle path see an ordinary dense tensor. The tail of the allocation beyond the extent's contiguous region is unused. That tail is the price of exact admission on a bounded tensor; a fully static tensor has no tail.

For an unbounded dimension there is no bound; allocation must be given an explicit runtime size, and this is only permitted where Section 5.9 allows unbounded tensors to live.

### 5.7 New HIR opcodes

| Opcode | Operands | Type stream | Lowering |
| --- | --- | --- | --- |
| `TensorWiden` | `o1 = source` | `type_idx = target type` | `memref.cast` |
| `TensorNarrow` | `o1 = source` | `type_idx = target type` | `memref.cast` (proof obligation discharged at type-check) |
| `TensorNarrowChecked` | `o1 = source` | `type_idx = target type` | dim / cmp / assert, then `memref.cast` |

No change to the 24-byte instruction. Each opcode gets a differential test (flat vs oracle, same lowering, same JIT, same result).

### 5.8 Lowering to MLIR

| Extent state | memref dim | Extra |
| --- | --- | --- |
| `Static(n)` | `n` | — |
| `Bounded(b)` | `?` | bound retained as a canonical `vx.bound` attribute (or an equivalent the tree already has) |
| `Unbounded` | `?` | no bound |

memref has no "bounded" state; boundedness is a frontend refinement that lowers to `?`. The bound must survive lowering because kernel generation can use it (tiling ceilings, unroll factors, register budgets), the admission checker and transfer-size computation read it, and diagnostics print it.

Allocation of a bounded tensor: the existing alloc path, size operands from the bound (comptime constants), extents written to the descriptor. Casts per Section 5.7.

### 5.9 Memory algebra and admission

Working set for any placed allocation:

```
bytes(t) = ∏ bound(i) × sizeof(elem)   rounded to the machine file's granule
```

**New rule:** a tensor with any `Unbounded` dimension cannot be placed in a memory space that declares a capacity (E-D). Host memory declares bandwidth but no capacity, so unbounded tensors are host-only, which is the intended use (I/O, FFI buffers).

The existing 8× GQA sizing correction stays; only the source of the numbers changes. The 90-cell matrix must reproduce 67 / 23 / 0. A different matrix is a bug.

### 5.10 Generics over shapes

Out of scope beyond this statement: if and when functions become generic over shape parameters, an instantiation with concrete bounds is minted exactly like `Vec<i32>` — base + digest of arguments — through the deferred path, and every instantiation the program names is emitted. No inference of specializations from observed call sites. See Section 4.5.

### 5.11 Ownership (assumption and recommendation — separable)

**Assumption to confirm:** `Tensor` is a memref and therefore a non-owning view; deallocation is explicit or handled by an existing mechanism.

**Recommendation (not in this RFC):** an owning buffer type, parameterized by shape (with bounds) and space, carrying the drop obligation and lending out `Tensor` views — the `Vec<T>` / `&[T]` split. Bounded allocation is naturally a property of the owning type. Its `NEEDS_DROP` bit in word 3 is the existing mechanism.

### 5.12 Struct fields and FFI

Rank is static, so the descriptor for a tensor field has a compile-time-known size regardless of extent state; an inline layout becomes possible with one descriptor shape per rank instead of one per tensor type. Not implemented here.

An `Unbounded` tensor has a C counterpart only as `(ptr, sizes...)`. No descriptor ABI is defined here. Crossing `extern "C"` with any tensor keeps today's behavior; if that behavior is "not allowed," keep it not allowed.

______________________________________________________________________

## 6. Worked examples (illustrative syntax)

### 6.1 KV cache

```
config Serving {
    max_batch:  u32 = 32,
    max_ctx:    u32 = 8192,
    n_kv_heads: u32 = 8,
    head_dim:   u32 = 128,
}

type KCache = Tensor<[?<=Serving.max_batch, ?<=Serving.max_ctx,
                      Serving.n_kv_heads, Serving.head_dim], f16, Memory::HBM>;
```

Working set: `32 × 8192 × 8 × 128 × 2 B = 536,870,912 B` per layer for K, before granule rounding. Admission compares this against the machine file's HBM. Runtime extent is irrelevant to admission.

### 6.2 Prefill produces a bounded KV; decode consumes it

```
fn prefill(tokens: Tensor<[?<=Serving.max_batch, ?<=Serving.max_ctx], u32, Memory::HBM>)
    -> (KCache, VCache)
{
    spawn on(Topology::GPU[0]) {
        // K, V allocated with bound = config, extent = (tokens.extent(0), tokens.extent(1))
    }
}

fn decode(k: KCache, v: VCache, ...) -> ... {
    spawn on(Topology::GPU[1]) { ... }
}
```

`transfer(k, ...)` moves the allocation, whose size is the bound-derived figure — the reason the demo's KV transfer matches the compile-time number byte-exactly becomes structural.

### 6.3 A shape-generic function, called with explicit widening

```
fn rmsnorm(x: Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>)
    -> Verified<Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>>
{ ... }

let a: Tensor<[512, 4096], f16, Memory::HBM> = ...;
let b: Tensor<[?<=2048, 4096], f16, Memory::HBM> = ...;

rmsnorm(a as Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>);   // 512 <= 8192
rmsnorm(b as Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>);   // 2048 <= 8192
rmsnorm(a);   // error: type mismatch; help: widen with `as`
```

Under the old design this function existed twice. Neither `as` emits runtime work.

### 6.4 Widening rejected when the constant is too large

```
let c: Tensor<[16384, 4096], f16, Memory::HBM> = ...;
rmsnorm(c as Tensor<[?<=Serving.max_ctx, 4096], f16, Memory::HBM>);
// error[E-A]: cannot widen Tensor<[16384, 4096], ...> to Tensor<[?<=8192, 4096], ...>
//   dimension 0: static extent 16384 exceeds bound 8192 (Serving.max_ctx, serving.vx:3)
```

### 6.5 Narrowing proved from a guard

```
fn f(t: Tensor<[?<=8192, 4096], f16, Memory::HBM>) {
    if t.extent(0) <= 2048 {
        let u = narrow<Tensor<[?<=2048, 4096], f16, Memory::HBM>>(t);   // proved from the guard
        small_kernel(u);
    } else {
        big_kernel(t);
    }
}
```

No runtime code for the cast; the guard is the check.

### 6.6 Narrowing that cannot be proved: host vs device

```
fn host_side(t: Tensor<[?, 4096], f32, Memory::CPU_DRAM>) {
    let u = narrow_checked<Tensor<[?<=8192, 4096], f32, Memory::CPU_DRAM>>(t);   // ok: runtime check + trap
}

fn device_side(t: Tensor<[?<=8192, 4096], f16, Memory::HBM>) {
    spawn on(Topology::GPU[0]) {
        let u = narrow_checked<Tensor<[?<=1024, 4096], f16, Memory::HBM>>(t);
        // error[E-C]: runtime-checked narrowing is not permitted in a device region (Topology::GPU[0])
        //   help: use `narrow` with a bound provable from a guard, a loop bound, or a declared invariant
    }
}
```

### 6.7 Unbounded tensor refused on device

```
let raw: Tensor<[?, 3], f32, Memory::CPU_DRAM> = read_points(path);
let on_gpu = transfer(raw, Memory::HBM);
// error[E-D]: cannot place Tensor<[?, 3], f32> in Memory::HBM
//   dimension 0 is unbounded; working set not computable against declared capacity 80 GiB (fleet/a100-80.vx:4)
//   help: narrow first, e.g. narrow_checked<Tensor<[?<=N, 3], ...>>(raw)
```

### 6.8 Bound from a machine-file figure

```
type Stage = Tensor<[?<=Machine.HBM.granule / 2], u8, Memory::CPU_DRAM>;
```

Whether machine-file figures are addressable from type-level expressions is a Phase 0 question. If not, config-derived bounds (6.1) suffice for the first cut.

### 6.9 Migration: before and after

```
// before
let kv: DynTensor<f16, Memory::HBM> = DynTensor::alloc([b, s, 8, 128]);

// after — bounded (preferred; admissible on device)
let kv: Tensor<[?<=Serving.max_batch, ?<=Serving.max_ctx, 8, 128], f16, Memory::HBM>
    = Tensor::alloc(extents: [b, s]);       // bounds come from the type; extent fixed here

// after — unbounded (host only; only when there truly is no bound)
let pts: Tensor<[?, 3], f32, Memory::CPU_DRAM> = Tensor::alloc(extents: [n]);
```

A `DynTensor` site on a device path that cannot be assigned a bound was already a latent admission hole and needs a real decision.

______________________________________________________________________

## 7. Diagnostics

Proposed additions to the seam taxonomy. Use the next free codes after E6014.

| Code | Trigger | Message shape |
| --- | --- | --- |
| E-A | `as` widening where a static extent exceeds the target bound, or bounds are not ordered | `dimension {i}: extent {n} exceeds bound {b} ({provenance})` |
| E-B | `narrow` the solver cannot discharge | `cannot prove extent({i}) <= {b} here; known facts: {list}` |
| E-C | `narrow_checked` in a device region | `runtime-checked narrowing is not permitted in a device region ({topology}); use narrow with a provable bound` |
| E-D | unbounded dimension placed in a capacity-declaring space | `dimension {i} is unbounded; working set not computable against declared capacity {cap} ({machine file})` |
| E-E | rank mismatch in any cast | `rank {r1} does not match rank {r2}; rank is static` |
| E-F | bound expression not evaluable from pre-freeze declarations | `bound must be a comptime constant from a declaration; `{expr}` depends on {body-local thing}` |

Existing type-mismatch on assignment/argument: add a help line suggesting the `as` when `⊑` holds.

Every message mentioning a bound prints its provenance — the same citation discipline the machine files use.

______________________________________________________________________

## 8. Invariants (checked at every phase checkpoint)

1. All existing tests pass (~700 in CI).
1. Emitted MLIR byte-identical at 1, 8, 32, 48 threads and with rayon off the path — over the existing corpus **and** over the corpus with the new-construct flag on.
1. TSan: zero data races, positive control still fires.
1. `src/` lint: zero locks; exempted atomics unchanged in count; no new `static` / `thread_local` / `LazyLock` registry.
1. Admission matrix: **67 / 23 / 0**.
1. Exactness: 64/64 token ids on every platform available.
1. Disaggregation KV transfer size unchanged at 442,368 B.
1. `llama-cpp.vx` and `flashattention.vx` do not decline to the oracle; the corpus-sweep gate reports no new declines.
1. Phase-scaling table: no new serial phase; any new pass appears in it and is schedule-controlled.

______________________________________________________________________

## 9. Implementation plan

### Phase 0 — Investigate and report (no code)

Report with file references:

1. How `Tensor` and `DynTensor` are represented in the frontend (GID digest form). How are static dims encoded as generic arguments today? Is rank stored? Is the memory space in the type? Which of `Tensor<f16>([8, 64])` / `Tensor<[8,64], f16, ...>` / `Ref<T, ...>` is current?
1. How integer/const arguments are digested into a GID, and whether a tag can distinguish the three extent states in the digest (Section 4.1).
1. How tensor types lower to memref types; any existing `?` dims; any existing bound-like attribute.
1. How `DynTensor` allocates and what it emits.
1. How comptime evaluates integer expressions, and separately how it invokes the solver: per query? any caching? any process-global state? (Section 4.7.)
1. What the solver can currently see about extents; how the admission matrix computes the KV working set and from which declarations.
1. Whether a config/serving declaration construct exists that can serve as the preferred bound source (Section 5.3); if not, whether `const`s suffice for the first cut.
1. Whether machine-file figures are reachable from type-level expressions, and how machine files reach the frozen session today.
1. Where `E6001`–`E6014` are emitted; infrastructure for provenance in messages.
1. Where `as` is implemented and how conversions are represented in the flat HIR.
1. Which tensor constructs the flat path currently accepts vs declines; where the differential tests live; how the corpus-sweep gate is defined.
1. How the benchmark generator is parameterized (the `--memalg` pattern) and where its content tests are.
1. Every `DynTensor` use in the tree: device path or host-only; bound derivable from an existing declaration or genuinely unbounded.
1. Whether a `Tensor` parameter occupies a word-2 borrow slot today (this RFC does not change word 2; confirm it is unaffected).

**Checkpoint:** a written report answering all fourteen. Do not proceed until the assumptions in 5.2, 5.6, 5.11 are confirmed against it.

### Phase 1 — Type representation

- Per-dimension extent state in the tensor type; tag-distinguished digest; rank static.
- Bounds evaluated once from pre-freeze declarations; evaluated constant in identity; provenance retained for diagnostics.
- Parsing for the illustrative forms, mapped to the real grammar.

**Checkpoint:** invariants 1–4; fully static tensors round-trip unchanged; MLIR byte-identical to pre-phase output.

### Phase 2 — Subtyping and explicit widening

- The `⊑` relation; `as` widening on the flat path (`TensorWiden`) and the oracle.
- E-A; help line on plain mismatch.

**Checkpoint:** invariants; differential test for `TensorWiden`; positive and negative tests for every row of 5.1.

### Phase 3 — Narrowing

- `narrow`: closed-world fact collection (Section 4.3), solver query per site with no caching, E-B on failure.
- `narrow_checked`: check + trap + cast.
- Placement rule, E-C. Rank mismatch, E-E.

**Checkpoint:** invariants; differential tests for both opcodes; tests for 6.5 and 6.6.

### Phase 4 — Memory algebra and admission

- Working set from bounds; E-D; provenance in messages.
- Machine-file-as-config regression test (Section 4.2).

**Checkpoint:** invariant 5 exactly. A different matrix is a bug in this phase.

### Phase 5 — Lowering and allocation

- `?` dims with canonical `vx.bound`; bounded allocation sized by bound, extent written, contiguous-in-extent strides.

**Checkpoint:** invariants 2, 3, 6, 7.

### Phase 6 — Corpus and gates

- Generator flag emitting bounded tensors, `as` widenings, and both narrowings; corpus-content test.
- Determinism gate over the flagged corpus at 1/8/32/48 threads and with rayon off the path.
- Demo no-decline gate.

**Checkpoint:** invariants 2, 8, 9.

### Phase 7 — Remove `DynTensor`

- Deprecation diagnostic with suggested replacement; mechanical rewrite of every site (demos, tests); delete.

**Checkpoint:** grep finds no `DynTensor`; all invariants.

### Phase 8 — Docs

- Language docs: the lattice, `as` widening, both narrowings, the placement rule, bound sources.
- One test file per diagnostic code. Update the overview and README example if they mention `DynTensor` or the old tensor syntax.

______________________________________________________________________

## 10. Test plan (minimum)

- Every row of the 5.1 ordering, positive and negative, through `as`.
- Plain mismatch without `as` at each site (argument, assignment, return, field store) → error with help line.
- `narrow` with each admissible fact source; one inadmissible (interprocedural) case that must **not** be accepted.
- `narrow_checked` on host (pass and trap) and in a device region (E-C).
- Unbounded on device (E-D); unbounded on host (ok).
- Bounded allocation: size equals the bound-derived figure; extent stored; strides contiguous in extent.
- Machine file as config: two compilations, same program, different `--machine`, different tensor-type GIDs, each invisible to the other.
- Differential (flat vs oracle) for `TensorWiden`, `TensorNarrow`, `TensorNarrowChecked`.
- Corpus-content: flag on emits the constructs, flag off emits none.
- Determinism: flagged corpus, byte-identical at 1/8/32/48 and with rayon off.
- Regression: 67/23/0; 64/64; 442,368 B; demos do not decline.

______________________________________________________________________

## 11. Open questions for the author

1. Naming: `narrow` / `narrow_checked`, or names that fit the existing grammar better?
1. Should `as _` infer its target from the expected type at the site (the same contextual mechanism literals use), so `rmsnorm(a as _)` keeps the conversion visible without restating a long type? This does not reintroduce implicit conversion — the `as` is still written — but it is a grammar decision.
1. Is a config/serving declaration construct in scope for this RFC, or do bounds come from `const`s first with configs following?
1. Should widening between two bounds (`?<=2048` → `?<=8192`) warn when the tighter bound is discarded at a site that will then allocate at the wider one? With explicit `as` the widening is visible, so this is now a lint question rather than a safety one.
