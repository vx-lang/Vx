# Heterogeneous Target Gap Analysis: TPU, B200, Rubin

**Status:** analysis, pre-release
**Scope:** what Vx can and cannot do against three real accelerator families, and what must land before the release that claims heterogeneous programming as the value proposition.

## 1. Why this document exists

Vx's day-one claim is *"One Language, Every Core."* This document tests that claim against three concrete targets — Google TPU (v4/v5-class), NVIDIA B200 (Blackwell), and NVIDIA Rubin / R200 — and converts the result into a prioritized work list with worked examples.

The finding in one sentence: **Vx already does something real and differentiated that nothing else does, and it is not the thing the README currently promises.** Closing that gap is mostly a positioning decision plus a handful of focused pieces of engineering, not a rewrite.

Section 9 gives a guided example for every work item — today's behaviour, proposed syntax, the diagnostic the compiler should produce, and where in the tree it lands. Work items carry stable IDs (`P0-1`, `P1-2`, …) used consistently across §7, §9 and §10.

## 2. Method

Every capability claim below was produced by running the checked-in compiler (`target/debug/vxc`) against hand-written `.vx` models of each machine, not by reading the source and inferring. Reproductions are in Appendix A; source evidence is indexed in Appendix B. Where a claim rests on reading code rather than executing it, it is marked *(static)*.

TPU architectural details are drawn from the Mosaic/XLA analysis in *TPU Through Open Source*. B200 details are public NVIDIA documentation. Rubin details are limited to what was publicly announced; its SM microarchitecture is **not** modeled here and should not be guessed at.

## 3. Executive verdict

| | TPU (v4/v5) | B200 (Blackwell) | Rubin / R200 |
|---|---|---|---|
| **Describe the memory hierarchy** | ✅ works today | ✅ works today | ✅ works today (system level) |
| **Verify placement & staging** | ✅ works today | ✅ works today | ✅ works today |
| **Express a fast kernel** | ❌ no async DMA | ❌ no warps, no async | ❌ same |
| **Backend to real silicon** | ❌ closed (libtpu) | ✅ **open** (NVPTX, upstream LLVM) | ✅ open, same path |
| **Realistic day-1 role** | Pallas/Mosaic frontend | placement layer over CUTLASS/cuBLAS; only target where a solo end-to-end demo is achievable | disaggregation modeling |

The two NVIDIA targets and the TPU fail for **opposite** reasons, and that is the most useful fact in this document:

- **TPU:** the language fits unusually well; the backend is a closed door. `libtpu.so` and XLA:TPU are proprietary, so `VxHardwarePlugin::lower_to_binary` cannot be implemented outside Google. The only open seam is emitting Mosaic MLIR (`stable_mosaic` versioned serialization) and handing it to jaxlib.
- **B200:** the backend is wide open — NVPTX is upstream LLVM, PTX is documented, `ptxas` and the driver API are public — but the *execution* model fits worse, because Blackwell performance is warp specialization and async pipelines, and Vx has neither.
- **Rubin:** too early to model the chip. But **Rubin CPX is the most Vx-shaped hardware NVIDIA has announced** (see §8.3), and that is a positioning opportunity, not an engineering one.

## 4. The verified capability baseline

This is the part worth protecting. All of it works today, on all three machines, with no compiler changes.

### 4.1 Machines are user-declarable

`Topology::Custom` and `MemorySpace::Custom` mean new cores and new memories are source-level declarations, not language edits. A TPU's TensorCore/SparseCore split and its VMEM/SMEM/CMEM/HBM spaces, or a Blackwell SM's SMEM/TMEM/L2/HBM3e, are written in `.vx` and require no work in `vxc`.

`MemoryDecl` carries `within:`, `capacity:`, `bandwidth:`, `granule:`, `managed:`, `scope:`, and `overcommit:` — enough to express the properties that actually matter on both machines. In particular `managed: explicit` is precisely the "no cache coherence, you move it yourself" contract that both the TPU scratchpads and Blackwell's SMEM/TMEM impose.

### 4.2 Multi-hop transfers are routed and staged automatically

Given only declared edges, `transfer(a, Memory::VMEM)` on a host tensor emits **two** `vx.transfer` ops (CPU_DRAM → HBM → VMEM). On the B200 model, `transfer(b, Memory::TMEM)` emits **three** (CPU_DRAM → HBM3e → SMEM → TMEM). The compiler found both routes from the declared cost graph. Nobody wrote the staging.

### 4.3 Scratchpad allocation happens at compile time

Each hop into a `granule`'d space gets a statically assigned, granule-rounded offset:

```mlir
// TPU: 128x128 f32 tiles into VMEM (4 KB granule)
{granule = 4096,  offset = 0,     slots = 16, scope = "sm", space = "VMEM", within = "HBM"}
{granule = 4096,  offset = 65536, slots = 16, scope = "sm", space = "VMEM", within = "HBM"}

// B200: 64x64 f32 into SMEM (1 KB granule), then TMEM (16 KB granule)
{granule = 1024,  offset = 16384, slots = 16, scope = "sm", space = "SMEM", within = "L2"}
{granule = 16384, offset = 0,     slots = 1,  scope = "sm", space = "TMEM", within = "L2"}
```

That last line is a compile-time `tcgen05.alloc`. Note that [`tests/frontend/pass/memory_hierarchy.vx`](../../tests/frontend/pass/memory_hierarchy.vx) — 192 GB HBM at 8 TB/s, 228 KB SMEM, 256 KB TMEM at 16 KB granule — **already is a B200**; 128 lanes × 32 columns × 4 B is exactly Blackwell's TMEM allocation granularity.

### 4.4 The canonical accelerator errors are compile-time errors

```
E6009: transferred tensor needs 268435456 bytes but memory space 'VMEM' has capacity 67108864 bytes
E6010: the working set placed in 'VMEM' (2 tiles) ... sums to 536870912 bytes, over its 67108864 byte capacity
E6007: memory space 'RF' (262144 bytes) is larger than its parent 'SMEM' (233472 bytes)
```

"Your tile doesn't fit in VMEM" is the single most common Pallas failure and it is a type error here. E6007 caught a genuine modeling mistake made while writing the B200 model (register file placed inside SMEM).

### 4.5 Shape polymorphism is already verified

Const generics are monomorphized **before** the capacity check, so shape-generic code is fully checked:

```rust
fn stage<const N : i32>() -> i32 {
  let a = Tensor<f32>([N, N]);
  let va = transfer(a, Memory::VMEM);
  return 0;
}
fn main() -> i32 { return stage<8192>(); }   // E6009: needs 268435456 bytes, capacity 67108864
```

This matters more than it looks: most of what people mean by "dynamic shapes" is really shape *polymorphism*, and that case is solved. See §6.

### 4.6 A formal seam checker already exists

The z3/QF_BV seam engine proves that a relaxed cross-device transfer cannot publish a stale buffer, with a concrete counterexample:

```
Error[E6004] relaxed transfer of 't' across the NPUHBM -> LocalSRAM seam violates the boundary
  contract ('t' == 1): the buffer carries no synchronizing release, so a consumer may read it stale
  note: z3 counterexample: ((tag_t #b11) (val_t #x0000000000000000))
```

This is the missing-`mbarrier`-wait bug and the missing-DMA-wait bug, proved. **Vx owns the hard half of the async-safety problem already** — and, as §6 shows, the hard half of the dynamic-shape problem too. It just has no syntax to attach either one to.

### 4.7 Assessment

Nothing else in the field does §4.2–§4.6. Pallas has no static capacity checking; `pl.BlockSpec` mistakes surface at trace time or silently pad. CUDA has no placement verification at all. Mojo has ownership but no memory-space algebra. This is a real, defensible, demonstrable-today differentiator — **and it is a placement-and-verification story, not a kernel-generation story.**

## 5. Bugs and regressions found

Fix these regardless of anything else in this document.

| # | Issue | Severity |
|---|---|---|
| B1 | **Subspace metadata is dropped on the flat path.** `offset`/`slots`/`granule`/`scope`/`space`/`within` appear on `vx.transfer` only under `--legacy-codegen`. On the default flat path they vanish silently. [`subspace_schedule.vx`](../../tests/middle_end/pass/subspace_schedule.vx) passes only on the AST path. | **Ship-blocker.** The flagship feature of §4.3 is invisible on the default code path. → **P0-1** |
| B2 | **Use-site seam proof does not reach declared spaces.** A `relaxed` edge in a user `Topology` yields `W1027` at the *declaration*. The built-in `to_device_relaxed`/`to_gpu_relaxed`/`to_sram_relaxed` intrinsics yield `E6004` at the *use site* with a per-buffer value contract and a counterexample. Declared spaces get the blunt version. | High — backwards. Declared spaces are the entire point of first-class memory. → **P0-3** |
| B3 | **Dynamic shapes silently skip verification.** `check_capacity` returns early when any dim is non-literal, with no diagnostic. A 40 GB tensor into a 64 MB VMEM compiles clean, and the emitted `vx.transfer` still carries `capacity`/`granule` while carrying no `offset`/`slots`. | **Ship-blocker.** Users believe they are covered when they are not. → **P0-4** |
| B4 | `Topology::Slice(..)` collapses to dispatch id 900 and does not round-trip as a runtime value, per [`docs/topology_representation.md`](../topology_representation.md). | Medium — blocks rack-scale. → **P2-3** |

## 6. Shapes: the axis that actually matters

PyTorch has dynamic shapes; XLA requires fully static ones; the natural reading is that Vx is drifting toward XLA. That framing is wrong for Vx, and getting it right is both a correctness fix and the strongest available positioning against XLA.

### 6.1 What Vx does today

Three tiers, verified by execution:

| Shape source | Example | Capacity checked? |
|---|---|---|
| Literal | `Tensor<f32>([8192, 8192])` | ✅ E6009 / E6010 |
| Const generic | `stage<8192>()` → `Tensor<f32>([N, N])` | ✅ monomorphized first |
| Runtime value | `fn k(n : i32) { Tensor<f32>([n, n]) }` | ❌ **silent no-op** |

`static_tensor_bytes` requires every dim to be a literal `Expr::Number` ([`src/hir/memory.rs:47`](../../src/hir/memory.rs#L47)); anything else returns `None` and `check_capacity` returns early. `k(100000)` — 40 GB into a 64 MB space — compiles without a word. Worse, the op it emits *asserts* the placement it never verified:

```mlir
"vx.transfer"(%2) {capacity = 67108864, granule = 4096, scope = "sm", space = "VMEM", within = "HBM"}
// capacity and granule stamped on; no `offset`, no `slots` — the allocator produced nothing
```

### 6.2 The reframe: bytes, not shapes

XLA needs static shapes because it wants *exact* shapes for kernel selection and fusion. **Vx's guarantee does not need shapes — it needs byte bounds.** Every check in §4.3–§4.4 is arithmetic over `bytes = f(shape, dtype, layout)`. Proving a KV tile fits in VMEM requires `seq_len <= 4096`. It never requires knowing that `seq_len` is 1723.

That is a far weaker requirement than XLA's, and it matches how accelerator kernels are actually written. Pallas kernels operate on fixed block shapes and mask the ragged tail. Flash attention tiles at a fixed block size and masks. Mosaic's `assume_multiple` is literally a bound hint. **The tile is static; the grid is dynamic.** The dynamism lives in the loop trip count, which is precisely where Vx can afford to let it live.

### 6.3 Four different things called "dynamic shapes"

Treating these as one problem is what makes the topic feel intractable. They have different answers:

| Kind | Example | Vx answer |
|---|---|---|
| **Shape polymorphism** | one kernel, many sizes | ✅ const generics, works today (§4.5) |
| **Bounded dynamism** | `seq_len` varies, `<= 4096` | **P1-1** — bound in the type, check against the bound |
| **Data-dependent** | MoE routing, top-k, NMS | **R-1** — declared capacity + forced overflow policy |
| **Structural** | network changes per step | out of scope; this is the eager penalty, and correctly so |

Bounded dynamism is ~90% of real serving need and is entirely tractable. Data-dependent is the genuinely hard case and has a Vx-shaped answer nobody else can express (§9.13).

### 6.4 The leverage: the solver is already there

Bound propagation over shape arithmetic is QF_LIA — strictly easier than the QF_BV staleness obligations [`hir::seam::Solver`](../lang/seam_obligations.md) already discharges at ~1.8 ms marginal per seam. `where` already exists as a keyword and already carries constraints (`where Transfer<A, B>`, [`src/parser/decl.rs:118`](../../src/parser/decl.rs#L118)), so the grammar extends rather than expands.

Two of the three largest open questions in this document — async safety (§7.1) and bounded shapes (P1-1) — collapse onto the same infrastructure.

### 6.5 Can Vx offer PyTorch-like programmability?

No, and it should not try. PyTorch's programmability *is* eager execution plus Python; chasing it costs AOT, topologies-as-types, and every guarantee in §4. The README's "eager penalty" framing is the right call.

But the thing users actually miss is rarely `print(x.shape)` mid-forward — it is **not recompiling for every shape**. That is shape polymorphism (solved) plus bounded dynamism (P1-1).

Worth noting that PyTorch is converging on the same point from the opposite direction: `torch.compile(dynamic=True)` uses symbolic shapes with recorded **guards**, recompiling when a guard fails. Same problem, weaker answer — a guard is a runtime check whose failure mode is a silent recompile and a performance cliff. Vx would reach the same programmability with a compile-time proof. **Guards versus proofs** is a sharp framing and one Vx wins.

## 7. Cross-cutting gaps, ranked

These affect all three targets. Ordering is by "how much does this block the value proposition." Worked examples for each are in §9.

### 7.1 No async transfer tokens — the single common denominator → P0-2

TPU needs it for DMA double-buffering. B200 needs it for TMA, `mbarrier`, and `tcgen05.mma` (all asynchronous; `tcgen05.mma` is issued by one thread and completes into TMEM later). Rubin needs it for KV-cache migration across a disaggregated rack. **It is the same feature every time.**

`transfer` is a blocking move. There is no `async`, `await`, `wait`, or `semaphore` token in the lexer *(static)*, and [`ROADMAP.md`](../../ROADMAP.md) §1 lists cross-topology synchronization as unimplemented. On a TPU this means correct-but-slow; on Blackwell it means you cannot express the machine at all.

The shape is small: `transfer` returns a linear token, `wait` consumes it, and the seam engine proves you did not skip the wait. `Type::is_linear()` already exists ([`src/syntax/types.rs:184`](../../src/syntax/types.rs#L184)), so "dropped token" is an error the borrow checker can already raise.

### 7.2 No layout in the type system → P1-2

`Type::Tensor(ElementType, Vec<Expr>, Option<Topology>)` has element type, shape, and placement — no tiling *(static)*. Unrepresentable as a result: the TPU's 8 × 128 vreg geometry, sublane/lane alignment and sub-32-bit packing; B200's TMEM 128-lane × 512-column geometry, SMEM swizzle and TMA box dimensions.

The sharpest irony in the analysis: "your shape is misaligned and you will lose 4–8×" is exactly the shift-left error Vx exists to give, and the type has no slot to put it in. [`docs/tensor_layout_conversions.md`](../tensor_layout_conversions.md) explicitly rejected `Layout::Permuted` in favour of reshape/transpose chaining; that door should be reopened.

### 7.3 No execution model below `spawn` → P2-2

`spawn on` is a *placement*; `for` is sequential. No grid, block, warp, lane, or subcore, and no iteration-space annotation.

- TPU: Pallas's `DimensionSemantics` (`PARALLEL` / `ARBITRARY` / `CORE_PARALLEL` / `SUBCORE_PARALLEL`) is what licenses DMA/compute overlap.
- B200: peak throughput requires warp specialization — producer warps issuing TMA, consumer warps issuing `tcgen05.mma`, rendezvousing on mbarriers. You cannot write a Blackwell GEMM in a language with no warps, and unlike the TPU you cannot delegate it, because there is no XLA doing it for you.

Largest single piece of work here, and the one most safely deferred.

### 7.4 The `Scope` ladder is too short, and `within:` is the wrong shape for lateral sharing → P1-3

`Scope` is `Device` / `Sm` / `Cta` / `Thread` *(static)* — pre-Hopper. Verified: `scope: cluster` is a parse error. B200 needs clusters and DSMEM (a CTA addressing a *peer* CTA's SMEM); the TPU's `VMemShared` (cross-subcore) has the same lateral shape. `within:` is a strict containment tree with capacity checks; neither is nested.

### 7.5 Low-precision numerics are absent, and the type set does not scale → P1-4

`ElementType` has `F16`/`F32`/`F64`/`BF16` and integers down to `I4`/`U4` — **no FP8 at all**, let alone FP6/FP4, and no notion of a microscaled block *(static)*. NVFP4 (16 values sharing an E4M3 scale) is the entire reason Blackwell exists. This is the marketing surface of every accelerator shipping in 2026 and it is unrepresentable.

The structural problem behind it is worse than the missing rows. Classical languages fixed their primitives at 1/8/16/32/64 and stopped; accelerators have reopened that question and will keep reopening it. In Vx, adding **one** element type today touches 10 sites across 6 files, nine of which are restatements of the same fact — and two of those restatements already disagree (`element_bits(I4) = 4` bits densely packed, `scalar_size_align(I4) = 1` byte padded; both correct, neither documented as such). `Topology` and `MemorySpace` were already converted to the "registry behind the enum" pattern; `ElementType` is the one axis still a closed enum with its facts smeared across the tree. Fixing that is a precondition for P1-4, not a follow-on — see [§9.8.1](#981-p1-4a--collapse-elementtype-to-a-descriptor-table).

### 7.6 No "who may address this" facet → P2-4

`visible:` says which topologies can see a space. It cannot say which *operations* may address it. TMEM is accumulator storage reachable only by `tcgen05`; the B200 model in Appendix A `transfer`s ordinary tensors into it, got capacity and granule right and semantics wrong, and nothing complained. TPU `SemaphoreMem` is the same shape.

### 7.7 No escape hatch → P1-5

All of [`ROADMAP.md`](../../ROADMAP.md) §6 is unchecked: no inline asm, no intrinsics *(static)*. Mosaic exposes ~90 ops; Vx emits `linalg` + `vector` + `memref`, perhaps a third. Without an escape hatch, every gap in §7.2–§7.5 is a hard wall rather than a slow path. [`docs/discussions/inline_mlir_macros.md`](inline_mlir_macros.md) already designs the answer.

## 8. Target-specific notes

### 8.1 TPU — good fit, closed door

The language fit is the best of the three: explicit, non-coherent, hierarchically scratchpadded, heterogeneous cores. There is already a test named `AcmeTPU` ([`custom_topology_decl.vx`](../../tests/middle_end/pass/custom_topology_decl.vx)).

The backend is the blocker and it is not an engineering problem. The only open seam is emitting Mosaic MLIR and handing it to jaxlib's Mosaic entry point — Vx links real MLIR (`melior` 0.27 / `mlir-sys` 220, LLVM 22), so building `tpu.*` ops via `OperationBuilder` is mechanically straightforward. That makes Vx **a Pallas alternative with a real type system**, a legitimate and differentiated position: Pallas has none of §4.2–§4.6.

Fallback, already proven in this repo: lower `spawn on(Topology::TPU)` to an FFI call into a JAX-hosted kernel, exactly as `runtime/npu_dispatch.mm` does for the ANE.

### 8.2 B200 — worse fit, open door, and the only achievable end-to-end demo

Beyond the cross-cutting gaps, B200 adds TMEM's non-scratchpad semantics (§7.6), NVFP4 (§7.5), clusters/DSMEM (§7.4), and the NV-HBI die pair presenting as one CUDA device that is NUMA underneath — Vx has no vocabulary for "one topology, non-uniform internal memory."

But the backend is real. `vx.spawn` already lowers to an outlined kernel plus `vx.launch` ([`VxLowering.cpp:37`](../../src/dialect/VxLowering.cpp#L37)) — the right skeleton. `--emit-llvm --target nvptx64` works ([`llvm_backends.vx`](../../tests/optimizations/pass/llvm_backends.vx)), though it retargets the *whole module*: a triple tag, not a GPU backend. There is no `nvvm` or `gpu.module` emission anywhere in `src/` *(static)*.

The path — `vx.launch` → `gpu.launch_func` → `gpu-to-nvvm` → `ptxas` → cubin, with CUTLASS or cuBLAS owning the MMA inner loop over FFI — is ordinary compiler engineering one person can finish. **The only one of the three targets where a real-silicon demo is achievable solo.**

### 8.3 Rubin / R200 — do not model the chip; model the rack

Rubin's SM microarchitecture is not public. Do not guess at TMEM's successor, `tcgen06`, or new numeric formats.

What *is* public and stable is the system shape: HBM4 (~288 GB, ~13 TB/s), NVLink 6, VR200 NVL144, Vera CPU over NVLink-C2C, Rubin Ultra NVL576 in 2027 — and **Rubin CPX**, a separate die for long-context *prefill* on 128 GB of GDDR7, paired with HBM4 Rubin dies for *decode*.

Rubin CPX is the most Vx-shaped hardware NVIDIA has announced. Two accelerators, two memory technologies, one workload, and an explicit KV-cache handoff whose cost determines whether the system is economical. Today that is Python-level disaggregated-serving config in Dynamo with no static checking anywhere. Worked example: §9.12. This needs none of the SM-level detail we lack. It does need B4 fixed.

## 9. Guided examples

One subsection per work item. Each gives **Today** (real, current behaviour), **Proposed** (new syntax, clearly not yet implemented), **Diagnostic** (what the compiler should say), and **Where it lands**.

> All `Proposed` syntax in this section is a design sketch for review, not implemented behaviour.

### 9.1 P0-1 — Carry subspace metadata on the flat path

**Today.** The AST path attaches the scheduler's output; the flat path drops it.

```console
$ vxc subspace_schedule.vx --action emit-mlir --legacy-codegen
"vx.transfer"(%alloc) {capacity = 233472, granule = 16384, offset = 0, slots = 4,
                       scope = "sm", space = "SMEM", within = "GPU_HBM"}

$ vxc subspace_schedule.vx --action emit-mlir          # default flat path
"vx.transfer"(%alloc) <{target_topology = 1727 : i32}>       # everything gone
```

**Root cause.** The decision was deliberate. [`src/codegen/flat.rs:1728`](../../src/codegen/flat.rs#L1728) says:

> *"The extra scheduling attrs the AST adds (`space`, `granule`, …) are discardable metadata and don't affect lowering."*

They are not discardable. They are the output of the sub-space bump allocator, and a device backend needs them to place tiles into VMEM/TMEM. The comment is the bug.

**Proposed.** Port the attribute block from [`src/codegen/lower/tensors.rs:173-235`](../../src/codegen/lower/tensors.rs#L173) into the `Opcode::Transfer` arm of `flat.rs`. The flat lowerer already has the target space (`ins.imm`) and the tensor shape (`ctx.tensors`); it needs the same per-function `subspace_offsets` bump map that `generator.rs` keeps ([`src/codegen/generator.rs:37`](../../src/codegen/generator.rs#L37)), cleared per function.

**Diagnostic.** None — this is a codegen fidelity fix. Verify by making `subspace_schedule.vx` pass without `--legacy-codegen`, and add a companion test asserting both paths emit byte-identical attribute sets.

**Where it lands.** `src/codegen/flat.rs` (transfer arm), `src/codegen/generator.rs` (share the allocator), `tests/middle_end/pass/subspace_schedule.vx`.

______________________________________________________________________

### 9.2 P0-2 — Async transfer tokens

**Today.** `transfer` is a blocking move. Double buffering is inexpressible.

```rust
let va = transfer(a, Memory::VMEM);
compute(va);                          // no way to overlap the next DMA
```

**Proposed.** `transfer_async` returns a linear `Token<T, Memory::X>`; `wait` consumes it and yields the placed tensor.

```rust
fn stream(tiles : &Vec<Tensor<f32, [128, 128]>>) -> void on Topology::TensorCore {
  let mut inflight = transfer_async(tiles[0], Memory::VMEM);
  for i in 1..tiles.len() {
    let next = transfer_async(tiles[i], Memory::VMEM);   // issue before consuming
    let cur  = wait(inflight);                           // token consumed here
    matmul_acc(cur);
    inflight = next;
  }
  matmul_acc(wait(inflight));
}
```

**Diagnostic.** Two new errors, both reusing machinery that exists:

```
E60xx: transfer token for 'inflight' is dropped without `wait`; the DMA may not have completed
  help: bind the result of `wait(inflight)` before the token goes out of scope

E6004: 'cur' is read before its transfer token is waited — a consumer may read stale data
  note: z3 counterexample: ((tag_cur #b11) (val_cur #x00))
```

The first is pure linearity — `Type::is_linear()` already returns `true` for `Tensor`/`Pinned`/`Ref` ([`src/syntax/types.rs:184`](../../src/syntax/types.rs#L184)), so `Token` joins that list and the borrow checker raises it. The second is the existing seam obligation with `Transfer::Relaxed` selected by "token not yet waited" instead of by the `_relaxed` method name.

**Lowering.** Two new dialect ops, `vx.transfer_async -> !vx.token` and `vx.wait(!vx.token)`, with per-target lowering:

| Target | `vx.transfer_async` | `vx.wait` |
|---|---|---|
| TPU | `enqueue_dma` + semaphore | semaphore wait |
| B200 | `cp.async.bulk.tensor` | `mbarrier.try_wait` |
| CPU | `memcpy` | no-op |

**Where it lands.** `src/lexer.rs` (2 keywords), `src/parser/expr.rs` (intrinsic forms), `src/syntax/types.rs` (`Type::Token`), `src/hir/expr.rs` (typecheck, linearity, seam selection), `src/dialect/VxDialect.cpp` + `VxLowering.cpp` (2 ops), `src/codegen/`.

______________________________________________________________________

### 9.3 P0-3 — Use-site seam proof for declared spaces

**Today.** A declared `relaxed` edge produces a blunt warning at the declaration; only the three built-in intrinsics get the per-buffer proof.

```rust
Topology TensorCore5 {
  memory: Memory::TMEM,
  transfer Memory::SMEM -> Memory::TMEM : 5 relaxed
}
fn f() -> i32 {
  let acc = transfer(b, Memory::TMEM);
  spawn on(Topology::TensorCore5) { assert(acc == 0.0); consume(acc); }
  return 0;
}
```

```
Warning[W1027]: topology 'TensorCore5': declared relaxed transfer SMEM -> TMEM does not
                preserve visibility; a consumer may read stale data
```

Fires once, at the declaration, whether or not any transfer crosses that edge, and names no buffer.

**Proposed.** No new syntax. Route declared-edge relaxation through the same path as `to_sram_relaxed`: when a `transfer` resolves to a hop whose `TransferEdge.sync == false`, set `pending_transfer_relaxed` before `check_transfer_seam`, exactly as the intrinsic arm does at [`src/hir/expr.rs:3440`](../../src/hir/expr.rs#L3440).

**Diagnostic.**

```
Error[E6004] at 7:13: relaxed transfer of 'acc' across the SMEM -> TMEM seam violates the
  boundary contract ('acc' == 0.0): the buffer carries no synchronizing release, so a
  consumer may read it stale
  note: z3 counterexample: ((tag_acc #b11) (val_acc #x00))
  help: declare the edge `sync`, or wait on the transfer token (see P0-2)
```

Keep W1027 as the declaration-time smell; add E6004 at the use site. Multi-hop already propagates the relaxed marker ([`src/hir/expr.rs:1730`](../../src/hir/expr.rs#L1730)), so a relaxed hop anywhere in a staged route taints the route.

**Where it lands.** `src/hir/expr.rs` — the `Transfer` arm around line 1692, plus the edge lookup in the multi-hop rewrite.

______________________________________________________________________

### 9.4 P0-4 — Do not silently skip unverified shapes

**Today.** Silence.

```rust
fn kernel(n : i32) -> i32 {
  let a = Tensor<f32>([n, n]);
  let va = transfer(a, Memory::VMEM);   // 64 MB space; n = 100000 is 40 GB
  return 0;
}
fn main() -> i32 { return kernel(100000); }
```

```console
$ vxc dyn.vx --action emit-mlir        # compiles clean, zero diagnostics
```

**Proposed.** No new syntax; make the existing early-return loud. `check_capacity` currently does:

```rust
let Some(raw) = crate::hir::memory::static_tensor_bytes(elem, dims) else {
    return;                                  // <- silent
};
```

**Diagnostic.**

```
Warning[W10xx] at 2:11: capacity of 'VMEM' not verified — 'a' has a dynamic shape
  note: dimension 0 is the runtime value 'n'
  help: add a bound to check it, e.g. `fn kernel(n : i32) -> i32 where n <= 512`
```

Emit only when the destination space actually declares a `capacity` — otherwise there was nothing to check and the warning is noise. Suppress under an explicit opt-out on the space (`unverified` alongside `overcommit`) for users who genuinely want the escape hatch.

This is small, and it converts a hole users fall into unknowingly into a documented boundary. It should ship even if P1-1 does not.

**Where it lands.** `src/hir/expr.rs::check_capacity` (~line 1325), `src/diagnostic.rs` (new code).

______________________________________________________________________

### 9.5 P1-1 — Bounded dynamic shapes

**Today.** Shapes are literal, const-generic, or unverified (§6.1).

**Proposed.** Two surfaces, the second sugar for the first.

*(a) Bound on a value, via the existing `where` clause:*

```rust
fn stage(n : i32) -> i32 where n <= 512 {
  let a = Tensor<f32>([n, 128]);        // bounded: 512 * 128 * 4 = 256 KB
  let va = transfer(a, Memory::VMEM);   // E6009 checked against the bound
  return 0;
}

fn main() -> i32 {
  stage(384);                            // ok: 384 <= 512 discharged statically
  stage(600);                            // E60xx
  return 0;
}
```

*(b) Bound in the type, for tensors crossing an API boundary:*

```rust
fn attend(q : Tensor<f32, [<=4096, 128]>,
          k : Tensor<f32, [<=4096, 128]>) -> Tensor<f32, [<=4096, 128]>
  on Topology::TensorCore
{ ... }
```

**Semantics.**

- `static_tensor_bytes` becomes `tensor_bytes(elem, dims, bounds) -> Bytes { exact: Option<u64>, upper: Option<u64> }`.
- E6009 (per-tile) and E6010 (working set) check `upper`.
- The granule allocator assigns `slots` from `upper` — worst-case reservation, which is what a real VMEM/SMEM allocator does anyway.
- Codegen is unchanged: `memref<?x128xf32>` plus masking. That already works.

**Diagnostics.**

```
Error[E60xx] at 9:9: cannot prove the bound 'n <= 512' at this call
  note: argument is the literal 600
  help: widen the bound on 'stage', or clamp the argument

Error[E6009] at 3:11: transferred tensor needs up to 2097152 bytes (bound: n <= 4096)
  but memory space 'VMEM' has capacity 67108864 bytes
```

**Discharge.** Reuse [`hir::seam::Solver`](../lang/seam_obligations.md). Bound propagation is QF_LIA — cheaper than the QF_BV obligations already running at ~1.8 ms marginal. Start with a syntactic fast path (literal ≤ literal, parameter with a declared bound) and fall back to z3 only when that fails, mirroring how seam checking is off by default behind `--verify-seams`.

**Where it lands.** `src/parser/decl.rs` (extend the `where` arm at line 118 beyond `Transfer<A, B>`), `src/parser/types.rs` (`<=N` in a dim position), `src/syntax/types.rs` (dim becomes `Dim::{ Exact(Expr), Bounded(Expr) }`), `src/hir/memory.rs` (`tensor_bytes`), `src/hir/expr.rs` (bound environment + call-site discharge).

______________________________________________________________________

### 9.6 P1-2 — Layout in the type system

**Today.** `Type::Tensor(ElementType, Vec<Expr>, Option<Topology>)` — no tiling. A `[129, 128]` tensor on a TPU wastes 127 of 128 lanes on the tail and nothing says so.

**Proposed.** Declare the native tile on the space; check shapes against it.

```rust
Memory VMEM {
  within: Memory::HBM, capacity: 64 MB, granule: 4 KB,
  managed: explicit, scope: sm,
  tile: [8, 128]                     // NEW: sublane x lane vreg geometry
}

Memory TMEM {
  within: Memory::L2, capacity: 256 KB, granule: 16 KB,
  managed: explicit, scope: sm,
  tile: [128, 32]                    // NEW: TMEM lanes x allocation columns
}
```

and, where a value needs a layout distinct from its space's default:

```rust
let w : Tensor<bf16, [256, 512], Layout::Tiled<[8, 128]>>;
```

**Diagnostic.**

```
Warning[W10xx] at 4:11: shape [129, 128] is not a multiple of VMEM's [8, 128] tile
  note: dimension 0 rounds up to 136, leaving 7 of 8 sublanes idle on the tail iteration
  help: pad to [136, 128], or reshape so dimension 0 is a multiple of 8
```

A warning, not an error — misalignment is a performance bug, not a correctness bug, and forcing it to an error would make the feature unusable on real code.

**Sub-byte packing belongs here too.** `element_bits` currently assumes dense packing, but int4 packs 8-per-32-bit-slot on TPU and 2-per-byte on Blackwell — the answer depends on the space, not the type. The `tile:` facet is where the machine-specific packing rule should live; see [§9.8.3](#983-packing-is-a-layout-property-not-a-format-property--see-p1-2). Land these two together or TPU's convention gets baked in as a language-wide constant.

**Where it lands.** `src/syntax/types.rs` (fourth field on `Type::Tensor`; `Layout` enum), `src/syntax/decl.rs` (`tile:` on `MemoryDecl`), `src/parser/decl.rs` + `types.rs`, `src/hir/expr.rs` (check alongside `check_capacity`). Note this reopens the decision in [`docs/tensor_layout_conversions.md`](../tensor_layout_conversions.md) — that document should get a follow-up section rather than being silently contradicted.

______________________________________________________________________

### 9.7 P1-3 — Longer scope ladder and lateral spaces

**Today.**

```console
$ vxc cluster.vx
Error at 2:92: `scope:` expects device/sm/cta/thread, got 'cluster'
```

And `within:` is strict containment, so DSMEM — a CTA addressing a *peer* CTA's SMEM — has no shape to be written in.

**Proposed.** Extend the ladder and add a lateral facet distinct from `within:`.

```rust
// B200: thread block cluster + distributed shared memory
Memory DSMEM {
  within: Memory::L2, capacity: 3648 KB, managed: explicit,
  scope: cluster,                    // NEW level, between sm and device
  shared_with: peers(16)             // NEW: lateral, 16 CTAs address each other
}

// TPU: cross-subcore shared VMEM
Memory VMemShared {
  within: Memory::HBM, capacity: 16 MB, managed: explicit,
  scope: subcore,                    // NEW level, below sm
  shared_with: peers(2)
}
```

`Scope` becomes `Device | Cluster | Sm | Cta | Subcore | Thread`, ordered broadest to narrowest so the existing narrowing check ([`src/syntax/decl.rs:183`](../../src/syntax/decl.rs#L183)) keeps working unchanged.

**Diagnostic.**

```
Error[E60xx]: 'DSMEM' is shared with 16 peers but 'SMEM' (scope: sm) is not;
  a peer-shared space cannot be narrower than the scope it is shared across
```

**Where it lands.** `src/syntax/decl.rs` (`Scope`, `shared_with` on `MemoryDecl`), `src/parser/decl.rs:352` (scope match arms), `src/hir/memory.rs` (narrowing + peer coherence).

______________________________________________________________________

### 9.8 P1-4 — Element types: the descriptor table, then the formats

Classical languages settled on 1/8/16/32/64-bit primitives and stopped. Accelerators have reopened the question — FP8, FP6, FP4, MX block formats — and will keep reopening it. So this item is two pieces: **P1-4a** makes adding a type cheap, **P1-4b** adds the types hardware ships today. Doing them in the other order triples the work.

#### 9.8.1 P1-4a — Collapse `ElementType` to a descriptor table

**Today.** Adding one element type touches **10 sites across 6 files**. Measured with `BF16` as the proxy (the most recent float added):

| Kind | Sites | Real work? |
|---|---|---|
| Width | [`hir/memory.rs:31`](../../src/hir/memory.rs#L31) (bits), [`layout.rs:74`](../../src/layout.rs#L74) (bytes) | table |
| "is it a float?" | [`syntax/types.rs:286`](../../src/syntax/types.rs#L286), [`codegen/flat.rs:59`](../../src/codegen/flat.rs#L59), [`codegen/lower/expr.rs:3055`](../../src/codegen/lower/expr.rs#L3055) | table |
| Name mapping | [`syntax/types.rs:297`](../../src/syntax/types.rs#L297) (Display), [`:324`](../../src/syntax/types.rs#L324) (parse), [`generator.rs:1326`](../../src/codegen/generator.rs#L1326), [`:1428`](../../src/codegen/generator.rs#L1428) | table |
| MLIR type handle | [`generator.rs:1098`](../../src/codegen/generator.rs#L1098) → `self.bf16_ty` | actual work |

**Nine of ten sites are pure data**, and the same fact is restated three or four times. The design does not scale — not because the problem is hard, but because it is bookkeeping.

It is already drifting. The two width tables disagree today:

```rust
element_bits(I4)      = 4        // a [8,8] i4 tensor is 32 bytes — densely packed
scalar_size_align(I4) = (1, 1)   // an i4 struct field is 1 byte — padded
```

Both are correct *for their purpose*, but nothing in the code says so. Adding a format means knowing to touch both **and** knowing they should differ. The failure is silent: a wrong width does not crash, it makes E6009 under-report on precisely the workloads the capacity checker exists for.

**Proposed.** One descriptor table, the same "registry behind the enum" pattern already used for `Topology::Custom` and `MemorySpace::Custom`.

```rust
pub struct ElementDescriptor {
    pub bits: u32,              // storage width; the only fact the capacity story needs
    pub class: Class,           // Float { exp, mant } | SInt | UInt | Bool
    pub surface_name: &str,     // as written in .vx           — "bf16"
    pub mlir_name: &str,        // as emitted to MLIR          — "bf16"; diverges for FP8+
    pub pack: Pack,             // Dense | ByteAligned — see 9.8.3
}
```

`element_bits`, `scalar_size_align`, the three `is_float` predicates, and both name maps all derive from this one table, with their differing purposes documented rather than implicit. Adding a format becomes **one row**.

Note `surface_name` and `mlir_name` are separate fields deliberately: they coincide for `bf16` but not beyond it — Vx would spell it `f8e4m3` where MLIR spells it `f8E4M3FN`.

#### 9.8.2 P1-4b — The formats, and the storage/compute split

**Proposed.** New element types, plus a scale facet on the tensor type.

```rust
// element types (table rows, per 9.8.1)
//   F8E4M3, F8E5M2      (Hopper/Blackwell FP8)
//   F6E3M2, F6E2M3      (Blackwell FP6)
//   F4E2M1              (Blackwell FP4)
//   F8E8M0              (MX scale type — scales only, never data)

// NVFP4: 16 values sharing an E4M3 scale
let w : Tensor<f4e2m1, [4096, 4096], Scale<f8e4m3, block: 16>>;

// MXFP4: 32 values sharing an E8M0 scale
let m : Tensor<f4e2m1, [4096, 4096], Scale<f8e8m0, block: 32>>;
```

**These are storage types, not arithmetic types, and the language should say so.** A memory space is inert — capacity and bandwidth are only ever *checked*. An element type must be *computed with*, and that is where the two concepts diverge:

| | Storage format | Computational type |
|---|---|---|
| What it carries | bit width, packing, scale-plane cost | arithmetic semantics, an MLIR type |
| Who needs it | capacity, staging, layout — Vx's differentiator | codegen |
| Extensible? | yes, safely | no — needs a real MLIR type behind it |

This split is not a compromise; it is an accurate model of the hardware. Nobody performs elementwise `a + b` on FP4 values. NVFP4 is stored 4-bit and computed by the MMA unit accumulating in FP32; the TPU packs bf16 two-per-slot and Mosaic's verifier enforces *"Expected matmul acc to be 32-bit"* — accumulation width is not a choice, it is the only hardware option. Low-precision formats are storage formats that get converted or fed to a matrix unit. They are never a scalar arithmetic domain.

So `f4e2m1` should be legal in a tensor, transferable, capacity-checked, and feedable to a matmul — while scalar arithmetic on it requires an explicit conversion. That is more honest than pretending `f4e2m1 + f4e2m1` has a meaning, and users will not fight it because the hardware already works this way.

**Byte accounting.** `element_bits` alone is not enough — a scaled tensor carries a scale plane, and the §4.4 checks must include it:

```
bytes = ceil(numel * data_bits / 8) + ceil(numel / block * scale_bits / 8)
```

For the NVFP4 example: `4096*4096*4/8 = 8 MB` data + `4096*4096/16*8/8 = 1 MB` scales = **9 MB, not 8**. Getting this wrong makes E6009 under-report on exactly the workloads people care about.

#### 9.8.3 Packing is a layout property, not a format property → see P1-2

Classical primitives assume width is a power of two and byte-aligned. Sub-byte formats break that, and *how* they break it depends on the machine, not the type:

- int4 packs 8-per-32-bit-slot on TPU, 2-per-byte on Blackwell.
- FP6 is the awkward case: 6 bits × 16 = 96 bits = 12 bytes densely, but hardware often stores FP6 padded or in fixed group sizes.

`element_bits` currently returns dense packing, which encodes one machine's convention as a universal truth. The `pack` field in 9.8.1 makes that assumption *named*; the machine-specific answer belongs on the memory space's `tile:` facet from [P1-2](#96-p1-2--layout-in-the-type-system), because it is a property of where the tensor lives.

**Wire these two together when P1-2 lands**, or TPU's int4 packing gets baked in as a language-wide constant.

#### 9.8.4 On future formats — do not build the extension point yet

`Format Foo { bits: 10, ... }` — a user-declared format mirroring `Memory` and `Topology` — is the obvious next step and should **not** be taken now:

- There is no second user. MLIR is itself the extensibility point for float formats and already carries `f8E4M3FN`, `f8E5M2`, `f4E2M1FN`, and `f8E8M0FNU`. If float10 ships in silicon, MLIR gets it and Vx adds a table row.
- The real risk is not the format, it is Vx's copy of the facts drifting from MLIR's — and a user-facing declaration form makes that drift permanent and public.

**But shape the table to admit it.** That is the cheap commitment: `Custom(Symbol)` is not needed on day one, only a representation where adding it later is additive rather than a rewrite. Same discipline that let `Topology::Custom` land without disturbing placement.

**Diagnostic.**

```
Error[E60xx]: matmul operands disagree on block scaling
  note: lhs is Scale<f8e4m3, block: 16>, rhs is unscaled f4e2m1
  help: both operands of a scaled matmul must carry the same block size

Error[E60xx]: 'f4e2m1' is a storage format and has no scalar arithmetic
  help: convert to a computational type (`as f32`), or feed both operands to a matmul
```

**Where it lands.** *P1-4a:* new `src/syntax/element.rs` (descriptor + table), with `src/hir/memory.rs::element_bits`, `src/layout.rs::scalar_size_align`, the three float predicates, and both name maps rewritten as lookups — a pure refactor, verifiable by asserting the existing tests are unchanged. *P1-4b:* table rows, `Scale` facet on `Type::Tensor` in `src/syntax/types.rs`, scale-plane arithmetic in `static_tensor_bytes`, `src/parser/types.rs`, and one melior type handle per new format in `src/codegen/generator.rs`.

______________________________________________________________________

### 9.9 P1-5 — Escape hatch (`mlir!`)

**Today.** No inline asm, no intrinsics. `dynamic_rotate`, `stochastic_convert`, `find_first_set`, `tpu.log`, `assume_multiple`, and the whole `tcgen05` / `cp.async.bulk.tensor` family are unreachable.

**Proposed.** Implement the design already written in [`docs/discussions/inline_mlir_macros.md`](inline_mlir_macros.md) — dialects as namespaces, Vx values as operands, explicit type overrides. No new design work needed; this is a build item.

```rust
// TPU: reach a Mosaic op with no Vx surface
fn rotate_lanes(v : Tensor<f32, [8, 128]>, amt : i32) -> Tensor<f32, [8, 128]> {
  mlir! {
    tpu::dynamic_rotate(v, amt) { dimension = 1 : i32 }
      : (vector<8x128xf32>, i32) -> vector<8x128xf32>
  }
}

// TPU: give the compiler an alignment fact it cannot derive
fn tail(n : i32) -> i32 {
  mlir! { tpu::assume_multiple(n) { multiple = 128 : i32 } : (i32) -> i32 }
}
```

**Why it is P1 rather than P2.** Every remaining gap in §7.2–§7.5 becomes a slow path instead of a wall the moment this lands. It is the cheapest single item in the plan measured by unblocked surface area.

**Where it lands.** `src/parser/` (macro grammar — the macro system already exists per [`macro_system_walkthrough.md`](macro_system_walkthrough.md)), `src/codegen/lower/` (splice operands into a generic op), verification via melior's parser so a malformed op fails at compile time rather than in the backend.

______________________________________________________________________

### 9.10 P2-1 — Complete the NVPTX backend

**Today.** `--emit-llvm --target nvptx64` retargets the whole module — a triple tag, not a GPU backend. No `nvvm`, no `gpu.module`, no launch configuration, no cubin.

**Proposed.** Connect the skeleton that already exists.

```
vx.spawn  ──(exists)──▶  outlined func + vx.launch      VxLowering.cpp:37
vx.launch ──(new)─────▶  gpu.launch_func + gpu.module
          ──(upstream)─▶  -gpu-to-nvvm  ▶  ptxas  ▶  cubin
```

Let CUTLASS or cuBLAS own the MMA inner loop over FFI; Vx owns placement, staging, capacity and launch. That division is what makes this finishable.

**Milestone to aim at.** The Appendix A.2 B200 model, plus a `linalg.matmul` inside the `spawn`, running on real hardware and producing the same result as the CPU path — the same parity bar [`flash_attention_placed.vx`](../../tests/backend/pass/flash_attention_placed.vx) already sets for the JIT.

**Where it lands.** `src/dialect/VxLowering.cpp` (a `LaunchOpLowering` targeting `gpu.launch_func`), `src/driver.rs` (pipeline + `ptxas` invocation), `src/plugin/` (a `CudaPlugin` implementing `VxHardwarePlugin::lower_to_binary`).

______________________________________________________________________

### 9.11 P2-2 — Execution model below `spawn`

**Today.** `spawn on` places; `for` is sequential; nothing in between.

**Proposed, stage 1 — annotated grid.** This is the piece that unlocks pipelining on both machines and is worth doing alone.

```rust
spawn on(Topology::TensorCore) grid([n_batch, n_heads, n_kv])
  semantics([parallel, parallel, arbitrary])
{
  let b = grid_index(0);
  let h = grid_index(1);
  ...
}
```

`arbitrary` on the KV axis is what tells the compiler the online-softmax accumulation is order-dependent while batch and head are free — exactly Pallas's `DimensionSemantics`, and directly usable as `gpu.launch_func` grid dimensions on the NVIDIA side.

**Proposed, stage 2 — roles (sketch, B200-specific).** Warp specialization needs producer/consumer separation inside one placement:

```rust
spawn on(Topology::TensorCore5) grid([m, n]) semantics([parallel, parallel]) {
  role producer(warps: 1) {
    let t = transfer_async(tile(a, m, n), Memory::SMEM);   // P0-2 token
    yield_to(consumer, t);
  }
  role consumer(warps: 4) {
    let s = await_from(producer);
    mma_acc(s, Memory::TMEM);
  }
}
```

Stage 2 is genuinely open design and should not be committed to before P2-1 gives it a backend to land on. Recorded here so the grid syntax in stage 1 is chosen with it in mind.

**Where it lands.** `src/parser/stmt.rs` (spawn grammar), `src/dialect/VxDialect.cpp` (`grid`/`semantics` attributes on `vx.spawn`), `VxLowering.cpp` (grid → `gpu.launch_func` dims).

______________________________________________________________________

### 9.12 P2-3 — Rack scale and the Rubin CPX demo

**Today.** `Topology::Slice(..)` collapses to dispatch id 900 and does not round-trip (B4), so `NPU[0..144]` cannot name an NVL144 domain.

**Proposed, part 1 — fix `Slice`.** Give it a real encoding (base id + extent) so `NPU[0..144]` and `NPU[0..72]` are distinct values, and make it round-trip through `topology_dispatch_id`.

**Proposed, part 2 — the disaggregation demo.** This is the positioning asset; it needs nothing beyond P1-1 and the `Slice` fix.

```rust
Memory GDDR7 { capacity: 128 GB, bandwidth: 2 TB/s }
Memory HBM4  { capacity: 288 GB, bandwidth: 13 TB/s }

Topology RubinCPX {                       // prefill die: compute-bound, cheap memory
  memory: Memory::GDDR7,
  visible: [Memory::GDDR7],
  transfer Memory::CPU_DRAM -> Memory::GDDR7 : 300,
  transfer Memory::GDDR7 -> Memory::HBM4 : 120     // the KV handoff
}

Topology Rubin {                          // decode die: memory-bound, fast memory
  memory: Memory::HBM4,
  visible: [Memory::HBM4]
}

fn serve(prompt : Tensor<f32, [<=131072, 128]>) -> Tensor<f32, [<=131072, 128]> {
  let kv = spawn on(Topology::RubinCPX) { prefill(prompt) };   // built in GDDR7
  let kv_hbm = transfer(kv, Memory::HBM4);                     // explicit, costed
  return spawn on(Topology::Rubin) { decode(kv_hbm) };
}
```

**Diagnostic — the demo's punchline.** Delete the `transfer` line and:

```
Error[E6003] at 12:34: 'kv' lives in GDDR7 but Topology::Rubin sees only [HBM4]
  help: transfer(kv, Memory::HBM4) before decoding — cost 120 on the declared path
```

Today that mistake is a Python config error in Dynamo that surfaces as a wrong answer or a silent PCIe stall. Here it is a compile error naming the cost.

**Where it lands.** `src/arch.rs` (`Slice` encoding), `src/syntax/types.rs`, plus a new `tests/backend/pass/rubin_disaggregated.vx`.

______________________________________________________________________

### 9.13 P2-4 / R-1 — Op-addressability, and the data-dependent shape question

**Today (P2-4).** `visible:` gates topologies, not operations, so the B200 model in Appendix A `transfer`s an ordinary tensor into TMEM and nothing complains — despite TMEM being addressable only by `tcgen05` instructions.

**Proposed (P2-4).**

```rust
Memory TMEM {
  within: Memory::L2, capacity: 256 KB, granule: 16 KB,
  managed: explicit, scope: sm,
  addressable_by: [mma]              // NEW: only matmul accumulation may target this
}

Memory SemaphoreMem {                // TPU
  capacity: 4 KB, managed: explicit,
  addressable_by: [sync]
}
```

```
Error[E60xx] at 24:13: 'TMEM' is addressable only by [mma]; a general transfer cannot target it
  help: accumulate into TMEM with a matmul, or stage through SMEM
```

**Today (R-1).** Data-dependent shapes — MoE routing, top-k, NMS — have no answer, and no bound analysis will produce one, because the extent depends on values rather than on input shapes.

**Proposed (R-1), research direction.** Production MoE already solves this with a capacity factor plus token dropping. That solution is Vx-shaped: the buffer is known, the occupancy is not, and the overflow behaviour is a policy decision currently buried in a YAML float.

```rust
Memory ExpertBuf {
  within: Memory::HBM, capacity: 8 MB, managed: explicit,
  on_overflow: drop                  // NEW: drop | spill(Memory::HBM) | recompute
}
```

```
Error[E60xx]: routing into 'ExpertBuf' has a data-dependent extent and no overflow policy
  help: declare `on_overflow:` on the space, or bound the routing extent
```

Nobody else can express this, because nobody else has capacities in the type system. It is also genuinely unproven — hence R-1, not P-anything. Do not block release on it; do write it down, because it is the strongest long-term claim in this document.

______________________________________________________________________

## 10. Recommended plan before release

The organizing principle: **ship the layer that works, with an honest seam to the layer that doesn't.** Do not ship a half-built kernel language.

### P0 — ship-blockers

| ID | Item | Why now | §9 |
|---|---|---|---|
| **P0-1** | Restore subspace metadata on the flat path (B1) | The flat path is the default; the §4.3 allocator is invisible there. Anyone evaluating Vx sees none of the flagship behaviour. | [9.1](#91-p0-1--carry-subspace-metadata-on-the-flat-path) |
| **P0-2** | Async transfer tokens | The common denominator across all three machines and the precondition for any credible performance claim. | [9.2](#92-p0-2--async-transfer-tokens) |
| **P0-3** | Use-site seam proof for declared spaces (B2) | Once P0-2 exists, this is the best demo in the project: *forgetting the wait is a compile error with a counterexample.* | [9.3](#93-p0-3--use-site-seam-proof-for-declared-spaces) |
| **P0-4** | Warn on unverified dynamic shapes (B3) | Integrity. Small fix; turns a silent hole into a documented boundary. | [9.4](#94-p0-4--do-not-silently-skip-unverified-shapes) |

### P1 — makes the story defensible rather than merely true

| ID | Item | §9 |
|---|---|---|
| **P1-1** | Bounded dynamic shapes | [9.5](#95-p1-1--bounded-dynamic-shapes) |
| **P1-2** | Layout parameter on `Tensor` | [9.6](#96-p1-2--layout-in-the-type-system) |
| **P1-3** | Longer `Scope` ladder + lateral spaces | [9.7](#97-p1-3--longer-scope-ladder-and-lateral-spaces) |
| **P1-4a** | Collapse `ElementType` to a descriptor table (refactor) | [9.8.1](#981-p1-4a--collapse-elementtype-to-a-descriptor-table) |
| **P1-4b** | FP8/FP6/FP4 + block scaling, as table rows | [9.8.2](#982-p1-4b--the-formats-and-the-storagecompute-split) |
| **P1-5** | Escape hatch (`mlir!`) | [9.9](#99-p1-5--escape-hatch-mlir) |

P1-1 is ahead of P1-2 in this ordering — it is what lets the placement story survive contact with real serving workloads, and it shares infrastructure with P0-2. P1-5 is the cheapest item by unblocked surface area and could reasonably be pulled forward.

**P1-4a is a pure refactor and could ship in P0** — it touches no semantics, is verifiable by the existing test suite passing unchanged, and every later format (including ones that do not exist yet) is one row afterward instead of ten edits. **P1-2 and P1-4 must land together or in that order**, because sub-byte packing is a property of the memory space, not the element type ([§9.8.3](#983-packing-is-a-layout-property-not-a-format-property--see-p1-2)).

### P2 — after release

| ID | Item | §9 |
|---|---|---|
| **P2-1** | Complete the NVPTX backend | [9.10](#910-p2-1--complete-the-nvptx-backend) |
| **P2-2** | Execution model below `spawn` (grid, then roles) | [9.11](#911-p2-2--execution-model-below-spawn) |
| **P2-3** | Fix `Slice` (B4) + Rubin CPX disaggregation demo | [9.12](#912-p2-3--rack-scale-and-the-rubin-cpx-demo) |
| **P2-4** | Op-addressability facet | [9.13](#913-p2-4--r-1--op-addressability-and-the-data-dependent-shape-question) |
| **R-1** | MoE capacity policy (research) | [9.13](#913-p2-4--r-1--op-addressability-and-the-data-dependent-shape-question) |

### Explicitly not before release

Mosaic emission for TPU, `tcgen05` intrinsics, NVFP4 microscaling arithmetic, warp roles, and anything requiring a vendor partnership. These are follow-ons whose absence does not undermine the day-1 claim, provided the claim is stated as in §11.

## 11. Positioning: what to claim on day one

This matters as much as the engineering, because the main risk is overclaiming. If the release says *"Vx programs TPUs and Blackwell GPUs,"* the first person who tries will discover there is no kernel backend, and that costs more credibility than a narrower claim would have earned.

**Claim this — it is true, demonstrable today, and unique:**

> Vx is the first language that type-checks *placement and staging* across a heterogeneous memory hierarchy. Declare your machine — cores, memories, capacities, bandwidths, granules, transfer costs — and the compiler routes multi-hop transfers, allocates scratchpad offsets at compile time, and rejects working sets that do not fit, before anything runs. Here is the same program checked against a TPU, a B200, and a disaggregated Rubin rack.

**On shapes, claim this** (once P1-1 lands; until then, claim the first sentence only):

> Vx does not need static shapes. It needs static *bounds*. `seq_len <= 4096` is enough to prove your KV tile fits in VMEM — you never have to know it is 1723. Where `torch.compile` records a guard and recompiles when it fails, Vx discharges a proof at compile time.

**Do not claim:** kernel generation, performance parity, or "replaces CUDA/Pallas." Say plainly that kernels currently reach silicon through existing backends (CUTLASS/cuBLAS, Pallas/Mosaic) over FFI, and that the execution model below `spawn` is the next milestone.

The three-machine demo is the asset. It is cheap to produce — the models in Appendix A are ~30 lines each — and no other language can run it.

## Appendix A — reproductions

Compiled with `target/debug/vxc <file> --action emit-mlir --legacy-codegen` (the `--legacy-codegen` requirement is bug B1 / P0-1).

### A.1 TPU (v4-class; figures representative)

```rust
Memory HBM  { capacity: 32 GB, bandwidth: 1200 GB/s }
Memory CMEM { within: Memory::HBM, capacity: 128 MB, bandwidth: 6 TB/s, managed: explicit }
Memory VMEM { within: Memory::HBM, capacity: 64 MB, granule: 4 KB, managed: explicit, scope: sm }
Memory SMEM { within: Memory::HBM, capacity: 1 MB, managed: explicit, scope: sm }

Topology TensorCore {
  memory: Memory::VMEM,
  visible: [Memory::VMEM, Memory::SMEM, Memory::CMEM, Memory::HBM],
  transfer Memory::CPU_DRAM -> Memory::HBM : 300,
  transfer Memory::HBM -> Memory::CPU_DRAM : 300,
  transfer Memory::HBM -> Memory::VMEM : 50,
  transfer Memory::VMEM -> Memory::HBM : 50,
  transfer Memory::HBM -> Memory::CMEM : 40
}
Topology SparseCore {
  memory: Memory::VMEM,
  visible: [Memory::VMEM, Memory::HBM],
  transfer Memory::HBM -> Memory::VMEM : 30
}

fn main() -> i32 {
  let a = Tensor<f32>([128, 128]);
  let b = Tensor<f32>([128, 128]);
  let va = transfer(a, Memory::VMEM);
  let vb = transfer(b, Memory::VMEM);
  spawn on(Topology::TensorCore) { let x = 1; }
  return 0;
}
```

Result: two-hop staging per tensor, VMEM offsets 0 and 65536 at 16 granule slots each. Enlarging the tiles to `[8192, 8192]` produces E6009 (per-tile) and E6010 (cumulative working set).

### A.2 B200 (Blackwell)

```rust
Memory HBM3e { capacity: 192 GB, bandwidth: 8 TB/s }
Memory L2    { within: Memory::HBM3e, capacity: 126 MB, bandwidth: 20 TB/s }
Memory SMEM  { within: Memory::L2, capacity: 228 KB, bandwidth: 128 B/cyc,
               granule: 1 KB, managed: explicit, scope: sm }
Memory TMEM  { within: Memory::L2, capacity: 256 KB,
               granule: 16 KB, managed: explicit, scope: sm }
Memory RF    { within: Memory::L2, capacity: 256 KB, managed: explicit, scope: thread }

Topology TensorCore5 {
  memory: Memory::TMEM,
  visible: [Memory::TMEM, Memory::SMEM, Memory::L2, Memory::HBM3e],
  transfer Memory::CPU_DRAM -> Memory::HBM3e : 300,
  transfer Memory::HBM3e -> Memory::CPU_DRAM : 300,
  transfer Memory::HBM3e -> Memory::SMEM : 40,
  transfer Memory::SMEM -> Memory::TMEM : 5,
  transfer Memory::TMEM -> Memory::SMEM : 5
}

fn main() -> i32 {
  let a = Tensor<f32>([64, 64]);
  let b = Tensor<f32>([64, 64]);
  let sa = transfer(a, Memory::SMEM);
  let acc = transfer(b, Memory::TMEM);
  spawn on(Topology::TensorCore5) { let x = 1; }
  return 0;
}
```

Result: three-hop staging into TMEM with per-stage granule assignment. Declaring `RF` `within: Memory::SMEM` (as originally written) correctly fails with E6007. Marking the SMEM→TMEM edge `relaxed` produces W1027 at the declaration — see P0-3 for why that should be an E6004 at the use site.

### A.3 Negative results

```console
$ vxc cluster.vx                 # Memory DSMEM { ..., scope: cluster }
Error at 2:92: `scope:` expects device/sm/cta/thread, got 'cluster'

$ vxc dyn.vx                     # Tensor<f32>([n, n]) with n = 100000 into a 64 MB VMEM
                                 # (no output — compiles clean; see B3 / P0-4)
```

### A.4 Const generics are checked

```rust
fn stage<const N : i32>() -> i32 {
  let a = Tensor<f32>([N, N]);
  let va = transfer(a, Memory::VMEM);
  return 0;
}
fn main() -> i32 { return stage<8192>(); }
```

```
Error[E6009]: transferred tensor needs 268435456 bytes but memory space 'VMEM' has capacity 67108864 bytes
```

## Appendix B — evidence index

| Fact | Location |
|---|---|
| `Topology::Custom`, `MemorySpace::Custom` | [`src/syntax/types.rs:41`](../../src/syntax/types.rs#L41), [`:55`](../../src/syntax/types.rs#L55) |
| `MemoryDecl` fields | [`src/syntax/decl.rs:207`](../../src/syntax/decl.rs#L207) |
| `Scope` = Device/Sm/Cta/Thread | [`src/syntax/decl.rs:188`](../../src/syntax/decl.rs#L188) |
| `Management` = Explicit/Cached | [`src/syntax/decl.rs:175`](../../src/syntax/decl.rs#L175) |
| `Topology` decl parser (`memory:`/`visible:`/`transfer` edges) | [`src/parser/decl.rs:196`](../../src/parser/decl.rs#L196) |
| `where` clause already exists (`Transfer<A, B>`) | [`src/parser/decl.rs:118`](../../src/parser/decl.rs#L118) |
| `Type::Tensor` — no layout slot | [`src/syntax/types.rs:149`](../../src/syntax/types.rs#L149) |
| `Type::is_linear()` — linearity machinery for tokens | [`src/syntax/types.rs:184`](../../src/syntax/types.rs#L184) |
| `ElementType` — no FP8/FP6/FP4 | [`src/syntax/types.rs:126`](../../src/syntax/types.rs#L126) |
| Element width restated in two tables that disagree on I4 | [`src/hir/memory.rs:31`](../../src/hir/memory.rs#L31) (4 bits, dense) vs [`src/layout.rs:67`](../../src/layout.rs#L67) (1 byte, padded) |
| The other 8 per-type match sites (float predicates, name maps, melior handle) | [`types.rs:286`](../../src/syntax/types.rs#L286), [`:297`](../../src/syntax/types.rs#L297), [`:324`](../../src/syntax/types.rs#L324), [`flat.rs:59`](../../src/codegen/flat.rs#L59), [`lower/expr.rs:3055`](../../src/codegen/lower/expr.rs#L3055), [`generator.rs:1098`](../../src/codegen/generator.rs#L1098), [`:1326`](../../src/codegen/generator.rs#L1326), [`:1428`](../../src/codegen/generator.rs#L1428) |
| `static_tensor_bytes` requires literal dims | [`src/hir/memory.rs:47`](../../src/hir/memory.rs#L47) |
| `check_capacity` silent no-op on dynamic shapes | [`src/hir/expr.rs:1307`](../../src/hir/expr.rs#L1307) |
| Subspace attrs attached (AST path only) | [`src/codegen/lower/tensors.rs:173`](../../src/codegen/lower/tensors.rs#L173) |
| Flat path discards them (with rationale comment) | [`src/codegen/flat.rs:1728`](../../src/codegen/flat.rs#L1728) |
| Per-function bump allocator | [`src/codegen/generator.rs:37`](../../src/codegen/generator.rs#L37) |
| Dispatch ids; `Slice` → 900 | [`src/arch.rs:154`](../../src/arch.rs#L154), [`docs/topology_representation.md`](../topology_representation.md) |
| `vx` dialect ops (spawn/transfer/yield/launch/return) | [`src/dialect/VxDialect.cpp`](../../src/dialect/VxDialect.cpp) |
| `vx.spawn` → outlined kernel + `vx.launch` | [`src/dialect/VxLowering.cpp:37`](../../src/dialect/VxLowering.cpp#L37) |
| Seam obligations (z3, QF_BV, E6004) | [`docs/lang/seam_obligations.md`](../lang/seam_obligations.md), [`src/hir/expr.rs:1483`](../../src/hir/expr.rs#L1483) |
| Declared relaxed edge → W1027 only | [`src/hir/expr.rs:1191`](../../src/hir/expr.rs#L1191) |
| Relaxed marker propagates across multi-hop | [`src/hir/expr.rs:1730`](../../src/hir/expr.rs#L1730) |
| Vendor plugin contract | [`src/plugin/hardware_trait.rs:56`](../../src/plugin/hardware_trait.rs#L56) |
| NVPTX triple support | [`src/driver.rs:907`](../../src/driver.rs#L907), [`tests/optimizations/pass/llvm_backends.vx`](../../tests/optimizations/pass/llvm_backends.vx) |
| Real MLIR bindings (melior 0.27 / mlir-sys 220) | [`Cargo.toml:30`](../../Cargo.toml#L30) |
| Subspace scheduler | [`tests/middle_end/pass/subspace_schedule.vx`](../../tests/middle_end/pass/subspace_schedule.vx) |
| B200-shaped memory hierarchy already in tests | [`tests/frontend/pass/memory_hierarchy.vx`](../../tests/frontend/pass/memory_hierarchy.vx) |
| Layout conversions design (rejected `Layout::Permuted`) | [`docs/tensor_layout_conversions.md`](../tensor_layout_conversions.md) |
| Inline MLIR macro design | [`docs/discussions/inline_mlir_macros.md`](inline_mlir_macros.md) |
