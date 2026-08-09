# GPU Llama + FlashAttention, then disaggregated inference (#321)

Plan of record for the demo campaign scoped in #319 and #321, written 2026-08-08 after a
full repo survey. Goal: show that Vx programs a distributed inference system *directly* —
prefill on one machine, decode on another, the KV-cache handoff written in Vx source, and
every placement admitted at compile time against the target's fleet file.

Related issues: #319 (single-GPU Llama + FA over the plugin ABI, the umbrella for M0–M3),
#320 (f16/bf16 scalar codegen), #251 (NVPTX kernel emission, the M6 track), #258 (device
address spaces on memrefs), #224 (declaration import for fleet files), #321 (M4/M5, the
disaggregation umbrella).

## Two run structures

1. **Flexibility** — generality is the metric, not throughput. Vendor libraries own the
   kernels (#319 Tracks A/B1); Vx owns placement, admission, staging, and the transport.
   Ends in a heterogeneous relay across {A100, x86, H100, MI300X}.
1. **Performance** — both-NVIDIA. Vx emits the device kernels itself (#251, generated from
   `flash_attention_v4.vx` source), f16 throughout (#320). The bar is batch-1 latency
   parity with vLLM on the same dtype and model, reported as *cost-of-admission* — not a
   throughput contest against batching machinery this deliberately does not have.

The property that makes the split cheap: **run 2 is run 1 with the kernel provenance
swapped.** Program text, placement annotations, admission, transport, and harness are
shared; only where the device code comes from changes.

## Verified baseline (survey of 2026-08-08)

What exists and is tested:

- The **compile-time half of disaggregation is done**: `spawn on` / `transfer` / `Pinned`
  parse, type-check, capacity-check (E6009/E6010), route transfers over the machine graph
  (`src/arch.rs` Dijkstra, `src/hir/memory.rs` derived costs), and emit costed
  `vx.transfer` ops. `--machine fleet/*.vx` admission runs live with `--diagnostics-json`.
  A prefill/decode-on-different-devices modelling test already exists:
  `tests/frontend/pass/rubin_disaggregated.vx`.
- A rich attention corpus in `tests/backend/pass/`: `flash_attention.vx` (online softmax),
  `flash_attention_conditional.vx` (FA-4 rescaling), `flash_attention_placed.vx` (placed +
  capacity-checked, runs correct on CPU fallback), `flash_attention_v4.vx` (device-bound
  `exp_poly`, vectorized `dot`/slices), backward pass, MQA, GQA.
- Three Llama2 implementations (`tests/backend/pass/llama2.vx`, `llama2_v2.vx`,
  `examples/llama.vx`) plus `stories15M.bin` and `tokenizer.bin` in-tree. **None executes
  in CI**: all are `NO_EXEC` or bitrotted on the #240 numeric-model tightening.
- `std::net` has TCP listener/stream and UDP backed by `stdlib/rust_core`;
  `tests/backend/pass/ffi_tcp_server.vx` executes a bind+drop in the JIT harness.
- dtypes: f32 executes end-to-end; bf16 executes on the AST path; f16 is declaration-only
  until #320; fp8 has no scalar codegen (correctly gated).
- The memalg instrument (`utils/memalg/`) is RunPod-hardened and has run on a rented H100;
  its container discipline (`run_m1.sh` preflight) is the template for GPU pods.

What is missing (the work):

- **Execution is host-CPU only.** `vx.launch` lowers to a host call
  `vx_plugin_dispatch_async(...)` (`src/dialect/VxLowering.cpp`); the only provider is
  `runtime/npu_dispatch.mm`, built on macOS only (`build.rs`). **On Linux nothing provides
  the symbol, so a placed program does not link.** A prerequisite #319 does not list.
- `vx.transfer` executes as host `memref.alloc` + `memref.copy`; the topology id is
  ignored at runtime. All placement machinery is verification and metadata today.
- No CUDA/HIP/NCCL/P2P code anywhere outside the hand-written memalg instrument.
- No test exercises a TCP accept/read/write loop across processes (the single-process
  harness cannot block on accept).
- Vx models **one representative device per declared kind**; a runtime device index falls
  back to device 0 with W1030 (`tests/frontend/fail/topology_index_constant_and_runtime.vx`).

## Milestones

### M0 — Linux bring-up (on the EC2 x86 build box, no GPU code yet)

The first Linux port of the execution path happens on the trusted EC2 build machine, not
the rented pod (see "Build & deployment discipline" below).

- vxc + MLIR/LLVM toolchain building on EC2 (extend `setup.sh` / `config.local`).
- A portable dispatch runtime: `vx_plugin_dispatch_async` with the libffi CPU fallback,
  buildable on Linux (today `build.rs` builds the shim only on macOS and `jit.rs` links it
  macOS-only).
- Un-bitrot the program base: fix the #240 fallout and promote one Llama to an *executing*
  test (drop `NO_EXEC`, `EXPECT:` on generated tokens). Base: `llama2_v2.vx` (idiomatic
  `@` matmul + `&mut f32` slices), not `examples/llama.vx`.
- **Acceptance:** `flash_attention_placed.vx` and the Llama program generate correct
  output on the EC2 CPU, from the packaged artifact rather than the source tree.

**Status: done except the executing-test item (2026-08-08).** Linux port landed in
5e83aa6a; it needed a portable dispatch shim (no provider for
`vx_plugin_dispatch_async` existed off macOS at all), a PATH-resolved `llvm-config`,
`-rdynamic` so `dlsym` can see outlined kernels in ELF, and `symbol-dce` before memref
finalization to stop dead `extern` declarations minting one dangling `@malloc_N` per
allocation site.

The Llama base changed as a result of what running it revealed. `llama2_v2.vx` prints
token pointers rather than text (#323), so `tests/backend/pass/llama2.vx` became the
reference program; 997d3d87 fixed four numerical defects in it and it now matches
llama2.c **token for token** (64/64 ids, greedy, from BOS). Prompted parity is blocked
by a character-level tokenizer, not by the transformer — see #323. The benchmark copies
still carry all four defects (#322).

Remaining: promoting it to an executing CI test needs a per-test environment mechanism.
The `EXPECT:` path JITs in-process, so it cannot set `LLAMA_TOKENS_CONFIG` the way a
`// RUN:` line (which goes through `sh -c`) can, and the default 1000-token budget is
too slow for CI.

### M1 — CUDA plugin: first light on the A100

**Prerequisite found 2026-08-08: the dispatch ABI carries no element type, rank or
shape.** `abiTagForType` described a memref as tag `0` and nothing more; the descriptor
pointer was opaque. cuBLAS selects its kernel by dtype and needs M/N/K, so library
routing could not be written against that boundary — `npu_dispatch.mm` only appeared to
manage it by hardcoding `float *`, rank 2 and 4x4, and by reinterpreting the last three
memref arguments as `[result, a, b]` by convention.

**Built, on the CPU, before renting anything** (all of it testable without a GPU):

- The tag now carries element type and rank (8708f73b), and bit 24 says the argument is
  a *slot* — storage holding a descriptor rather than elements, which is what
  `c = a @ b` publishes its result through (e5d94139). A slot described verbatim is a
  rank-0 memref with no element type, byte-identical to an opaque pointer, so a plugin
  computing the result itself could not have known what to write there.
- The payload names the operation (`kind=matmul`, b6850f27) and which operand plays
  which role (`roles=a:0,b:2,out:5` plus `outkind=slot|buffer`, f0581473). Shapes cannot
  answer the second question: for square operands every assignment of A and B conforms,
  and swapping them yields a plausible matrix of wrong numbers.
- `runtime/vx_dispatch_plan.h` decodes a dispatch into a GEMM plan, with no vendor API
  in it so the decision is testable on any machine, and refuses on every axis where the
  facts do not line up. `tests/runtime/gemm_plan_test.cpp` exercises it against
  hand-built descriptors in the exact shape the compiler emits.
- `runtime/cuda_dispatch.cpp`: `cudaMalloc`/`cudaMemcpy2D` staging, cuBLAS `Sgemm`,
  `Dgemm` and `GemmEx` for f16/bf16 with f32 accumulation, the row-major-as-transpose
  call convention, descriptor writeback into the slot, and the libffi host path for
  everything unrecognised. `build.rs` selects it wherever a CUDA toolkit is installed;
  no GPU is needed to build it, and a build with it runs correctly on a machine without
  one.
- `tests/backend/pass/gpu_matmul_roles.vx` is the acceptance test and is
  backend-independent by construction: the same expected numbers whether the host loop
  nest or cuBLAS computed them. Its operands are chosen so a plugin that swaps A and B,
  or transposes the result, prints different numbers.

**Remaining, and what actually needs the A100:**

- Run the acceptance test on hardware and confirm cuBLAS produces the same numbers as
  the host path.
- cuDNN fused SDPA for the attention kernel (FlashAttention-2 library as fallback).
- `src/plugin/cuda.rs`, mirroring `apple_npe.rs`.
- Replace the Apple path's `[result, a, b]` convention and 4x4 shape match with the same
  decoder now that the facts exist.
- macOS CI stays green: `REQUIRES: cuda` gating in `compile_test.rs` if any test needs
  a GPU (none does so far, which is the point).
- **Acceptance:** placed FA and a placed matmul produce GPU-executed results matching the
  CPU oracle, with no fallback message.

**Found along the way, not blocking M1:** dispatch carries no device index (#331), so a
plugin cannot tell `Topology::GPU[0]` from `GPU[1]`. That is the whole of M4, and the
fix is one more payload entry.

### M2 — Device residency (the performance gate)

- Route `vx.transfer` to `vx_device_alloc`/`vx_device_copy` when the target memref
  carries a device address space (#258 annotations already exist), instead of host
  alloc+copy. Weights transfer once and stay resident; the KV cache lives on-device
  across the decode loop.
- **Acceptance:** transfer-count instrumentation shows weights H2D exactly once per run;
  the decode loop beats the CPU path. **No performance number is published before this
  lands** — naive per-spawn H2D/D2H loses to CPU at stories15M sizes.

### M3 — Run-1 single GPU: placed Llama + FA on the A100

- The serving program with placement annotations: GEMMs and FA inside
  `spawn on(Topology::GPU)`, explicit `transfer`s, admitted via
  `--machine fleet/a100-80.vx` (or `a100-40.vx` — check the rented part with
  `nvidia-smi`). The sibling A100 file gives a genuine admit/reject boundary without
  renting an A10.
- **Acceptance:** token-for-token parity with the CPU reference on stories15M;
  `--diagnostics-json` prediction captured alongside the run. Two-layer configs are free
  (stories15M is 6 tiny layers; `n_layers=2` clamp available for paper diagrams).

### M4 — Run-1 disaggregation: prefill on one machine, decode on another (#321)

Design that sidesteps the one-representative-device limit cleanly: **one process per
role**, each seeing one GPU (`CUDA_VISIBLE_DEVICES` per process; every process is
device 0). One program text; the role (prefill | decode) picked at runtime.

- Prefill computes the KV cache for the prompt, then ships {header: n_layers, kv_dim,
  pos, dtype} + raw KV bytes + last token over `std::net` TCP. The handoff is Vx source —
  that is the claim.
- Decode receives, continues autoregressively.
- **M4a** CPU↔CPU across two processes (no GPU dependency; also closes the TCP
  accept/read/write test gap).
- **M4b** prefill pod → decode pod (A100 ↔ A100 over the real network). Admission per
  role per machine file; predicted KV-transfer cost (NIC edges as modelled in
  `fleet/node-8gpu.vx`) recorded next to the measured one — feeds the memory-algebra
  calibration campaign.
- **Acceptance:** same tokens as the single-machine reference; both processes admitted at
  compile time; KV handoff size and latency logged, predicted vs measured recorded.

### M5 — Run-1 finale: the combo relay {H100 → A100 → MI300X → x86}

- `runtime/rocm_dispatch.cpp`: hipBLAS port of the CUDA plugin, same ABI, same shape.
- The demo: a KV-cache relay — prompt prefilled on H100, decode continues a few tokens on
  A100, then MI300X, then x86 CPU. One unchanged program text, four machine files, four
  compile-time admissions, coherent text across all four.
- Rent H100/MI300X pods only after A100↔A100 works; container scripts staged in advance.
- **Acceptance:** the relay transcript plus the four admission JSONs, archived together.

### M6 — Run-2 performance track (parallel, once M3 lands)

- #251: `vx.launch` → `gpu.launch_func` → NVPTX/cubin via MLIR's gpu pipeline; the FA
  kernel emitted from `flash_attention_v4.vx` source; fused decode kernels
  (RMSNorm+GEMV, RoPE); f16 throughout; tensor cores as stretch.
- Bar: batch-1 latency parity with vLLM on the same dtype/model, reported as
  cost-of-admission.
- Reuses the M3/M4 harness unchanged; only kernel provenance flips.

## Build & deployment discipline

Source does not ship to rented machines. The execution path is already AOT (`src/jit.rs`
runs mlir-translate → opt → llc → clang and executes a native binary), and admission is
compile-time, so a pod never needs the compiler, the repo, or any `.vx` source.

- **Build on the EC2 x86 box** (trusted): vxc, the demo binaries, `cuda_dispatch.so`
  (link against CUDA toolkit stubs — no GPU needed to build), the `--diagnostics-json`
  admission verdicts. Match the pod's OS/glibc (Ubuntu LTS) and keep the CUDA toolkit at
  or below the pod's driver version.
- **Ship an artifact, not a checkout**: one tarball via `scp` — demo binary, runtime
  `.so`s, model assets, run scripts, admission JSONs. Never `git clone` on a pod; no
  `.git`, no credentials, no source. If a stray source file is ever genuinely needed,
  it goes in the archive deliberately, not via the repo.
- **Iteration loop**: fix on EC2, rebuild, re-scp the tarball. The pod only ever runs
  and reports.
- **Teardown**: nuke the working directory when done and terminate the pod; results are
  pulled back (scp) before teardown, following the `utils/memalg/` results-directory
  convention.
- Bonus: the EC2 x86 box doubles as the **x86 leg of the M5 relay** — the machine that
  builds the artifact is also a participant, admitted against its own machine file.

## Risks, priced in advance

- **Residency before benchmarks.** See M2 acceptance; this is the largest single term in
  any delta and has nothing to do with kernel quality.
- **cuDNN SDPA layout mismatch** with our KV-cache layout may force the FA-2 library or a
  layout shim — contained inside M1.
- **Shape-pattern dispatch must be generalized, not extended.** The ANE path's hardcoded
  4x4 matching is a dead end for kernels of many shapes.
- **Admission is not a deployment budget.** The `gpu_memory_utilization` gap documented in
  `fleet/README.md` means verdicts near a SKU ceiling are unresolved; present margins,
  never byte-exact OOM claims.
- **Bitrot is real and unmeasured** (#319): treat any in-tree Llama as a starting point
  only after it executes.

## Open design decisions (not blockers)

1. Role selection: CLI argument vs two thin `main`s over a shared module. Leaning CLI
   argument — one program text is the paper's claim.
1. KV wire format versioning: keep the header dumb; add a magic + version byte so M5's
   cross-vendor relay can evolve dtype without renegotiating.
1. cuDNN fused SDPA vs FlashAttention-2 library for M1: cuDNN ships in the RunPod
   container; decide by what the container images actually have.
1. Whether M4b uses two pods (real network; stronger story) or one 2-GPU pod (simpler
   ops). Leaning two pods — A100 PCIe pods are cheap and the network hop is the claim.

## Hardware schedule

| Phase | Machines |
|---|---|
| M0 | EC2 x86 build box only (no rental) |
| M1–M3 | EC2 (build) + 1x A100 PCIe pod (run artifact; iterate freely, cheap) |
| M4a | EC2 + laptop, or two processes on EC2 (no rental) |
| M4b | EC2 (build) + 2x A100 PCIe pods, briefly |
| M5 | + H100 pod + MI300X pod, short rentals, scripts staged beforehand; EC2 is the x86 leg |
| M6 | the A100 pod again; H100 for the final numbers |

## Out of scope

Multi-GPU sharding execution (the #284/#290 V1 modelling rule stands), continuous
batching / PagedAttention comparisons, and any claim that Vx out-performs vendor kernels.
