# Vx: bundled LLVM, llvm-sass, GLM on GPU, and visibility/versioning design

*Chat export, October 2026*

______________________________________________________________________

## 1. Shipping llvm-project with Vx

**Aditya:** I am planning to ship full llvm-project with Vx. Is this a good idea? https://github.com/vx-lang/Vx/issues/1269 is the issue.

**Claude:**

Yes, bundling a pinned LLVM is the right call, but the issue as written isn't really "full llvm-project," and it shouldn't be. The plan in #1269 is already a trimmed build, and I'd trim it further before it goes into a release archive.

**Why bundling is right.** The LLVM C++ API and MLIR have no stability guarantees, and Enzyme only loads into the exact LLVM it was built against. That alone settles it. Rust, Zig, Julia, Swift and Triton all ship their own LLVM for the same reasons. Depending on whatever LLVM 22 the user happens to have also undercuts the prebuilt-binary story you need for macOS arm64 and Linux x86-64.

**Separate the dev toolchain from the release payload.** The issue builds one install directory for both. FileCheck, `opt`, `llc` and `mlir-translate` belong in the dev/CI toolchain, not in what users download. Since `vxc` already links LLVM and MLIR, it can run the lowering pipeline in-process instead of shelling out. That's faster, gives better errors, and removes four binaries from the archive. The mechanism to use is `LLVM_DISTRIBUTION_COMPONENTS` with the `install-distribution` target. It installs exactly the named components, which is what "build everything Vx runs and nothing else" needs. You can define two distributions, one for dev and one for release.

**Clang is the expensive part.** A clang binary is easily 100+ MB and is most of your archive size. If `src/jit.rs` only uses it as a linker driver, you can ship `lld` instead and invoke it directly, or use the system `cc` the way rustc does. On macOS you need the SDK from Xcode's Command Line Tools to link anyway, so the system toolchain is already present there. If clang actually compiles C (runtime shims, headers), that changes things.

**Static vs. shared LLVM, given Enzyme.** If `vxc` links LLVM statically, which is nicest for a single-binary release and works with your static z3/zstd plan, a dlopen'd `LLVMEnzyme.so` won't find LLVM's symbols unless you export them from `vxc`, and that gets fragile. The cleaner options are linking Enzyme statically into `vxc`, or shipping `libLLVM.so`/`.dylib` and linking both against it. Decide this before the CI work, because it changes what you cache.

**Tie it to your existing launch blockers.** Build LLVM in a manylinux_2_28-style container (or an older glibc), or the bundled libs will raise your glibc floor on their own. Set `LLVM_ENABLE_ZSTD` to use the same static zstd you're linking for `vxc`, so you don't end up with two copies or a stray `libzstd.so` dependency. Turn off `LLVM_ENABLE_TERMINFO`/`LLVM_ENABLE_LIBXML2` too, so the binaries don't pick up host libraries you didn't intend to depend on.

**CI.** Don't build LLVM in the Vx workflow. Build it in a separate workflow, or a small `vx-lang/llvm-build` repo, keyed on a hash of the pinned commit plus the CMake flags. Publish per-platform tarballs as release assets and have `setup.sh` and CI download them. Rust's `download-ci-llvm` and Julia's `LLVM_jll` work this way. The GitHub Actions cache is 10 GB per repo and LLVM builds churn through it, so release assets are more reliable. Use ccache/sccache for the occasional rebuild. You may want assertions on for the dev toolchain and off for release.

**Smaller points.** Pin to a release tag (`llvmorg-22.x.y`) rather than an arbitrary main commit unless you need an unreleased MLIR fix, and keep any local patches as files in the repo. The release source tarball is checksum-verifiable and faster than a sparse clone. Add a `VX_USE_SYSTEM_LLVM` escape hatch, because Homebrew and Linux distro packagers strongly prefer not to vendor LLVM. On targets, host plus NVPTX is right for v0.0.3, but the planned second GPU backend (SPIR-V for Vulkan/SYCL) will mean adding `SPIRV` to `LLVM_TARGETS_TO_BUILD`. The build should make that a one-line change. Licensing is easy: LLVM and Enzyme are Apache-2.0 with the LLVM exception. Just include their `LICENSE.TXT` files in the archive alongside Vx's MIT license.

What does `jit.rs` actually use clang for: only linking, or compiling C as well? That decides whether clang stays in the release.

______________________________________________________________________

## 2. Dialect C++, runtime, and developer build times

**Aditya:** We have C++ programs that are part of the Vx MLIR dialect. We also have runtime code (`runtime/`, `src/dialect/`). A big concern is build times for software developers. I don't think that should increase, only the install time.

**Claude:**

You can meet that constraint: developers download the same prebuilt toolchain CI uses and never build LLVM. "Install" then costs one download of a few hundred MB compressed, not a multi-hour compile.

**What developers' builds look like.** Today they compile the dialect and runtime against a system LLVM 22. After the change they compile against a downloaded, pinned LLVM/MLIR. The per-change compile cost is the same, and so are incremental builds. The only full rebuild comes when you bump the pinned commit, which already happens whenever their system LLVM upgrades. The dev toolchain artifact needs more than the release does:

- MLIR and LLVM headers.
- The CMake package files (`MLIRConfig.cmake`, `LLVMConfig.cmake`).
- `mlir-tblgen` and `llvm-tblgen`. The dialect's ODS `.td` files have to be generated by the tblgen matching the pinned headers. Missing tblgen is the most common way this setup breaks.

**Compile the dialect and runtime with the bundled clang.** Set it as `CMAKE_CXX_COMPILER` in the dev toolchain. That keeps the C++ ABI and standard library consistent with how the prebuilt LLVM was built, so clang stays in the dev toolchain. A recent clang with bundled `lld` links large static LLVM/MLIR archives much faster than GNU ld or gold, and linking is usually where `vxc`'s build time actually goes. Add `-gsplit-dwarf` for debug builds.

**Protect incremental Rust builds.** If `build.rs` drives the C++ build through the `cc` or `cmake` crate, be strict with `rerun-if-changed`: point it at `src/dialect` and `runtime` only. Put sccache in front of both rustc and the C++ compiler.

**Whether clang ships to end users depends on the runtime.** If runtime C++ isn't compiled on the user's machine, prebuild it per target and ship it:

- **Host side:** a static archive.
- **Device side:** LLVM bitcode, linked in-process by `vxc`, the way CUDA's libdevice works.
- **Release payload:** `vxc` plus `lld`, with no clang and no headers.

If the runtime has templates or headers that user programs instantiate, clang has to ship.

**Two small things.** Publish an assertions-enabled variant of the dev toolchain (costs test runtime, not build time). Keep the `config.local` override so anyone working on LLVM itself can point at their own build.

______________________________________________________________________

## 3. Runtime is built only when Vx is built

**Aditya:** Only when Vx itself is built. When users are developing, those runtimes/codegen are already built and linked to `vxc`.

**Claude:**

Then clang can come out of the release entirely. Two things to replace:

**Linking user programs (the `src/jit.rs` path).**

- **In-process JIT:** no linker needed. ORC (or MLIR's `ExecutionEngine`) resolves symbols in-process; the runtime is already in `vxc`'s address space.
- **Executables on disk:** link lld into `vxc` as a library (`lld::lldMain` with ELF and Mach-O drivers). You lose the clang driver's search logic for `crt1.o`/`crti.o`/`crtbegin.o`, libc/libgcc paths, and the macOS SDK and `libSystem` (`xcrun --show-sdk-path`).
- **System `cc` as the linker driver:** what rustc does. Most robust; requires a system C toolchain (build-essential, or Xcode CLT, which macOS needs anyway).

Suggestion: ORC for JIT, system `cc` for AOT in v0.0.3, embedded lld later if needed.

**The MLIR runner libraries.** `ExecutionEngine` normally `dlopen`s `libmlir_runner_utils` and `libmlir_c_runner_utils`. Either ship them next to `vxc` and resolve their path relative to the executable, or link them statically and register their symbols yourself.

**Enzyme** should be linked statically into `vxc`.

**If user AOT executables call into the runtime,** ship it as a prebuilt static archive per target (and bitcode for the device side).

| | Dev toolchain | Release |
|---|---|---|
| Who gets it | Vx contributors and CI | Users |
| Contents | clang, lld, `mlir-tblgen`/`llvm-tblgen`, headers, CMake configs, static libs, FileCheck, optional assertions build | `vxc` (statically linked LLVM, MLIR, dialect, runtime, Enzyme), runtime archives if AOT needs them, license files |
| Size | Several GB unpacked; downloaded once per LLVM pin | Probably tens of MB compressed |
| Build-time impact | None vs. today | None; nothing compiled at install |

`opt`, `llc` and `mlir-translate` move to the dev side once `vxc` runs those passes in-process.

______________________________________________________________________

## 4. The real reason: llvm-sass

**Aditya:** We use clang as a linker. The real reason I'm doing it is that I want to ship the llvm-sass backend at some point.

**Claude:**

That makes bundling LLVM required, not just convenient. A system LLVM will never contain your SASS backend, and LLVM has no stable plugin interface for codegen targets.

**Where the pin lives.** Vx and llvm-sass must build against exactly the same LLVM commit. A separate toolchain-build repo would own the pin, build LLVM + MLIR + Enzyme + llvm-sass together, and publish dev and release artifacts. A toolchain version = (LLVM commit, llvm-sass commit).

**Developer build time stays flat for Vx contributors.** They never compile llvm-sass. If llvm-sass is an LLVM target, each change needs a toolchain rebuild before Vx sees it; treat it like an LLVM bump.

**Keep PTX in the release.** SASS is tied to one SM architecture; PTX is forward-compatible through the driver JIT. Embed both: SASS for validated targets plus PTX fallback, like nvcc fatbins.

**Make it a build-time option in `vxc`** (Cargo feature or CMake option), so `vxc` still builds without llvm-sass.

**CI needs a GPU.** GitHub-hosted runners can't execute SASS. Use a self-hosted runner or a manual hardware gate per claimed SM target.

On clang as the linker: llvm-sass doesn't depend on it. Use the system `cc` as linker driver in v0.0.3, possibly embedded lld later.

______________________________________________________________________

## 5. Keeping llvm-sass inside Vx only

**Aditya:** For now llvm-sass is in my private repo. I don't want to ship it separately — huge risk of Vx becoming an llvm-sass fork and nobody using Vx. llvm-sass + llvm-nvptx together can emit a complete CUDA binary while we gradually emit llvm-sass completely. Timeline unknown.

**Claude:**

The key design rule: Vx should never offer an entry point that turns `vxc` into a general-purpose SASS emitter (arbitrary LLVM IR or PTX in, SASS out). The SASS path should only be reachable from Vx source. The stronger moat: llvm-sass knows exact register counts, shared memory and occupancy at compile time, feeding admission verdicts.

**Hybrid SASS + PTX granularity.** Split per kernel: each entry point plus its transitive device functions goes to one backend.

- **Linking constraint:** device calls can't cross between llvm-sass and ptxas cubins without relocatable device code and nvlink.
- **Fallback:** automatic per kernel; `vxc` records which path each kernel took.
- **Testing:** differential testing — compile both ways, run on GPU, compare.
- **Forward compatibility:** keep PTX embedded alongside SASS.

**Two toolchain variants.** Public dev toolchain without llvm-sass (public CI, fork PRs); release toolchain with llvm-sass built in a protected workflow with a deploy key.

**Open-source story.** A closed component in an MIT binary needs upfront disclosure and its own license notice in the archive.

**Publication.** Artifact-evaluation "available" badges usually require public availability.

______________________________________________________________________

## 6. Open sourcing llvm-sass in a combined toolchain

**Aditya:** I can release it in public as a combined toolchain. Open sourcing Vx has provided value: ~200 stars, invited to present at Paradigm Frontiers 2026. Making Vx the one compiler that has the potential to be ptxas-free can be pretty big.

**Claude:**

I'd open source it, as part of the combined toolchain in the vx-lang org rather than a standalone repo.

**The fork risk is smaller than it feels.**

- Being ignored is the bigger danger than being forked. A closed blob inside an MIT language becomes a demo nobody can build on (Mojo's closed compiler drew constant criticism).
- Outside use is mostly a win: the work still says vx-lang, and most users will track upstream rather than maintain a SASS backend.
- The hard-to-copy parts live in Vx: per-kernel hybrid selection, exact resource numbers feeding admission verdicts, differential testing.

**Living in the combined toolchain.** Apache-2.0 with the LLVM exception keeps upstreaming possible and fits next to MIT.

**Before announcing:**

1. **Legal read on the ISA question.** SASS encodings come from reverse engineering; NVIDIA's CUDA license terms restrict reverse engineering of the SDK. Get advice from a software-licensing lawyer before the code goes public.
1. **Make the claim precise.** Mesa's NAK already compiles to native NVIDIA ISA without ptxas (for Vulkan shaders). Google's gpucc still relied on ptxas. Defensible claim: "an open path from a general-purpose language to complete CUDA binaries, with ptxas becoming optional per kernel," ideally with a measured coverage number.
1. **Don't tie the release to the talk** (15-minute slot). A live demo plus "code lands with v0.0.3" is strong.

Motivation example: the LLVM tracker has a ptxas miscompile involving divergent branches and convergent operations that was hard to diagnose without ptxas source.

______________________________________________________________________

## 7. What to build next for a larger audience

**Aditya:** We shipped bundled LLVM; llvm-sass ships next release. What high-value library/utility should we ship? Someone from Paradigm built GLM with Vx + Python without vLLM, CPU-only so far.

**Claude:**

Start from the GLM prototype and generalize it.

**Hybrid CPU+GPU inference for large MoE models.** Attention and KV cache on GPU, cold experts in host memory on CPU. A heterogeneous placement problem — what Vx's memory-space and topology types are for. Admission checking answers "does this model fit my machine at this context length, and with what split?" at compile time.

**Make it general where it's cheap.** A Vx kernel library of shared operators (RMSNorm, RoPE, GQA/MLA attention with paged KV cache, gated MLPs, MoE routing + grouped GEMM, FP8/INT4 matmuls, sampling) plus a Hugging Face `config.json` + safetensors loader.

**Packaging reaches the most people.** `pip install vx` wheel with bundled toolchain (manylinux, macOS arm64) and zero-copy DLPack interop with PyTorch/NumPy.

**Avoid** building a vLLM competitor (serving stack). Be the layer underneath.

**Order:** GLM on GPU with logit checks → extract kernel library + second model family → wheel → hybrid offload with admission verdicts → honest benchmarks vs llama.cpp and vLLM.

______________________________________________________________________

## 8. Plan / GitHub issue for GPU bring-up

**Aditya:** I'll find a cheaper GPU on RunPod. A100 is $1.8/hr, so I can test on that machine with AWS EC2 as host. Give me a plan for a GitHub issue.

**Claude:**

A RunPod GPU can't use an EC2 instance as its host — a pod is its own machine. Hybrid offload runs on the pod's CPU and RAM, so check pod RAM/vCPUs. Use EC2 for builds/CI only. Do bring-up on a cheaper card (L4/A10/RTX 4090); use the A100 for perf and the MoE milestone. Keep weights on a network volume; stop pods when idle.

```markdown
# GPU bring-up for GLM inference, then generalize into a Vx LLM kernel library

## Context
@<author> built GLM inference with Vx + Python, no vLLM, CPU only. This issue
takes it to NVIDIA GPUs, extracts the reusable pieces into a kernel library,
and builds toward hybrid CPU+GPU inference for MoE models, using admission
verdicts to decide placement.

## Goals
- GLM runs on GPU with outputs matching the Hugging Face reference
- Shared LLM operators live in a reusable Vx library, not model-specific code
- A second model family runs on the same library with mostly config changes
- Hybrid CPU+GPU MoE inference, with `vxc` reporting at compile time whether
  a model fits a machine at a given context length, and with what split

## Non-goals
- Serving stack (continuous batching, scheduler, OpenAI API); a minimal demo
  server at most
- Multi-GPU / multi-node
- Training

## Hardware
- Bring-up: a single cheaper GPU on RunPod (L4 / A10 / RTX 4090)
- Perf + MoE milestone: A100 80GB on RunPod; record host RAM and vCPUs for
  every benchmark, since offload depends on them
- Model weights on a RunPod network volume
- Builds / CI: EC2 or GitHub runners; GPU tests run only on the pod

## Milestones

### M0: Infrastructure
- [ ] Pod setup script: Vx toolchain, CUDA driver check, weights on volume
- [ ] Reference harness: run the HF transformers model, dump logits for a
      fixed prompt set to a file we compare against
- [ ] Choose bring-up model: <smallest GLM checkpoint, dense if available>

### M1: GLM on GPU (correctness first)
- [ ] Port prototype ops to GPU kernels (BF16)
- [ ] Prefill + decode with a contiguous KV cache
- [ ] Pass criteria: greedy decode matches the HF reference token for token
      for the first <N> tokens on all reference prompts; per-position logit
      max-abs-diff under <tol>
- [ ] Perplexity on a small fixed eval set within <x>% of HF

### M2: Extract the kernel library
- [ ] Operators: RMSNorm, RoPE, GQA attention, gated MLP, MoE routing +
      grouped GEMM, sampling
- [ ] Paged KV cache
- [ ] Hugging Face loader: config.json + safetensors → model definition
- [ ] Per-op tests against PyTorch reference implementations
- [ ] GLM rewritten on top of the library, M1 tests still pass

### M3: Second model family
- [ ] <Qwen or Llama>, added mostly through config, same M1-style criteria
- [ ] List anything that needed new code; that's the library's next gap

### M4: Hybrid CPU+GPU MoE
- [ ] Target: <MoE GLM checkpoint that doesn't fit in 80GB>
- [ ] Attention + KV cache on GPU, experts in host memory, CPU expert path
- [ ] Admission verdict: given model config + machine (GPU memory, host RAM),
      `vxc` reports fit / no-fit, max context, and the placement it chose
- [ ] Correctness vs HF reference on a subset of prompts

### M5: Quantization + benchmarks
- [ ] FP8 and INT4 weight matmuls
- [ ] Benchmarks against llama.cpp and vLLM on the same pod, same model,
      same precision: prefill tokens/s, decode tokens/s, time to first token,
      peak memory
- [ ] Publish scripts and raw numbers, including runs where Vx loses

## Open questions
- Which GLM checkpoints does the prototype support today?
- Does the prototype author want to co-own M1/M2?
- Where should the library live: in-tree (`lib/llm`) or a separate repo?
```

Packaging (`pip install vx` wheel with DLPack interop) is better as its own linked issue.

______________________________________________________________________

## 9. Problems with Rust's `pub` visibility

**Aditya:** What is the problem with Rust's pub visibility? They have private, crate and external.

**Claude:**

The levels are fine; the problems are where the boundaries fall, and that `pub` doesn't actually mean "part of the public API."

- **`pub` means "visible if reachable," not "exported."** A `pub` item in a private module is effectively crate-private, but the declaration doesn't say so. The `unreachable_pub` lint patches over this.
- **No level between crate and world.** Sibling crates in a workspace can only share via full `pub`, creating semver obligations — hence `#[doc(hidden)] pub mod __private` in serde, tokio, etc.
- **Macros make it worse.** `macro_rules!` expands in the user's crate, so `$crate::` references must be `pub`.
- **Fields can't be read-only.** Fully public or private-with-getter.
- **Trait impls have no visibility.** Preventing outside impls requires the "sealed trait" pattern.
- **Type privacy and name privacy are separate checks** (RFC 2145; now lints `private_interfaces`, `private_bounds`).
- **`pub(in path)` only works for ancestor modules** — no "friend."

For Vx: explicit package-level export semantics, a package/workspace visibility level, read-only fields, and a sanctioned way for macros and generated code to reach internals.

______________________________________________________________________

## 10. Proposed Vx visibility syntax and semantics

**Aditya:** So how should the syntax and semantics look?

**Claude:**

**Four levels, named for the boundary they open**

```
fn helper()            // private: this module only (default)
pkg fn plan()          // anywhere in this package
ws fn lower_to_sass()  // packages in this workspace group
pub fn compile()       // exported API
```

- **`pub` means exported, and the compiler enforces it.** A `pub` item unreachable from outside is a hard error.
- **`ws`** — visible to packages in a manifest-declared workspace group; group packages publish at the same version (registry-enforced), so `ws` interfaces carry no semver promises.
- **One rule for signatures.** A signature can only mention types at least as visible as itself; opaque returns (`-> impl Trait`) excepted.

**Fields: separate read and write visibility**

```
struct Stream {
    pub get len: u64,            // readable by anyone, writable in this module
    pub get pkg set cursor: u64, // readable by anyone, writable in this package
    pkg buf: DevicePtr<u8, Global>,
}
```

**Traits and impls**

```
pub sealed trait Backend { ... }   // usable anywhere, implementable only in this package
pkg impl Debug for Plan { ... }    // impl usable only inside the package
```

A `pkg impl` still claims its coherence slot globally, so nobody can write a competing impl.

**Macros and generated code get definition-site privileges** (what Rust intended for macros 2.0). Same mechanism for MLIR-dialect-generated code.

**Tooling:** `vx api` prints a package's public API; `vx api diff v0.3 v0.4` flags semver breaks; `@unstable` for experimental APIs.

**Keep visibility separate from placement** — a `pkg` kernel is as launchable from the host as a `pub` one.

Alternative: an explicit export list in the package root (Haskell, Go internal packages). Recommendation: the reachability error plus `vx api` for the overall view.

______________________________________________________________________

## 11. Visibility at the ABI level

**Aditya:** The programming world reasons in terms of object file, static library, DSO/shared object, and executable. Visibility at the ABI level should reflect one of these. Am I thinking it right?

**Claude:**

Half right. Visibility should determine ABI behavior but not be *defined* by artifact types. Linkers enforce only two boundaries:

- **Translation unit / object file:** internal-linkage symbols (`static`, `STB_LOCAL`).
- **Linked module (DSO or executable):** hidden vs. default-visibility exported symbols.

Static libraries are bundles of object files; the same package may be built as any of these, so visibility must not change with output type. Keep language levels and *map* them:

| Level | Object / static lib | DSO | Executable |
|---|---|---|---|
| private | internal linkage | internal | internal |
| `pkg` | hidden | hidden | hidden / internalized by LTO |
| `ws` | hidden | **exported, private version tag** | internalized |
| `pub` | hidden until linked | exported, public version | internalized unless `@export` |

`ws` across DSOs mirrors glibc's `GLIBC_PRIVATE` version tag: exported, no compatibility promise. Emit `VX_<group>_PRIVATE` for `ws` and normal versioned nodes for `pub`. Mach-O: exported-symbols list + reserved mangling prefix; Windows: `.def` file.

- **`pub` across DSOs needs a stable ABI** (mangling, calling convention, generics via caller monomorphization). `pkg`/`ws` can stay unstable.
- **Executables internalize everything** except entry points and `@export`.
- **Device code is another linked module.** Kernel entry points are externally visible for `cuModuleGetFunction`; treat "kernel" as an implicit `@export` toward the device module.

Rule of thumb: language visibility sets the most a symbol may be exported; build mode can only narrow it.

______________________________________________________________________

## 12. Covariance between visibility and versioning

**Aditya:** This also partly addresses package management. We need algorithms for a covariance (partial order) relation among visibility and versioning. This will help with (1) fewer exported symbols, (2) lower binary load time, (3) semantic versioning of shipped binaries.

**Claude:**

Benefit 3 is the big one; 1 and 2 are real but secondary.

**The core relation.**

- **Visibility:** `private < pkg < ws < pub`; with read/write field splits (`get ≥ set`) and sealed traits (`use ≥ impl`) it becomes a product lattice (partial order).
- **Version obligation:** `none < patch < minor < major`.

Each change δ has a kind and the visibility of the item it touches. Define b(kind, vis), **monotone in visibility**. The release's required bump is the join:

```
bump(release) = ⊔ { b(kind(δ), vis(δ)) | δ ∈ diff(old, new) }
```

Prior art at source level: Elm's `elm bump`.

**Source vs. binary compatibility need separate functions, b_src and b_abi.** E.g., adding a private field to a `pub` struct is source-compatible but ABI-breaking. `pub` types crossing DSOs must declare frozen layout or resilient (opaque, accessor-based) — see Swift's library-evolution mode (`@frozen`, resilience).

**Three algorithms.**

1. **Export set as a least fixed point:** start from `pub` symbols, add everything reachable from bodies consumers instantiate or inline (Swift's `@usableFromInline`). Cross-DSO inlining trades speed for exports/ABI surface.
1. **Per-symbol version tags:** ELF symbol versioning (`compile@VX_1.2`, `compile@@VX_1.3`); keeping compat symbols turns some breaking changes into minor ones; the compiler can generate shims.
1. **Inferred version requirements:** a binary's minimum required version = max version of the symbols it imports (as with glibc). Resolution stays NP-complete in general, but false conflicts largely disappear.

**On benefits 1 and 2.** GNU hash lookups are near constant time; load time depends more on relocations/imports. The bigger win: hidden symbols can't be interposed, enabling direct calls, inlining, and dropping PLT stubs (see Drepper, *How to Write Shared Libraries*). Load-time wins show mainly in very large DSOs.

**Practical order:** `vx api diff` with b_src, b_abi and the join on Vx's own releases → export fixed point and version-script emission → inferred requirements (needs a registry). The monotone obligation function, src/abi split and export fixed point could make a PL paper.
