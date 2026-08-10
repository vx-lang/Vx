# M4: prefill on one H100, decode on the other

Session of 2026-08-10, continuing
[gpu_disaggregated_inference.md](implementation_plans/gpu_disaggregated_inference.md)
(#347). [M2/M3](walkthrough_gpu_campaign_m2_m3_2026_08_10.md) put a placed Llama on a
single A100 with token parity. This session ran prefill on one GPU and decode on another,
from one program text, with the KV cache crossing NVLink between them.

It also found four ways that run could have been silently a one-GPU run reporting two,
which is most of what the session was actually about.

## The premise, restated by the user

*"When we do 'spawn on', the device may be anywhere but the programmer should not worry
about it as it can be abstracted away. It's magic."*

That is stronger than the M4 written in the plan, which had the KV handoff as TCP written
in Vx source with `std::net` — *"the handoff is Vx source, that is the claim"*. Under the
stated vision that is backwards: a program that opens a socket has stopped being
location-transparent and become a network application that compiles with Vx. Sockets
belong below the plugin ABI.

The distinction that keeps this honest, and the reason a compiler can make the claim at
all: **the abstraction hides the mechanism, not the price.** A local peer copy is
microseconds; the same copy over InfiniBand is milliseconds. The fleet models already
declare those edges and `src/arch.rs` already routes across them with Dijkstra, so the
compiler can price a remote placement before the machine is rented. *The device may be
anywhere, and the compiler tells you what anywhere costs.*

## The compiler half already worked

Checked before writing anything, because the alternative was discovering it on a rented
pod:

- `Topology::substitute` handles GPU, so a const generic reaches a `spawn` placement.
- A const generic forwards through a call chain: `outer<0>` → `inner<0>` → `topology(500)`.
- A *monomorphized* function still classifies. `matmul<0>` and `matmul<1>` emit two
  kernels with identical `kind=matmul` and `roles=a:0,b:1,out:2` differing only in
  `topo=500` against `topo=501`.

None of those were obvious in advance and all three were prerequisites. The compiler
needed no changes at all.

## Four ways the runtime would have lied

Every one of these produces a plausible answer rather than a failure, which is the
category that matters when the only evidence a demo has is *which device did the
arithmetic*.

**1. W1030 never fired for `Topology::GPU` (#345, `81af092b`).** `non_constant_index` —
the function the warning consults — matched `NPU` and `AccCore` and fell through on
`GPU`. So a runtime GPU index took the device-0 fallback in silence, while the doc
comment three lines above used `GPU[i]` as its example of exactly that case. Nothing
about a device index is kind-specific, which is why it survived: the NPU tests covered
the behaviour and passed throughout.

**2. The cuBLAS handle was process-wide (#346, `fcfd81cf`).** A handle belongs to the
device current at `cublasCreate`, and using it after `cudaSetDevice` moved elsewhere does
not fail — it keeps computing in the original context. A dispatch to GPU 1 would call
`select_device(501)`, print `[Vx CUDA] device 1`, and hand the GEMM to device 0. The
worst of the four, because it is the one that produces *numbers*.

**3. `vx_plugin_alloc_and_transfer` discarded its `topology_id`.** The parameter existed,
callers passed it, llama2.vx's comment said the weights were staged "into the memory of
the topology that will read them", and they were staged into whichever device happened to
be current. Accidentally right with one GPU. `vx_plugin_free` discarded it the same way.

**4. No device-to-device edge existed at all.** Nothing in the plugin ABI could express a
movement whose two endpoints are both devices — which is precisely what a disaggregated
placement is made of. `vx_plugin_transfer_peer` is new; CUDA answers it with
`cudaMemcpyPeer`, which needs no peer access enabled and stages through the host itself
where no P2P path exists, so one implementation covers an NVLink pod and a PCIe-only one.
The host and NPE backends answer with a copy, so a program naming the movement still runs
on a machine that does not need it.

`select_device` was correct throughout, and its comment already said what it was for —
"the whole of running prefill on one device and decode on another". It was called from
exactly one place.

## The program (#347, `1e89612c`)

`transformer` and `matmul` become const-generic over a device index, so `transformer<0>`
and `transformer<1>` are one body instantiated twice. The device must be a const generic
rather than a parameter because placement is a compile-time fact — an index arriving at
runtime cannot select a device, which is what #345 is about.

`memory_map_weights` takes the topology to stage into and is called twice. Two replicas
is what disaggregation *is*, as distinct from sharding: each worker holds the whole model,
and prefill's GPU cannot read decode's HBM.

The generation loop splits at the prompt boundary — which the single-phase loop already
tested for on every iteration; it has only been lifted out, because a boundary that a
*placement* changes at has to separate two loops rather than branch inside one.

### Making the handoff load-bearing

The design decision that gives the demo teeth. Prefill and decode hold **separate**
`RunState`s and therefore separate KV caches; decode's starts empty and is filled only by
the transfer.

Sharing one array and copying it device-to-device on the side would produce identical
output with the transfer deleted — theatre. Deleting it now produces garbage, and that is
the acceptance test. Verified on CPU before renting anything, and again on the GPUs:

```
with handoff:     ... "I'm so hungry, Mommy!" / Ooama smiled and said, "Let's go find
without handoff:  ... Later that day, Timmy and his friends were all over the playground.
```

`VX_LLAMA_DISAGG=0` runs the same code with the destination index 0 instead of 1, so the
single-device run *is* the disaggregated program with both indices equal. That makes it a
usable oracle, and it means the default run exercises the handoff — a broken transfer
fails on a laptop rather than waiting for a pod.

## Two defects in the Apple backend, found because they blocked local checking

Not multi-device, but in the way of verifying any of it without hardware.

It wrote its narration to **stdout**, unconditionally, so a Llama run's story came out
interleaved character-by-character with `[Vx Dispatcher] Intercepted kernel dispatch` and
neither could be read. That is why token parity had previously been checked on rented
hardware, where the CUDA backend happens to be quiet. Both other backends already gate the
same messages on `VX_DISPATCH_VERBOSE` and write to stderr.

And a kernel it could not find printed `DEBUG: Could not find JIT kernel %s, skipping execution` followed by `return 1` — success — leaving the caller's buffer holding whatever
it held before. Both other backends abort.

## Getting there: the build box and the bundle

The EC2 box was fresh. `scripts/setup_linux.sh` provisioned LLVM 22.1.8 and Rust 1.97.1,
and then `source config.local && cargo build` failed with **`cargo: command not found`**.

A real bug in the repository's own bring-up path (`b1adbed2`): `config.template` set
`CARGO_HOME` but never put cargo on `PATH`, and `setup_linux.sh` installs to `$HOME/.cargo`
while the template named `$PROJECT_DIR/.cargo`. Nobody had hit it because a developer's
shell rc puts cargo on PATH before `config.local` is ever sourced. Two directories were
being treated as one: cargo's *state* (registry, aliases, toolchains — legitimately
in-repo) and the cargo *binary* (wherever rustup installed its shims). Now answered
separately, with macOS behaviour preserved exactly.

Source reached the box by `rsync` of `git ls-files` rather than a clone — the repo is
private and the box has no credentials, and that file list is exactly a checkout minus
build artefacts (6.9 MB). Model assets are gitignored and travel separately. The 110 MB
bundle went EC2 → pod streamed through the local machine in one pass, so the pod's private
key never touched the build box, md5-verified on both ends.

**One false start.** The first pod advertised as multi-GPU had one A100. `nvidia-smi -L`
listed one device and `CUDA_VISIBLE_DEVICES` was unset, so nothing was masked. This is
what `run_disagg_demo.sh` now refuses on, before the upload rather than after.

## The run

2x H100 80GB HBM3, **NV18** between them — eighteen bonded NVLink 4 links, a direct peer
path. CUDA 12.8, driver 580.126.09. 64 tokens of stories15M, greedy.

| | device 0 | device 1 | KV transfer |
|---|---|---|---|
| `VX_LLAMA_DISAGG=0` | 2775 | **0** | `peer 0 -> 0, 442368 bytes` x2 |
| `VX_LLAMA_DISAGG=1` | 1601 | **1174** | `peer 0 -> 1, 442368 bytes` x2 |

Tokens identical between the runs, and identical to macOS CPU and Linux CPU. **The zero in
the first row is what makes the second row evidence** — a plugin that always reported two
devices would look the same in run 2 alone.

Byte-exact against the compile-time figure: `n_layers x seq_len x kv_dim x 4` =
`6 x 64 x 288 x 4` = 442,368, twice, for keys and values.

The dispatch counts close too. Both runs show **2752 GEMM dispatches**, which is exactly
43 per token (4 qkvo and 3 FFN projections across 6 layers, plus `lm_head`) times 64
tokens, and **2775 device selections** — the 23 extra being weight staging for two
replicas and the handoff's allocate, copy, read-back and free calls. The device-0/device-1
split of 1601/1174 corresponds to 37 prefill tokens and 27 decode, consistent with a
38-token character-level tokenization of the prompt (#323).

Switch points land where the program says: prefill replica staged on device 0, decode
replica on device 1, prefill's GEMMs on 0 through dispatch 1606, device 1 from 1607 (the
handoff), briefly back to 0 at 1611 to free the source buffers, device 1 from 1613.

### Two results not to misread

**Disaggregation was slower: 2707 ms against 2437 ms.** Expected, not a defect. At batch 1
the run pays for the handoff and a second weight staging and gains nothing, because there
is no concurrent load for prefill and decode to stop competing over. Disaggregation buys
throughput under load, which one request cannot demonstrate.

**The declared interconnect was wrong by 7x, in the right direction.**
`fleet/node-2gpu-h100.vx` declares the peer edge at 63 GB/s (PCIe Gen5), chosen before the
pod was rented because a pod does not say what it will give you. The pod gave NVLink at
~450 GB/s per direction. The declaration was **not** corrected to match; the observation is
recorded beside it with date and driver version. A bound the hardware beats is the correct
kind of wrong, and editing it afterwards would destroy the only property that made the
number worth writing: that it predated the machine.

## The harness lied, and that is the part worth remembering

`run_disagg_demo.sh` read the plugin's traces from stderr, which is where the C++ writes
them. But `vxc --run` *executes* the program it just built, and the executed program's
stderr arrives on vxc's stdout. All 5,529 `[Vx CUDA]` lines landed in the token file.

The first report read:

```
=== 1. Same tokens? ===
  DIFFER -- this is a failure, not a curiosity:
=== 3. Did the KV cache cross? ===
  NONE -- the handoff did not happen
```

Both false. The tokens matched exactly and the KV cache had crossed. **An experiment that
worked, reported as one that failed** — and had the run been marginal rather than clean,
the obvious response would have been to go debug a compiler that was fine.

Fixed to split by line prefix rather than by file descriptor, which does not care how many
processes are involved, and to refuse outright when a trace holds no dispatches —
otherwise every check passes by vacuity and "zero device-1 dispatches" reads as *no
disaggregation* rather than *no data*. Re-verified by replaying the fixed analysis over
the same captured bytes that defeated the original.

This is the third instrument failure of the campaign, after the vacuous FileCheck prefix
and the stale-binary negative control below. The pattern is consistent enough to state as
a rule: **an instrument that has never been shown to fail has not been shown to work.**

## Groundwork for location transparency (#348)

With two devices working in one box, the question became what changes when the second
device is in another rack. Less than expected.

**`Topology::Custom` already names a worker.** A name declared in a machine file and
spawned on in the program resolves, participates in admission, classifies, and carries its
own dispatch id:

```
spawn on(Topology::PrefillWorker)  ->  topo=1669
spawn on(Topology::DecodeWorker)   ->  topo=1113
```

Zero errors, two distinct ids, no new syntax. Undeclared names are refused; declared but
unreachable memory gives a proper E6003 naming the space and the fix. **This is the
location transparency the user described, and it arrives as a consequence of the
`--host`/`--machine` split rather than as an addition to it**: the program names a *role*,
the machine file says whether that role is the GPU beside this one or one in another rack.

It also kills the `Topology::GPU[node, dev]` idea — a second index bakes fleet layout into
program text, which is the thing being avoided.

Two small prerequisites landed:

**The payload carries the declared name, not only the id.** For a machine-file topology the
id is `1000 + fnv32(name) % 1000` — one-way, a thousand slots wide. A plugin holding
`topo=1113` cannot recover `DecodeWorker`, so it cannot resolve the worker against a fleet
manifest to find out which machine it is; and two names can collide with nothing to notice.
The name is the identity, the id a shortcut. Not hypothetical: reading the id-only payload,
I took `topo=1669` for PrefillWorker because it was emitted first. Kernels come out in
reverse; it is DecodeWorker.

**`NicRam` and `RemoteHbm` get bands 800 and 900.** Both answered 300, which is
`Topology::AMX` — so a `transfer` into a peer's memory reached a plugin as a request for
Apple's matrix coprocessor, in range and therefore unobjectionable to any diagnostic, and
the two network spaces were indistinguishable from each other. The comment in place read
"kept at 300 (overlaps AMX) for now".

The negative control for the payload change was invalid on the first attempt: disabling the
emission left a dangling reference, the build failed, and the test passed against a *stale
binary* — reporting "still passed" for a compiler that had never been rebuilt. Perturbing
the key instead of deleting the block gives a build that succeeds and a check that fails,
which is what a control has to be.

## State at the end of M4

Done: prefill and decode on two physical GPUs from one program text, KV cache crossing
NVLink, byte-exact against prediction, token parity across four configurations, handoff
proven load-bearing by deletion.

Remaining for the network form, in order: a plugin that resolves a topology name to an
endpoint against a fleet manifest; **marshalling** — `device_args[i]` points into this
process, so a remote dispatch must serialise memrefs rather than pass addresses, and this
is the large one and the point where a wire format has to be chosen; a remote agent to
receive dispatches.

Unchanged and still the ceiling on everything: attention, softmax, RoPE and RMSNorm run on
the host (#251), so the KV cache's working copy is host memory and the outer legs of the
handoff exist only because of that. Strided sub-views of a placed tensor (#344) are what
would put the cache on the device and read it there.
