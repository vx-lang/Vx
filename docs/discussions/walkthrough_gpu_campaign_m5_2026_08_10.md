# M5: what disaggregation costs, on two A100s in one box

Session of 2026-08-10, continuing
[gpu_disaggregated_inference.md](implementation_plans/gpu_disaggregated_inference.md)
(#348). [M4](walkthrough_gpu_campaign_m4_2026_08_10.md) ran prefill on one H100 and decode
on the other and showed the tokens were identical. This session asked the next question,
which is what that cost.

Two A100-SXM4-80GB in one pod, NV12 between them, both idle at the start. One box rather
than two matters: every fleet run before this went between rented machines through an SSH
tunnel, which is why none of their timings were ever quoted. Loopback removes the network
from the measurement, so `fleet - disagg` is the marshalling stack and nothing else.

## The measurements

Timing is by **slope**, not by total. `vxc --run` compiles, JITs, links and loads a 60 MB
checkpoint before the first token, and that fixed cost is larger than the effect being
measured. Sampling several sizes and fitting puts startup in the intercept and the
per-unit cost in the slope.

### GEMM, square, f32

Fitted from 50 and 450 iterations of the same program, so compilation, the host-side fill
loop and process startup all cancel.

| path | N | ms/GEMM | GFLOP/s | staged MB/GEMM |
|---|---|---|---|---|
| GPU 0, plugin | 512 | 0.767 | 350 | 3.0 |
| GPU 0, plugin | 1024 | 3.517 | 611 | 12.0 |
| GPU 0, plugin | 2048 | 10.918 | 1574 | 48.0 |
| GPU 1, plugin | 1024 | 3.472 | 618 | 12.0 |
| GPU 1, plugin | 2048 | 10.210 | 1683 | 48.0 |
| GPU 1, over the wire | 1024 | 12.265 | 175 | 12.0 |
| GPU 1, over the wire | 2048 | 40.513 | 424 | 48.0 |

All three paths agree on the value at every size — 133.82478, 1072.1694, 8583.648 — and
the worker served exactly 1500 dispatches, which is (50+450) × 3 sizes.

Two things fall out of this and neither is about the A100:

**Vx's GEMM path reaches about 8% of the device.** An A100 does ~19.5 TFLOP/s of f32
SGEMM; this is 1.6. The arithmetic says where the rest goes: at N=2048 a GEMM is 17.2
GFLOP, which is 0.88 ms of compute, against 10.2 ms measured. The remaining 9.3 ms moves
48 MiB, or 5.4 GB/s — which is the rate of a pageable `cudaMemcpy2D`, not of PCIe Gen4.
Operands are staged per dispatch (#321) out of pageable memory. Residency is the large
fix and pinned staging buffers are the cheap one.

**The wire runs at about 1.6 GB/s on loopback.** N=2048 costs 30.3 ms more remotely for
50.3 MB moved; N=1024 costs 8.8 ms more for 12.6 MB. Both land near 1.6 GB/s, which for
loopback TCP is slow — it is the read/write loops and the staging copies, not the socket.

### Llama, per decode token

Prefill runs for as many steps as the prompt is long — 37, here — so the token count
varies decode steps only, and the slope is the cost of one decode step: the phase that
moves. Fitted on the fastest run at each size, three reps, after an untimed warmup.

| tokens | single | disagg | fleet |
|---|---|---|---|
| 48 | 3.025 | 3.033 | 3.094 |
| 64 | 3.045 | 3.490 | 3.359 |
| 96 | 3.278 | 3.645 | 4.667 |
| 128 | 3.621 | 4.190 | 5.738 |
| **ms/decode step** | **7.6** | **13.0** | **34.4** |

All three produce identical tokens. Splitting across two devices costs +5.4 ms/token
(1.7×); reaching the second device over the wire instead of through the plugin costs a
further +21.4 ms (2.6×).

The wire figure is explicable: llama moves small operands very often, so it pays round
trips rather than bandwidth — roughly six per dispatch (two TRANSFER, DISPATCH, FETCH, two
FREE) at 43 dispatches per token. Carrying small operands inline is the known fix and is
not done.

**The 1.7× is not explained, and three plausible explanations are ruled out by the data.**
It is not device switching: the whole 64-token run contains 5 switches across 5529 device
lines, one of them the phase boundary. It is not per-dispatch re-staging: 15 `stage` lines
serve 2752 dispatches, so operands are resident as intended. And it is not that GPU 1 is
slower, because the GEMM table above has the two devices within 7% of each other at every
size that measures cleanly. What remains — cross-NUMA staging for many small transfers
(the pod puts GPU 0 on node 0 and GPU 1 on node 1), or a per-dispatch cost that only
appears once a process holds contexts on two devices — this experiment does not
distinguish. Worth saying rather than picking the most plausible-sounding one.

## What was wrong with the instrument, again

M4 found four ways a one-GPU run could report two. This session found six more, and it is
worth being blunt that every one of them was in the measuring apparatus rather than in the
thing measured.

**A manifest naming workers the program never mentions.** `run_fleet_demo.sh` wrote
`PrefillWorker` and `DecodeWorker`; `llama2.vx` spawns on `Topology::GPU[D]`, so it
dispatches to `GPU[0]` and `GPU[1]`. A name absent from a manifest means *local*, by
design — so run 3 executed entirely on the driving machine, produced identical tokens
because it was the identical computation, and printed `IDENTICAL`. The A100-to-A100 result
was real because that manifest was written by hand. This script would have claimed the
same thing without leaving the box. Matching output cannot distinguish a distributed run
from a local one; it is the one thing both are guaranteed to agree on. Only the far side
knows, so the workers now narrate and the script asserts each served dispatches.

**A trace that could not tell an allocation from a dispatch.** `[Vx CUDA] device 1` was
printed by staging, dispatch, peer copy, fetch and free alike, and the demo's "did it use
two GPUs?" check counted those lines. At 32 tokens the entire generation is prefill, so
decode never executes and the only marks on device 1 are the decode replica's weight
allocations — thirteen of them. The check reported two devices and six switches between
them; the token comparison agreed. Both halves of the evidence passed for a run that
disaggregated nothing. The line now names its purpose, the check counts dispatches, and a
run with no decode in it is an error. At 64 tokens it reports 1591 and 1161 with one
switch at 1592, and 37×43 and 27×43 are exactly those.

**A witness grepping for a line that was never written.** The worker's own narration
honoured `--verbose` while the plugin linked into it honoured `VX_DISPATCH_VERBOSE`. A
harness setting the variable got thousands of `[Vx CUDA]` lines and no `[Vx worker]` ones,
so a check for messages *served* found none and reported the run had never left the host —
with 77064 device operations in the same file. I believed it for several minutes.

**`grep -c ... || echo 0`.** `grep -c` exits 1 on zero matches, so the count became
`"0\n0"`, which the integer test then rejected as malformed instead of reporting the
failure it existed to report.

**A demo with a decorative second worker.** `run_fleet_demo.sh` never set
`VX_LLAMA_DISAGG=1`, so decode targeted `GPU[0]` exactly as prefill did. It reported 2752
dispatches on worker 1, none on worker 2, and identical tokens -- all true, and describing
a run that used one worker twice. Fixed, it reports 1591 and 1161 across two processes
over TCP, which are 37x43 and 27x43.

**A fit contaminated by one slow run.** These runs share a box, so interference can only
add time: the distribution has a floor and a long right tail. An 8.7 s sample beside a
3.9 s neighbour at the same size is not noise in both directions, and averaging it in
moves the estimate away from the quantity of interest. It produced a *negative*
per-token cost in one run, and made disaggregation look 2.5× faster than not
disaggregating in another. Both are arithmetically impossible rather than merely
surprising, which is the tell. The harness now takes an untimed warmup per configuration
and fits on the minimum at each size.

## Two real defects, found by load

### A remote result that never came home

The first GEMM benchmark was written `c = a @ b`. The worker served 560 dispatches and
every answer came back **zero** — the value the tensor was initialised with, returned as
the product, exit status 0, no diagnostic.

The mechanism is exact. A result written into a caller's buffer is `outkind=buffer`: the
worker fills the copy it was staged and the client fetches it back. A result the kernel
*allocates* is `outkind=slot`: the plugin allocates on the far side and publishes a
descriptor naming memory in another process, and no message in this protocol retrieves it.
The dispatch was sent anyway.

It now declines to route and says so once, which costs the distribution and keeps the
arithmetic — the trade this backend makes everywhere else. The benchmark was rewritten to
use `matmul_into`, which is why the table above has a working remote column.

Worth noting what caught it: not the timings, which looked plausible, but a check that the
three paths agree on the *value*.

### A worker that could serve 380 tokens, ever

The fleet row failed at 128 tokens, three times out of three, with a SIGSEGV inside
`cudaMemcpy2D` — on the *host*, three frames below anything that named a cause. The
obvious reading was "128 tokens is too many". It was not.

`vx_remote_table_alloc` is a bump allocator over a 47-bit offset space, and it charges
every region a gap of at least `VX_REMOTE_MIN_GAP` so that an overrun reaching the next
region's base is caught rather than mistaken for a valid handle. That gap was 4 GiB. The
space is never reused — deliberately, so a stale handle stays dead. So:

```
128 TiB / 4 GiB = 32768 allocations, for the life of the process
```

A dispatch stages two operands, so ~16384 dispatches; llama2 issues 43 per token, so
about **380 tokens** — not per request, but total. The worker died after 16145 dispatches,
having served the 48-, 64- and 96-token runs correctly first, which is exactly why it
presented as a token-count limit. Fresh workers run 128 tokens without complaint: worker 1
serves 1591 (37×43) and worker 2 serves 3913 (91×43, and 128−37=91).

And the *way* it failed is the more serious half. TRANSFER returned handle 0, the client
could not encode its arguments, `vx_remote_dispatch` reported failure — and failure there
meant "not routed", so the caller ran the dispatch locally while holding operands that
lived on the worker. Address-space exhaustion on one machine arrived as a segmentation
fault on another. Three changes:

- The gap floor is 1 MiB, giving 134 million allocations, at which point what binds is the
  regions that are genuinely large (llama2 stages ~121 MB of stride per token, so ~1M
  tokens). Overrun detection is unaffected: the gap is `max(size, floor)`, so a region
  always gets a gap at least its own size, and the floor governs only regions smaller than
  it. With it, the fleet row runs 128 tokens three times out of three, workers serving
  20683 and 24725 dispatches — both past the old ceiling.
- Declining after operands have been staged is now fatal and says which worker and why.
  Declining *before* touching the wire is still a safe local fallback; the distinction is
  the whole point.
- `stage` refuses a remote handle with a message naming the worker, instead of faulting.

Removing the ceiling properly means putting the generation in the handle so freed space
can be recycled without resurrecting stale ones. That is not done.

## What can and cannot be claimed

FlashAttention **does not run on the GPU**, and no number here says otherwise. The trace
is unambiguous:

```
[Vx CUDA] device 0 stage        x4   (Q/K/V/O to HBM)
[Vx CUDA] vx_npu_kernel_0 (kind=<unclassified>) not routed; running on the host
```

Only `kind=matmul` routes to cuBLAS; a fused online-softmax loop nest is not classified as
one, so the operands are transferred to the A100 and the kernel executes on the pod's
Xeon. Timing that and calling it FlashAttention-on-A100 would be a Xeon number with a GPU
label. This is #251 — kernel emission — and it remains the ceiling on everything here.

The declared PCIe figure in `fleet/node-2gpu-a100.vx` is again pessimistic against the
hardware rented: the pod reports NV12, twelve bonded NVLinks. Recorded, not edited to
match, and the same open question as #349.

## Files

- `scripts/run_perf_matrix.sh` — llama along three paths, warmup, fit on minima
- `scripts/run_gemm_bench.sh` — GEMM along three paths, agreement check on values
- `tests/backend/pass/gpu_gemm_bench.vx` — the first matmul here sized to be measured
- `tests/runtime/remote_region_test.cpp` — the allocation ceiling, on any machine
