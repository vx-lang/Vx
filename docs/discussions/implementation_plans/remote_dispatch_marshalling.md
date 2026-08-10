# Marshalling a dispatch to a remote device

Design for the last piece of #348. Written 2026-08-10, after
[M4](../walkthrough_gpu_campaign_m4_2026_08_10.md) put prefill and decode on two GPUs in
one box and established that everything *except* this already works for a device in
another rack.

## What is already true

Worth stating precisely, because it determines how small this is.

- **A worker can be named.** `Topology::Custom` — a name declared in a machine file,
  spawned on in the program — resolves, admits, classifies, and carries a dispatch id.
  `spawn on(Topology::DecodeWorker)` compiles today.
- **The name reaches the plugin.** `toponame=DecodeWorker` is in the payload, so a plugin
  can resolve a worker against a manifest without the compiler knowing any address.
- **Every argument is self-describing.** `arg_tags[i]` carries kind, element type, rank
  and a slot bit; `vx_memref_sizes` / `vx_memref_strides` / `vx_memref_offset` decode a
  descriptor. This is the groundwork the whole document rests on — nothing needs to be
  added to the ABI to *know* what an argument is.
- **The seam exists.** `vx_plugin_dispatch_async` and `vx_plugin_transfer_peer` both take
  topology ids and hide the mechanism. Neither signature changes here.

## The actual problem

`device_args[i]` is a pointer into *this* process. libffi reconstructs the call locally
because the callee is in this address space. Both assumptions fail across a machine
boundary, and they fail differently:

1. **Values must travel.** A memref argument is a descriptor plus the buffer it points
   at. The descriptor is small and fixed; the buffer is the model.
1. **Results must come back.** `outkind=buffer` means the caller's buffer is filled;
   `outkind=slot` means the kernel allocates and publishes a descriptor. Only the first
   has an obvious remote meaning, and the second is the interesting case.
1. **The callee must exist there.** `vx_host_kernel_symbol(kernel_name)` does `dlsym` in
   this process. A remote agent needs the same outlined kernel, which means it needs the
   same compiled artifact.

## The decision that shapes everything: what does a remote worker hold?

Two models, and the choice is not close.

**A. Stateless RPC.** Every dispatch ships all its operands and receives its results.
Simple, and correct by construction. Also unusable: llama2's `wq` is 2 MB and is read
once per layer per token, so a 64-token generation would ship it 384 times. At batch 1
this is not a slow implementation of disaggregation, it is a fax machine.

**B. Resident handles.** A buffer transferred to a worker stays there and is named by an
opaque handle; a dispatch ships handles, not bytes. This is what
`vx_plugin_alloc_and_transfer` already means locally — it returns a device pointer the
program keeps and passes to later dispatches — so the remote form is the same idea with
a wider notion of "pointer".

**B, and it is already the local design.** The whole reason M3 was not slower than the
CPU is that weights are staged once and passed by pointer thereafter. A remote worker is
the same story with a handle in place of an address.

This has a consequence worth being explicit about: **a remote pointer is not
dereferenceable by the host.** Today `is_device_ptr` distinguishes host from device and
the program never dereferences a device pointer by accident because the type system and
the placement analysis keep it out of host code. The same discipline must extend, and the
failure mode if it does not is the `matmul_ane` segfault from M2/M3 — a host loop
dereferencing a staged pointer — but across a network, where it is a wild read rather
than a fault.

## Wire format

Three message types. Deliberately few.

```
TRANSFER   handle, dtype, rank, sizes[rank], strides[rank], bytes[]   -> ack
DISPATCH   payload_blob, [arg_tag, arg_body]*                         -> results
FREE       handle                                                     -> ack
```

`arg_body` is discriminated by `arg_tag`, which the ABI already defines:

- **scalar** (`VX_ABI_KIND_I32`, `F32`, ...): the value, little-endian, `vx_dtype_bytes`
  wide. There are no other cases; the tag's kind byte enumerates them.
- **memref** (`VX_ABI_KIND_MEMREF`): a handle plus the descriptor's shape metadata. Not
  the buffer — the buffer arrived in an earlier TRANSFER and is resident. Sizes and
  strides do travel, because a view over a resident buffer has its own extents and a
  strided sub-view (#344) will have its own strides.
- **slot** (`VX_ABI_IS_SLOT`): a handle to storage the *worker* writes a descriptor into,
  plus the element type and rank it will hold. The result comes back in the DISPATCH
  reply rather than being read out of the slot afterwards, because a slot's contents are
  meaningless on this side.

**Endianness is fixed little-endian on the wire and not negotiated.** Every target in the
fleet is little-endian; a big-endian one would need conversion anyway, and a negotiation
that has never been exercised is a bug waiting rather than portability.

**No framing library, no schema compiler.** The messages are fixed-layout with one
variable-length tail each, and the ABI header is already the schema. A dependency here
would be the second place the argument layout is written down, and the two would drift.

## Handles

**Not a pointer on the remote device, and not an opaque token either.** The first draft
of this document said "opaque `uint64_t`", which is wrong for a reason that only shows up
by reading the program it has to support.

### The constraint the program imposes

`llama2.vx` does pointer arithmetic on staged pointers, seven times per layer:

```
let d_wq : *mut f32 = vx_plugin_alloc_and_transfer(wq_n * f32_bytes, p2, topology_id);
...
let wq_l : *mut f32 = vx_advance_ptr(w.wq, wq_off);   // this layer's slice
```

`w.wq` *is* `d_wq`. The seven weight blobs are contiguous and the per-layer slicing is a
host-side offset computation on a device address, which works today because a CUDA device
pointer is a real address in a unified space and `+ offset` means what it says.

An opaque handle has no meaningful `+ offset`. Adding a byte offset to a table index
produces either a different valid handle or a lookup miss, and the first is worse. Since
the acceptance criterion for #348 is that this program runs *unchanged*, any handle design
that cannot be offset is disqualified before it is evaluated.

### Why not simply the worker's own pointer

Tempting — it fits in 64 bits, needs no table, and offsets natively. Four reasons not to:

1. **Indistinguishable from a host pointer.** `is_device_ptr(p)` asks CUDA whether `p` is
   device memory. For a remote address it answers *no*, so the plugin stages it — reading
   host memory at an address that is really a remote one. Silent, and the read succeeds.
1. **Stale handles look live.** A buffer freed on the worker leaves an address that is
   still a plausible address. With an indirection, a freed handle is a lookup miss and can
   abort naming the worker.
1. **It pins the representation.** The worker can never relocate, spill or re-place a
   buffer without the caller knowing, which is precisely the transparency this is for.
1. **Not unique across workers.** Two workers independently allocate the same numeric
   value. Identity is `(topology_id, value)` and never `value` alone, so any cache keyed on
   the value would alias two machines' memory.

### What it is

**A worker-assigned address in a synthetic 64-bit space, partitioned per worker**, with
the plugin resolving `(topology_id, address)` to a region by interval lookup — the
containing region plus the offset within it. Offsetting is plain integer addition, so
`vx_advance_ptr` and every existing pointer expression keep working untouched, while the
worker keeps an indirection it can move bytes behind.

**Mint them non-canonical** — bits 48..63 set. On x86-64 and AArch64 an address with a
non-canonical top half is never returned by `malloc`, `cudaMalloc`, or `mmap`, so:

- Collision with a genuine pointer is impossible *by construction*, not by convention,
  which is what makes the `is_device_ptr` hazard above go away rather than being
  documented around.
- An accidental host dereference **faults immediately** instead of reading garbage. That
  matters more than it sounds: the `matmul_ane` segfault in M2/M3 was a host loop
  dereferencing a staged device pointer, and the only reason it was diagnosable is that it
  faulted. A remote handle that is a plausible host address would have read whatever was
  there and produced wrong numbers.

The offset arithmetic stays inside the region's span, so it does not disturb the tag: a
region is at most a few GiB and the tag lives 48 bits up.

### Is this a GID?

The question a reader of this codebase will ask, and the answer is *structurally yes,
semantically no* — worth writing down because reusing `Gid` would be a plausible-looking
mistake.

The shape is shared: a wide integer standing in for a thing resolved through a registry
rather than dereferenced, with bit-fields carrying provenance. And one GID rule should be
copied outright: **one codec, the only place the word is read or written**. That rule
exists in `gid.rs` because word 2 got triple-booked with two disagreeing flags (#193), and
a handle packing a partition tag beside an address is the identical hazard.

The two defining GID properties are ones a handle must not have:

| | GID | remote handle |
|---|---|---|
| minting | content-addressed; workers agree *without coordinating* | authority-minted by one worker |
| same value in two places | the same entity, by construction | *different memory on different machines* |
| lifetime | immutable identity | allocated then freed; can dangle |
| arithmetic | meaningless | required, by `vx_advance_ptr` |

A GID answers *what is this thing, canonically, everywhere*. A handle answers *where are
these bytes, on one machine, right now* — identity against location.
`content_addressed_workers_agree_without_coordinating` is the GID system's headline
guarantee and is exactly what must be **false** here: two workers allocating independently
must produce values meaning different memory, which is why identity is
`(topology_id, address)` and never the address alone.

So: borrow the codec discipline and the registry pattern; do not borrow the identity
model. Reusing `Gid` would import content-addressing into something that must be
authority-minted, and the failure mode is silent aliasing between two machines' memory.

### What still has to be checked

Pointer arithmetic that leaves a region — `vx_advance_ptr(w.wq, huge)` — must not silently
resolve into the *next* region. Interval lookup gives this for free if regions are not
made adjacent in the synthetic space; leaving a guard gap between them turns an
out-of-bounds offset into a lookup miss instead of a valid handle for the wrong buffer.
This is the same argument as a guard page and should be spelled the same way.

## The agent

A process on the worker machine that owns its GPUs and speaks the three messages. It
needs the same outlined kernels the caller has, which is the honest cost of this design:
**the artifact ships, once, at setup.** That is already how a rented pod works
(`make_gpu_bundle.sh`), so it is not a new class of problem — but it does mean a remote
worker is not a bare machine, and pretending otherwise would be the same kind of
optimism as declaring NVLink bandwidth before seeing the box.

Not in scope, and worth writing down so it is not silently assumed: authentication,
multi-tenancy, reconnection, partial failure. A dispatch that fails to reach its worker
should abort with the worker's name, in the same spirit as `select_device`'s abort when a
launch names a GPU the machine does not have. A demo that hangs is worse than one that
stops.

## Staging

Each step is separately testable, and each one is useful even if the next never lands.

1. **A loopback agent, no network.** The plugin talks to an in-process agent over the
   same three messages, with handles in a table. This exercises marshalling, handle
   lifetime and the slot round-trip on a laptop, and it is where the encoding gets
   debugged rather than on two rented pods.
1. **Two processes on one machine**, over a unix socket. Adds framing, and the first real
   test that the artifact assumption holds.
1. **Two machines.** Adds only the transport. If step 2 was honest, this is a change of
   address.
1. **llama2.vx unchanged.** The acceptance criterion for the whole issue: prefill on a
   worker in one place and decode on a worker in another, with the *same program text*
   that ran on two GPUs in one box, differing only in what the machine file says those
   two workers are.

Step 4 is what makes the claim. Steps 1 through 3 are what make step 4 not a debugging
session on rented hardware — which is the lesson M1 through M4 kept re-teaching, most
expensively when an instrument reported a working run as a failed one.

## What this does not change

The compiler. No new syntax, no new placement rule, no change to admission. A program
that runs on two GPUs in one box should run on two machines with a different machine
file and an unchanged source, and if any part of this design requires editing
`llama2.vx`, that part is wrong.

That is not a slogan; it has already done work. The first draft of the handles section
said "opaque `uint64_t`", and the thing that refuted it was reading the program: seven
`vx_advance_ptr` calls per layer, doing arithmetic on a staged pointer. The criterion
found the defect before any code was written, which is the argument for stating it
up front rather than discovering it as a porting cost.

## Open questions

- **Does a remote `spawn` need a different cost model?** The fleet files already declare
  NIC and remote edges and `src/arch.rs` routes across them, so admission can price it.
  Whether a placement whose handoff does not fit its declared link budget should be
  *refused* — as an over-capacity tile already is — or merely costed, is a design
  question, and refusing is the more Vx answer.
- **What happens to `outkind=slot` when the result is large?** The reply carries it back
  by value. For a projection that is 288 floats; for something that is not, this becomes
  the thing to fix, probably by leaving the result resident and returning a handle.
- **Does the handle table belong in the plugin or the runtime?** In the plugin, on the
  argument that `vx_plugin_*` is the whole seam and a second registry above it would be a
  second source of truth about where a buffer lives.
