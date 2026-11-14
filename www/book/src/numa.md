# NUMA and the host's memory domains

Every other memory space in this book belongs to an accelerator you have to attach. A two-socket
server has two of them already, and they are the hardest case for the model to describe well —
because unlike a GPU's HBM against a host's DRAM, the two memories here are physically identical.

Node 0's memory and node 1's memory are the same DDR4, at the same speed, from the same order.
Nothing about the bytes differs. The only thing separating them is **which core is asking**.

That makes NUMA a good test of whether placement in the type system is describing locality or
merely labelling hardware. It also makes it the one heterogeneous memory you can experiment with on
a machine you may already own.

## It is the same relation as two GPUs

A NUMA domain is a memory that some execution units reach cheaply and others reach across a link.
`fleet/node-8gpu.vx` already says that about two GPUs, as `HBM` and `PEER_HBM` with a priced edge
between them. A two-socket host is the same shape:

```
Memory HBM {
  capacity: 96 GiB, bandwidth: 140 GB/s, managed: explicit, scope: device, node: 0
}

Memory PEER_HBM {
  capacity: 96 GiB, bandwidth: 140 GB/s, managed: explicit, scope: device, node: 1
}

Topology Device {
  arch: x86_64,
  memory: Memory::HBM,
  visible: [Memory::HBM, Memory::PEER_HBM, Memory::L2],
  transfer Memory::CPU_DRAM -> Memory::HBM : 140 GB/s,
  transfer Memory::HBM -> Memory::L2,
  transfer Memory::HBM -> Memory::PEER_HBM : 62 GB/s,
  transfer Memory::PEER_HBM -> Memory::HBM : 62 GB/s
}
```

The names are roles rather than materials, as everywhere else in `fleet/` — there is no HBM within a
mile of this part, and `HBM` means "this SKU's device memory". What the pair says is that a tile
lives in one domain and reaching it from the other crosses a link with a price.

`fleet/xeon-8275cl.vx` is this file in full, for the Xeon Platinum 8275CL that AWS sells as
`c5.metal`.

## `node:` is the field that is obeyed

Every other field on a `Memory` describes the space so the compiler can admit or refuse a
placement. `node:` is different: a backend has to **act** on it.

```
Memory PEER_HBM {
  capacity: 96 GiB, bandwidth: 140 GB/s, managed: explicit, scope: device, node: 1
}
```

The space's name cannot carry this. A declared name reaches the runtime as an FNV hash, by design,
and a hash cannot be turned back into a node number — while `mbind` needs exactly that. So a space
declaring a node is given a banded dispatch id instead (`arch::NUMA_DISPATCH_BASE + N`, in the
1100..1999 range), and the backend decodes it.

Two consequences follow, and both are visible:

- `transfer(t, Memory::PEER_HBM)` binds the allocation to node 1, through the `mbind` syscall. No
  libnuma dependency and no link flag; it compiles out entirely off Linux.
- A dispatch whose arguments are placed pins itself to that node before running the kernel.

Zero is a valid node and means the first one, so `node: 0` is a declaration rather than an absence.
The upper bound is the dispatch band's rather than the hardware's; a node number past it is a parse
error rather than an id that would collide with something else.

## What it buys at compile time

The compile-time half needs no hardware at all. Declaring the domains separately means a tile too
large for one of them is refused, where a model that flattened the box into a single space would
admit it:

```
Error[E6009]: transferred tensor needs 42949672960 bytes
              but memory space 'HBM' has capacity 32212254720 bytes
```

That program was being told something false before. The box really does have 60 GiB; no *node* has
40 GiB, so the tile could not have been local however it was allocated.

The hop between domains is priced like any other edge, and `--diagnostics-json` will hand you the
per-route figures:

```
CPU_DRAM -> HBM      | 8 GiB | 61.4 ms  | link_rate
HBM -> PEER_HBM      | 8 GiB | 138.5 ms | link_rate
```

## Checking the model against the machine

`utils/campaign/run_numa_probe.sh` measures all four (cpu node, memory node) pairs and compares the
measured remote/local ratio against the one the compiler derives, reading the prediction out of
`--diagnostics-json` so it cannot drift from the machine file.

```bash
sudo apt install numactl
./utils/campaign/run_numa_probe.sh                          # defaults to fleet/xeon-8275cl.vx
MACHINE=fleet/xeon-e5-2666v3.vx ./utils/campaign/run_numa_probe.sh
```

It compares **ratios rather than absolute times**, deliberately. Declared bandwidths are memory
controller peaks, and a copy moves two bytes of traffic per byte copied, so the model over-predicts
any single edge by at least a factor of two. Both sides of a ratio carry that error and it cancels.

On a `c5.metal` the model predicts 2.258 and the hardware gives 2.348, a 4% error.

## The probe refuses some machines, and that is the point

A virtualized instance can report two NUMA nodes, honour `--membind`, and still spread the pages
across both sockets underneath. Neither `/proc/PID/numa_maps` nor `move_pages()` will tell you.
Both answer truthfully about *guest* nodes, and it is the guest nodes that are not backed by
locality — `move_pages()` reports 64 of 64 probed pages on the requested node on a `c5.metal` where
a remote read really does cost 2.35×, and the identical 64 of 64 on a `c4.8xlarge` where it costs
nothing. The kernel's own placement query cannot separate the two.

An AWS `c4.8xlarge` does exactly this, on two separately provisioned instances. CPU lists, node
sizes and ACPI distance table all look right, and then all four pairs measure identically — at a
bandwidth **above what one socket can deliver**, which is the only thing in the whole picture that
cannot be explained away.

So the probe tests the machine before it reports anything: bind to one node, interleave across
both, and if the two agree within 10% then the bind confined nothing and every number above it is
measuring one undivided pool. It says so and exits.

If you want NUMA on EC2, use a bare-metal instance. There is nothing between the guest and the
sockets there.

## What placement is worth, honestly

Measured on `c5.metal`, 8 GiB, 96 threads:

| Placement | Bandwidth | vs interleaved |
| --------------------------- | ---------- | -------------- |
| First-touch (naive default) | 96.8 GB/s | 0.54× |
| `numactl --interleave=all` | 180.5 GB/s | 1.00× |
| Placed per node | 242.1 GB/s | 1.34× |

**1.34× is the figure, not 2.50×.** Interleaving costs nothing, needs no source change, and is what
a competent operator already does on a two-socket box. Comparing against naive first-touch would be
comparing against the worst case.

That 1.34× also needs the *threads* placed as well as the pages. Vx's host backend runs an
outlined kernel on a pool of threads. After a placed dispatch, the pool uses only the CPUs of the
NUMA node that holds the arguments. `VX_HOST_THREADS=n` sets the number of threads.

`utils/campaign/placement_bench.c` is the benchmark those numbers come from.

## Seeing it happen

```bash
VX_DISPATCH_VERBOSE=1 vxc prog.vx --host default --machine $HOME/.vx/current/fleet/xeon-8275cl.vx --run
```

```
[Vx x86] staged 268435456 bytes on NUMA node 0
[Vx x86] handed 268435456 bytes to NUMA node 1
[Vx x86] pinned to NUMA node 1 for this dispatch
```

`VX_NUMA_NO_AFFINITY=1` turns the pinning off, for a process that manages its own affinity — and
for telling the two states apart with one binary when measuring.

A binding that cannot be honoured — an offline node, a kernel without NUMA support, a machine that
is not Linux — warns once and leaves the memory unplaced. The program is correct in every one of
those cases and only slower, so it is not an error; but a run expected to be placed should not
quietly read as one that was.

## Where to next

- [Machine files](machine-files.md) — the full field reference for `Memory` and `Topology`
- [Topologies and memory](heterogeneous.md) — how placement and reachability are checked
