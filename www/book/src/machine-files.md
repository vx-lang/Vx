# Machine files

Most compilers hard-code a cost model. Vx reads one.

A **machine file** describes the memory hierarchy and interconnect of a real part. The compiler
admits or rejects your placements against it, and derives transfer costs from it, before a binary
exists.

## An example

```rust
Memory HBM  { capacity: 80 GiB, bandwidth: 3.35 TB/s, managed: explicit, scope: device }
Memory L2   { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s, managed: cached }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc,
              clock: 1.98 GHz, replicas: 132, granule: 1 KiB, scope: sm }

Topology Device {
    arch: nvptx64,
    memory: Memory::HBM,
    transfer Memory::CPU_DRAM -> Memory::HBM : 63 GB/s,
}
```

That is an H100, in eleven lines.

## The fields

**`capacity`** — how much the space holds. Checked against the working set of anything you place
there.

**`bandwidth`** — either a rate (`3.35 TB/s`) or a per-cycle figure with a `clock` (`128 B/cyc` at
`1.98 GHz`). The second form is how scratchpads are usually specified in vendor documentation.

**`within`** — containment. `L2` sits inside `HBM`. The containment relation must be acyclic, a
child may not exceed its parent's capacity, and scope narrows as you descend.

**`managed`** — `explicit` if software moves the data, `cached` if hardware does.

**`scope`** — who can see it. `device` is visible to the whole accelerator; `sm` is private to one
streaming multiprocessor.

**`replicas`** — how many copies of the space exist. 132 SMs means 132 scratchpads, and a placement
is admitted against one of them, not their sum.

**`granule`** — the allocation quantum. A 1 KiB granule means a 100-byte tensor occupies 1 KiB, and
admission rounds accordingly.

## Units are exact

SI prefixes are decimal and IEC prefixes are binary:

| | |
| --- | --- |
| `GB` | 10⁹ bytes |
| `GiB` | 2³⁰ bytes |
| `TB/s` | 10¹² bytes per second |

Conversions are exact integer arithmetic, never floating point. A figure copied off a vendor
datasheet means precisely what the datasheet meant, and does not drift by a fraction of a percent on
the way through the compiler.

## What the compiler derives

From that declaration alone, before any code is generated:

**Admission** — whether a tensor's working set fits the space it is placed in, with granule
rounding applied. A rejection carries the space, the amount required, the amount available and the
margin.

**Routing** — the cheapest legal path between two spaces, over the declared transfer graph. If you
transfer from a space that has no declared route to the destination, that is an error rather than a
silently inserted staging copy.

**Transfer cost** — a roofline over the containment tree. A containment hop is charged at *both*
endpoints, because data has to leave the parent as well as enter the child.

**Coherence of the model itself** — `within` is acyclic, no child exceeds its parent, scope narrows
downward, and every edge has exactly one cost source. A machine file that contradicts itself is
rejected as a machine file, before your program is even considered.

## The bundled fleet

The repository ships machine files for real parts under `fleet/`: H100, H200, B200, A100, MI300X,
Apple M4 unified memory, multi-GPU nodes and hosts. Each one cites its sources, and marks figures
that have not been validated against hardware as unverified.

That last part matters. A declared bandwidth is often the vendor's *hardware peak* rather than an
achievable rate — Apple's 120 GB/s for the base M4 is the memory system's ceiling, not what a copy
loop will reach. The fleet files state the vendor's claim rather than a number already fitted to a
measurement, so the gap between prediction and reality stays visible instead of being quietly
tuned away.

## Using one

```bash
vxc --machine fleet/h100-sxm.vx program.vx -o program
```

`--machine` prepends the declarations to your compilation unit. They are visible to capacity and
placement checks exactly as if you had written them inline. A name declared by both the machine file
and the program is an error, never a silent shadow.

`--machine` describes an accelerator and says nothing about the host it hangs off. If your program
stages through host memory, declare the host too:

```bash
vxc --machine fleet/h100-sxm.vx --host default program.vx -o program
```

`--host default` means "the machine doing the compiling". A host file declares no `capacity`, on
purpose: host memory is virtual, and a tensor larger than physical RAM pages rather than failing. A
hard limit there would reject programs that actually run.

## Writing your own

Start from the closest file in `fleet/` and change the numbers. The compiler will tell you if the
result is incoherent, which makes the edit-check loop fast. Two habits worth keeping:

1. **Cite every figure.** The fleet files carry the source of each number in a comment. When a
   prediction is wrong, the first question is always whether the model or the measurement is at
   fault, and a citation answers it in seconds.
1. **Mark what you have not verified.** A number from a datasheet and a number from a benchmark are
   different kinds of thing, and the difference should survive in the file.
