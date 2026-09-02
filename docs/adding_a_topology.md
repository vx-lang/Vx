# Adding a topology

*Step by step, for someone who has not done it before.*

Related: [`lang/hosts_and_machines.md`](lang/hosts_and_machines.md) (the `--machine` and `--host`
flags), [`../fleet/README.md`](../fleet/README.md) (the SKU files and their shared space names),
[`memory_algebra.md`](memory_algebra.md) (the model these declarations feed), and
[`custom_transfer_contract.md`](custom_transfer_contract.md) (writing your own code for an edge).

______________________________________________________________________

## The short answer

**A topology is a declaration, not a compiler change.** You write a `Topology` block, and the
compiler picks it up. New hardware is a new file.

Adding a *built-in* topology kind (a new variant inside the compiler, like `GPU` or `NPU`) is a
different and much rarer job. It is the last section of this document, and you almost certainly do
not need it. Everything before that section needs no Rust at all.

______________________________________________________________________

## Part 1 — declaring a topology

### Step 1: declare the memory it holds

A topology has to name a memory space, and that space has to exist. Declare it first:

```rust
Memory TPU_HBM {
  capacity: 32 GiB, bandwidth: 1200 GB/s, scope: device, managed: explicit
}
```

`capacity` and `bandwidth` are what the admission checks read. `scope` says where the memory lives
on the part, and `managed` says whether the hardware keeps it coherent. See
[the memory fields](#memory-fields) below.

### Step 2: declare the topology

```rust
Topology TPU {
  arch: nvptx64,
  memory: Memory::TPU_HBM,
  visible: [ Memory::TPU_HBM ],
  transfer Memory::CPU_DRAM -> Memory::TPU_HBM : 200 copy_engine,
  transfer Memory::TPU_HBM -> Memory::CPU_DRAM : 200 copy_engine,
}
```

`memory:` is the only required field. It names the space this device holds — the one a value gets
when the program says `Topology::TPU` and nothing more.

`visible:` lists every space this device can address. Leaving it out means the device sees exactly
its own memory. A space a device cannot see is a space it cannot read, and that is the rule behind
the `E6003` diagnostic that tells a program to transfer first.

`transfer` clauses are the edges of the cost graph: which moves are possible, and what they cost.
A device with no edge to host memory can be declared, and the compiler will warn (`W1026`) that
nothing can reach it.

### Step 3: name a host, if data stages through host memory

A transfer whose other end is `Memory::CPU_DRAM` needs a declared host. Without one:

```
Error[E6014]: this program stages through host memory, but no host was declared:
  the transfer CPUDRAM -> Custom("TPU_HBM") has an end nothing describes
  note: pass --host <file> to name the host, or --host default for the machine
        compiling this. `--machine` describes the accelerator only.
```

`--host default` describes the machine doing the compiling, which is what you want while
developing. A real host file declares `Memory::CPU_DRAM` and nothing else; see
`fleet/host-x86-e5-2666v3.vx`.

### Step 4: use it

```rust
fn main() -> i32 {
  let a : Tensor<f32, [4, 4]> = Tensor<f32, [4, 4]>::new();
  let on_tpu = transfer(a, Memory::TPU_HBM);
  let out = spawn on(Topology::TPU) {
    on_tpu
  };
  return 0;
}
```

```
$ vxc program.vx --host default --action emit-mlir
```

The emitted module carries the topology through by name:

```mlir
%2 = "vx.transfer"(%cast) <{target_topology = 737069050 : i32}>
       {capacity = 34359738368 : i64, managed = "explicit", scope = "device", space = "TPU_HBM"}
%3 = vx.spawn topology(912088492) { vx.yield %2 } {arch = "nvptx64", topology_name = "TPU"}
```

The dispatch id is a hash of the declared name, taken from the range at and above
`CUSTOM_DISPATCH_ID_BASE` (3000), so a declared topology can never collide with a built-in one.

### Step 5: move it into a machine file, when it describes a SKU

A `Topology` block in the program file is fine for a test or a one-off. To admit the *same program*
against different hardware, put the declarations in their own file and pass it with `--machine`:

```
$ vxc program.vx --machine fleet/my-part.vx --host default
```

A machine file declares hardware and nothing else — no functions. It is a peer of the program
rather than an import, so the program never names it, which is what lets one program text be
checked against several parts. `fleet/` has thirteen worked examples.

If you are adding a SKU that programs should be portable across, use the space names the other
fleet files already use (`HBM`, `L2`, `SMEM`) rather than inventing new ones — that shared
vocabulary is what makes one program text admissible against every file in the directory.

______________________________________________________________________

## Field reference

### Topology fields

| Field | Meaning |
|---------------------|-------------------------------------------------------------------------|
| `memory:` | The space this device holds. **Required.** |
| `visible: [..]` | Every space it can address. Defaults to just its own memory. |
| `arch:` | The instruction set it executes: `x86_64`, `aarch64`, `nvptx64`, `amdgcn`. |
| `transfer A -> B` | An edge in the cost graph. Repeatable. |

A `transfer` clause takes an optional cost (`: 300`, or a bandwidth like `: 64 GB/s`), an optional
consistency grade (`relaxed` or `sync`, default `sync`), and an optional `copy_engine` marker.
`copy_engine` declares that hardware can drive the hop, and it is what makes `raw::async_copy`
legal in a lowering for that edge. Omitting the cost declares that the move is possible and leaves
the number to be derived from the endpoints.

### Memory fields

`capacity`, `bandwidth`, `clock`, `granule`, `within`, `scope`, `managed`, `replicas`, `crossing`,
`sequenced`, `streamed`, and the per-level `device` / `sm` / `cta` / `thread` forms.

`scope:` takes `device`, `sm`, `cta`, or `thread`, and is how a space maps to a target address
space. `within:` nests one space inside another and is how a capacity check knows that a tile in
`SMEM` is also inside `HBM`.

______________________________________________________________________

## What will stop you

These are the errors a first attempt actually hits, in the order you are likely to meet them.

| Diagnostic | Cause |
|-------------|--------------------------------------------------------------------------------|
| `E6014` | A transfer touches host memory and no host was declared. Pass `--host default`. |
| `E6016` | The name is already a built-in (`Topology GPU { .. }`), or two declared names collide on a dispatch id. |
| `E6003` | A value is read on a device whose `visible:` list does not include the space it lives in. |
| `E6009` | A tensor does not fit the declared `capacity`. This one is the point of the exercise. |
| `W1026` | The topology's memory has no transfer edge from the host, so nothing can reach it. |

One more, which only appears on the AST backend (`--legacy-codegen`) and only when a tensor is
*placed* in the space rather than transferred into it:

```
Codegen Error: topology 'Acc_RAM' has no memory space that maps to this target's
address spaces; declare its memory with a `scope:` (device/sm/cta/thread)
```

Declaring `scope:` on the memory fixes it. A space with no `scope:` has no honest target address
space, and defaulting one would put the data somewhere the program did not ask for.

______________________________________________________________________

## A caveat worth knowing

Place values on your new topology with `transfer(x, Memory::TPU_HBM)`, or by writing the space in
the type (`Tensor<f32, [4, 4], Memory::TPU_HBM>`). Both work.

Writing the *device* in the type (`Tensor<f32, [4, 4], Topology::TPU>`) does not work yet for a
declared topology: the value is refused as unreachable from the very device it sits on. The
placement's space is filled in during name resolution, and the type checker runs before that, so
the checker still sees the provisional like-named guess (`Memory::TPU`) instead of the declared
`Memory::TPU_HBM`. Tracked as Vx#431. Built-in topologies are unaffected, since their space is
correct from the moment the placement is built.

______________________________________________________________________

## Part 2 — adding a built-in topology kind

Only do this for hardware the compiler has to reason about structurally, where a declaration cannot
express what it needs to know. A declared topology already gets a dispatch id, an address space, a
cost graph, visibility rules and codegen. Prefer Part 1.

If you are sure, every site that needs an arm:

**`src/syntax/types.rs`**

1. A variant on `Topology`. Add an index (`Box<Expr>`) if more than one such device can be named.
1. A variant on `TopologyKind`.
1. An arm in `Topology::kind()`.
1. An arm in the hand-written `PartialEq`. This one has a tripwire: a forgotten variant reaches the
   fallback arm and compares unequal *to itself*, so a `debug_assert` fires there saying exactly
   that. Do not silence it. If you add a `Hash` impl, it must agree with this equality.

**`src/parser/types.rs`**

5. A name arm in `parse_topology`. Note that any unmatched identifier becomes `Topology::Custom`,
   so there is no "unknown topology" error to catch a typo — and adding a built-in silently changes
   the meaning of that name for any program that was declaring it.

**`src/arch.rs`**

6. A band in `topology_dispatch_id`. The hundreds 0–700 are taken, slices use 2000–2999, and
   declared names start at 3000.
1. An arm in `builtin_default_space` — the space this kind holds.
1. An arm in `first_device_of` — device zero of this kind.
1. An entry in `builtin_descriptors` — its default space and its visibility list.
1. An arm in `builtin_space_owner`, if the kind brings a new built-in memory space with it.

A new built-in memory space also needs a name in `MemorySpace::from_name` and its inverse
`MemorySpace::name`, in `src/syntax/types.rs`.

Most of these the compiler will make you do: `kind()`, `topology_dispatch_id`,
`builtin_default_space`, `first_device_of` and `builtin_space_owner` all match exhaustively, so a
new variant fails to compile until each has an arm. Three sites will stay quiet, and they are the
ones to check by hand:

- `parse_topology` ends in a catch-all, so until you add the name arm your new kind parses as a
  `Custom` topology with the same name and behaves almost right.
- `builtin_descriptors` is a table of inserts rather than a match, so a missing entry is a topology
  with no descriptor rather than a compile error.
- `PartialEq` ends in a catch-all too. That one at least shouts: the `debug_assert` fires the first
  time the variant is compared with itself in a debug build.
