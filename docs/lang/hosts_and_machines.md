# Hosts and machines

*How a Vx compilation learns what hardware it is compiling for, and what happens when it is not
told.*

Related: [`fleet/README.md`](../../fleet/README.md) (SKU vocabulary and provenance),
[`semantics.md`](semantics.md) (placement and memory spaces), [`abi.md`](abi.md)
(the plugin boundary), and
[`first_class_memory_spaces.md`](../discussions/implementation_plans/first_class_memory_spaces.md).

______________________________________________________________________

## The claim this exists to support

Vx's distinguishing claim is that it refuses a program *before* the machine is rented:

> `--machine fleet/a100-40.vx` → `E6009: transferred tensor needs 61440000000 bytes but memory space 'HBM' has capacity 42949672960`
> `--machine fleet/a100-80.vx` → admitted

One program text, one flag changed, and a verdict that can be checked against silicon afterwards.
A refusal that is later proven right cannot be cherry-picked the way a throughput number can.

A verdict is only as good as what it was computed from. Everything below is about making sure the
compiler never answers a question about hardware it was not told about.

______________________________________________________________________

## Two flags, two different things

```
vxc prog.vx --machine fleet/a100-80.vx --host fleet/host-x86-e5-2666v3.vx
                      ─────┬─────────         ─────────┬──────────────
                           │                           │
              the accelerator: what           the machine it hangs off:
              compute is placed on,           the memory a program stages
              and its device memory           through on the way there
```

Neither is assumed. A compilation with neither flag is an ordinary native build and asks no
fleet-level question at all.

| | `--machine` | `--host` |
|---|---|---|
| Declares | `HBM`, `L2`, `SMEM`, `Topology` | `CPU_DRAM` |
| Capacity | yes — a device's memory is a wall | **no** — a host's memory is virtual |
| Example | `fleet/a100-80.vx` | `fleet/host-x86-e5-2666v3.vx` |
| Loaded as | peer module | peer module |

Both load as *peer modules*: declarations the program never names. That is what makes the same
program text portable across a matrix — swap a flag, not a line of code.

### Why a host declares no capacity

A device's HBM is a hard limit. Ask for 61 GB of an A100-40 and the allocation fails, so a
compile-time refusal is a prediction that will come true.

Host memory is virtual. A tensor larger than physical RAM does not fail — it pages. Declaring
60 GiB and rejecting at 60 GiB would refuse programs that *run*, and a refusal that is wrong is
worth much less than no refusal at all. So a host file states that a host exists and how fast its
memory is; it does not state a budget.

This is a deliberate asymmetry, and it is the reason `--host` is not simply "another `--machine`".

### The same silicon, two files

`fleet/xeon-e5-2666v3.vx` and `fleet/host-x86-e5-2666v3.vx` describe one physical box. The first
describes it as a *machine* — an x86 target compute is placed on, with a `Topology Device` and a
memory hierarchy. The second describes it as a *host* — memory a program stages through.

Both are true. Which one a file says is decided by which question is being asked of it, and passing
the wrong one is caught rather than silently accepted:

```
$ vxc prog.vx --machine fleet/a100-80.vx --host fleet/xeon-e5-2666v3.vx
Error[E6012]: memory space 'HBM' is declared in more than one input
Error[E6012]: topology 'Device' is declared in more than one input
```

A file that declares `HBM` and a `Topology Device` is describing a device. The collision is the
system telling the truth.

______________________________________________________________________

## When a host is required

```
                      ┌─────────────────────────────┐
                      │  does the program stage     │
                      │  through host memory?       │
                      │  (a transfer with CPU_DRAM  │
                      │   on either end)            │
                      └──────────┬──────────────────┘
                            no   │   yes
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
          ┌───────────────┐          ┌────────────────────────┐
          │ no host       │          │ is a machine model in  │
          │ required      │          │ force? (--machine)     │
          │               │          └───────┬────────────────┘
          │ e.g. a kernel │             no   │   yes
          │ whose tensors │      ┌───────────┴──────────┐
          │ are already   │      ▼                      ▼
          │ device-       │  ┌────────────┐   ┌──────────────────┐
          │ resident      │  │ native     │   │ --host REQUIRED  │
          └───────────────┘  │ build:     │   │ else E6014       │
                             │ no host    │   └──────────────────┘
                             │ required   │
                             └────────────┘
```

The rule is deliberately narrow. `--host` exists because no host is assumed, not because every
compilation is a fleet question — a rule that fired on every program would be ceremony, and
ceremony gets suppressed.

### The diagnostic

```
$ vxc prog.vx --machine fleet/a100-80.vx
Error[E6014]: this program stages through host memory, but no host was declared:
              the transfer CPUDRAM -> Custom("HBM") has an end nothing describes
  help: pass --host <file> to name the host, or --host default for the machine
        compiling this. `--machine` describes the accelerator only.
```

### `--host default`

Means *the machine compiling this program*. It is synthesised rather than read from a file, because
for a native host the statement is simply "this one":

```rust
// src/driver.rs
if self.options.host.as_deref() == Some("default") {
    program_arr.push(Program {
        module_path: Symbol::from("<native-host>"),
        memories: vec![MemoryDecl {
            name: Symbol::from("CPU_DRAM"),
            capacity: None,        // virtual memory: no wall to declare
            bandwidth: None,       // unmeasured on an arbitrary build machine
            managed: Management::Explicit,
            ..
        }],
        ..
    });
}
```

`--host default` is an *assertion*, not a shortcut: the program is being compiled for the machine
in front of you, and you are saying so.

______________________________________________________________________

## What this fixed

Before `--host`, `Memory::CPU_DRAM` arrived from the built-ins with no declaration anywhere:

```rust
// src/arch.rs — the built-in host space
MemorySpace::CPUDRAM => 0,                      // dispatch id
MemorySpace::CPUDRAM => Some(AddressSpace::Host),
...
_ => MemorySpace::CPUDRAM,                      // and the default for an unknown placement
```

Every SKU in `fleet/` declares the *link* into it —

```
transfer Memory::CPU_DRAM -> Memory::HBM : 31.5 GB/s
```

— so the edge had a rate while the endpoint had nothing. The compiler was costing a seam against a
source it could not describe, and the last line above is the sharper version of the problem: an
unknown placement *became* the host, silently. Not a wrong number so much as an unasked question,
which is the class of defect [#329](https://github.com/hiraditya/Vx/issues/329) is about.

______________________________________________________________________

## Worked example: the demo matrix

```
                    │ --host default   │ host-x86-e5-2666v3.vx
────────────────────┼──────────────────┼───────────────────────
(none)              │ native build     │  —
fleet/a100-40.vx    │ E6009 at 40 GiB  │ E6009 at 40 GiB
fleet/a100-80.vx    │ admitted         │ admitted
fleet/h100-sxm.vx   │ admitted         │ admitted
fleet/xeon-e5…v3.vx │ E6009 at 60 GiB  │  — (collides: it is a machine)
```

The row that matters is the first `--machine` column: **the same program, the same host, a
different accelerator, a different verdict.** That is the claim, and the host column is what makes
the verdict a statement about a whole machine rather than about a card.

______________________________________________________________________

## Implementation map

| Piece | Where |
|---|---|
| `--host` flag | `src/driver.rs`, `Options::host` |
| host file loaded as peer module | `src/driver.rs::load_and_expand` |
| `--host default` synthesis | `src/driver.rs::load_and_expand` |
| the requirement + `E6014` | `src/hir/check/transfer.rs` |
| diagnostic code | `src/diagnostic.rs`, `DiagnosticCode::E6014` |
| host file | `fleet/host-x86-e5-2666v3.vx` |
| tests | `tests/frontend/fail/host_not_declared.vx`, `tests/optimizations/pass/host_flag_scope.vx` |

The requirement is checked where the transfer path is resolved, and it needs no plumbing from the
CLI. It asks only about declarations:

```rust
let touches_host = source_mem == MemorySpace::CPUDRAM || target_mem == MemorySpace::CPUDRAM;
let host_declared    = env.memories.values().any(|m| m.name.as_ref() == "CPU_DRAM");
let machine_declared = env.memories.values().any(|m| matches!(m.scope, Some(Scope::Device)));

if touches_host && machine_declared && !host_declared { /* E6014 */ }
```

`--host default` satisfies it by *being* a declaration, so there is no flag to consult and no second
code path to keep in agreement with the first.

______________________________________________________________________

## The target *is* the host

There is no `--target`. There was, and it was the source of a confusion worth recording, because
GCC and clang have both carried the same one.

"Target" was coined when a compilation had one machine to name: the one the output runs on. A
heterogeneous program has two, and the word stops being able to pick one. Vx names them:

| | runs what | classical name |
|---|---|---|
| `--host` | `main`, the dispatch calls, the outlined kernels' C interfaces | **the target** |
| `--machine` | the device kernels, once emission lands (#251) | the *offload* target |

The host is the target. It is the machine the program runs on and drives the rest of the system
from; everything else is a machine it dispatches to. Clang reached the same shape from the other
direction -- `-triple` plus `-aux-triple`/`--offload-arch` -- because one flag cannot describe two
machines.

### What the flag did, and why it went

`--target` tagged the *whole module* with one triple, whichever machine that triple named:

```
$ vxc prog.vx --emit-llvm --target nvptx64
target triple = "nvptx64-nvidia-cuda"
```

on a module containing `main` and two calls to `vx_plugin_dispatch_async`. NVPTX has no `main` and
no libffi dispatch. The deleted test `llvm_backends.vx` asserted exactly this as correct
behaviour —

```
// NVPTX: target triple = "nvptx64-nvidia-cuda"
// NVPTX: define i32 @main
```

— so the confusion was not only in the flag; it was pinned as expected. Nothing related the flag to
`--machine` or `--host` either, so `--host <x86 file> --target aarch64` was accepted: a program
admitted against one machine and emitted for another, with the compiler holding both beliefs and
comparing them never.

### What it should be

```
   --host  fleet/host-x86-e5-2666v3.vx  ──▶  host triple    x86_64-unknown-linux-gnu
                                             host layout    e-m:e-p270:32:32-...
                                                  │
                                                  ▼
                                        the module's llvm.target_triple:
                                        main, the dispatch calls, the
                                        outlined kernels' C interfaces

   --machine fleet/a100-80.vx           ──▶  device triple  nvptx64-nvidia-cuda
                                             device layout  e-i64:64-i128:128-...
                                                  │
                                                  ▼
                                        the gpu.module's target, once
                                        kernel emission lands (#251)
```

Three things fall out, and they are the reason this is worth doing rather than tidy:

**Cross-compilation becomes ordinary.** Build on a laptop for an x86 host with an A100 attached.
The bundle shipped to a rented pod today (`scripts/make_gpu_bundle.sh`) exists partly because the
compiler can only emit for the machine it runs on.

**The incoherent combination stops being expressible.** A triple derived from a declaration cannot
disagree with it. If `--target` survives at all it becomes an override that must agree, and a
mismatch is a diagnostic rather than a silently divergent artifact.

**A host file gains a reason to name its architecture.** `fleet/host-x86-e5-2666v3.vx` describes
bandwidth and says nothing about the instruction set, which is a gap only because nothing needed it
yet. The device side already has the vocabulary — `src/arch.rs` maps a `MemorySpace` to an NVPTX
address space (#258), so the address-space half of a device data layout is derivable now.

### The rule for getting there

`--target` staying a flag is right; what changes is where its value comes from and what happens
when it disagrees.

1. **Derive by default.** With no `--target`, the triple and layout come from `--host` (for the
   module) and `--machine` (for device code). A compilation that was told what machine it is for
   does not need to be told twice.
1. **An explicit `--target` is an assertion, and it is checked.** If it disagrees with the
   declarations, that is an error. Not a warning, and not "the flag wins" — a compilation holding
   two beliefs about its target is one that will produce an artifact matching neither.
1. **Reject any combination not understood.** Not "fall back to native", not "use the first one".
   An unrecognised or unmodelled configuration is refused, because a compiler that guesses here
   produces a binary that is wrong in a way nothing downstream can detect. This is the same
   principle the rest of the system already follows: a placement with no declared space is
   `E6003`, a duplicate declaration is `E6012`, a host that was never named is `E6014`. A target
   nobody can vouch for belongs in that list.

The shape is an `arch:` declaration in machine and host files, the triple looked up from it, and
disagreement — or absence of a mapping — made a diagnostic rather than a default.

______________________________________________________________________

## Open

- **Host-to-device is the only direction implemented.** The reverse still lowers to a host copy
  ([#339](https://github.com/hiraditya/Vx/issues/339)).
- **No arm64 host file** yet ([#341](https://github.com/hiraditya/Vx/issues/341)).
- **NUMA is not modelled.** `fleet/xeon-e5-2666v3.vx` flattens two 30 GiB domains into one space;
  `node-8gpu.vx` is the shape that models several memories with edges between them.
- **The default placement remains.** `src/arch.rs` still resolves an unknown placement to
  `CPUDRAM` in four places. `--host` makes the host declarable; it does not yet make the *default*
  go away ([#329](https://github.com/hiraditya/Vx/issues/329)).
