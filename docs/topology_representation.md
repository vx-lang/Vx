# Topology: Representation & Semantics

`Topology` names a hardware execution context — a CPU, a GPU, an NPU tile, the
Apple Neural Engine, and so on. In Vx it plays **two distinct roles**, and the
same `Topology::X` surface syntax is interpreted differently depending on where
it appears:

1. **A compile-time placement concept.** Where computation runs and where data
   lives. This drives `spawn on`, `Pinned<T, Topology>`, the `on Topology::X`
   function annotation, memory-space affinity, and transfer checking. These uses
   are resolved and enforced entirely at compile time; nothing survives to run
   time except the device dispatch id baked into the emitted `vx.spawn`.

1. **A first-class runtime value** (added in
   [#206](https://github.com/vx-lang/Vx/issues/206)). A topology can be stored,
   passed, compared, and dispatched on at run time — its value is a small integer
   **discriminant**. This is what makes `Vec<Topology>` and value-based device
   dispatch possible.

This document explains both roles, the concrete representation of each, and how
they coexist safely.

> [!NOTE]
> For the *placement* engine (context tracking, memory spaces, transfer
> obligations) see [`hardware_topology_plan.md`](hardware_topology_plan.md) and
> [`spawn_on.md`](spawn_on.md). This document focuses on **representation** — how
> a topology is spelled, typed, and lowered — and on the runtime-value role.

## Syntax positions

`Topology` and `Topology::X` mean different things by position. The parser
disambiguates on the token that follows the `Topology` keyword.

| Position | Example | Parsed as | Role |
|---|---|---|---|
| **Placement annotation** | `fn k() on Topology::GPU`, `Pinned<T, Topology::GPU>`, `spawn on (Topology::GPU)` | a `Topology` enum (compiler-internal), via `parse_topology` | compile-time |
| **Type position** | `let t : Topology`, `Vec<Topology>`, `-> Topology` | `Type::Scalar(I32)` — the discriminant type | runtime value |
| **Value position** | `Topology::GPU` (assigned, pushed, compared, returned) | `Expr::Topology`, typed `i32` | runtime value |

The rule in the parser: a **bare** `Topology` keyword *not* followed by `::` is
the value type (an `i32`); `Topology::X` in a placement slot is consumed by the
dedicated placement parsers before it reaches the general type parser.

```vx
// placement (compile-time): pins the function to the GPU context
fn gpu_kernel() on Topology::GPU -> void { /* ... */ }

// value (runtime): `target` is an i32 discriminant; `Topology::GPU` is 500
fn batch_for(target : Topology) -> i32 {
  if target == Topology::GPU { return 1024; }
  if target == Topology::ANE { return 256; }
  return 32;
}
```

## Runtime representation: the discriminant

A topology **value** lowers to a stable `i32` **dispatch id** — the single source
of truth is `arch::topology_dispatch_id` (`src/arch.rs`). The same id is used by
the backend to target a device in `vx.spawn topology(N)`, so a value and a
placement of the same topology agree by construction.

| Variant | Dispatch id |
|---|---:|
| `CPU`, `Current` | `0` |
| `NPU[i]` | `100 + i` |
| `AccCore[i]` | `200 + i` |
| `AMX` | `300` |
| `ANE` | `400` |
| `GPU` | `500` |
| `CpuAvx512` | `600` |
| `CpuNeon` | `700` |
| `Slice(..)` | `2000 + FNV(base, start, end) % 1000` |
| `Custom(name)` | `3000 + FNV(name) % (INT32_MAX - 2999)` |

A declared name's id is a hash, so two names can still collide; E6016 refuses that
rather than resolving it. The range used to be `1000..1999`, where a thousand slots
collided constantly — 40 declared names collided 55% of the time, and a 250-space
corpus lost 42 of its memory descriptors. `runtime/vx_manifest.h`
mirrors the hash, so the two must change together.

`Topology::Current` is resolved to the enclosing `active_topology` at type-check
time *before* it becomes a value, so `let d = Topology::Current` inside a
`Topology::GPU`-pinned function yields `500`, not a "current" sentinel.

The type of a topology value is `i32`. There is deliberately **no distinct
nominal `Topology` type** at the value level today — a `Topology` value and an
`i32` are interchangeable. This keeps `Vec<Topology>` == `Vec<i32>` and requires
zero new codegen. (See [Limitations](#limitations--future-work).)

### Lowering

```
Topology::GPU   (Expr::Topology)
  ── type-check ─▶  Type::Scalar(I32)        // src/hir/expr.rs
  ── codegen ────▶  %c500_i32 = arith.constant 500 : i32   // src/codegen/lower/expr.rs
```

Before #206, `TopologyExpr` lowering `panic!`ed ("Should not be evaluated
directly") — a topology could never reach run time. Now it emits its dispatch-id
constant.

## Placement representation (compile-time)

In its placement role a topology is carried in **types and statements**, never as
an SSA value:

- `Pinned<T, Topology>` — a value resident in a topology's memory. `Type::Pinned`
  holds the `Topology` directly.
- `fn f() on Topology::X` — pins a function's body to a context. Calling an
  `on Topology::X` function from a different context is a hard error
  (`E6001: requires topology 'X', but is called from 'Y'`); the legal way to
  reach it is inside a matching `spawn on (Topology::X) { ... }`.
- `spawn on (Topology::X) { ... }` — runs a block in a context. It lowers to
  `vx.spawn topology(N) { ... vx.yield }` with `N` = the dispatch id (and an
  optional device `plugin` attribute, e.g. `{plugin = "Apple_NPE_v1"}` for ANE).
- **Memory affinity.** Each topology has a default `MemorySpace`
  (`TransferCostGraph::default_memory_for`); crossing spaces needs an explicit
  transfer (`.to_device`, `.to_host`, `transfer`). Address spaces:
  `CPUDRAM=0`, `NPU/GPU HBM=1`, `LocalSRAM=2`, `NIC/Remote=3`.

Comptime comparisons of topologies (`Topology::Current == Topology::CPU_AVX512`,
common in `stdlib/std/tensor.vx`) are **folded away before codegen**, so they
never rely on the runtime discriminant. The comptime `if` branch that doesn't
match is eliminated, which is what lets device-specific code compile only for its
device.

## Value-based device dispatch

The two roles compose: dispatch on a runtime topology **value**, then **place**
work with `spawn on`. `impl dispatch for topology_test` in
[`tests/optimizations/pass/topology_dispatch.vx`](../tests/optimizations/pass/topology_dispatch.vx):

```vx
impl dispatch for topology_test {
  fn run(self : &topology_test, target : Topology) -> Topology {
    if target == Topology::GPU {                    // runtime: %arg1 == 500
      spawn on (Topology::GPU) { gpu_kernel(); };   // placement: vx.spawn topology(500)
      return Topology::GPU;
    }
    if target == Topology::ANE {                    // %arg1 == 400
      spawn on (Topology::ANE) { ane_kernel(); };   // vx.spawn topology(400) {plugin = "Apple_NPE_v1"}
      return Topology::ANE;
    }
    spawn on (Topology::CPU) { cpu_kernel(); };      // vx.spawn topology(0)
    return Topology::CPU;
  }
}
```

Lowered (abridged):

```mlir
func.func @topology_test$run(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
  %c500_i32 = arith.constant 500 : i32
  %0 = arith.cmpi eq, %arg1, %c500_i32 : i32          // dispatch on the value
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  vx.spawn topology(500) { func.call @gpu_kernel() : () -> () ; vx.yield }
  ...
}
```

> [!NOTE]
> `spawn` lowers to the MLIR `async` dialect in the JIT path, which the JIT's
> `mlir-translate` does not register — so `spawn`-based dispatch is verified via
> `--action emit-mlir` + FileCheck, not JIT execution. Pure value dispatch (no
> `spawn`, e.g. `batch_for` above) JIT-executes normally; see
> [`tests/backend/pass/vec_topology.vx`](../tests/backend/pass/vec_topology.vx).

## Why the two roles are safe together

Making `Topology::X` a runtime value did **not** disturb placement, because:

- Placement forms (`Pinned`, `on`, `spawn on`) parse the topology through the
  dedicated placement parsers and store it in the type/statement — they never go
  through `Expr::Topology`'s value type.
- Comptime placement comparisons are folded before codegen and never reach
  `TopologyExpr::lower`.
- If a topology comparison *did* reach run time, an `i32` discriminant comparison
  (`500 == 0`) yields the same boolean the comptime fold would have.

## Limitations & future work

- **No nominal value type.** A `Topology` value is an `i32`; the type system does
  not distinguish `Topology` from `i32`. A dedicated nominal enum (unifying with
  the ADT work in [#111](https://github.com/vx-lang/Vx/issues/111) /
  [#98](https://github.com/vx-lang/Vx/issues/98)) would give type safety and
  exhaustiveness at the cost of new codegen.
- **Parametrized variants: constant indices only.** `NPU[i]` / `AccCore[i]`
  encode a *constant* index into the id (`100 + i`, `200 + i`), so `NPU[0]` and
  `NPU[1]` are distinct values. A **non-constant** index (a runtime expression)
  falls back to index `0` (`topology_index`), and `Slice(..)` collapses to `900` —
  those cases do not round-trip through a runtime value.
- **Discriminant stability.** The ids in `topology_dispatch_id` are an ABI-ish
  contract shared with the backend dispatcher; renumbering them changes emitted
  `vx.spawn topology(N)` and any persisted topology values.

## Implementation reference

| Concern | Location |
|---|---|
| Discriminant table | `arch::topology_dispatch_id` (`src/arch.rs`) |
| Default memory space / address space | `TransferCostGraph::default_memory_for`, `arch::topology_address_space` |
| Parse bare `Topology` as value type | `Parser::parse_type` (`src/parser/types.rs`) |
| Parse placement `Topology::X` | `Parser::parse_topology` (`src/parser/types.rs`) |
| `Topology::X` value type-check (`i32`, `Current` resolution) | `check_expr_type_flag` → `Expr::Topology` (`src/hir/expr.rs`) |
| `Topology::X` value codegen (dispatch-id constant) | `impl LowerToMelior for TopologyExpr` (`src/codegen/lower/expr.rs`) |
| `spawn on` lowering | `src/codegen/lower/` (emits `vx.spawn topology(N)`) |

## References

- [`hardware_topology_plan.md`](hardware_topology_plan.md) — placement/context engine
- [`spawn_on.md`](spawn_on.md) — `spawn on` semantics
- [`docs/lang/types.md`](lang/types.md) — the type system
- Issue [#206](https://github.com/vx-lang/Vx/issues/206) — first-class runtime Topology
- Tests: [`tests/optimizations/pass/topology_dispatch.vx`](../tests/optimizations/pass/topology_dispatch.vx),
  [`tests/backend/pass/vec_topology.vx`](../tests/backend/pass/vec_topology.vx)
