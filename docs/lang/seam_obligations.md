# Seam Obligations (cross-device transfer safety)

A **seam** is one hop of a cross-device transfer — each `MemorySpace` edge the
`TransferCostGraph` produces between the source and the destination of a transfer
(e.g. `CPUDRAM -> NPUHBM`, `NPUHBM -> LocalSRAM`, `CPUDRAM -> GpuHbm`). At each
seam the compiler can discharge a **boundary obligation**: a proof that the value
contract the consumer relies on still holds after the transfer's memory-ordering
effects are applied.

This catches the classic heterogeneous-offload bug: a *relaxed* cross-device copy
that omits the synchronizing release/acquire (or DMA-completion wait) can let the
consumer observe a **stale** buffer.

## Transfer methods

| Method | Target memory | Synchronizing? |
|---|---|---|
| `.to_device()` / `.to_device_relaxed()` | NPU HBM | yes / **no** |
| `.to_gpu()` / `.to_gpu_relaxed()` | discrete GPU HBM (NVPTX) | yes / **no** |
| `.to_sram()` / `.to_sram_relaxed()` | accelerator scratchpad | yes / **no** |

The `_relaxed` variants are the escape hatch: they omit the release, so the
published buffer loses its visibility guarantee across the seam.

## The obligation

For each seam the compiler emits a quantifier-free bit-vector (QF_BV) query and
discharges it with `z3`. The reached abstract state is a bit-packed lattice cell
per location (2-bit tag `BOT`/`CONST`/`TOP` + value field). The seam's transfer
function `F` is the identity for a synchronizing transfer, and sends published
buffers to `TOP` (possibly-stale) for a relaxed one. The query asks whether the
post-transfer state can violate the contract:

- **`unsat`** ⇒ the contract is guaranteed → **accept**.
- **`sat`** ⇒ a concrete counterexample (e.g. a stale read) → **reject** with
  diagnostic `E6004`.

Two contract forms are extracted automatically, in priority order:

1. **Value contract** — when the buffer's required value is statically known, the
   obligation *pins that value* (`buffer == v`), and a violation yields a
   value-level counterexample. The value comes from either:
   - a **consumer assertion** `assert(buf == N)` (see below), or
   - the **producer's** own compile-time-known constant.
1. **Visibility contract** — for an opaque buffer (e.g. a tensor whose contents
   are unknown), the obligation falls back to the coarsest sound abstraction:
   the buffer must be *definite* (not `TOP`) after the transfer.

### Consumer assertions and ordering

The seam at a transfer is checked *before* the consumer (`spawn` body) that reads
the buffer. To use a contract the consumer states downstream, the compiler runs a
**pre-scan** of the function body that records `assert(var == const)` facts
(descending into `spawn`/`if`/loop blocks). At the transfer, the value the
consumer requires of the produced buffer (matched by the `let` binding it feeds,
e.g. `local_a`) is consulted as the contract's conclusion.

```vx
fn f(a: Tensor<i32, [4]>) -> Pinned<Tensor<i32, [4]>, Topology::NPU[0]> {
    let local_a = a.to_device_relaxed();   // seam checked here
    spawn on(Topology::NPU[0]) {
        assert(local_a == 42);             // recovered by the pre-scan
        k(local_a)
    }
}
// vxc --verify-seams  =>  E6004: relaxed transfer of 'a' across the
//   CPUDRAM -> NPUHBM seam violates the boundary contract ('a' == 42)
//   z3 counterexample: ((tag_a #b11) (val_a #x00))
```

Using `.to_device()` (synchronizing) instead accepts.

## Enabling the check

Seam verification is **off by default**, so ordinary compilation pays nothing and
needs no solver. Enable it with:

```
vxc --verify-seams <file.vx>
```

When enabled it requires `z3` on `PATH`; if `z3` is absent the check **fails open**
(compilation proceeds without seam verification). One persistent solver is spawned
per compilation and reused across seams, so the marginal cost is one solver
round-trip per seam (~tens of microseconds), scaling with the number of seams
rather than program size. With `--verify-seams`, the driver reports
`[seam] N obligation(s): solver init X ms (once) + Y ms solving ...`.

## Diagnostic

`E6004` (Topology/Hardware group) — *transfer violates the boundary contract at a
seam*. The message names the buffer and the `src -> dst` seam, includes the
expected value when a value contract was extracted, and attaches the solver's
counterexample as a note.

## Implementation

- Obligation engine: `src/hir/seam.rs` (QF_BV emission, persistent `z3` `Solver`).
- Wiring: `src/hir/expr.rs` — `check_transfer_expr` (per-hop seam check),
  `collect_assert_contracts` (pre-scan), `const_value_of`, `run_seam_hop`.
- Flag: `--verify-seams` (`TypeChecker::verify_seams`).
