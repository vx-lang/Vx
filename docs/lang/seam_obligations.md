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

1. **Value contract** — when the compiler knows two numbers, the obligation compares
   them, and a violation yields a value-level counterexample. The two numbers come
   from two different places:
   - the **producer's** number is the premise. It is the whole-number literal in the
     `let` that declares the buffer, as in `let a : Tensor<i32, []> = 0;`. A float
     literal such as `42.0` is a whole number too, when the element type holds it
     with no rounding.
   - the **consumer's** number is the conclusion. It is a **consumer assertion**
     `assert(buf == N)` (see below).

   A synchronizing transfer keeps the producer's number. The obligation holds only
   when the two numbers are equal. A relaxed transfer can give the consumer any
   number, so the obligation does not hold.
1. **Visibility contract** — when the compiler does not know both numbers (e.g. a
   call gives the buffer its value), the obligation falls back to the coarsest sound
   abstraction: the buffer must be *definite* (not `TOP`) after the transfer. This
   obligation does not compare a value. An assert the compiler cannot connect to a
   producer's number stays a run-time check.

### Consumer assertions and ordering

The seam at a transfer is checked *before* the consumer (`spawn` body) that reads
the buffer. To use a contract the consumer states downstream, the compiler runs a
**pre-scan** of the function body that records unconditional `assert(var == const)`
facts. It skips closure bodies and a short-circuited logical right operand; an
`if` or `match` contributes only facts on which every branch agrees, and loop
bodies contribute none because they may not run. Loop invariants are collected.
At the transfer, the value the consumer requires of the produced buffer (matched
by the `let` binding it feeds, e.g. `local_a`) is consulted as the contract's
conclusion.

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

Using `.to_device()` (synchronizing) instead accepts: `a` is a parameter, so the
compiler does not know its number.

When the compiler knows the producer's number, a synchronizing transfer can also
violate the contract:

```vx
fn g() -> i32 {
    let a : Tensor<i32, []> = 0;           // the producer gives 0
    let local_a = a.to_device();           // seam checked here
    spawn on(Topology::NPU[0]) {
        assert(local_a == 42);             // the consumer requires 42
    }
    return 0;
}
// vxc --verify-seams  =>  E6004: the transfer of 'a' across the CPU_DRAM -> NPU_HBM
//   seam violates the boundary contract. 'a' is 0, so 'local_a' is 0 after the
//   transfer. The program asserts 'local_a' == 42
//   z3 counterexample: ((tag_a #b01) (val_a #x0000000000000000))
```

### Which names carry a number

The pre-scans read names, not values. A name carries a number only when the function
declares the name one time, with a `let` that has no `mut`. A second `let` of the
name in any block, a parameter of the function or of a closure, a generic parameter,
a loop variable, a `match` binding and a top-level `const` are all second
declarations. Such a name can mean two values, so the compiler uses no number for it.

The pre-scan for asserts stops at the first statement that is a `return` or has a
`return` inside it. An assert after that statement is not a fact about every run. A
call that never returns, such as `todo()`, does not stop the pre-scan.

The consumer's number applies only to the transfer that is the whole initializer of
the `let`. In `let n = f(a.to_device());` the name `n` is the result of `f`, so an
assert on `n` says nothing about the transfer. A staged transfer checks the value
contract one time, on its last hop.

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
seam*. The message names the buffer and the `src -> dst` seam, and attaches the
solver's counterexample as a note. For a relaxed transfer the message includes the
number the contract requires, when the compiler knows one. For a synchronizing
transfer the message gives the producer's number and the consumer's number.

## Implementation

- Obligation engine: `src/hir/seam.rs` (QF_BV emission, persistent `z3` `Solver`).
- Wiring: `src/hir/check/transfer.rs` — `check_transfer_expr` (per-hop seam check),
  `collect_assert_contracts` and `collect_single_lets` (pre-scans), `run_seam_hop`.
- Flag: `--verify-seams` (`TypeChecker::verify_seams`).
