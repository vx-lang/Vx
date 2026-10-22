# Declared writes for `mlir!` blocks in `across` loops

Status: proposed. Issue: #252. Builds on the L4 race proof (`src/hir/check/across_proof.rs`,
#2073) and on the contract clauses of `contracts_for_extern_and_unsafe_code.md` (X1, X3).

The L4 proof refuses an `mlir!` block, a method call, an indirect call, and a call to any
function whose body it does not read, in an `across` loop: "... may write memory that the
checker cannot see". It also refuses a `&mut` borrow. That keeps every tensor-core kernel out of
`across`, because wgmma, TMA and the mbarrier waits are `mlir!` blocks, usually behind a device
function. glm-vx has 7 such kernels: a block-FP8 GEMM for H100 (two tile sizes), four tensor-core
probes and a repro.

This note gives `mlir!` blocks and device functions a contract that says what they write. The
proof uses the contract in place of the body. Most of a contract is checked. The rest is trusted,
and the audit lists it.

## 1. What the kernels do

The FP8 GEMM, written with `across`, is one region per output tile:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
for n in 0..tiles_n across blocks { for m in 0..tiles_m across blocks {
  let mut ring_a = unsafe { Tensor<i32, [8192], Memory::SMEM, aligned(1024)>::uninit() };
  let mut ring_b = unsafe { Tensor<i32, [16384], Memory::SMEM, aligned(1024)>::uninit() };
  let mut bars = unsafe { Tensor<i64, [8], Memory::SMEM>::uninit() };
  init_barriers(&mut bars);                      // lane 0, then the block barrier
  for w in 0..5 across subgroups {
    if w == 4 {                                  // the producer warp
      for l in 0..32 across lanes {
        if l == 0 {
          for kb in 0..blocks_k {
            let _w = mbar_wait(&bars, 4 + kb % 4, parity(kb) ^ 1);  // the stage is empty
            let _t = tma_load(&mut ring_a, &mut ring_b, kb % 4, maps, &bars, kb);
          }
        }
      }
    } else {                                     // 4 consumer warps
      for l in 0..32 across lanes {
        let mut acc = Tensor<f32, [64], Memory::REG>::fill(0.0);
        for kb in 0..blocks_k {
          let _w = mbar_wait(&bars, kb % 4, parity(kb));            // the stage is full
          let _m = wgmma(&mut acc, &ring_a, &ring_b, kb % 4);
          let _a = mbar_arrive(&bars, 4 + kb % 4);
        }
        for j in 0..16 { for h in 0..2 { for e in 0..2 {    // the wgmma fragment
          out[m * 64 + w * 16 + h * 8 + l / 4, n * 128 + j * 8 + (l % 4) * 2 + e] =
              acc[j * 4 + h * 2 + e];
        } } }
      }
    }
  }
} }
```

The epilogue's `l / 4` and `l % 4` need L4 to read `/` and `%` by a number (section 6,
question 4).

The writes are of four kinds:

| Kind | In the GEMM | What makes it race-free |
| --- | --- | --- |
| A tensor that one pass makes | `acc` (REG), written by wgmma | No other pass sees it. |
| Elements at an affine index | `out` in the epilogue | L4, once it reads `/` and `%` by a number. |
| Atomic operations | `bars`: `mbarrier.init`, `arrive`, `try_wait` | The hardware. |
| Data whose order a protocol gives | `ring_a`, `ring_b`: TMA writes, wgmma reads | The `full`/`empty` mbarrier handshake, not `barrier()`. |

The last kind is the one that no static proof covers today. It is also the one that the kernel's
speed depends on: the producer and the consumers overlap, with no block barrier in the loop.

## 2. The clauses

**D1. No clause means the worst case** (X1). An `mlir!` block or a device function without a
clause may write everything that it can reach. In an `across` loop it is refused, as today.

**D2. The clauses are `ensures` predicates** (X3). Vx gets no new keyword. A device function
states them after its signature. An `mlir!` block states them in a new `ensures:` entry, next to
`inputs:` and `returns:`:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn wgmma(acc: &mut Tensor<f32, [64], Memory::REG>, a: &Tensor<i32, [8192], Memory::SMEM>,
         b: &Tensor<i32, [16384], Memory::SMEM>, stage: i32) on Topology::Device -> i32
    ensures writes_only(acc)
    ensures ordered(a)
    ensures ordered(b)
{
  return mlir!( inputs : (..), returns : i32, dialects : ["nvvm", "llvm"],
                ensures : [writes_only(acc), ordered(a), ordered(b)]) { .. };
}
```

| Predicate | Meaning | How the proof uses it |
| --- | --- | --- |
| `writes_only()` | Writes no memory. A fence or a register read, such as `%tid.x`. | No access. |
| `reads_only(t)` | Reads `t`, writes nothing of it. | A read of all of `t`. |
| `writes_only(t)` | May write any element of `t`. | A write of all of `t`. Race-free only when this pass made `t`. |
| `writes_only(t[e0, ..])` | Writes only the elements at those indices. An index is an expression or a range `lo..lo + k` with `k` a number. | Accesses at those indices, which L4 checks as it checks element writes. A range is one more digit. |
| `atomic(t)` | Each access to `t` is an atomic operation. | No race with another atomic access. A plain access to `t` in the same loop is refused. |
| `ordered(t)` | Each access to `t` is ordered with the accesses of other passes by an mbarrier protocol (D5). | No race with another `ordered` access to `t`. A plain access to `t` in the same loop is refused. |

A tensor that the clauses do not name is not touched. Several clauses combine.

**D3. A pass-local tensor needs nothing.** A tensor that the pass makes (`Binding::Tensor` in
the proof) can be borrowed `&mut` and passed to a call or a block. Nothing outside the pass can
see it. Today the proof refuses every `&mut` borrow in the loop; this narrows that to tensors
from outside the pass.

## 3. Device functions

**D4. The contract of a device function is checked against its body.** The compiler checks once,
for each function, that the accesses of its body are covered by its clauses: its element
accesses, its calls (through their contracts) and its `mlir!` blocks (through theirs). A call in an
`across` loop then uses only the contract. The proof substitutes the arguments for the
parameters, so `wgmma(&mut acc, &ring_a, &ring_b, s)` is a write of the pass-local `acc` and
`ordered` accesses to `ring_a` and `ring_b`. A `&mut` borrow of a tensor from outside the pass is
allowed as an argument when the callee's clauses for that parameter are not the worst case.

So the only trusted text is the clause list of each `mlir!` block (D6), and what D5 says.

## 4. Ordered tensors

**D5. `ordered(t)` is a claim about the protocol, and it is trusted.** It says that the code
which accesses `t` follows an mbarrier handshake: a pass writes a stage only after a wait shows
that its readers are done, and reads it only after a wait shows that the write completed. The
compiler does not prove the handshake. It checks around it:

1. `t` is in `Memory::SMEM` or `Memory::HBM`, and it is made outside the loop.
1. Every access to `t` in the loop is `ordered`. A plain read or write of `t` there is an error.
1. The loop has at least one `atomic` tensor of `i64` in `Memory::SMEM`: the mbarriers.
1. The audit (A1 of the contracts note) lists each `ordered` clause, with the tensor and the
   block or function that states it.

Naming the barrier tensor in the clause (`ordered(t, by: bars)`) was considered. It would let the
check confirm that the reader and the writer use the same barriers. But wgmma does not take the
barriers, and the pairing of a stage with its barrier is index arithmetic that the check cannot
follow. So this note leaves the barrier out.

## 5. What the proof checks and what it trusts

**D6. Checked:**

- Each name in a clause is an input of the block or a parameter of the function, and it is a
  tensor.
- `atomic(t)` has an integer element type of 32 or 64 bits. `ordered(t)` satisfies D5 (1) to (3).
- The element sets of `writes_only(t[..])` pass L4 together with the loop's other accesses.
- Each device function's body is covered by its contract (D4).
- In the MLIR of an `mlir!` block, an op that writes a memref input (`memref.store`,
  `vector.store`, an `nvvm` op with a memref operand) writes only an input that its clauses name
  with a write predicate.

**D7. Trusted:**

- That an address input (`i64`, `!llvm.ptr`) points into the tensor that the clause names. TMA
  and wgmma take addresses, not memrefs. So a block that takes an address and writes memory must
  name, in a clause, each tensor that the address can reach. A block that takes an address and
  names nothing gets D1.
- That the protocol of D5 holds.
- That the PTX of the block does what the clauses say.

The audit lists both kinds, as it lists `!keeps` today.

## 6. What the `across` stack does today

The answers below describe #2072 to #2074 as they are. The open choices are for the design's
owner, so section Decisions stays empty.

1. **Roles in one loop: works.** L8 (`uniform_around` in `src/hir/check/across.rs`) accepts an
   `if` on a subgroup index around a lane loop. A lane loop directly in a subgroup loop ends with
   a subgroup barrier (`vx.barrier {subgroup}`), not a block barrier. So the producer warp and the
   consumer warps do not wait for each other.
1. **Lane loops of different widths in one subgroup loop: refused.** In a subgroup loop, a lane
   loop must have exactly the subgroup's lane count of passes, and no lane loop may be inside
   another. The masking of N2 applies only to a nest directly in the blocks loop. In a subgroup
   loop a pass is `lane % width`, so a loop of 1 would run on all 32 lanes. The producer is
   therefore a loop of 32 with `if l == 0` inside, as in the example.
1. **Warpgroup ops: no level for them.** `across subgroups(4)` gives 128 lanes, but its barriers
   are per warp, so the convergence that wgmma needs is not proved. Open: a warpgroup level
   (`across warpgroups`), which wgmma and `setmaxnreg` would need.
1. **Indices: `/` and `%` are not read.** Nested lane loops for `lane / 4` and `lane % 4` hit the
   rule of 2 inside a subgroup loop; directly in the blocks loop, up to 3 lane axes nest. Open:
   teach L4 floor division and remainder by a number, as affine maps have them, and keep the
   nesting rule. The example assumes that. A one-axis grid (`tile = m + tiles_m * n`) can also
   stay one axis then.

## Decisions

None yet.

## The work, in order

1. D1 to D3 in the proof: the `ensures:` entry of `mlir!`, the predicates, pass-local `&mut`.
1. D4: contracts on device functions, with the body check.
1. D5 and D6: `ordered` and `atomic`, the MLIR-level check, and the audit entries.
1. The glm-vx FP8 GEMM and probes move to `across`, and `grid(..) block(..)` (#2065) can go.

The code goes on the `across` stack (#2072, #2073, #2074).
