# Where a tile matmul puts its result

Issues: #2312 (tensor-core instructions in `spawn on`). Related PRs: #3646 (fp8 tiles), #3648
(an f16 tile by an fp8 tile). Decided on 2026-11-19.

This note records how much control Vx gives a programmer over where the operands and the result
of a matmul of tiles live, and what the compiler decides by itself.

## The question

A lane matmul in a subgroup loop multiplies two tiles:

```vx
acc[w] = acc[w] + (sa[w] as f32) @ (sb[w] as f32);
```

Today the checker gives the product the memory of its first operand. So when `a` is in HBM and
`sb` is in shared memory, this line is an error (E3004), because `acc` is in shared memory:

```vx
acc[w] = (a[blk] as f32) @ (sb as f32);   // a is in Memory::HBM, sb and acc in Memory::SMEM
```

For a batch of one, a kernel wants to read the weights `a` from HBM straight into the registers
of the tensor cores. The kernel reads each weight byte once, so a copy into shared memory first
only adds time. glm-vx does this with inline PTX in `mlir!`. Its MoE down kernel on an H200 took 11.4 µs
this way, against 18.6 µs for the kernel that copies the weights first.

So: if A is in HBM and B is in shared memory, where does C go, and who decides?

## The decision

**The programmer decides the memory level and the sizes. The compiler checks them, and decides
everything below that level.**

### What the programmer decides

Where a tile lives and how large it is are part of the algorithm. An accumulator of 64 x 64 in
registers or in shared memory decides how many blocks fit on an SM, and which threads can read
it. Vx already treats these as the programmer's choice:

- The type of a tile names its memory, as in `Tensor<f32, [32, 16], Memory::SMEM>`.
- The checker refuses tiles that do not fit in their memory (E6010, and E6027 across calls), and
  an access from a thread that cannot reach the memory.

So the declared target decides where the product goes. When `acc` is in shared memory, the
product goes to shared memory. The compiler checks that it fits there, and that the threads that
read it can see it.

### No default

A default, for example "the product goes to HBM, which every thread can reach", is correct but
slow. Nothing in the program text would show the cost. Vx avoids this elsewhere: an operation on
fp8 values needs a result type that the programmer writes
(`www/book/src/low-precision-numbers.md`). In the same way, a product of tiles in two memories
with no target in a known memory is an error that asks for one. So there is no hidden copy and
no hidden slow path.

### What the compiler decides

The compiler decides how the matmul runs:

- The instruction: `mma.sync`, `wgmma`, or a loop on the lanes.
- The register layout of each fragment, and the order of K in a fragment, for example 16
  contiguous bytes for one load.
- `ldmatrix`, conversions such as `cvt.rn.f16x2.e4m3x2`, and pipelining.

This follows an earlier decision: the language has no syntax for one chip, and details such as
warpgroups and `wgmma` are passes in MLIR. Kernel writers choose these by hand today, in CUDA,
CUTLASS and inline PTX. But each choice belongs to one chip. The fp8 work showed this: an e4m3
`mma.sync` is native on `sm_89` (`QMMA.16832`), and on `sm_90` `ptxas` unpacks it to f16
(`F2FP.F16.E4M3.UNPACK_B` and `HMMA.16816`). A program that names memories and sizes stays
correct and fast on both. A program that names instructions does not.

For a kernel that needs exact PTX, `mlir!` stays the way out.

## The rule for the A operand from HBM

1. In a lane matmul in a subgroup loop, an operand tile can be in any memory that the subgroup
   can see, for example the HBM view `a[blk]`.
1. The product goes to the memory that the type of the target names.
1. When the target names no memory, the checker reports an error.
1. The compiler loads the fragments of an operand in HBM from HBM into registers, with no
   copy in shared memory.

The matmul runs in the subgroup, wherever its operands are, so the rule hides no computation on
another device. It is about one expression and its target, and it uses no fact from another
body, so it keeps the rules in
[`language_rules_for_a_parallel_compiler.md`](../language_rules_for_a_parallel_compiler.md). It
converts no value: the product is a new value, and the program says where it goes.

## Open: registers as a target

`Memory::REG` has the scope of one thread. An accumulator of a subgroup "in registers" means that
each lane holds its part of the tile. The compiler already does this when it can prove that
nothing else reads the tile (`k_loop` in `lane_matmuls_on_the_gpu.vx`), but no rule guarantees it.

A later change can let a programmer write `Memory::REG` on a tile of a subgroup, with this
meaning. The compiler would then check that the part of each lane fits in the register budget,
and that only that subgroup reads the tile. This needs its own design.
