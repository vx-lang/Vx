# A lane's elements of a tile in registers

Status: done. Issue: #3469. Change: #3475.

A flash-decoding kernel keeps each lane's share of the query, of the output sums and of the softmax
state for the whole loop over the tokens. In Vx that state is a tile in shared memory. Before
#3475, every pass of the token loop loaded the tiles from shared memory and stored them again. This
note shows, in pictures, how the compiler now keeps them in registers. The example is the `decode`
kernel of `tests/optimizations/pass/lane_values_stay_in_registers_between_lane_loops.vx`.

## 1. The kernel

```text
   for w in 0..2 across subgroups {
     ┌─ lane loop ─────────────────────┐
     │ load the query into qr,         │   once
     │ zero acc, set the softmax st    │
     └─────────────────────────────────┘
     for t in 0..tokens {                   ◄── the token loop, runs many times
       ┌─ lane loop ─────────────────┐
       │ part = qr · row(t)          │        each lane: its own share of the dot product
       └─────────────────────────────┘
       x = lanes::sum(&part[w][h], 32)      ◄── sum across the 32 lanes
       ┌─ lane loop ─────────────────┐
       │ update st and acc with x    │        online softmax
       └─────────────────────────────┘
     }
   }

   qr, acc, st, part are tiles: Tensor<.., Memory::SMEM>
   Every access is  t[w][..][l]   with  l = this lane
                    ▲          ▲
             my subgroup    my lane
   So no lane ever touches another lane's element.
```

## 2. Before: the tiles in shared memory

```text
        lane 0   lane 1   lane 2  ...  lane 31
          │        │        │            │
          ▼        ▼        ▼            ▼
   ┌──────────────────────────────────────────────┐
   │ SHARED MEMORY   qr[w][h][e][0..31]           │
   │                 acc[w][h][e][0..31]          │◄── every pass of the token loop:
   │                 st[w][h][s][0..31]           │    load qr, acc, st ... compute ...
   │                 part[w][h][0..31]            │    store them back
   └──────────────────────────────────────────────┘

   token t:   ld.shared ─► math ─► st.shared ─► sync ─► ld.shared ─► ...
   token t+1: ld.shared ─► math ─► st.shared ─► sync ─► ld.shared ─► ...
              ▲─────────────── slow round trips, thousands of times ──┘
```

## 3. After: each lane keeps its elements in its own registers

The frontend marks each access `t[w][..][l]` or `t[..][l]` with `vx.lane_own`, when `l` is the
index of a lane loop over all the lanes. The device lowering checks that every use of the tile has
the mark. Then it gives each lane a private array, with one place for each of its elements.

```text
   shared tile qr [2][4][2][32]                   lane 5's registers
   ┌──────┬──────┬──────┬─────┬──────┐            ┌───┬───┬───┬───┬───┬───┬───┬───┐
   │ l=0  │ l=1  │ ...  │ l=5 │ ...  │  ───────►  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
   └──────┴──────┴──────┴─────┴──────┘            └───┴───┴───┴───┴───┴───┴───┴───┘
     (h=0..3, e=0..1 for each lane)                 8 places = 4 h × 2 e

   Which place?   offset = w·256 + h·64 + e·32 + l
                  place  = (offset / 32) % 8   =  h·2 + e
                                                  ▲
                         w·256 is a multiple of 8×32, so it drops out

   The checker unrolls `for h`, `for e`  →  every place is a constant
   →  LLVM puts the 8 values in 8 registers. No memory at all.

   token t:   math in registers ─► math in registers ─► ...
```

## 4. The sum across the lanes: shuffles, not shared memory

```text
   x = lanes::sum(&part[w][h], 32)

   Before:  every lane stores part ─► sync ─► lanes add from shared memory ─► sync

   After:   each lane passes its register value directly to another lane
            (shfl.sync.bfly), 5 times:

   step 1   0 ⇄ 16   1 ⇄ 17   ...      (distance 16)
   step 2   0 ⇄ 8    1 ⇄ 9    ...      (distance 8)
   step 3   0 ⇄ 4    ...               (distance 4)
   step 4   0 ⇄ 2    ...               (distance 2)
   step 5   0 ⇄ 1    ...               (distance 1)
            ─────────────────────────
            every lane now holds the full sum, in a register

   decode kernel PTX:  20 shfl.sync.bfly  (4 sums × 5 steps)
                        4 bar.warp.sync    (cheap: one warp only)
                        0 local memory, 0 shared memory accesses in the token loop
```

## 5. A number of places that is not a power of two

The MLA decode kernel of glm-vx has tiles with 18 places in each lane.

```text
   place = (offset / 32) % 18     offset contains  w · 18 · 32

   %8   →  LLVM turns it into a bit mask   →  the place is a constant  ✓
   %18  →  no bit mask, LLVM could not fold it
                                           →  the place is computed at run time
                                           →  array goes to LOCAL memory  ✗
                                              (216 bytes, 227 registers)

   Fix: the place repeats every 18 × 32 elements, so drop any term that
        is a multiple of 18 × 32 (the w · 18 · 32 term) before the %
        →  the place is a constant again
        →  0 bytes of local memory, 112 registers
```

## 6. Result on an H200

GLM-5.3 MLA decode step (64 heads, rank 512, rope 64), measured by glm-vx. The flash rows time the
flash kernel and its merge of the splits together.

```text
   selected   mla_heads_at (before)          mla_flash_at + merge (after)
   tokens
     256      ██████████████████████  52.3   █████████  22.1 µs   2.4×
    1000      ██████████████████████  81.6   ████████   28.9 µs   2.8×
    2048      ██████████████████████ 119.8   ███████    40.2 µs   3.0×
              (bars are scaled within each row)

   Output: within 1.8e-7 of mla_chunks_at, within 4.6e-7 (relative) of float64.
```

SGLang's FlashMLA takes 13.9 µs for the same step. The remaining gap is in the kernel: the flash
kernel waits on two dependent loads for each token, and the merge is a separate launch.

## 7. No pragmas, no hints

The program does not say "keep this in registers", "unroll this loop" or "use shuffles". The
compiler finds all three from the program alone:

```text
   What the compiler must know           Where it comes from in Vx
   ───────────────────────────           ─────────────────────────
   which lane owns an element      ◄──   t[w][..][l], with l the index of a
                                         lane loop over all the lanes
   which subgroup owns a row       ◄──   w, the index of a subgroup loop
   that no other lane reads it     ◄──   every access to the tile has that form
   which loops to unroll           ◄──   their variables index such an element
   how to sum across the lanes     ◄──   lanes::sum on a row that the lanes own
```

In other GPU languages, the programmer usually gives these facts by hand:

- In CUDA, each thread has its own variables, and a warp sum is a loop of `__shfl_xor_sync`
  calls, or a library such as CUB. To keep an array in registers, its indexes must be constants,
  so programmers often add `#pragma unroll`.
- In OpenMP or OpenACC offload, the programmer adds clauses such as `private` and `reduction`.

In Vx, the program uses one tile for the whole block or subgroup, and `across lanes` and
`across subgroups` say which level of the GPU runs each loop. The index `l` of a lane loop then
says which lane owns each element. Because the compiler sees the loops and the indexes together,
it can prove that each lane touches only its own elements, and it needs no annotation. The same
program still runs on the CPU, with the same result.

The abstractions give the compiler two kinds of facts:

- Space: where each element lives and who owns it. The type gives the memory (`Memory::SMEM`) and
  the shape (`[2, 4, 2, 32]`). The index gives the subgroup and the lane.
- Time: when each lane uses it. The lane loops run in the order of the code, the end of each lane
  loop is a barrier, and the token loop repeats them.

The same kernel in C, C++ or Rust hides both facts:

```text
   C++ (CUDA)                                   Vx
   ──────────                                   ──
   __shared__ float s[512];                     Tensor<f32, [2, 4, 2, 32], Memory::SMEM>
   s[w*256 + h*64 + e*32 + tid % 32]            qr[w][h][e][l]
     a flat address, the shape is gone            the shape is in the type
   tid = threadIdx.x, a run-time value          l = the index of a lane loop
   __syncthreads() anywhere in the code         a barrier at the end of each lane loop
```

To prove there that each thread touches only its own elements, the compiler must first get the
dimensions back from the flat address. LLVM has a delinearization analysis for this, which guesses
the dimensions from the address expression. It often fails when the sizes are parameters. Then a
polyhedral tool such as Polly must model the accesses as sets of integer points, through the
barriers and with alias facts for the pointers. In Vx the check is a match on the index: is the
last index the variable of a lane loop over all the lanes? The high-level abstractions make the
analysis simple.

## 8. The other option: a sum across the lanes inside a lane loop

#3469 named two fixes. Option (a) was a sum across the warp that a lane loop can use as a value.
Option (b) was per-lane values that stay in registers from one lane loop to the next. #3475 is
option (b). Option (a) is not needed, for two reasons:

- A program must mean the same with `across` removed, with the passes run one at a time. A sum
  inside a lane loop breaks that rule: pass 0 would need the value of pass 31. So the compiler
  would have to split the lane loop at the sum, which is what a program with option (b) already
  does.
- In a subgroup, option (b) gives the PTX that option (a) would: shuffles between registers and a
  warp barrier (section 4).

One part of option (a) still has a cost. Code between lane loops that calls a method, such as
`.exp()` in the online softmax, runs on lane 0 alone, and its result then goes to the other lanes.
#3476 lets every lane run such a pure method itself, with no new syntax.
