# Unrolling short loops on NVIDIA GPUs

Issues: #3587 (the change, PR #3588), #3570 (the same change for Metal), #3589 (early SROA).

This note records what we learned when we made the NVIDIA code generator unroll short loops.
Most of it is about measurement: three of the first four results were wrong for a reason
other than the one they seemed to show.

## Why unroll

A top-k kernel has this loop:

```vx
for j in 0..8 {
  let mut bits: i32 = c0;
  if j == 1 { bits = c1; }
  ...
  if j == 7 { bits = c7; }
  // about 40 more instructions
}
```

LLVM unrolls a loop with a constant count by itself only when the body is small, and this body
is too large. Rolled, each pass picks its value with seven compares and selects. Unrolled, `j`
is a constant in each copy, and the picks fold away:

```
 Rolled: 8 passes, each pays for the pick       Unrolled: 8 copies, no pick left

 +-> j = 0..7                                   copy j=0:  bits = c0  --> body
 |     bits = c0                                copy j=1:  bits = c1  --> body
 |     j == 1 ? c1 : bits    7 compares         copy j=2:  bits = c2  --> body
 |     j == 2 ? c2 : bits    and 7 selects        ...
 |     ...                   in every pass      copy j=7:  bits = c7  --> body
 |     j == 7 ? c7 : bits
 |     body
 +---- j < 8 ?
```

## What the compiler does now

`unrollShortLoops` in `src/dialect/VxLowering.cpp` runs on the LLVM IR of the kernels, before
the `-O3` pipeline. It marks a loop with `llvm.loop.unroll.full` when both of these are true:

- The loop runs at most 16 passes. The Metal code generator uses the same limit.
- The loop holds no other loop.

`-O3` then unrolls each marked loop.

```
  Vx --> MLIR --> LLVM IR of the kernels
                       |
                       v
  +---------------- unrollShortLoops ---------------------------------+
  |                                                                   |
  |   kernel function ----- copy ----> copy of the function           |
  |         |                               |                         |
  |         |                               v                         |
  |         |                     SROA: stack slots -> registers      |
  |         |                               |                         |
  |         |                               v                         |
  |         |                     count the passes of each loop       |
  |         |                     (SCEV), keep the loops that         |
  |         |                     - run 1 to 16 passes                |
  |         |                     - hold no other loop                |
  |         |                               |                         |
  |         |   same header block           v                         |
  |   mark the loop  <------------------ loop of the copy             |
  |   llvm.loop.unroll.full                                           |
  |         |                     (the copy is deleted)               |
  +---------|---------------------------------------------------------+
            v
     LLVM -O3 (unrolls the marked loops) --> PTX --> ptxas --> GPU code
```

## What went wrong on the way

```
  version 1        version 2               version 3                  final
  mark loops  -->  SROA on the kernel, --> + innermost loops only --> SROA on a copy,
  (no marks        then mark               + count fixed by one        mark the kernel
   came out)       (spills in               (spills in                 (93 of 96 kernels
                    mla_heads_at)            topk_mb_list_at)           unchanged)
```

### 1. The marks did nothing

The first version found almost no loop with a known count. At that point, the IR keeps each
`let mut`, and the loop counter too, in a stack slot (`alloca`). LLVM's trip count analysis
(SCEV) cannot follow a value through memory:

```
  Before SROA                              After SROA

  %j = alloca i32                          header:
  store 0, %j                                %j = phi [0, entry], [%j1, latch]
  header:                                    %c = icmp slt %j, 8      <- SCEV sees
    %v = load %j      <- SCEV stops here     ...                         0, 1, 2 ... 7
    %c = icmp slt %v, 8                    latch:
  latch:                                     %j1 = add %j, 1
    store (%v + 1), %j
```

A debug print of the counts showed this at once. SROA moves stack slots into registers, so the
second version ran SROA first.

### 2. SROA on the kernel changed the kernel

The second version gave the right marks, but a sweep of 96 kernels showed spills in
`mla_heads_at`: 16 bytes before, 52 bytes after. That kernel has no short loop.

The cause was SROA, not unrolling. A compiler with SROA and no marks gave the same spills.
Running SROA before `-O3` changes the IR that `-O3` gets, and `-O3` does not always reach the
same result from it. The same sweep showed register drops of 2 to 16 in about 20 other kernels,
and rises in a few. All of these went away with the early SROA. #3589 tracks them.

Three compilers separated the two causes:

```
                               mla_heads_at          index_scores_tc_at
  A. no new step (= main)      16 B spill            142 registers
  B. SROA, no marks            52 B spill  <-- SROA  142 registers
  C. SROA on a copy + marks    16 B spill            80 registers  <-- unrolling
```

The fix: SROA runs on a copy of each function, only to count the passes. SROA with
`PreserveCFG` keeps the blocks of the copy, so a loop of the copy and a loop of the function
with the same header are the same loop. The mark goes on the loop of the function, and the
function is otherwise unchanged. After this fix, 93 of the 96 kernels gave the same PTX as
before the change.

**Lesson:** to measure a change, compare it with a compiler that has every part of it except the
part you want to measure.

### 3. The count was one too high

The loops come out of the code generator with the test at the top. LLVM counts how often the
header runs, and the header runs once more than the body:

```
  for j in 0..16              header runs:  j = 0, 1, 2, ..., 15, 16   -> 17 times
                                            |___________________|  |
     +-> header: j < 16 ? --no--> exit      body runs 16 times     last test fails
     |        | yes
     |      body
     +---- latch: j = j + 1   (the latch does not exit)
```

So `0..16` counted as 17 and did not unroll. The code now subtracts one when the latch does not
exit the loop.

### 4. Outer loops added spills

The Metal code generator unrolls outer loops too. On NVIDIA that gave `topk_mb_list_at` 64
bytes of spills, against 4 before. Innermost loops only gave 28 bytes (24 bytes after the fix in
section 2). So the NVIDIA rule takes innermost loops only:

```
  for r in 0..2 {          <- holds a loop: not marked
    for t in 0..n { }      <- count unknown: not marked
    for j in 0..4 { }      <- innermost, 4 passes: marked
  }
```

LLVM can still unroll an outer loop by itself after it unrolls the inner loops.

## Why unrolling can need more registers

Copies of a loop body can reuse the same registers, so unrolling does not need more registers
by itself. The cost comes from scheduling. In a rolled loop, the values of one pass die before
the next pass starts. Unrolled, the passes are straight-line code. The scheduler moves the loads
of a later copy above the work of an earlier copy, so that they wait on memory at the same time:

```
  Rolled: one pass live at a time           Unrolled and scheduled: passes overlap

  time -->                                  time -->
  pass 0: [load ...wait... use]             copy 0: [load ...wait... use]
  pass 1:                     [load ...wait... use]
                                            copy 1:   [load ...wait... use]
  live:   |  a  |                                     ^^^^^^^^^^^^^
          one set of registers              live:     both sets here
```

That overlap is the reason to unroll: the two waits on memory become one. It costs a few
registers, not twice as many, because only the values in flight overlap.
`topk_mb_write_at` went from 29 to 32 registers this way.

### Most of the "rise" was a register cap

`topk_mb_list_at` showed 32 registers with 4 bytes of spills before the change, and 24 bytes
after. Its PTX says `.maxntid 1024`: at most 1024 threads in a block. ptxas fits two such blocks
on one H100 core:

```
  one H100 core (SM): 65,536 registers, up to 2,048 threads

  +--------------------------+--------------------------+
  |  block 1: 1024 threads   |  block 2: 1024 threads   |
  +--------------------------+--------------------------+
     65,536 registers / 2,048 threads = 32 registers per thread
```

So many of these kernels report exactly 32 registers. `-maxrregcount 64` did not lift the cap.
Only a change to the PTX did:

| `topk_mb_list_at`, ptxas 13.1, sm_90 | Before | After |
| --- | --- | --- |
| As built (`.maxntid 1024`) | 32 regs, 4 B spill | 32 regs, 24 B spill |
| Without the `.maxntid` line | 44 regs, 0 spill | 44 regs, 0 spill |
| With `.minnctapersm 1` also | 60 regs, 0 spill | 56 regs, 0 spill |

```
  registers the kernel needs:     before  ############################################  44
                                  after   ############################################  44
  cap from .maxntid 1024:                 ################################|             32
                                                                          '-> the rest spills
```

Without the cap, the unrolled kernel needs no more registers than before. Both versions need
more than 32, so both spill under the cap, and only the amount of the spill differs. If such a
kernel runs one block per core, `.minnctapersm 1` removes the spills in both versions.

An earlier build (48a3229d9) gave 40 registers against 44 without the cap. That build still ran
SROA on the kernel itself.

**Lesson:** before you blame a change for spills, find the register limit that ptxas used, and
measure without it.

## Testing an unroll

Two things made the first test pass without the change:

- **LLVM unrolled the test loop by itself.** A loop of 16 passes with a small body is under
  LLVM's own limit. The test passed with the marks turned off. The fix was a larger body, so
  that only the mark can unroll it. Check this with a compiler that has no marks.
- **LLVM unrolls a loop by two when it cannot unroll it fully.** A test `j == 11` in a rolled
  loop can show in the PTX as a test against 10. The checks accept either value. A fully
  unrolled loop has neither:

```
  fully unrolled            unrolled by two                   rolled
  (no test left)            j = 0, 2, 4, ... 16               j = 0, 1, 2, ... 16
                            copy A: j      == 11 ?  never     j == 11 ?
                            copy B: j + 1  == 11 ?
                                    -> PTX: j == 10
```

`tests/optimizations/pass/nvidia_gpu_short_loops_of_constant_bounds_are_unrolled.vx` has the
final test. Each of its checks failed under its own sabotage: no marks, a count one too high, a
limit of 17, and marks on outer loops.

## Open

- No GPU timings yet. A top-k timing on an H200 will show whether the unrolled kernels are
  faster, and whether the extra spill in `topk_mb_list_at` costs time.
- #3589: why SROA before `-O3` lowers registers in many kernels, and adds spills to one.
- A `.minnctapersm 1` hint for kernels that run one block per core would remove the spills in
  `topk_mb_list_at`. Nothing in Vx sets it today.
