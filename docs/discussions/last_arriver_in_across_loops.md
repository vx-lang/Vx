# The last-arriver pattern in `across` loops

Status: option C is built (#2139, item 5). The checker records the memory for the count, and the
flat code generator writes the count, the fences and the guard. The block of the last pass runs
the code after the nest, so that code may hold lane loops. A nest with no passes runs no code
after it, and a nest with two axes of blocks cannot have code after it yet. Code in the outer loop
of two axes, after the inner loop, is the code of a group: each pass of the outer loop has its own
count, and the last inner pass of the group runs the code.

## The pattern

A kernel that reduces over blocks in one launch, such as glm-vx's MLA decode merge, does this in
CUDA:

1. Each block writes its partial result.
1. Each block takes a release fence and adds 1 to a counter.
1. The block that sees `n - 1` takes an acquire fence and reads every partial result.

In Vx today, step 3 is a read of a tensor that the other passes of the blocks loop wrote, so the
race proof (L4) refuses it. The checker also refuses code after the blocks loop: "a region with
`across` loops does nothing after its nest of blocks" (E6032). A second `spawn` region works,
but it is a second launch.

## What makes the read safe

The read in step 3 is safe only when all of these hold:

1. The counter is 0 before the first pass.
1. Each pass adds 1 to it exactly once, after all its writes, with a release ordering or a
   release fence of device scope. Inside a block, every lane that writes must be ordered before
   that add.
1. The reader's add is acquire, and the reader tests the old value against the trip count minus 1.
1. Nothing else in the loop touches the counter.

The proof can check 2 to 4 from the syntax. It cannot check 1: the counter is a run-time value.

## Options

### A. The proof learns the pattern

The program writes the counter, the fences and the test. The proof matches the shape and accepts
reads of any tensor the loop wrote inside `if old == n - 1 { .. }`.

- Close to CUDA, and the program controls every instruction.
- The proof must trust that the counter is 0 at first. A counter left at 3 by an earlier launch
  gives a race that no check finds.
- The shape match is brittle: a counter passed through a helper, or a trip count that is an
  expression, needs more rules.

### B. A library routine with a contract

`if last_block(&count, n) { .. }`, where `last_block` takes the fences and the add, and says a new
clause such as `ensures last_arriver(count)`. The proof reads the clause.

- Smaller rules than A, because the routine fixes the shape.
- Condition 1 still rests on the program.

### C. Code after the blocks loop, in the same kernel (recommended)

Lift the E6032 limit. Code after the nest of blocks in a `spawn` region runs once, after every
block:

```vx
spawn on(Topology::Device) {
  for b in 0..n across blocks {
    for l in 0..32 across lanes { partial[b * 32 + l] = .. }
  }
  out[0] = merge(&partial);
}
```

The code generator builds the tail into the same kernel with the last-arriver pattern. The
compiler owns the counter: it zeroes it before the launch and puts the fences in. On the CPU, the
tail runs after the loop.

- The race proof needs no new rule: the tail is sequential code after the loop, as the program
  reads.
- The program has no counter, so condition 1 cannot go wrong.
- It fits the rule that chip details belong in the compiler, not in the language. A target
  without the needed atomics can use two launches for the same program.
- It is the most work in the code generator: a counter buffer for each launch, the release and
  acquire fences, and one block that runs the tail. A tail with its own `across` lanes loop needs
  the whole last block, not one thread.

## Decision

Option C, chosen on 2026-10-24. A and B leave condition 1 to the program, and the proof's answer
then depends on a value it cannot see.
