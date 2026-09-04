# Working-set capacity: a peak over scope chains

How the compiler decides whether the tiles a function places in a memory space fit that space, and
why the answer is a peak rather than a sum.

## The question

A `Memory` declaration states a `capacity`. A program places tiles into that space with `transfer`.
The compiler refuses a program whose tiles do not fit — E6009 for a single tile larger than the
space, E6010 for a set of them that together exceed it.

"Together" is the whole difficulty. Two tiles occupy a space at the same time only if their lifetimes
overlap, so the quantity that has to fit is the **peak** residency over the program, not the sum of
everything ever placed. Summing describes a moment the program never has.

## What makes the peak computable here

Three properties, and the analysis needs all three:

1. **Placement is in the type.** `transfer(t, Memory::W)` produces a value whose type names `W`, so
   the compiler knows which budget a tile draws on without any escape analysis or points-to
   reasoning.
1. **Spaces declare a capacity.** The budget is a number in the machine model rather than a property
   of the machine the program happens to run on.
1. **Release follows the block structure, for the scopes that can be bypassed.** The lowering's
   rule is dominance rather than nesting: a block that dominates the function's exits has its
   frees hoisted to those exits, and only a block that can be skipped is freed at its own end.
   For an `if` arm or a loop body -- which can be -- the tile's lifetime is exactly that block,
   which is what turns "do these overlap" into a question about the syntax tree.

The third is the one that does the work, and it is worth being precise about it. From the emitted IR
for a tile placed inside an `if` body and another placed after it:

```
^bb1:                                            // the `if` body
  %11 = call @vx_plugin_alloc_and_transfer(...)
  call @vx_plugin_free(%11, ...)                 // released before the branch
  br ^bb2
^bb2:
  %17 = call @vx_plugin_alloc_and_transfer(...)  // allocated only now
  call @vx_plugin_free(%17, ...)
```

The two tiles never coexist. A sum reports two; the run has one.

Release is at block end rather than at last use, which the ordering confirms — with a use later in
the same block, the release follows the use:

```
  %11 = call @vx_plugin_alloc_and_transfer(...)
  %14 = call @vx_plugin_alloc_and_transfer(...)
  %15 = call @vx_plugin_transfer_device_to_host(%11, %14, ...)   // the use
  call @vx_plugin_free(%11, ...)                                 // after it
```

### Which scopes actually release

Nesting and lifetime are not the same thing here, and reading them as the same was a real
unsoundness. A `spawn` region ends in an unconditional branch, so its block dominates the exits and
its tile lives to the function's return:

```
^bb1:                                            // the spawn region
  %11 = call @vx_plugin_alloc_and_transfer(...)
  br ^bb2                                        // no free
^bb2:
  %17 = call @vx_plugin_alloc_and_transfer(...)
  call @vx_plugin_free(%11, ...)                 // both released only here
  call @vx_plugin_free(%17, ...)
```

An `if` arm nests exactly the same way in the source and behaves the opposite way, because it can be
bypassed and therefore does not dominate:

```
^bb1:
  %11 = call @vx_plugin_alloc_and_transfer(...)
  call @vx_plugin_free(%11, ...)                 // freed at the end of its own block
  br ^bb2
```

So the chain records only the scopes whose end the lowering frees at. `push_releasing_scope` marks
those; `push_scope` -- the default -- does not, which keeps a scope's tiles resident outside it. The
default is the conservative direction: it can refuse a program that would fit, and cannot admit one
that does not.

Opted in, each verified against emitted IR rather than reasoned about: **`if` arms** and **loop
bodies**. Left conservative: `spawn` regions, `comptime` and `unsafe` blocks, match arms, and closure
bodies. A match arm is probably releasing on the same grounds as an `if` arm; it stays out until
someone checks the IR, because the cost of being wrong is asymmetric.

## The algorithm

Every lexical scope gets an id when it is entered and gives it up when it is left, so at any point
during checking there is a **scope chain**: the ids of the blocks currently open, outermost first.
`push_scope` / `pop_scope` maintain it, which covers a control-flow body, a loop body, a spawn
region, and the function itself — the scopes that become blocks in the IR.

Each placement records three things:

| field | meaning |
|---|---|
| `bytes` | the tile's size, rounded up to the space's granule |
| `scope` | the scope chain at the moment it was placed |
| `order` | its position in program order |

At the end of a function, for each space:

```
placements ← the space's tiles, sorted by order
peak ← 0
for each tile T in placements:
    live ← Σ { U.bytes : U placed no later than T, and U.scope is a prefix of T.scope }
    peak ← max(peak, live)
```

`U.scope is a prefix of T.scope` is the coexistence test. A tile placed in a block that encloses
`T`'s block is still open when `T` is placed; a tile in a sibling block was released when that block
closed and is not resident. The "no later than" bound is what distinguishes a tile placed in an outer
scope *before* an inner block from one placed *after* that block has finished.

Sorting by order makes the scan linear in placements and quadratic in the worst case, which is fine —
the input is the placements in one function, not the program.

### Why the ordering test is needed as well as the prefix test

Prefix alone would count a tile placed in the enclosing scope *after* the inner block closed as
coexisting with the inner one. It does not: by then the inner tile is gone. Program order settles it.

### Degenerate case, deliberately preserved

A function whose placements all sit in one block is unchanged: every chain is a prefix of every
other, so the peak equals the sum. That is the honest answer for that shape — those tiles really are
resident together until the function's block ends, whether or not they are still read.

## Granularity: scope, not last use

The obvious refinement is to release a tile at its last use, and the machinery for that already
exists — `compute_block_liveness` gives a per-block last-use index and `borrow_cx::is_variable_used_after`
is the NLL predicate the borrow checker runs on. Using it here would be wrong.

The runtime releases at block end. A tile whose last read is early in a block is dead to the borrow
checker and still allocated until the block closes. Accounting at last-use granularity would be finer
than what the runtime does, and the check would admit programs that then exceed the space — trading a
conservative refusal for an unsound admission.

The rule is that the accounting must model the release discipline the lowering implements, not the
tightest one imaginable. If the release point ever moves to last use, this analysis moves with it,
and not before.

## Soundness

The analysis assumes a tile is gone once its block closes. That is only safe if a placement cannot
outlive its block. Two things prevent it, and neither is part of this analysis:

- **A placed value carries its space in its type**, so binding it to something outside the block
  requires that binding to have the placed type. Assigning it into an ordinary host binding is a
  cross-topology error, refused by the checker.
- **An `if` expression yielding a transfer does not lower.** It fails MLIR verification (`type mismatch for bb argument`), so no such program reaches a runtime where the assumption could be
  observed.

If either changes — an escape hatch that lets a placement leave its block, or that verification gap
being closed by making the shape lower — this analysis needs an escape check before it is still
sound.

## What this does not cover

- **Across calls.** The budget is per function and is taken at the end of each one, so a tile held by
  a caller is not counted against a callee's placements. See Vx#444; the quantity wanted there is the
  peak over the call tree, which needs a call graph and a decision about recursion.
- **Loops.** A loop body is checked once, so a placement inside it counts once. That matches the
  runtime, where the release at the end of the body runs every iteration — but it means the analysis
  says nothing about a loop that accumulates tiles into an outer structure.

## Why this is not the usual state of affairs

The comparison worth drawing is not that other toolchains get this wrong, but that they are not in a
position to ask the question at compile time.

- In C, C++ or Rust, a heap allocation is not associated with a bounded named space. Capacity is
  discovered at run time, when an allocator returns null or the OOM killer arrives. There is nothing
  to compare a lifetime against.
- In CUDA, static `__shared__` is bounded per kernel and the bound is enforced at launch, not
  composed across a program's scopes; dynamic shared memory is a launch parameter, so the check moves
  to the host and to run time. Device global memory has no compile-time budget at all.
- Frameworks that place tensors on devices generally learn the element type, the shape, and the
  device at the point of dispatch, so a capacity failure surfaces as an allocation error during
  execution, on the machine that lacks the memory.

What makes it tractable in Vx is the conjunction listed at the top: the space is in the type, the
capacity is declared, and the lifetime is lexical. Any one of the three missing and the peak stops
being a compile-time quantity.

## Tests

`tests/integration_test/working_set_peak_test.rs` asserts the admitted/refused verdict and the
reported peak for a matrix of shapes: sibling blocks, nested blocks, both arms of an `if`, a loop
body, placements before and after an inner block, chains several deep, and the degenerate
single-block case. The two `.vx` tests alongside it read as documentation of the two ends:

- `frontend/pass/working_set_is_a_peak_not_a_sum.vx` — the sibling-block case that a sum refuses.
- `frontend/fail/working_set_counts_shadowed_placements.vx` — one block, where the sum is the peak.

Each was checked by breaking what it tests rather than by watching it pass.
