# Three tiles that each fit and together do not

## The mistake

Several buffers, none of them individually too large, resident at once in a space that cannot hold
the sum.

## Vx

```
Error[E6010]: the working set placed in memory space 'HBM' (3 tiles) sums to 51539607552 bytes,
over its 42949672960 byte capacity; place fewer/smaller tiles or declare it `overcommit`
```

The diagnostic names the space, the tile count, the sum, and the bound. `overcommit` on the memory
declaration downgrades it to W1028, which is the escape hatch for a space that genuinely pages.

## CUDA

Compiles, and allocates until one call fails. The error it reports is about the allocation that
happened to cross the line -- typically the third -- which is the one piece of information least
useful for fixing the problem. Nothing is wrong with the third buffer; the set is wrong.

## Why this is the strongest capacity pair

Pair 03 is a case CUDA reports clearly, just later and only for the card in front of it. This one is
different in kind: **CUDA has no notion of a working set at all**, so there is nothing it could have
checked. The failure is attributed to the wrong object, and the attribution is not a quality-of-
implementation issue -- a per-allocation API cannot say anything about a set of allocations.

This is the pair to lead with in a table, and pair 03 is the one to include next to it so the
difference between "later" and "not expressible" is visible.

## The card matters

`EXPECT_GPU` pins this to an A100-40, for the same reason as pair 03: the 40 GiB bound comes from
the cited model, and running against an 80 GiB card would compare a refusal against hardware the
refusal was not computed for.
