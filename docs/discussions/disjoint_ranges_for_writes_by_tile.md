# Disjoint ranges for writes by tile

Issues: #3657 (this design), #3562 (`Distinct`), #3561 (iteration domains). Discussed on
2026-11-19.

This note records how a kernel can show the race check that its blocks write rows that do not
overlap, when the rows come from a table that another kernel wrote. It also records what owns
what: the claim, and the output that the kernel writes.

## The problem

The race check of `across` loops accepts a write when its index fixes the loop variable. A tile
whose rows come from a table at run time has no such index. The grouped block-FP8 GEMM of the
routed experts (glm-vx) is the example. A plan kernel turns the experts' row offsets into tiles
of 64 rows. Each block of the GEMM writes the rows of its tile:

```vx
let start: i32 = tiles[2 + 3 * t];   // first row of this tile, read from memory
let end: i32 = tiles[3 + 3 * t];
out[(start + r) * n + col] = acc;    // for r < end - start
```

The tiles do not overlap, because the plan made them so. That is a contract between two kernels,
held in data, and nothing in the GEMM's source shows it. If the plan wrote two tiles with the same
start, two blocks would write the same rows. So the check refuses the write, and the kernel stores
through raw addresses today, as `unsafe` would in Rust.

A second case looks the same, but is different. A dense GEMM computes its tile from the block
number alone, as `m_tile = bid % tiles_m` and `n_tile = bid / tiles_m`. That split is one to one,
so the writes do not overlap, but the check does not reason about `%` and `/`. That case needs no
new type: nested loops, `for n_tile in .. across blocks { for m_tile in .. across blocks { .. } }`,
give indices that fix each variable.

## Why `Distinct` is not enough

`Distinct<..>` claims that no two elements of an index tensor are equal, so the check treats
`y[idx[b]]` as `y[b]`. A tile covers a range of rows, and distinct starts do not make ranges
that do not overlap: starts 0 and 10 are distinct, but rows 0 .. 64 and 10 .. 74 overlap. The
plan's guarantee is of another kind: no two `[start, end)` ranges overlap.

## Two questions: the claim, and the access

**The type owns the claim's data, and gives only reads.** `Distinct<Tensor<i32, [?]>>` owns its
index values, so nothing can break the claim after the check. A kernel takes
`&Distinct<..>`, and the borrow checker stops other writes while that borrow lives. The claim owns
nothing of the output. It is a fact about values that the check uses.

`DisjointRanges<..>` is the same kind of type: a table of `[start, end)` ranges, owned by the
type, with only reads (`start(t)`, `end(t)`, `len()`). Owned or viewed makes no difference to the
claim. A part of a set of distinct values is still distinct, and a subset of ranges that do not
overlap still does not overlap. So a view of part of either table keeps its claim.

**The access to the output does not move.** The kernel holds `&mut out`, as today. The claim only
lets the check prove that the passes write apart:

```vx
fn grouped(out : &mut Tensor<f32, [?, ?], Memory::HBM>, tiles : &DisjointRanges<Tensor<i32, [?, 2], Memory::HBM>>) -> void {
  spawn on(Topology::Device) {
    for t in 0..tiles.len() across blocks {
      for r in 0..64 across lanes {
        if r < tiles.end(t) - tiles.start(t) {
          out[tiles.start(t) + r][0] = 1.0;   // fixes t, because the ranges do not overlap
        }
      }
    }
  }
  return;
}
```

## Mutable views: Rust's way

Rust has no reasoning about indices, so a split must show in the types. `chunks_mut` and
`split_at_mut` give one `&mut` view of each part. The views own nothing: each borrows a part of the
data, and the owner cannot use it while the views live.

Vx proves the split from the indices, so a kernel does not need view objects. A view can come later
as a convenience, such as `out.rows_mut(&tiles, t)` for the rows of range `t`. Its safety comes
from the `DisjointRanges` claim, not from new borrow rules.

## The rule

The check treats `out[start(t) + r]`, with `0 <= r < end(t) - start(t)`, as an index that fixes
`t`. In the z3 query of #3561, `start` and `end` are unknown functions. For the two passes that the
query compares, it adds the claim `t1 != t2 => end(t1) <= start(t2) or end(t2) <= start(t1)`, so
the query stays quantifier-free.

## How a program gets one

- A library routine that makes the ranges by construction returns one with no check. The plan
  of the grouped GEMM, and the tiles of a partition by key, are examples.
- `DisjointRanges::check(t)` checks a table on the host, as `Distinct::check` does.
- `unsafe { DisjointRanges(t) }` makes one with no check, for a program that made the table
  itself.

The claim is in the signature, so a body never needs another body to be checked, and the rules of
[`language_rules_for_a_parallel_compiler.md`](../language_rules_for_a_parallel_compiler.md) hold.
