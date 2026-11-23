# Distinct tensors for writes by index

Issues: #3562 (this design), #3450 (the mixture-of-experts kernel that needed it), #3561
(iteration domains). Merged in #3565 (the type and the race check), #3567 (`Distinct::check`)
and #3569 (`nn_moe_group` returns one). The book has it in `gpu.md`, "Rows from a tensor of
indices", and the blog post `an-escape-hatch-the-size-of-one-fact.html` gives the background.

This note records the design: what the type claims, who owns what, and the one rule that the
race check adds.

## The problem

The race check of `across` loops (E6032) accepts a write when its index fixes the loop variable.
A write at a row that comes from a tensor has no such index:

```vx
for b in 0..idx.len() across blocks {
  for l in 0..64 across lanes(64) {
    y[idx[b]][l] += x[b][l];   // E6032: idx may hold the same row twice
  }
}
```

The values of `idx` exist only at run time, so no analysis of the kernel can find them. The
kernel of one expert in a mixture-of-experts layer needs this write. Each token is in the list of
an expert once, so the program has no race. The fact that shows it is not in the source.

## The claim, in the type

`Distinct<Tensor<i32, [?], Memory::HBM>>` is a tensor of integers with one dimension, whose
elements are all different. The type has the same family as `Verified<..>` and `Pinned<..>`: a
tensor with a claim about it.

```vx
fn scatter(y : &mut Tensor<f32, [?, ?], Memory::HBM>, idx : &Distinct<Tensor<i32, [?], Memory::HBM>>) -> void {
  spawn on(Topology::Device) {
    for b in 0..idx.len() across blocks {
      for l in 0..32 across lanes(32) {
        y[idx[b]][l] = 1.0;
      }
    }
  }
  return;
}
```

## Who owns what

- **The type owns the values, and gives only reads.** A write to an element is E4006, because a
  write could make two elements equal. `len()` gives the length. `unsafe { Distinct(t) }` and
  `Distinct::check(t)` move `t`, so no other name can change the values.
- **A borrow keeps the claim.** A kernel takes `&Distinct<..>`. The borrow checker stops other
  writes while the borrow lives.
- **The claim owns nothing of the output.** The kernel still takes `&mut y`. The claim is a fact
  about values that the race check uses, not a split of `y`.
- **A `transfer` keeps the claim.** A `transfer` of a `Distinct` tensor gives a `Distinct` tensor
  in the new memory.

## The rule in the race check

The fact is: if `idx[a] == idx[b]`, then `a == b`. In `src/hir/check/across_proof.rs`, a read
`idx[e]` of a `Distinct` parameter becomes a new variable of the pass, and that variable fixes
whatever `e` fixes. So `idx[b]` fixes `b`, as `b` does.

The new variable then takes part in the old rules. `(idx[b] as i64) * 4 + (l as i64)`, with `l`
below 4, fixes `b`, as `b * 4 + l` does. The range of the variable is the range of the element
type. With `i64` elements, `idx[b] * 4` can wrap, so the write is still E6032.

These stay E6032:

- `y[idx[b / 2]]`, because `b / 2` does not fix `b`.
- A read of `y[b]` or `y[idx[b + 1]]` in the same loop, because another pass may write that row.
- A `Distinct` local or field, or a parameter whose name the function binds again. Only a
  parameter counts, so the claim is in the signature, and the check of a body never needs another
  body. The rules of
  [`language_rules_for_a_parallel_compiler.md`](../language_rules_for_a_parallel_compiler.md)
  hold.

In the iteration-domain form of #3561, `idx` becomes an unknown function. The race query compares
two passes, so it needs the fact only for those two: `idx(b1) == idx(b2) => b1 == b2`. The query
stays quantifier-free.

## How a program gets one

- `Distinct::check(t)` on the host returns `Option<Distinct<..>>`. It sorts a copy of the values
  as `i64` and compares each value with the next (`stdlib/core/distinct.vx`). It takes `i32`,
  `i64`, `u32` and `u64` tensors with one dimension in CPU memory. Check first, then `transfer`.
- A library routine that makes distinct values by construction returns one with no check.
  `nn_moe_group` in `std::nn` groups the token-expert pairs by expert on the GPU, and returns the
  order as a `Distinct` tensor.
- `unsafe { Distinct(t) }` makes one with no check, for a program that made the values itself. If
  two elements are equal, two passes write the same element. That is a data race, and Vx makes no
  promise about the result.

## What it does not cover

- Two passes that add to the same row. In a mixture-of-experts layer, a token goes to several
  experts, and their results add into the row of the token. That needs atomics or a separate pass.
- Ranges of rows. Distinct starts do not make ranges that do not overlap. A table of `[start, end)` ranges needs another claim: see
  [`disjoint_ranges_for_writes_by_tile.md`](disjoint_ranges_for_writes_by_tile.md).
