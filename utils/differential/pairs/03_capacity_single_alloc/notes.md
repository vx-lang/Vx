# One allocation larger than the device has

## The mistake

A working set that does not fit the part it is staged onto.

## Vx

Refused at compile time against `fleet/a100-80.vx`, in bytes:

```
Error[E6009]: transferred tensor needs 137438953472 bytes but memory space 'HBM' has capacity 85899345920 bytes
```

The check is plain arithmetic -- element size times extents, rounded by the declared granule,
compared against the declared capacity. No solver is involved, and the paper should not imply one
is.

## CUDA

Compiles. `cudaMalloc` returns `out of memory` on the card.

## The honest reading of this pair

CUDA does *report* this one, and reports it clearly. The difference is not that Vx catches an error
CUDA misses; it is **when**, and **against what**. Vx answers before the machine is rented, from a
declared model that can be checked against the vendor's specification afterwards. CUDA answers on
the rented machine, and only for the machine it is standing on.

That distinction is the whole claim for this pair, and overstating it as "CUDA misses this" would be
wrong. Pair 01 and pair 02 are the ones where CUDA genuinely does not object.

There is a second-order point worth a sentence: this program checks the return value of
`cudaMalloc`, so it fails gracefully. Code that does not check proceeds with a null pointer and
faults later, somewhere less obviously connected to the cause.

## The card matters

`EXPECT_GPU` pins this to an A100-80, and the bound in the Vx half comes from `fleet/a100-80.vx`.
Running it against a card with a different capacity would compare a refusal computed for one part
against hardware that is another, so the runner refuses rather than producing a row that looks fine.
Moving this pair to a different card means changing both the tile size and the `--machine` file, and
the two have to move together.
