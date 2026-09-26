# A size type that never wraps

> **Status: proposal, not implemented.** Nothing in this note exists in the compiler yet. It
> records a design and the reasons for it, for issue #794.

## The problem

C, C++ and Rust use an unsigned integer as the size type: `size_t` and `usize`. An unsigned
integer is a set of bits. Adding one to the largest value gives zero, and the language says that
result is correct. So when a loop counter has that type, the compiler must keep the wrap-around as
a possible outcome. It cannot prove the counter only goes up, cannot work out how many times the
loop runs, and cannot fold the counter into an address. Loop optimizations then either add guards
for the wrap case or give up.

C avoids this on signed integers by making overflow undefined, which lets the compiler mark every
add as never wrapping. Rust gets the same mark only when LLVM can infer from a nearby comparison
that the add cannot wrap. That works for a plain `0..n` loop and fails for many others.

Vx is behind both today. Every integer operator wraps, on signed and unsigned types alike, and the
code generator emits no no-wrap flag on any operation. LLVM sees a Vx loop counter the way it sees
an unsigned counter in C, whatever its declared type.

The optimizer needs a promise that overflow never produces a value. The sign of a type does not
give that promise. This note proposes a size type that does.

## What Vx does today

- There is no size type. `Vec::len`, `String::len` and tensor extents return `i32`. Iterator
  lengths, `sizeof` and `raw::extent` return `i64`. The core library plan, §8 of
  [`core_library.md`](../implementation_plans/core_library.md), proposes `i64` everywhere and
  leaves `usize` for later.
- `+`, `-` and `*` wrap at the width of the type, on every integer. `core::num` documents this,
  and its `wrapping_add` is written as `self + rhs`. Compile-time evaluation refuses an
  overflowing result instead of wrapping it.
- Both code generators emit `arith.addi`, `arith.subi` and `arith.muli` with no flags
  (`src/codegen/flat.rs` and `src/codegen/lower/mod.rs`).
- A `for i in a..b` loop keeps its counter in a stack slot of the range's element type, compares
  with the signed `slt`, and increments with an unflagged add (`lower_for` in
  `src/hir/flatten.rs`). No `scf.for` is involved. Only LLVM's own passes, at `-O3`, move the slot
  into a register.
- Tensor indexing converts the index with `arith.index_cast`, which sign-extends, even when the
  index type is unsigned (`src/codegen/flat/emit/tensor.rs`). A `u32` index with its top bit set
  becomes a negative `index`.
- `Vec::get` checks bounds through an out-of-line Rust function, so LLVM cannot remove a check
  even when the counter is plainly in range.
- The checker already gives the prover `i >= lo` and `i < hi` for a range loop
  (`src/hir/stmt.rs`), and uses them to discharge the bounds obligations of `raw::` accesses.

## The proposal

### The type

A new scalar type, called `size` in this note. It is 64 bits wide on every target. Its values are
the integers from zero up to the largest positive `i64`, so the top bit of the representation is
always clear.

Why 64 bits everywhere rather than the pointer width: a struct holding a `size` then has the same
layout on the host, on a GPU and on a microcontroller, and crosses a `transfer` unchanged. Rust's
`usize` and MLIR's `index` change width per target. The cost is 64-bit arithmetic on a 32-bit
microcontroller, where code that cares can use `u32`.

Why half the range: a value with the top bit clear reads the same as signed and as unsigned.
Signed and unsigned comparison agree on it, sign extension and zero extension agree, and both of
LLVM's no-wrap flags hold at once. It is also the largest size Rust allows an object to have,
`isize::MAX`, so a `size` passes to Rust's `usize` over FFI with no conversion.

### The rule

Overflow on `size` is a bug. A program in which a `size` operation leaves the range has no
defined result, and the compiler may assume it does not happen.

Each `+`, `-` and `*` on `size` carries an obligation: the result is in range. The checker tries
to discharge it with the prover it already has. An obligation it discharges costs nothing at run
time. For one it cannot discharge, the compiler emits one compare and a branch to a trap, as
Rust's debug builds do. Either way the emitted operation is marked as never wrapping, and the mark
is sound: the value was either proved in range or the trap took the other path.

What the prover sees in practice:

- `i + 1` in `for i in 0..n`. The loop condition gives `i < n`, and `n` is itself in range, so
  `i + 1` fits. No check is emitted.
- `v.len() - 1` for a `Vec` the prover knows nothing about. Nothing says `len >= 1`, so the
  subtraction traps when `len` is zero. In Rust's release mode `0usize - 1` is `usize::MAX`, and
  `for i in 0..len - 1` runs until some later bounds check panics.
- `i * stride` with unknown operands. It traps if the product leaves the range.

Code that wants a wrap, a saturation or a checked result says so with the `wrapping_*`,
`saturating_*` and `checked_*` families, which `core::num` already has for the fixed-width types
and gains for `size`. The stdlib's own iterator adapters use them wherever a value can reach the
limit, as Rust's do: `nth` under `step_by`, and the last step of an inclusive range.

Monotonic loop counters need no rule of their own. A counter that cannot wrap and moves by a
positive step only goes up, and LLVM's scalar evolution derives that from the flags. `step_by`
already rejects a step of zero.

### Conversions

- `size` widens to `i64` and `i128` freely, and `From` says so.
- An `as` cast into `size` from a signed type, or from `u64` or `u128`, carries the same
  obligation as arithmetic: the value is in range. The prover discharges it or a trap guards it.
  Unsigned types up to `u32` widen freely.
- An untyped literal takes `size` from its context, as it takes any other type today. The
  counter of `0..v.len()` is therefore a `size` without an annotation.
- `size` and the fixed-width types do not mix in one expression without an `as`, following the
  no-implicit-conversion rule.

### Lowering

`size` lowers to a signless `i64`. The three operators lower to `arith.addi`, `arith.subi` and
`arith.muli` with `overflow<nsw, nuw>`. Comparisons use the unsigned predicates. At a memref
boundary the value becomes `index` through `arith.index_castui`. No new operation in the Vx
dialect is needed: `arith` already carries the flags, and the conversion to the LLVM dialect keeps
them. A loop step looks like this:

```mlir
%next = arith.addi %i, %c1 overflow<nsw, nuw> : i64
%more = arith.cmpi ult, %next, %n : i64
%ix   = arith.index_castui %i : i64 to index
```

Checked on the LLVM 22.1.4 this repository builds against: `mlir-opt --convert-arith-to-llvm`
turns the first line into `llvm.add %i, %c1 overflow<nsw, nuw> : i64`.

An unproved obligation lowers to the operation, then one signed compare of the result against
zero, then a `cf.cond_br` to a trap block. Because both operands have their top bit clear, a
result outside the range is exactly a result with its top bit set. Multiplication also needs the
high word from `arith.mulsi_extended` to be zero.

### What the optimizer can then do

- With the flags, LLVM computes trip counts, widens or narrows counters, strength-reduces address
  arithmetic and vectorizes, all without a guard for the wrap case.
- The Vx prover can remove bounds checks before LLVM runs. Once `Vec::get` checks bounds inline
  instead of through the Rust call, `i < len` from the loop condition discharges the check on
  `v[i]` in a loop over `0..v.len()`. This is the part that happens while the program is still
  Vx, and it is why the guarantee lives in the language rather than only in the flags.

## What changes, roughly in order

1. Type checker: the type, literal typing, the `as` rules, and the range obligation on the three
   operators and on casts into `size`.
1. Prover: model `size` values as integers in the range, so a length is known to be non-negative
   and the loop facts carry over.
1. Code generation: the flags, unsigned predicates, `index_castui`, and the trap sequence for an
   unproved obligation. Fix the sign-extending `index_cast` on unsigned indices at the same time.
1. Standard library: `Vec::len`, `Vec::get` and `Vec::set`, `String::len`, tensor `len` and
   `extent`, iterator `len`, `sizeof` and `raw::extent` take or return `size`. `Range` and its
   adapters move from `i64` to `size`. This replaces the plan in `core_library.md` §8 to
   standardize on `i64`.
1. The range loop: type the counter as `size` when the range is, and emit the flagged step.
   Moving the counter out of its stack slot into a block argument, which the AST path already
   does, is a separate improvement.
1. `Vec::get`: check bounds inline, so both the prover and LLVM can see the check.

## Open decisions

- **The name.** `size` says what the type is, and it is also a common variable name. `usize` is
  what every Rust programmer will type, and the stdlib port refers to Rust's `usize` throughout,
  but its leading `u` says "bit pattern", which is the reading this type exists to break. This
  note uses `size`; the choice is for review.
- **Whether the fixed-width integers keep wrapping.** Treating `i8` through `i128` and `u8`
  through `u128` as bit patterns is consistent with this proposal, and the obligation machinery
  extends to them later if wanted. Issue #794 asks the general question; this note answers it for
  `size` and leaves the rest open.
- **Trap or undefined behaviour for an unproved obligation.** The flags are sound either way.
  This note says trap: the project rule is to crash rather than fail silently, and a trap at the
  subtraction names the bug where it happens.

## Alternatives considered

- **`i64` everywhere**, the current plan. It removes the mixed `i32` and `i64` lengths but keeps
  wrap-around, so LLVM still sees no flags, and a negative length becomes representable.
- **An alias of `u64`.** It loses the one thing the type is for.
- **Undefined overflow, as in C.** Same flags, silent failure. The project rule rejects it.
- **A `vx.size_add` operation in the Vx dialect.** It hides the arithmetic from every upstream
  MLIR pass and needs its own folding and lowering. The flags on `arith` already reach LLVM
  unchanged.
- **Pointer width, like `usize` and `index`.** The layout of a value would differ across a
  `transfer`, which is the discipline Vx is built around.
