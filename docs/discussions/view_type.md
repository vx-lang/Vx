# A view type for tensors

> **Status: proposal, for review.** This note answers the design questions of #1163, part of
> #400. Nothing in it is built beyond what the section "What exists today" describes. #1164
> builds the type from this note once it is agreed.

## The decision in brief

- **A view is a reference to a tensor:** `&Tensor<T, [dims]>` for a shared view and
  `&mut Tensor<T, [dims]>` for a mutable one. There is no new `View` type.
- **`&q[i]`, `&q[a..b]` and `&h.t` make views,** as `&v[a..b]` makes a slice in Rust. `q[i]` on its
  own names a place, not a value that can be moved out.
- **Views are contiguous.** A row, or a range of rows, is a contiguous window of a row-major
  tensor. A column or a strided window needs a different lowered type and gets its own spelling
  later (#1167).
- **A view carries its owner's placement,** in the tensor type it refers to. Taking a view is not
  a read and is allowed anywhere; reading through it is checked where the read happens.
- **The borrow rules are the ones `&` already has.** Lifetimes, exclusivity, escape (E4005),
  storage in struct fields (#1112) and disjoint constant indices (#1119) all apply unchanged.
- **A view never becomes an owning tensor behind the programmer's back.** Where a `Tensor` is
  expected, a view is a type error that asks for `.clone()`.

## What exists today

Checked on `main` at 42ada79c.

- **A row is a view at run time.** `q[i]` lowers to a `memref.reinterpret_cast` of `q`'s buffer:
  no copy.

- **The checker treats it as a borrow.** `let r = q[i]` records a borrow of `q`
  (`src/hir/check/views.rs`, and §4 of
  [`borrow_checker_architecture.md`](borrow_checker_architecture.md)). The type the checker gives
  `r` is still `Tensor<T, [D]>`, the name of an owning value. A view is told apart from an owner
  by where it came from, not by its type.

- **The explicit spelling already works.** `&q[1]` compiles on the default code generator, passes
  to a `&Tensor<f32, [4]>` parameter, and gets every borrow rule a reference gets:

  | Program | Result today |
  | --- | --- |
  | `let r = &mut q[1]; q[1][0] = 2.0; r[0] = 3.0;` | refused, E4002 |
  | `let a = &mut q[0]; let b = &mut q[1];` | accepted (#1119) |
  | `return &q[1];` with `q` a local | refused, E4005 |
  | `return &q[1];` with `q : &Tensor<..>` | accepted |

- **A borrowed tensor is a descriptor.** The default code generator passes `&Tensor` as the
  tensor's memref descriptor, and lays out a `&Tensor` struct field the same way (#1120). A row
  passed to a `&Tensor` parameter has its data pointer moved to the row first, so the callee can
  treat it as a plain memref.

- **A row keeps its owner's placement.** A row of a tensor in `Memory::GPU_HBM` read in host
  code is refused (E6003), like a read of the tensor.

- **Range slicing crashes the compiler** (#1158). `q[0..2]` type-checks, and both code generators
  then fail.

- **The legacy code generator recognises a view by matching MLIR type text** for
  `strided` (`is_slice_operand` in `src/codegen/lower/expr.rs`; #1165).

So the representation, the lowering and the borrow rules are built. What is missing is a type
that says "this is a view", and range slicing.

## Decisions

### 1. The spelling is `&Tensor`, not a new `View` type

A view is a non-owning, borrow-checked handle to tensor storage. `&Tensor<T, [dims]>` already is
exactly that, and the rest of the compiler already handles it.

Alternatives considered:

- **A new `View<T, [dims]>` type, beside `Tensor`.** It would need its own rules in unification,
  assignability, the borrow checker, the escape check, both code generators, mangling, metadata
  and the layout of struct fields. All of them exist for `&Tensor` today. Worse, the language
  would then have two spellings for one thing: `&q[1]` and a `View` would both mean "a window
  onto `q`", and every function would have to pick one.
- **A wrapper, `View<Tensor<..>>`.** This repeats `Ref<Tensor<..>, Memory::X>`, removed by #429
  because every consumer of a tensor type had to peel the wrapper first, and the ones that forgot
  were bugs.
- **A flag inside `Type::Tensor`** (owning or view). It avoids a wrapper, but a function parameter
  must also say whether the view is mutable, so the flag needs three states, and it duplicates
  what `&` and `&mut` already say.

`&Tensor` is also what Rust does: a slice is a reference to unsized storage (`&[T]`), not a
separate struct. #400 asked for Rust's lifetime semantics, and this is where they come from.

The name `View` stays useful in prose and diagnostics: "a view of `q`".

### 2. `q[i]` is a place; `&q[i]` is the view

In Rust, `v[i]` names a place, and moving a non-`Copy` value out of it is refused. Vx follows
that:

- `&q[i]` and `&mut q[i]` are views of row `i`.
- `q[i].clone()` is an owning copy of the row.
- `p[0] = q[1]` copies the row into `p`'s row, as it does today. An assignment into an element
  reads its source.
- `q[i]` passed where a `Tensor` is expected stays the E4011 error it is today, and the message
  offers `&q[i]` or `.clone()`.

`let r = q[i]` and `let mut r = q[i]` mean a shared and a mutable view today, and existing code
uses them. They keep that meaning as shorthand for `let r = &q[i]` and `let r = &mut q[i]`, and
the type of `r` is printed as the reference it is. Whether the shorthand should later warn, and
then be refused, is an open question below.

### 3. Views are contiguous; strided views come later, with their own spelling

A `&Tensor` parameter lowers to a memref with the identity layout: a unit stride in the last
dimension and the rest row-major. Code generated for the callee relies on that; vector loads and
the `dot`/`sum` kernels do. A row, and a range of rows, of a row-major tensor are contiguous, so
they fit after the data pointer is moved to the window's start, which the code generators
already do for a row.

A column, or a window with a step, is not contiguous. Admitting it as `&Tensor` would mean either
copying it at every call, which a view must not do, or lowering every `&Tensor` parameter with
unknown strides, which slows every callee to support a few. So:

- `&Tensor` is a contiguous view.
- Ranges are allowed on the leading dimension, where the window stays contiguous: `&q[a..b]` on
  `Tensor<T, [N, D]>` is `&Tensor<T, [b - a, D]>`, with `[?, D]` when an end is not a constant.
  A range on a later dimension, `q[i][a..b]`, is contiguous too, since it is a range of the last
  dimension of a row.
- Columns and strided windows are #1167. That issue chooses their spelling. The requirement from
  here is that it is a different type from `&Tensor`, so a function says in its signature whether
  it accepts a strided view.

### 4. A view carries its owner's placement

The placement is part of the tensor type, so `&q[1]` of `q : Tensor<f32, [2, 4], Memory::GPU_HBM>`
is `&Tensor<f32, [4], Memory::GPU_HBM>` with no new mechanism. Every check that reads a placement
through a reference keeps working, and one that does not is a bug (#1160).

Taking a view reads nothing. Today `let r = &g;` with `g` in GPU memory is refused in host code
(E6003 at the `&`). That is too strict, and it is the same mistake #742 fixed for `as_ptr()`:
the address may cross, the load may not. #1164 lets a view of a placed tensor be taken anywhere;
a read through it is checked where the read is.

A view moves no bytes, so the traffic counter counts it as zero and counts reads through it
against the owner's memory (#1161).

### 5. Mutability and aliasing are the reference rules

- `&` views may overlap. A `&mut` view excludes every other use of the part of its owner it
  covers.
- Two `&mut` views of one tensor may be live together when the checker can see they are
  disjoint: different constant indices (#1119), and non-overlapping constant ranges (#1166). Any
  index or range with a non-constant end may overlap anything.
- Nothing like Rust's `split_at_mut` is needed for the first version: constant ranges cover the
  split-K kernels #400 names. A function that splits at a run-time point can come later.

### 6. Device code follows placement

A view inside `spawn on` is a reference inside `spawn on`; no rule is added. A read through it is
allowed where the owner's memory is visible and refused (E6003) where it is not, as for the owner.
A view itself cannot change placement: `transfer` moves an owning tensor. `transfer` of a view
would be an explicit copy of the window into a new owning tensor; it is not needed for #1164 and
is left out until a program needs it.

### 7. A view may be stored in a struct

A struct field of type `&Tensor<..>` holds a view. #1112 already makes a struct that holds a
reference keep its source borrowed while the struct is used, and the escape check from #860
refuses returning such a struct when it points into the function's frame. The field is laid out
as the tensor's descriptor (#1120). The legacy code generator cannot hold a tensor in a struct
field at all yet (#1125).

### 8. A function that returns a view says so

A function returns a view as `-> &Tensor<..>`, and it may only return a view of memory its caller
owns (E4005 already says this). Returning a view from a function whose declared return type is
the owning `Tensor<..>` becomes a type error that asks for `.clone()` or a `&Tensor` return
type. A handful of tests do this today, for example `second_row` in
`tests/backend/pass/tensor_views_used_by_the_rules.vx`; #1164 changes their signatures.

### 9. `reshape`

A reshape without a mode, or with `PadMode::Trim`, reads a window of the source's storage and is a
view: `t.reshape([..])` has type `&Tensor<T, [new dims]>` and borrows `t`. A reshape with
`PadMode::Pad` is an owning tensor made by a copy, whatever its size, and so is `t.pad([..])`; both
are built that way already (#1183).

### 10. What happens to the provenance test

`view_of` in `src/hir/check/views.rs` decides today whether an expression is a view by looking at
its shape. Once the shorthand in decision 2 is rewritten to an explicit borrow, a view is any
value of type `&Tensor`, and the general reference machinery records its borrow. `views.rs`
shrinks to the shorthand rewrite and the E4011 diagnostics; its other rules are already the
reference rules.

### 11. Relation to slices (#534)

#534 asks for `&[T]` and `&mut [T]` over a flat sequence of any `T`, for the core library. That is
the same idea for `Vec` and arrays that this note is for tensors: a reference to storage of a
length the reference carries. The two should share the borrow rules and the spelling family, a
`&` in front of the storage type, and need not share a representation: a slice is a pointer and a
length, and a tensor view is a memref descriptor.

## What #1164 does

In order:

1. Type `&q[i]`, `&h.t` and `t.reshape(..)` as `&Tensor` references with the rows' dims and the
   owner's placement. `&q[a..b]` comes with #1158.
1. Rewrite `let r = q[i]` and `let mut r = q[i]` to the explicit borrow, so the type of `r` is a
   reference and the general rules record the borrow.
1. Allow taking a view of a placed tensor in code that cannot read it.
1. Refuse a view where an owning `Tensor` is expected, with E4011 asking for `.clone()`, including
   at a `return`.
1. Remove the provenance test from `views.rs`.
1. Update `docs/lang/types.md` and §4 of `borrow_checker_architecture.md`.

The code generators need little: `&Tensor` already lowers to a descriptor on the default path. The
legacy path has the gaps listed in #1162 and #1125.

## Alternatives rejected

| Alternative | Why not |
| --- | --- |
| A new `View<T, [dims]>` type | Two spellings for one thing, and every rule `&Tensor` already has written a second time |
| `View<Tensor<..>>` as a wrapper | The `Ref<Tensor<..>>` problem #429 removed: every consumer peels it |
| An owning/view flag in `Type::Tensor` | Needs a third state for mutability, which `&`/`&mut` already give |
| Strided views as `&Tensor` | Every `&Tensor` callee would lose its unit-stride layout, or every call would copy |
| Views copied automatically where a `Tensor` is wanted | Hides a copy of possibly large data in an ordinary call; Rust makes the same choice |

## Open questions

- **Should the `let r = q[i]` shorthand stay?** It is not Rust, and `let mut r = q[i]` uses `mut`
  for the reference's mutability rather than the binding's. Keeping it avoids churn now. A
  warning that suggests `&q[i]`, and later a refusal, would make the spelling uniform.
- **A run-time split** (`split_at_mut`) for kernels that split at a point known only at run time.
- **`transfer` of a view,** as an explicit copy of the window to another memory.
- **Columns and strided windows:** the spelling, in #1167.
