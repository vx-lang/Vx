# Standard library reference

Every public type and function in the shipped library modules, taken from their signatures, with
the documentation each one carries in the source.

Import a module with its path, then use the names it declares:

```rust
import std::vec;

fn main() -> i32 {
    let mut v = Vec<i32>::new();
    v.push(10);
    v.push(32);
    return v.get(0) + v.get(1);
}
```

The toolchain also ships a `graph` library outside `std`, imported as `graph::traversal` and
friends.

> This page is generated from `stdlib/core/*.vx` and `stdlib/std/*.vx` by
> `scripts/tools/gen_stdlib_reference.py`. Signatures are exactly what the source declares, and
> the prose under each is its `///` comment. An item with no description has none in the source.

## Contents

- [`core::clone`](#coreclone) — `Clone`, an explicit duplicate of a value.
- [`core::cmp`](#corecmp) — Ordering and equality: `PartialEq`, `Ord`, `PartialOrd` and `Ordering`.
- [`core::convert`](#coreconvert) — `From`, the conversions that cannot fail and lose nothing.
- [`core::default`](#coredefault) — `Default`, the value a type starts from.
- [`core::iter`](#coreiter) — The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.
- [`core::marker`](#coremarker) — The traits that say something about a type without giving it a method.
- [`core::mem`](#coremem) — Moving values around without looking at what they are.
- [`core::num`](#corenum) — The integer and float methods, stamped over every width.
- [`core::ops`](#coreops) — The callable types a closure literal lowers into.
- [`core::option`](#coreoption) — `Option<T>`, for a value that may be absent.
- [`core::ptr`](#coreptr) — Raw pointers: making one, and reading or writing through it.
- [`core::result`](#coreresult) — `Result<T, E>`, for an operation that may fail.
- [`std::alloc`](#stdalloc) — Raw allocation and deallocation.
- [`std::box`](#stdbox) — `Box<T>`, a single-owner heap allocation. Required for recursive types.
- [`std::fs`](#stdfs) — Files and directories.
- [`std::googletest`](#stdgoogletest) — Assertions for tests written in Vx.
- [`std::hash_map`](#stdhash_map) — `HashMap<K, V>`.
- [`std::hash_set`](#stdhash_set) — `HashSet<T>`.
- [`std::io`](#stdio) — Standard input, output and error.
- [`std::iter`](#stditer) — The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.
- [`std::libc`](#stdlibc) — Direct bindings to the C library.
- [`std::llama`](#stdllama) — Helpers used by the Llama 2 example.
- [`std::mmap`](#stdmmap) — Memory-mapped files.
- [`std::net`](#stdnet) — TCP and UDP sockets.
- [`std::rand`](#stdrand) — Seeded pseudo-random numbers, one stream per `Rng`.
- [`std::simd`](#stdsimd) — SIMD vector types and operations.
- [`std::string`](#stdstring) — `String` and text manipulation.
- [`std::tensor`](#stdtensor) — Operations on `Tensor`, including shape queries and elementwise maths.
- [`std::time`](#stdtime) — Clocks and durations.
- [`std::vec`](#stdvec) — `Vec<T>`, a growable array.

## `core::clone`

`Clone`, an explicit duplicate of a value.

**Types**

- `trait Clone`

**`trait Clone` methods**

- `fn clone(self : &Self) -> Self`<br>
  A duplicate of this value, made explicitly.
  The only required method. A type that can copy itself by a plain read still needs one,
  because a body written over `T : Clone` has to be able to call it.
- `fn clone_from(self : &mut Self, source : &Self) -> void`<br>
  Replace this value with a duplicate of `source`.
  A default written over `clone`. Override it for a type that can overwrite itself more
  cheaply than it can build a fresh copy; nothing in `core` needs to.

**`Clone for Option<T>` methods**

- `fn clone(self : &Option<T>) -> Option<T>`<br>
  The option with its value cloned, if it has one.

**`Clone for Result<T, E>` methods**

- `fn clone(self : &Result<T, E>) -> Result<T, E>`<br>
  The result with whichever side it holds cloned.

**`Clone for T` methods**, stamped for 12 instantiations

- `fn clone(self : &T) -> T`<br>
  A read, since a value of this width is copied by reading it.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `Ordering`

## `core::cmp`

Ordering and equality: `PartialEq`, `Ord`, `PartialOrd` and `Ordering`.

**Types**

- `enum Ordering`
- `trait PartialEq`
- `trait Ord`
- `trait PartialOrd`<br>
  A comparison that may answer with nothing, which is what a float needs: NaN is neither
  less than, equal to, nor greater than anything, itself included.

**`Ordering` methods**

- `fn is_lt(self : Ordering) -> bool`<br>
  Is this Less?
- `fn is_gt(self : Ordering) -> bool`<br>
  Is this Greater?
- `fn is_eq(self : Ordering) -> bool`<br>
  Is this Equal?
- `fn is_ne(self : Ordering) -> bool`<br>
  Is this anything but Equal?
- `fn is_le(self : Ordering) -> bool`<br>
  Is this Less or Equal?
- `fn is_ge(self : Ordering) -> bool`<br>
  Is this Greater or Equal?
- `fn reverse(self : Ordering) -> Ordering`<br>
  Less becomes Greater and back; Equal stays.
- `fn then_with(self : Ordering, f : Closure0<Ordering>) -> Ordering`<br>
  `then`, with the second comparison left uncomputed unless it is needed.
- `fn then(self : Ordering, other : Ordering) -> Ordering`<br>
  This one unless it is Equal, in which case the other. Chains comparisons: order by
  the first field, and on a tie by the second.

**`trait PartialEq` methods**

- `fn eq(self : &Self, other : &Self) -> bool`<br>
  Are the two values equal?
  The only required method of this trait. "Partial" is Rust's name for the fact that
  equality need not be reflexive: a NaN is not equal to itself, and `f32` implements this
  and not `Ord` for that reason.
- `fn ne(self : &Self, other : &Self) -> bool`<br>
  Are the two values different? The negation of `eq`, and not spelled `!=`, which answers
  false for a NaN on both sides (Vx#716).

**`trait Ord` methods**

- `fn cmp(self : &Self, other : &Self) -> Ordering`<br>
  Where this value sits relative to the other: Less, Equal or Greater.
  The only required method. Every other method of this trait is a default written over it,
  so a type joins the ordering by writing this one and nothing else.
  The order must be total, which is why the floats do not implement this trait: a NaN
  compares to nothing, and `PartialOrd` is where they answer.
- `fn lt(self : &Self, other : &Self) -> bool`<br>
  Is this value less than the other?
- `fn le(self : &Self, other : &Self) -> bool`<br>
  Is this value less than or equal to the other?
- `fn gt(self : &Self, other : &Self) -> bool`<br>
  Is this value greater than the other?
- `fn ge(self : &Self, other : &Self) -> bool`<br>
  Is this value greater than or equal to the other?
- `fn max(self : Self, other : Self) -> Self`<br>
  The greater of the two, taking both by value and handing one back.
  A method rather than the free function Rust also has, because the compiler reads the
  bare names `max` and `min` as the tensor reductions (Vx#223).
- `fn min(self : Self, other : Self) -> Self`<br>
  The lesser of the two.
- `fn clamp(self : Self, lo : Self, hi : Self) -> Self`<br>
  This value brought inside the range, so `lo` below it and `hi` above it.
  # Panics
  When `lo` is greater than `hi`, which asks for a range no value can be in.

**`trait PartialOrd` methods**

- `fn partial_cmp(self : &Self, other : &Self) -> Option<Ordering>`<br>
  Where this value sits relative to the other, or nothing when they do not compare.
  Nothing is what a float answers against a NaN: it is neither less than, equal to, nor
  greater than anything, itself included. Over a total order this always answers `Some`,
  and the integer widths implement it that way so a body written over `PartialOrd` works
  for every number.

**`PartialOrd for $t` methods**

- `fn partial_cmp(self : &$t, other : &$t) -> Option<Ordering>`<br>
  Always an answer, since this type is totally ordered.

**Functions**

- `fn max_by<T>(a : T, b : T, f : Closure2<T, T, Ordering>) -> T`<br>
  The greater of two values by `f`, and the lesser. They are free functions because they
  take the comparison rather than reading it off the type. Rust spells them `max_by` and
  `min_by`; the plain `max` and `min` are `Ord` methods here, since those two names are
  the compiler's tensor reductions.
- `fn min_by<T>(a : T, b : T, f : Closure2<T, T, Ordering>) -> T`<br>
  The lesser of the two by `f`, answering `a` when they compare equal.

**`PartialEq for T` methods**, stamped for 9 instantiations

- `fn eq(self : &T, other : &T) -> bool`<br>
  Equality at this width.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`

**`Ord for T` methods**, stamped for 9 instantiations

- `fn cmp(self : &T, other : &T) -> Ordering`<br>
  The three-way comparison at this width.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`

**`PartialEq for T` methods**, stamped for 9 instantiations

- `fn eq(self : &T, other : &T) -> bool`<br>
  Equality at this width. A NaN is equal to nothing, itself included.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`

**`PartialOrd for T` methods**, stamped for 9 instantiations

- `fn partial_cmp(self : &T, other : &T) -> Option<Ordering>`<br>
  The comparison, or nothing when either side is a NaN.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`

## `core::convert`

`From`, the conversions that cannot fail and lose nothing.

**Types**

- `trait From<T>`
- `enum Infallible`<br>
  An enum with no variants, so no value of it can be built. It is the error type of a
  conversion that cannot fail.

**`trait From<T>` methods**

- `fn from(v : T) -> Self`<br>
  This type built from a `T`, losing nothing.
  A static method, called through the target: `i32::from(x)`. Only implemented where every
  value of `T` fits, so there is no failure to report; the narrowing direction is
  `TryFrom`'s, which Vx does not have yet.

**`From<T> for Option<T>` methods**

- `fn from(v : T) -> Option<T>`<br>
  The value wrapped in `Some`.

**Functions**

- `fn identity<T>(x : T) -> T`<br>
  Returns its argument. Useful where a function is wanted and nothing should happen.

**`From<T> for U` methods**, stamped for 48 instantiations

- `fn from(v : T) -> U`<br>
  The value widened, which cannot lose anything at these two widths.

(T, U) = `i8 → i8`, `i16 → i16`, `i32 → i32`, `i64 → i64`, `u8 → u8`, `u16 → u16`, `u32 → u32`, `u64 → u64`, `f32 → f32`, `f64 → f64`, `bool → bool`, `i8 → i16`, `i8 → i32`, `i8 → i64`, `i16 → i32`, `i16 → i64`, `i32 → i64`, `u8 → u16`, `u8 → u32`, `u8 → u64`, `u16 → u32`, `u16 → u64`, `u32 → u64`, `u8 → i16`, `u8 → i32`, `u8 → i64`, `u16 → i32`, `u16 → i64`, `u32 → i64`, `i8 → f32`, `u8 → f32`, `i16 → f32`, `u16 → f32`, `i8 → f64`, `u8 → f64`, `i16 → f64`, `u16 → f64`, `i32 → f64`, `u32 → f64`, `f32 → f64`, `bool → i8`, `bool → i16`, `bool → i32`, `bool → i64`, `bool → u8`, `bool → u16`, `bool → u32`, `bool → u64`

## `core::default`

`Default`, the value a type starts from.

**Types**

- `trait Default`

**`; the language does not derive it. fn default() -> Self; } /// Zero, spelled at the width by the return type. } impl<T> Default for Option<T>` methods**

- `fn default() -> Self`<br>
  The value this type starts from: zero for a number, `false`, `None`.
  A static method, so it is called through the type -- `i64::default()`, not `x.default()`.
  A struct gets one by writing the impl; the language does not derive it.
- `fn default() -> Option<T>`<br>
  `None`, whatever `T` is: an option's default is the absent one.

**`Default for T` methods**, stamped for 11 instantiations

- `fn default() -> T`<br>
  Zero, spelled at this width.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`

## `core::iter`

The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.

**Types**

- `trait Iterator<Item>`
- `struct Range`<br>
  The numbers from `at` up to but not including `end`.
  A half-open span of `i64`, from `at` up to but not including `end`.
- `struct Map<I, Item, U>`<br>
  `f` over every item.
  An iterator over another one's items with `f` applied to each. Built by `map`.
- `struct Filter<I, Item>`<br>
  Only the items `keep` accepts.
  An iterator over the items of another that `keep` accepts. Built by `filter`.
- `struct Take<I, Item>`<br>
  The first `left` items, then nothing.
  `Item` is carried by the struct although `Take` never stores one: without associated
  types it is the only place the impl can read the element type from.
  An iterator over at most `left` items of another. Built by `take`.
- `struct Skip<I, Item>`<br>
  Everything after the first `drop` items.
  An iterator over another's items with the first `drop` of them discarded. Built by `skip`.

**`trait Iterator<Item>` methods**

- `fn next(self : &mut Self) -> Option<Item>`<br>
  The next item, or nothing once the sequence is finished.
  The only required method. Every other method of this trait is a default written over it,
  so a type becomes iterable by writing this one.
  Calling it again after it has answered nothing is allowed and answers nothing again; an
  iterator that would resume is not something this trait promises either way.
- `fn count(self : &mut Self) -> i64`<br>
  How many items are left, consuming them all to find out.
- `fn last(self : &mut Self) -> Option<Item>`<br>
  The final item, consuming the sequence. Nothing when it is already finished.
- `fn nth(self : &mut Self, n : i64) -> Option<Item>`<br>
  The item `n` places along, counting the next one as zero, discarding those before it.
  Nothing when the sequence finishes first. The items skipped are consumed either way.
- `fn any(self : &mut Self, f : Closure1<Item, bool>) -> bool`<br>
  Does `f` accept any item? Stops at the first it does, leaving the rest unconsumed.
- `fn all(self : &mut Self, f : Closure1<Item, bool>) -> bool`<br>
  Does `f` accept every item? Stops at the first it does not.
  True for a sequence that is already finished, which is the usual convention: there is no
  item to disagree.
- `fn find(self : &mut Self, f : Closure1<Item, bool>) -> Option<Item>`<br>
  The first item `f` accepts, or nothing. Stops there, so the rest is unconsumed.
- `fn position(self : &mut Self, f : Closure1<Item, bool>) -> Option<i64>`<br>
  How far along the first item `f` accepts is, counting the next one as zero.
- `fn for_each(self : &mut Self, f : Closure1<Item, i32>) -> i32`<br>
  Hand every item to `f`, consuming the sequence.
  `f` answers an `i32` rather than nothing, and this returns the last of them, because no
  closure literal can return void yet (Vx#711). Both signatures become Rust's when it can.

**`Iterator<i64> for Range` methods**

- `fn next(self : &mut Range) -> Option<i64>`<br>
  The next value in the span, or nothing once `end` is reached.

**`Iterator<U> for Map<I, Item, U>` methods**

- `fn next(self : &mut Map<I, Item, U>) -> Option<U>`<br>
  The inner iterator's next item with `f` applied.

**`Iterator<Item> for Filter<I, Item>` methods**

- `fn next(self : &mut Filter<I, Item>) -> Option<Item>`<br>
  The inner iterator's next item that `keep` accepts.

**`Iterator<Item> for Take<I, Item>` methods**

- `fn next(self : &mut Take<I, Item>) -> Option<Item>`<br>
  The inner iterator's next item, until `left` of them have been handed out.

**`Iterator<Item> for Skip<I, Item>` methods**

- `fn next(self : &mut Skip<I, Item>) -> Option<Item>`<br>
  The inner iterator's next item, once the skipped ones have been consumed.

**Functions**

- `fn map<I, Item, U>(inner : I, f : Closure1<Item, U>) -> Map<I, Item, U>`<br>
  The adaptors are built through these rather than by writing the struct literal: a
  closure literal is coerced to `Closure1<A, B>` in an argument but not in a field
  initializer (Vx#648).
- `fn filter<I, Item>(inner : I, keep : Closure1<Item, bool>) -> Filter<I, Item>`<br>
  An iterator over the items of `inner` that `keep` accepts.
- `fn take<I, Item>(inner : I, left : i64) -> Take<I, Item>`<br>
  An iterator over at most `left` items of `inner`.
- `fn skip<I, Item>(inner : I, drop : i64) -> Skip<I, Item>`<br>
  An iterator over `inner` with its first `drop` items discarded.
- `fn range(at : i64, end : i64) -> Range`<br>
  The numbers `at .. end`.

## `core::marker`

The traits that say something about a type without giving it a method.

**Types**

- `trait Copy`<br>
  A type that is duplicated rather than moved when it is assigned.
  Opt-in, as Rust's is: a type is copyable only when it says so, and only when every field
  already is. That is what keeps it away from placement -- a tensor or a placed buffer cannot
  declare it, so they stay linear without a special case, and duplicating placed data stays an
  explicit `transfer`.
  Enforced for a struct and for a payload-free enum. A generic enum is not treated as linear
  at all, so `Option` and `Result` survive a move whether or not they declare this (Vx#715).
- `trait Send`<br>
  A type that may be moved to another thread. Declared, not enforced: there are no threads
  to send a value between yet.
- `trait Sync`<br>
  A type that may be referenced from several threads at once. Declared, not enforced, for
  the same reason as `Send`.
- `trait Sized`<br>
  A type whose size is known at compile time, which every Vx type is. The bound exists to be
  written, as Rust's does; nothing is excluded by it.
- `struct PhantomData<T>`<br>
  A field that records a type without storing a value of it.
  An empty struct occupies nothing, so a struct carrying one is no larger. It is how a type
  parameter that appears in no field is still named by the type.

## `core::mem`

Moving values around without looking at what they are.

**Functions**

- `fn size_of<T>() -> i64`<br>
  The size of `T` in bytes.
- `fn swap<T>(a : &mut T, b : &mut T) -> void`<br>
  Each value ends up where the other one was.
- `fn replace<T>(dest : &mut T, src : T) -> T`<br>
  `src` goes in, and what was there comes back.
- `fn drop<T>(_x : T) -> void`<br>
  Consumes the value. With no `Drop` in the language this frees nothing; it says that the
  caller is finished with it, and the move checker holds them to it.
- `fn forget<T>(_x : T) -> void`<br>
  Consumes the value without running anything. The same as `drop` until `Drop` exists.
- `fn needs_drop<T>() -> bool`<br>
  Whether dropping a `T` does any work. False for every type, since nothing has a
  destructor yet.

## `core::num`

The integer and float methods, stamped over every width.

**`$t` methods**

- `fn min_value(self : $t) -> $t`<br>
  Zero, at every unsigned width.
- `fn max_value(self : $t) -> $t`<br>
  The largest value of this width.
- `fn bits(self : $t) -> $t`<br>
  How many bits this width has.
- `fn count_ones(self : $t) -> $t`<br>
  How many bits are set.
- `fn count_zeros(self : $t) -> $t`<br>
  How many bits are clear.
- `fn leading_zeros(self : $t) -> $t`<br>
  Zero bits above the highest set bit. All of them, for zero.
- `fn trailing_zeros(self : $t) -> $t`<br>
  Zero bits below the lowest set bit. All of them, for zero.
- `fn is_power_of_two(self : $t) -> bool`<br>
  Zero is not a power of two.
- `fn abs_diff(self : $t, other : $t) -> $t`<br>
  The distance between two values, which is never negative and so always fits.
- `fn pow(self : $t, exp : $t) -> $t`<br>
  By squaring. Overflow wraps, as every arithmetic operator here does.
- `fn div_euclid(self : $t, rhs : $t) -> $t`<br>
  Plain division: an unsigned quotient is already the Euclidean one.
- `fn rem_euclid(self : $t, rhs : $t) -> $t`<br>
  Plain remainder, which at this width is never negative.
- `fn ilog2(self : $t) -> $t`<br>
  Rounded down. Refused at zero, which has no logarithm.
- `fn next_power_of_two(self : $t) -> $t`<br>
  One for anything at or below one. Refused above the top power of two, which is the
  half of the range that has no next power to reach.
- `fn rotate_left(self : $t, n : $t) -> $t`<br>
  Wrapping round. No mask, unlike the signed rotate: `>>` brings in zeros here.
- `fn rotate_right(self : $t, n : $t) -> $t`<br>
  The bits rotated right.
- `fn swap_bytes(self : $t) -> $t`<br>
  The bytes reversed.
- `fn reverse_bits(self : $t) -> $t`<br>
  The bits reversed.
- `fn checked_add(self : $t, rhs : $t) -> Option<$t>`<br>
  Nothing if it would not fit. The bound is rearranged so the check cannot overflow.
- `fn checked_sub(self : $t, rhs : $t) -> Option<$t>`<br>
  Nothing if it would go below zero, which is where an unsigned width ends.
- `fn checked_mul(self : $t, rhs : $t) -> Option<$t>`<br>
  Checked by dividing back out, exact when the product fit.
- `fn checked_div(self : $t, rhs : $t) -> Option<$t>`<br>
  Nothing on division by zero, which is the only division that fails here.
- `fn checked_rem(self : $t, rhs : $t) -> Option<$t>`<br>
  Refused on a zero divisor, as `checked_div` is.
- `fn saturating_add(self : $t, rhs : $t) -> $t`<br>
  Held at the top of the range instead of wrapping past it.
- `fn saturating_sub(self : $t, rhs : $t) -> $t`<br>
  Held at zero instead of wrapping below it.
- `fn wrapping_add(self : $t, rhs : $t) -> $t`<br>
  The sum, wrapping round at the width. Vx's `+` already wraps.
- `fn wrapping_sub(self : $t, rhs : $t) -> $t`<br>
  The difference, wrapping round at the width, so subtracting past zero lands near
  the top.
- `fn wrapping_mul(self : $t, rhs : $t) -> $t`<br>
  The product, keeping the low bits and discarding the rest.
- `fn wrapping_neg(self : $t) -> $t`<br>
  Zero minus this, wrapping.
- `fn saturating_mul(self : $t, rhs : $t) -> $t`<br>
  Clamped to the top of the width rather than wrapping. There is no other end to
  clamp to without a sign.
- `fn leading_ones(self : $t) -> $t`<br>
  How many set bits the value starts with, counting from the top.
  The complement is taken with `^` against an all-ones value rather than with `!`,
  which the checker types as a `bool` on an integer and the two code generators
  lower two different ways (Vx#717).
- `fn trailing_ones(self : $t) -> $t`<br>
  How many set bits the value ends with, counting from the bottom.
- `fn sqrt(self : $t) -> $t`<br>
  The positive square root.
- `fn abs(self : $t) -> $t`<br>
  The distance from zero, so the sign is dropped.
- `fn exp(self : $t) -> $t`<br>
  e raised to this.
- `fn exp2(self : $t) -> $t`<br>
  Two raised to this.
- `fn exp_m1(self : $t) -> $t`<br>
  `exp` minus one, kept accurate for a small argument where the subtraction would
  lose every significant digit.
- `fn ln(self : $t) -> $t`<br>
  The natural logarithm.
- `fn log2(self : $t) -> $t`<br>
  The logarithm to base two.
- `fn log10(self : $t) -> $t`<br>
  The logarithm to base ten.
- `fn ln_1p(self : $t) -> $t`<br>
  `ln` of one plus this, kept accurate for a small argument.
- `fn sin(self : $t) -> $t`<br>
  The sine of this many radians.
- `fn cos(self : $t) -> $t`<br>
  The cosine of this many radians.
- `fn tan(self : $t) -> $t`<br>
  The tangent of this many radians.
- `fn asin(self : $t) -> $t`<br>
  The angle in radians whose sine is this, between -pi/2 and pi/2.
- `fn acos(self : $t) -> $t`<br>
  The angle in radians whose cosine is this, between 0 and pi.
- `fn atan(self : $t) -> $t`<br>
  The angle in radians whose tangent is this. `atan2` is the form that keeps the quadrant.
- `fn sinh(self : $t) -> $t`<br>
  The hyperbolic sine.
- `fn cosh(self : $t) -> $t`<br>
  The hyperbolic cosine.
- `fn tanh(self : $t) -> $t`<br>
  The hyperbolic tangent.
- `fn floor(self : $t) -> $t`<br>
  The largest whole number no greater than this.
- `fn ceil(self : $t) -> $t`<br>
  The smallest whole number no less than this.
- `fn round(self : $t) -> $t`<br>
  The nearest whole number, halves going away from zero.
- `fn trunc(self : $t) -> $t`<br>
  The whole part, so the fraction is dropped and the sign is kept.
- `fn fract(self : $t) -> $t`<br>
  The fractional part, which carries this value's sign.
- `fn powf(self : $t, n : $t) -> $t`<br>
  This raised to `n`.
- `fn atan2(self : $t, x : $t) -> $t`<br>
  The angle to the point (`x`, this), which is `atan` with the quadrant kept.
- `fn copysign(self : $t, sign : $t) -> $t`<br>
  This value's magnitude with `sign`'s sign.
- `fn recip(self : $t) -> $t`<br>
  One divided by this.
- `fn to_degrees(self : $t) -> $t`<br>
  This many radians in degrees.
- `fn to_radians(self : $t) -> $t`<br>
  This many degrees in radians.
- `fn is_nan(self : $t) -> bool`<br>
  Is this the value that is equal to nothing, itself included?
  Spelled as the negation of an equality rather than as `self != self`, which is
  how Rust writes it: `!=` between floats lowers to the ordered predicate and so
  answers false for a NaN, while `==` is ordered as it should be (Vx#716).
- `fn signum(self : $t) -> $t`<br>
  One with this value's sign, or the value itself when it is a NaN. Zero answers 1
  rather than 0, which is Rust's rule and not `signum`'s in every language.
- `fn is_finite(self : $t) -> bool`<br>
  Is this a real number, rather than an infinity or a NaN?
- `fn is_infinite(self : $t) -> bool`<br>
  Is this an infinity, of either sign?

**`T` methods**, stamped for 4 instantiations

- `fn min_value(self : T) -> T`<br>
  The smallest value of this width.
- `fn max_value(self : T) -> T`<br>
  The largest value of this width.
- `fn bits(self : T) -> T`<br>
  How many bits this width has.
- `fn count_ones(self : T) -> T`<br>
  How many bits are set. The hardware instruction, so a negative operand is
  counted right; a loop shifting right would copy the sign bit forever.
- `fn count_zeros(self : T) -> T`<br>
  How many bits are clear.
- `fn leading_zeros(self : T) -> T`<br>
  Zero bits above the highest set bit. All of them, for zero.
- `fn trailing_zeros(self : T) -> T`<br>
  Zero bits below the lowest set bit. All of them, for zero.
- `fn is_power_of_two(self : T) -> bool`<br>
  Zero and the negatives are not.
- `fn is_positive(self : T) -> bool`<br>
  Is this greater than zero? Zero is neither positive nor negative.
- `fn is_negative(self : T) -> bool`<br>
  Is this less than zero?
- `fn abs(self : T) -> T`<br>
  Refused at the smallest value, which has no positive counterpart.
- `fn signum(self : T) -> T`<br>
  -1, 0 or 1, by sign.
- `fn abs_diff(self : T, other : T) -> T`<br>
  The distance between two values.
- `fn pow(self : T, exp : T) -> T`<br>
  By squaring. Overflow wraps, as every arithmetic operator here does.
- `fn rem_euclid(self : T, rhs : T) -> T`<br>
  Never negative, whatever the signs: -7 % 4 is -3 where this is 1.
- `fn div_euclid(self : T, rhs : T) -> T`<br>
  The quotient pairing with `rem_euclid`.
- `fn ilog2(self : T) -> T`<br>
  Rounded down. Refused at zero and below.
- `fn next_power_of_two(self : T) -> T`<br>
  One for anything at or below one. The top power of two does not fit in a
  signed width, so a value beyond it is refused.
- `fn rotate_left(self : T, n : T) -> T`<br>
  Wrapping round. The right half is masked because `>>` copies the sign bit.
- `fn rotate_right(self : T, n : T) -> T`<br>
  The bits rotated right.
- `fn swap_bytes(self : T) -> T`<br>
  The bytes reversed. The mask is a parameter because 255 does not fit in an
  `i8`; it is accepted there and wraps to -1, which is right by luck.
- `fn reverse_bits(self : T) -> T`<br>
  The bits reversed.
- `fn checked_add(self : T, rhs : T) -> Option<T>`<br>
  Nothing if it would not fit. The bound is rearranged so the check itself
  cannot overflow.
- `fn checked_sub(self : T, rhs : T) -> Option<T>`<br>
  Nothing if it would not fit.
- `fn checked_mul(self : T, rhs : T) -> Option<T>`<br>
  Checked by dividing back out, exact when the product fit. The two divisions
  that would themselves overflow are ruled out first.
- `fn checked_div(self : T, rhs : T) -> Option<T>`<br>
  Nothing on division by zero, or on the one division that overflows.
- `fn checked_rem(self : T, rhs : T) -> Option<T>`<br>
  Refused in the same two cases as `checked_div`.
- `fn checked_neg(self : T) -> Option<T>`<br>
  Nothing for the smallest value.
- `fn saturating_add(self : T, rhs : T) -> T`<br>
  Held at the end of the range instead of wrapping past it.
- `fn saturating_sub(self : T, rhs : T) -> T`<br>
  Held at the end of the range instead of wrapping past it.
- `fn wrapping_add(self : T, rhs : T) -> T`<br>
  The sum, wrapping round at the width. Vx's `+` already wraps, so the body is the
  operator; the name is what a reader who wants that on purpose looks for, and what
  a hash function is written in.
- `fn wrapping_sub(self : T, rhs : T) -> T`<br>
  The difference, wrapping round at the width.
- `fn wrapping_mul(self : T, rhs : T) -> T`<br>
  The product, keeping the low bits and discarding the rest.
- `fn wrapping_neg(self : T) -> T`<br>
  Zero minus this, wrapping. The smallest value negates to itself, because its
  positive is one past the largest.
- `fn saturating_mul(self : T, rhs : T) -> T`<br>
  Clamped to the width rather than wrapping. Which end it clamps to is the sign the
  product would have had, which is whether the two operands agree in sign.
- `fn leading_ones(self : T) -> T`<br>
  How many set bits the value starts with, counting from the top.
  The complement is taken with `^` against an all-ones value rather than with `!`,
  which the checker types as a `bool` on an integer and the two code generators
  lower two different ways (Vx#717).
- `fn trailing_ones(self : T) -> T`<br>
  How many set bits the value ends with, counting from the bottom.

T = `i8`, `i16`, `i32`, `i64`

## `core::ops`

The callable types a closure literal lowers into.

**Types**

- `struct Closure0<Ret>`
- `struct Closure1<Arg, Ret>`
- `struct Closure2<Arg1, Arg2, Ret>`
- `struct Closure3<Arg1, Arg2, Arg3, Ret>`

## `core::option`

`Option<T>`, for a value that may be absent.

**Types**

- `enum Option<T>`

**`Option<T>` methods**

- `fn is_some(self : &Option<T>) -> Bool`<br>
  Is there a value?
- `fn is_none(self : &Option<T>) -> Bool`<br>
  Is there no value?
- `fn unwrap(self : Option<T>) -> T`<br>
  The value, or a stop. Reach for `unwrap_or` where there is a sensible answer for
  the absent case; this one ends the program.
- `fn unwrap_or(self : Option<T>, default : T) -> T`<br>
  The value, or the given one. `default` is evaluated by the caller either way, so
  keep it cheap; Rust's `unwrap_or_else` is the form that does not, and it takes a
  closure whose type parameter this cannot yet spell.
- `fn or(self : Option<T>, other : Option<T>) -> Option<T>`<br>
  This one if it holds a value, otherwise the other. Both sides are the same type,
  which is why this fits while `and_then` does not.
- `fn and(self : Option<T>, other : Option<T>) -> Option<T>`<br>
  The other one if this holds a value, otherwise nothing.
- `fn xor(self : Option<T>, other : Option<T>) -> Option<T>`<br>
  Whichever one holds a value, and nothing when both do or neither does.
- `fn map<U>(self : Option<T>, f : Closure1<T, U>) -> Option<U>`<br>
  The value with `f` applied, if there is one.
- `fn and_then<U>(self : Option<T>, f : Closure1<T, Option<U>>) -> Option<U>`<br>
  `map` for an `f` that answers with an `Option` of its own, without the nesting.
- `fn filter(self : Option<T>, p : Closure1<T, bool>) -> Option<T>`<br>
  The value if it is there and `p` accepts it, otherwise nothing.
- `fn map_or<U>(self : Option<T>, default : U, f : Closure1<T, U>) -> U`<br>
  `f` applied to the value, or the given answer when there is none. Both are the same
  type, which is what separates this from `map`.
- `fn unwrap_or_else(self : Option<T>, f : Closure0<T>) -> T`<br>
  The value, or the answer `f` gives. Unlike `unwrap_or`, nothing is computed when
  there is a value.
- `fn is_some_and(self : Option<T>, p : Closure1<T, bool>) -> bool`<br>
  Is there a value, and does `p` accept it?
- `fn is_none_or(self : Option<T>, p : Closure1<T, bool>) -> bool`<br>
  Is there no value, or does `p` accept the one there is? The mirror of `is_some_and`.
- `fn or_else(self : Option<T>, f : Closure0<Option<T>>) -> Option<T>`<br>
  This one if it holds a value, otherwise what `f` gives. `or` is the form that
  evaluates the other side either way.
- `fn map_or_else<U>(self : Option<T>, d : Closure0<U>, f : Closure1<T, U>) -> U`<br>
  `f` applied to the value, or what `d` gives when there is none. `map_or` is the form
  that takes the fallback as a value.
- `fn take(self : &mut Option<T>) -> Option<T>`<br>
  The value, leaving nothing behind.
- `fn replace(self : &mut Option<T>, v : T) -> Option<T>`<br>
  The value, leaving `v` behind.

## `core::ptr`

Raw pointers: making one, and reading or writing through it.

**Functions**

- `fn null<T>() -> *const T`<br>
  A pointer to nothing.
- `fn null_mut<T>() -> *mut T`<br>
  A mutable pointer to nothing.
- `unsafe fn read<T>(p : *const T) -> T`<br>
  The value the pointer addresses. The caller promises there is one.
- `unsafe fn write<T>(p : *mut T, v : T) -> void`<br>
  Puts a value where the pointer addresses. The caller promises it may.

## `core::result`

`Result<T, E>`, for an operation that may fail.

**Types**

- `enum Result<T, E>`

**`Result<T, E>` methods**

- `fn is_ok(self : &Result<T, E>) -> bool`<br>
  Did it succeed?
- `fn is_err(self : &Result<T, E>) -> bool`<br>
  Did it fail? The negation of `is_ok`.
- `fn unwrap(self : Result<T, E>) -> T`<br>
  The value, or a stop. `unwrap_or` is the form with an answer for the failing case.
- `fn unwrap_or(self : Result<T, E>, default : T) -> T`<br>
  The value, or the given one on failure.
  `default` is evaluated by the caller either way, so keep it cheap; `unwrap_or_else` is
  the form that computes nothing when there is a value.
- `fn ok(self : Result<T, E>) -> Option<T>`<br>
  The success dropped, leaving what there is of one.
- `fn err(self : Result<T, E>) -> Option<E>`<br>
  The failure as an `Option`, the mirror of `ok`.
- `fn map<U>(self : Result<T, E>, f : Closure1<T, U>) -> Result<U, E>`<br>
  `f` over the success, the failure untouched.
- `fn map_err<F>(self : Result<T, E>, f : Closure1<E, F>) -> Result<T, F>`<br>
  `f` over the failure, the success untouched.
- `fn and_then<U>(self : Result<T, E>, f : Closure1<T, Result<U, E>>) -> Result<U, E>`<br>
  `map` for an `f` that may itself fail, without the nesting.
- `fn unwrap_or_else(self : Result<T, E>, f : Closure1<E, T>) -> T`<br>
  The value, or what `f` makes of the failure.
- `fn unwrap_err(self : Result<T, E>) -> E`<br>
  The failure, or a stop. The mirror of `unwrap`.
- `fn is_ok_and(self : Result<T, E>, p : Closure1<T, bool>) -> bool`<br>
  Did it succeed, and does `p` accept the value?
- `fn is_err_and(self : Result<T, E>, p : Closure1<E, bool>) -> bool`<br>
  Did it fail, and does `p` accept the failure?
- `fn and<U>(self : Result<T, E>, other : Result<U, E>) -> Result<U, E>`<br>
  The other one if this succeeded, otherwise this failure. The success types differ,
  which is why this takes a type parameter where `Option::and` does not.
  Bound first and returned once: a `match` whose arms each return an enum is declined
  by the flat path, and `other` is a starting value that needs no default of its own.
- `fn or<F>(self : Result<T, E>, other : Result<T, F>) -> Result<T, F>`<br>
  This success if there is one, otherwise the other result. The failure types differ.
- `fn map_or<U>(self : Result<T, E>, default : U, f : Closure1<T, U>) -> U`<br>
  `f` applied to the value, or the given answer when there is none.
- `fn map_or_else<U>(self : Result<T, E>, d : Closure1<E, U>, f : Closure1<T, U>) -> U`<br>
  `f` over the value, or `d` over the failure. Both answer with the same type.

**`Option<T>` methods**

- `fn ok_or<E>(self : Option<T>, err : E) -> Result<T, E>`<br>
  The value as a success, or the given failure.
- `fn ok_or_else<E>(self : Option<T>, f : Closure0<E>) -> Result<T, E>`<br>
  The value as a success, or the failure `f` gives. Nothing is computed when there is
  a value.

## `std::alloc`

Raw allocation and deallocation.

**Functions** *(bound directly to C)*

- `fn malloc(size : i64) -> *mut i8`
- `fn realloc(ptr : *mut i8, size : i64) -> *mut i8`
- `fn free(ptr : *mut i8) -> i32`

## `std::box`

`Box<T>`, a single-owner heap allocation. Required for recursive types.

**Types**

- `struct Box<T>`

**`Box<T>` methods**

- `fn new(val : T) -> Box<T>`
- `fn free(self : &mut Box<T>) -> i32`

## `std::fs`

Files and directories.

**Types**

- `struct File`

**`File` methods**

- `unsafe fn open(path : *const i8, mode : i32) -> File`
- `unsafe fn read(self : *mut File, buffer : *mut u8, len : i64) -> i64`
- `unsafe fn write(self : *mut File, buffer : *const u8, len : i64) -> i64`
- `fn seek(self : *mut File, offset : i64, whence : i32) -> i64`

**Functions**

- `unsafe fn file_drop(file : *mut File) -> void`

**C bindings** *(the native functions this module is built on)*

- `fn vx_file_open(c_path : *const i8, mode : i32) -> *mut i8`
- `fn vx_file_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`
- `fn vx_file_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64`
- `fn vx_file_seek(ptr : *mut i8, offset : i64, whence : i32) -> i64`
- `fn vx_file_drop(ptr : *mut i8) -> i32`
- `fn fopen(path : *const i8, mode : *const i8) -> *mut i8`
- `fn fread(ptr : *mut u8, size : i64, nmemb : i64, stream : *mut i8) -> i64`
- `fn fclose(stream : *mut i8) -> i32`
- `fn fileno(f : *mut i8) -> i32`
- `fn fseek(f : *mut i8, offset : i64, whence : i32) -> i32`
- `fn ftell(f : *mut i8) -> i64`
- `fn mmap(addr : *mut i8, len : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8`
- `fn munmap(addr : *mut i8, len : i64) -> i32`

## `std::googletest`

Assertions for tests written in Vx.

**Types**

- `trait GoogletestEq`

**`trait GoogletestEq` methods**

- `fn expect_eq(self : Self, expected : Self) -> i32`

**`GoogletestEq for f32` methods**

- `fn expect_eq(self : f32, expected : f32) -> i32`

**`GoogletestEq for i32` methods**

- `fn expect_eq(self : i32, expected : i32) -> i32`

**Functions**

- `fn expect_eq<T : GoogletestEq>(actual : T, expected : T) -> i32`

**C bindings** *(the native functions this module is built on)*

- `fn vx_googletest_expect_eq_f32(actual : f32, expected : f32) -> i32`
- `fn vx_googletest_expect_eq_i32(actual : i32, expected : i32) -> i32`

## `std::hash_map`

`HashMap<K, V>`.

**Functions** *(bound directly to C)*

- `fn vx_hash_map_new_i32_i32() -> *mut i8`
- `fn vx_hash_map_insert_i32_i32(ptr : *mut i8, key : i32, val : i32) -> i32`
- `fn vx_hash_map_get_i32_i32(ptr : *mut i8, key : i32) -> *mut i8`
- `fn vx_hash_map_contains_key_i32_i32(ptr : *mut i8, key : i32) -> Bool`
- `fn vx_hash_map_len_i32_i32(ptr : *mut i8) -> i32`
- `fn vx_hash_map_drop_i32_i32(ptr : *mut i8) -> i32`
- `fn vx_hash_map_new_i32_f32() -> *mut i8`
- `fn vx_hash_map_insert_i32_f32(ptr : *mut i8, key : i32, val : f32) -> i32`
- `fn vx_hash_map_get_i32_f32(ptr : *mut i8, key : i32) -> *mut i8`
- `fn vx_hash_map_contains_key_i32_f32(ptr : *mut i8, key : i32) -> Bool`
- `fn vx_hash_map_len_i32_f32(ptr : *mut i8) -> i32`
- `fn vx_hash_map_drop_i32_f32(ptr : *mut i8) -> i32`

## `std::hash_set`

`HashSet<T>`.

**Functions** *(bound directly to C)*

- `fn vx_hash_set_new_i32() -> *mut i8`
- `fn vx_hash_set_insert_i32(ptr : *mut i8, val : i32) -> i32`
- `fn vx_hash_set_contains_i32(ptr : *mut i8, val : i32) -> Bool`
- `fn vx_hash_set_len_i32(ptr : *mut i8) -> i32`
- `fn vx_hash_set_drop_i32(ptr : *mut i8) -> i32`

## `std::io`

Standard input, output and error.

**Functions**

- `unsafe fn stdout_write(buffer : *const u8, len : i64) -> i64`
- `unsafe fn stderr_write(buffer : *const u8, len : i64) -> i64`
- `unsafe fn stdin_read(buffer : *mut u8, len : i64) -> i64`

**C bindings** *(the native functions this module is built on)*

- `fn vx_stdout_write(buffer : *const u8, len : i64) -> i64`
- `fn vx_stderr_write(buffer : *const u8, len : i64) -> i64`
- `fn vx_stdin_read(buffer : *mut u8, len : i64) -> i64`

## `std::iter`

The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.

**Types**

- `trait Iterator<T, Item>`

**`trait Iterator<T, Item>` methods**

- `fn next(self : &mut T) -> Option<Item>`

**`Iterator<Map<I, F, Item, NewItem>, NewItem> for Map<I, F, Item, NewItem>` methods**

- `fn next(self : &mut Map<I, F, Item, NewItem>) -> Option<NewItem>`

**`Map<I, F, Item, NewItem>` methods**

- `fn collect(self : &mut Map<I, F, Item, NewItem>) -> Vec<NewItem>`

## `std::libc`

Direct bindings to the C library.

**Functions** *(bound directly to C)*

- `fn open(path : *const i8, flags : i32) -> i32`
- `fn close(fd : i32) -> i32`
- `fn lseek(fd : i32, offset : i64, whence : i32) -> i64`

## `std::llama`

Helpers used by the Llama 2 example.

**Types**

- `struct LlamaConfig`
- `struct TransformerWeightOffsets`
- `struct Tokenizer`

**`LlamaConfig` methods**

- `fn load(filepath : *const i8) -> LlamaConfig`

**`TransformerWeightOffsets` methods**

- `fn calculate(c : &LlamaConfig) -> TransformerWeightOffsets`
- `fn load_all_weights(filepath : *const i8, c : &LlamaConfig) -> Tensor<f32, [?, ?]>`

**`Tokenizer` methods**

- `fn load(filepath : *const i8, vocab_size : i32) -> Tokenizer`
- `fn decode(self : &Tokenizer, prev_token : i32, token : i32) -> String`

**C bindings** *(the native functions this module is built on)*

- `fn vx_load_config(filepath : *const i8) -> *mut i32`
- `fn vx_load_weights(filepath : *const i8) -> *mut f32`
- `fn vx_build_tokenizer(filepath : *const i8, vocab_size : i32) -> *mut i8`
- `fn vx_decode_token(tokenizer_ptr : *mut i8, prev_token : i32, token : i32) -> *const i8`
- `fn vx_encode_prompt(tokenizer_ptr : *mut i8, text_ptr : *const i8) -> *mut i32`
- `fn vx_read_prompt_file(filepath : *const i8) -> *const i8`
- `fn vx_get_llama_config() -> *mut i32`

## `std::mmap`

Memory-mapped files.

**Functions** *(bound directly to C)*

- `fn mmap(addr : *mut i8, length : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8`
- `fn munmap(addr : *mut i8, length : i64) -> i32`

## `std::net`

TCP and UDP sockets.

**Types**

- `struct TcpStream`
- `struct UdpSocket`
- `struct TcpListener`

**`TcpStream` methods**

- `unsafe fn connect(addr : *const i8) -> TcpStream`
- `unsafe fn read(self : *mut TcpStream, buffer : *mut u8, len : i64) -> i64`
- `unsafe fn write(self : *mut TcpStream, buffer : *const u8, len : i64) -> i64`

**Functions**

- `unsafe fn tcp_stream_drop(stream : *mut TcpStream) -> void`
- `unsafe fn udp_socket_drop(socket : *mut UdpSocket) -> void`
- `unsafe fn tcp_listener_drop(listener : *mut TcpListener) -> void`

**`UdpSocket` methods**

- `unsafe fn bind(addr : *const i8) -> UdpSocket`
- `unsafe fn recv(self : *mut UdpSocket, buffer : *mut u8, len : i64) -> i64`
- `unsafe fn send_to(self : *mut UdpSocket, buffer : *const u8, len : i64, addr : *const i8) -> i64`

**`TcpListener` methods**

- `unsafe fn bind(addr : *const i8) -> TcpListener`
- `fn accept(self : *mut TcpListener) -> TcpStream`

**C bindings** *(the native functions this module is built on)*

- `fn vx_tcp_stream_connect(c_addr : *const i8) -> *mut i8`
- `fn vx_tcp_stream_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`
- `fn vx_tcp_stream_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64`
- `fn vx_tcp_stream_drop(ptr : *mut i8) -> i32`
- `fn vx_udp_socket_bind(c_addr : *const i8) -> *mut i8`
- `fn vx_udp_socket_recv(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`
- `fn vx_udp_socket_send_to(ptr : *mut i8, buffer : *const u8, len : i64, c_addr : *const i8) -> i64`
- `fn vx_udp_socket_drop(ptr : *mut i8) -> i32`
- `fn vx_tcp_listener_bind(c_addr : *const i8) -> *mut i8`
- `fn vx_tcp_listener_accept(ptr : *mut i8) -> *mut i8`
- `fn vx_tcp_listener_drop(ptr : *mut i8) -> i32`

## `std::rand`

Seeded pseudo-random numbers, one stream per `Rng`.

**Types**

- `struct SplitMix64`<br>
  SplitMix64: a counter, scrambled. One multiply-xor-shift chain, no rejection, no loop.
  Its job here is to turn one seed word into the four `Rng` needs, which is what it was
  written for. It is a usable generator on its own where 64 bits of state are enough.
- `struct Rng`<br>
  A stream of pseudo-random numbers. Seed it, then draw from it.
  `spare` holds the second of the pair `normal` produces, since the polar method makes two
  normals at once and handing one back would throw half the work away.

**Functions**

- `fn sqrt_f64(x : f64) -> f64`
- `fn ln_f64(x : f64) -> f64`

**`SplitMix64` methods**

- `fn seeded(seed : u64) -> SplitMix64`
- `fn next_u64(self : &mut SplitMix64) -> u64`<br>
  Advance the counter by the golden-ratio constant, then scramble the value taken. The
  three constants are 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9 and 0x94D049BB133111EB,
  spelled in decimal because Vx has no hex literal.

**`Rng` methods**

- `fn seeded(seed : u64) -> Rng`<br>
  A stream from one seed word. Every seed is allowed, zero included.
- `fn next_u64(self : &mut Rng) -> u64`<br>
  The next draw, uniform over the whole 64-bit range. Every other method is built on it.
  The value handed back is computed from the state *before* the state advances, which is
  what xoshiro256\*\* specifies; returning the new state instead is a different and worse
  generator.
- `fn next_u32(self : &mut Rng) -> u32`<br>
  The narrower widths take the *high* bits of a draw. The low bits of a xoshiro draw are
  the weakest ones, and a `% 256` would hand back exactly those.
- `fn next_u16(self : &mut Rng) -> u16`
- `fn next_u8(self : &mut Rng) -> u8`
- `fn next_i64(self : &mut Rng) -> i64`<br>
  Uniform over the signed range, negatives included: the bits are reinterpreted, not
  clamped, so half the draws are below zero.
- `fn next_i32(self : &mut Rng) -> i32`
- `fn next_i16(self : &mut Rng) -> i16`
- `fn next_i8(self : &mut Rng) -> i8`
- `fn next_f64(self : &mut Rng) -> f64`<br>
  Uniform in \[0, 1), built from the top 53 bits because that is f64's mantissa. Taking
  more would round, and rounding up at the top of the range returns exactly 1.0 -- which
  a caller scaling into a half-open range does not expect.
- `fn next_f32(self : &mut Rng) -> f32`<br>
  Uniform in \[0, 1), on the same terms with f32's 24 bits.
- `fn next_f16(self : &mut Rng) -> f16`<br>
  Uniform in \[0, 1) over an evenly spaced grid: 2048 points at f16 and 256 at bf16, each
  of them exact at that width. The draw is built from that many bits rather than narrowed
  from an f32, so every point is equally likely.
- `fn next_bf16(self : &mut Rng) -> bf16`
- `fn next_bool(self : &mut Rng) -> bool`<br>
  One bit, from the top of a draw.
- `fn chance(self : &mut Rng, p : f64) -> bool`<br>
  True with probability `p`. Outside [0, 1] it is always false or always true.
- `fn below(self : &mut Rng, bound : u64) -> u64`<br>
  Uniform in \[0, bound), with no bias.
  A plain `next_u64() % bound` is biased whenever `bound` does not divide 2^64: the first
  `2^64 % bound` values come up once more often than the rest. The draws that would land
  in that overhang are rejected and taken again. `floor` is where the overhang ends, and
  `0 - bound` is `2^64 - bound` -- the subtraction wraps, which is what makes 2^64
  expressible in a 64-bit word at all.
- `fn range_i64(self : &mut Rng, lo : i64, hi : i64) -> i64`<br>
  Uniform in \[lo, hi), `lo` included and `hi` not.
  The span is measured in `u64` so that a range spanning zero, or the whole of `i64`,
  is still one subtraction: `hi - lo` in `i64` would overflow for the widest of them.
- `fn range_f64(self : &mut Rng, lo : f64, hi : f64) -> f64`<br>
  Uniform in \[lo, hi). With lo > hi the range runs backwards and the result is in
  (hi, lo\], which is the same arithmetic and rarely what a caller meant.
- `fn range_f32(self : &mut Rng, lo : f32, hi : f32) -> f32`
- `fn normal(self : &mut Rng) -> f64`<br>
  One draw from the standard normal distribution: mean 0, standard deviation 1.
  Marsaglia's polar method. A point is drawn from the square until it lands inside the
  unit circle (about four tries in five), and that point yields *two* normals. The second
  is kept in the struct for the next call, so the loop runs once per two draws.
  This is the method to reach for when filling something that stands in for model data.
  Weights, activations and KV entries are roughly Gaussian, and a uniform fill exercises
  a range of magnitudes real data never has -- which matters most in f16, where the
  interesting failures are overflow and underflow at the tails.
- `fn normal_around(self : &mut Rng, mean : f64, stddev : f64) -> f64`<br>
  A normal draw moved and stretched: mean `mean`, standard deviation `stddev`.
- `fn fill_f32(self : &mut Rng, out : *mut f32, count : i32) -> i32`<br>
  Fill a buffer in place. One call per buffer rather than one per element, which is what
  makes filling a real tensor practical -- a 256x8192 one is two million values.
  `count` elements are written, so the buffer must hold that many.
- `fn fill_range_f32(self : &mut Rng, out : *mut f32, count : i32, lo : f32, hi : f32) -> i32`
- `fn fill_normal_f32(self : &mut Rng, out : *mut f32, count : i32, mean : f32, stddev : f32) -> i32`
- `fn fill_f16(self : &mut Rng, out : *mut f16, count : i32) -> i32`
- `fn fill_normal_f16(self : &mut Rng, out : *mut f16, count : i32, mean : f64, stddev : f64) -> i32`<br>
  The normal fill at half precision. The draw and the scaling happen in f64 and narrow
  once at the store, so a tail value is rounded rather than computed twice.

## `std::simd`

SIMD vector types and operations.

**Functions**

- `unsafe fn simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `unsafe fn simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `unsafe fn simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `unsafe fn simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `unsafe fn simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32`

**C bindings** *(the native functions this module is built on)*

- `fn vx_simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `fn vx_simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `fn vx_simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `fn vx_simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`
- `fn vx_simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32`

## `std::string`

`String` and text manipulation.

**Types**

- `struct String`

**`String` methods**

- `fn new() -> String`
- `unsafe fn from_c_str(c_str : *const i8) -> String`
- `unsafe fn push_c_str(self : *mut String, c_str : *const i8) -> i32`
- `fn len(self : *mut String) -> i32`
- `fn as_c_str(self : *mut String) -> *const i8`
- `fn drop(self : *mut String) -> i32`

**`i32` methods**

- `fn to_string(self : i32) -> String`

**Functions**

- `unsafe fn string_length(s : *const i8) -> i32`
- `unsafe fn string_compare(s1 : *const i8, s2 : *const i8) -> i32`
- `unsafe fn parse_int(s : *const i8) -> i32`

**C bindings** *(the native functions this module is built on)*

- `fn vx_string_new() -> *mut i8`
- `fn vx_string_from_c_str(ptr : *const i8) -> *mut i8`
- `fn vx_string_push_c_str(ptr : *mut i8, c_str : *const i8) -> i32`
- `fn vx_string_len(ptr : *mut i8) -> i32`
- `fn vx_string_as_c_str(ptr : *mut i8) -> *const i8`
- `fn vx_string_free_c_str(ptr : *const i8) -> i32`
- `fn vx_string_drop(ptr : *mut i8) -> i32`
- `fn vx_i32_to_string(val : i32) -> *mut i8`

## `std::tensor`

Operations on `Tensor`, including shape queries and elementwise maths.

**`Tensor<T, [?, ?]>` methods**

- `fn from_ptr_1d(ptr : *mut T, d1 : i32) -> Tensor<T, [?, ?]>`
- `fn from_ptr_2d(ptr : *mut T, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`
- `fn slice_2d(self : &Tensor<T, [?, ?]>, row : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`
- `fn slice_2d_from_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`
- `fn slice_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32) -> Tensor<T, [?, ?]>`
- `fn fill(self : &mut Tensor<T, [?, ?]>, val : T) -> void`
- `fn copy(self : &mut Tensor<T, [?, ?]>, src : &Tensor<T, [?, ?]>) -> void`
- `fn assign(self : &mut Tensor<T, [?, ?]>, val : T) -> void`
- `fn compare(self : &Tensor<T, [?, ?]>, other : &Tensor<T, [?, ?]>) -> bool`

**`Tensor<T, [N, M]>` methods**

- `fn fill_static(self : &mut Tensor<T, [ N, M ]>, val : T) -> void`

## `std::time`

Clocks and durations.

**Functions**

- `fn now() -> f32`
- `fn sleep(seconds : f32) -> i32`
- `fn unix_timestamp() -> f64`
- `unsafe fn bench_report(name : *const i8, unit : *const i8, value : f32) -> i32`

**C bindings** *(the native functions this module is built on)*

- `fn vx_get_time() -> f32`
- `fn vx_sleep(seconds : f32) -> i32`
- `fn vx_unix_timestamp() -> f64`
- `fn vx_bench_report(name : *const i8, unit : *const i8, value : f32) -> i32`

## `std::vec`

`Vec<T>`, a growable array.

**Types**

- `struct Vec<T>`
- `struct VecIter<T>`
- `struct VecMap<T, NewItem>`

**`Vec<T>` methods**

- `fn new() -> Vec<T>`
- `fn with_capacity(capacity : i32) -> Vec<T>`
- `fn free(self : &mut Vec<T>) -> i32`
- `fn as_mut_ptr(self : &Vec<T>) -> *mut T`
- `fn as_mut_slice(self : &mut Vec<T>) -> &mut T`
- `fn as_slice(self : &Vec<T>) -> &T`
- `fn push(self : &mut Vec<T>, val : T) -> i32`
- `fn get(self : &Vec<T>, index : i32) -> T`
- `fn set(self : &mut Vec<T>, index : i32, val : T) -> i32`
- `fn len(self : &Vec<T>) -> i32`
- `fn iter(self : &Vec<T>) -> VecIter<T>`

**`Iterator<VecIter<T>, T> for VecIter<T>` methods**

- `fn next(self : &mut VecIter<T>) -> Option<T>`

**`VecIter<T>` methods**

- `fn map<NewItem>(self : VecIter<T>, f : Closure1<T, NewItem>) -> VecMap<T, NewItem>`

**`Iterator<VecMap<T, NewItem>, NewItem> for VecMap<T, NewItem>` methods**

- `fn next(self : &mut VecMap<T, NewItem>) -> Option<NewItem>`

**`VecMap<T, NewItem>` methods**

- `fn collect(self : &mut VecMap<T, NewItem>) -> Vec<NewItem>`

**C bindings** *(the native functions this module is built on)*

- `fn vx_vec_alloc(elem_size : i64, cap : i64) -> *mut i8`
- `fn vx_vec_grow(ptr : *mut i8, old_cap : i64, new_cap : i64, elem_size : i64) -> *mut i8`
- `fn vx_vec_free(ptr : *mut i8, cap : i64, elem_size : i64) -> i32`
- `fn vx_vec_bounds_check(index : i64, len : i64) -> i32`

______________________________________________________________________

607 functions across 30 modules.
