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
- [`core::iter::adapters`](#coreiteradapters) — The iterators `Iterator`'s adaptor methods build: `Map`, `Filter`, `Chain` and the rest.
- [`core::iter::traits`](#coreitertraits) — The `Iterator` trait: one required `next`, and the methods written over it.
- [`core::iter`](#coreiter) — `Range` and `range`; importing it brings the trait and the adaptors too.
- [`core::marker`](#coremarker) — The traits that say something about a type without giving it a method.
- [`core::mem`](#coremem) — Moving values around without looking at what they are.
- [`core::num`](#corenum) — The integer and float methods, stamped over every width.
- [`core::ops`](#coreops) — The callable types a closure literal lowers into.
- [`core::option`](#coreoption) — `Option<T>`, for a value that may be absent.
- [`core::ptr`](#coreptr) — Raw pointers: making one, and reading or writing through it.
- [`core::result`](#coreresult) — `Result<T, E>`, for an operation that may fail.
- [`core::tuple`](#coretuple) — The structs tuple syntax stands for, `Tuple2` to `Tuple6`; imported by any module that writes a tuple.
- [`std::alloc`](#stdalloc) — Raw allocation and deallocation.
- [`std::box`](#stdbox) — `Box<T>`, a single-owner heap allocation. Required for recursive types.
- [`std::fs`](#stdfs) — Files and directories.
- [`std::googletest`](#stdgoogletest) — Assertions for tests written in Vx.
- [`std::hash_map`](#stdhash_map) — `HashMap<K, V>`.
- [`std::hash_set`](#stdhash_set) — `HashSet<T>`.
- [`std::io`](#stdio) — Standard input, output and error.
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

## `core::iter::adapters`

The iterators `Iterator`'s adaptor methods build: `Map`, `Filter`, `Chain` and the rest.

**Types**

- `struct Map<I, F>`<br>
  An iterator over another one's items with `f` applied to each. Built by `map`.
- `struct Filter<I, P>`
- `struct Take<I>`<br>
  An iterator over at most `left` items of another. Built by `take`.
- `struct Skip<I>`<br>
  An iterator over another's items with the first `drop` of them discarded. Built by `skip`.
- `struct StepBy<I>`<br>
  An iterator over every `step`th item of another, starting with its first. Built by `step_by`.
- `struct Chain<A, B>`<br>
  An iterator over the items of one iterator, then those of another. Built by `chain`.
- `struct TakeWhile<I, P>`<br>
  An iterator over another's items for as long as `keep` accepts them. Built by `take_while`.
- `struct SkipWhile<I, P>`<br>
  An iterator over another's items from the first one `skip` refuses. Built by `skip_while`.
- `struct MapWhile<I, F>`<br>
  An iterator over what `f` answers for another's items, until it answers nothing. Built by
  `map_while`.
- `struct Inspect<I, F>`<br>
  An iterator that hands each of another's items to `f` on its way past. Built by `inspect`.
- `struct Scan<I, St, F>`<br>
  An iterator over what `f` answers for another's items while it carries a state between
  them, ending where `f` first answers nothing. Built by `scan`.
- `struct Fuse<I>`<br>
  An iterator that answers nothing forever once another has answered nothing once. Built by
  `fuse`.
- `struct Peekable<I, T>`<br>
  An iterator whose next item can be looked at without taking it. Built by `peekable`.
  `T` is always `I::Item`: a struct field cannot name `I::Item`, so the item it holds on to
  is typed by a parameter of its own, which `peekable` fills in.
- `struct FlatMap<I, F, U>`<br>
  An iterator over the items of the iterators `f` answers for another's items, one after
  another. Built by `flat_map`.
  `U` is what `f` answers, held while its items are handed out, and a parameter of its own
  for the reason `Peekable`'s `T` is.
- `struct Flatten<I, U>`<br>
  An iterator over the items of each of another's items, which are iterators themselves.
  Built by `flatten`.
- `struct Rev<I>`<br>
  An iterator over another's items from the back. Built by `rev`.
- `struct Zip<A, B>`<br>
  An iterator over pairs of two others' items, taken in step, ending when either does.
  Built by `zip`.
- `struct Enumerate<I>`<br>
  An iterator over another's items, each paired with how far along it is, counting from zero.
  Built by `enumerate`.

**`Iterator for Map<I, Closure1<I :  : Item, U>>` methods**

- `fn next(self : &mut Map<I, Closure1<I :  : Item, U>>) -> Option<U>`<br>
  The inner iterator's next item with `f` applied.

**`DoubleEndedIterator for Map<I, Closure1<I :  : Item, U>>` methods**

- `fn next_back(self : &mut Map<I, Closure1<I :  : Item, U>>) -> Option<U>`<br>
  The inner iterator's last item with `f` applied.

**`ExactSizeIterator for Map<I, Closure1<I :  : Item, U>>` methods**

- `fn len(self : &Map<I, Closure1<I :  : Item, U>>) -> i64`<br>
  As many as the inner iterator has.

**`Iterator for Filter<I, Closure1<&I :  : Item, bool>>` methods**

- `fn next(self : &mut Filter<I, Closure1<&I :  : Item, bool>>) -> Option<I :  : Item>`<br>
  The inner iterator's next item that `keep` accepts.

**`DoubleEndedIterator for Filter<I, Closure1<&I :  : Item, bool>>` methods**

- `fn next_back(self : &mut Filter<I, Closure1<&I :  : Item, bool>>) -> Option<I :  : Item>`<br>
  The inner iterator's last item that `keep` accepts.

**`Iterator for Take<I>` methods**

- `fn next(self : &mut Take<I>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, until `left` of them have been handed out.

**`ExactSizeIterator for Take<I>` methods**

- `fn len(self : &Take<I>) -> i64`<br>
  The inner iterator's count, up to `left`.

**`DoubleEndedIterator for Take<I>` methods**

- `fn next_back(self : &mut Take<I>) -> Option<I :  : Item>`<br>
  The last of the items `next` would hand out, passing over the inner iterator's items
  beyond them.

**`Iterator for Skip<I>` methods**

- `fn next(self : &mut Skip<I>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, once the skipped ones have been consumed.

**`ExactSizeIterator for Skip<I>` methods**

- `fn len(self : &Skip<I>) -> i64`<br>
  The inner iterator's count less the ones still to be skipped.

**`DoubleEndedIterator for Skip<I>` methods**

- `fn next_back(self : &mut Skip<I>) -> Option<I :  : Item>`<br>
  The inner iterator's last item, while it is not one of the skipped ones.

**`Iterator for StepBy<I>` methods**

- `fn next(self : &mut StepBy<I>) -> Option<I :  : Item>`<br>
  The inner iterator's first item, and after that the one `step` places further on.

**`ExactSizeIterator for StepBy<I>` methods**

- `fn len(self : &StepBy<I>) -> i64`<br>
  How many of the inner iterator's items land on a step.

**`DoubleEndedIterator for StepBy<I>` methods**

- `fn next_back(self : &mut StepBy<I>) -> Option<I :  : Item>`<br>
  The last item that lands on a step, passing over the inner iterator's items after it.

**`Iterator for Chain<A, B>` methods**

- `fn next(self : &mut Chain<A, B>) -> Option<A :  : Item>`<br>
  The first iterator's next item, or the second's once the first is finished.

**`DoubleEndedIterator for Chain<A, B>` methods**

- `fn next_back(self : &mut Chain<A, B>) -> Option<A :  : Item>`<br>
  The second iterator's last item, or the first's once the second is finished.

**`Iterator for TakeWhile<I, Closure1<&I :  : Item, bool>>` methods**

- `fn next(self : &mut TakeWhile<I, Closure1<&I :  : Item, bool>>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, until the first `keep` refuses; nothing from then on.

**`Iterator for SkipWhile<I, Closure1<&I :  : Item, bool>>` methods**

- `fn next(self : &mut SkipWhile<I, Closure1<&I :  : Item, bool>>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, once the leading ones `skip` accepts are consumed.

**`Iterator for MapWhile<I, Closure1<I :  : Item, Option<U>>>` methods**

- `fn next(self : &mut MapWhile<I, Closure1<I :  : Item, Option<U>>>) -> Option<U>`<br>
  `f` of the inner iterator's next item, which is nothing once `f` says so.

**`Iterator for Inspect<I, Closure1<&I :  : Item, i32>>` methods**

- `fn next(self : &mut Inspect<I, Closure1<&I :  : Item, i32>>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, after `f` has seen it.

**`DoubleEndedIterator for Inspect<I, Closure1<&I :  : Item, i32>>` methods**

- `fn next_back(self : &mut Inspect<I, Closure1<&I :  : Item, i32>>) -> Option<I :  : Item>`<br>
  The inner iterator's last item, after `f` has seen it.

**`ExactSizeIterator for Inspect<I, Closure1<&I :  : Item, i32>>` methods**

- `fn len(self : &Inspect<I, Closure1<&I :  : Item, i32>>) -> i64`<br>
  As many as the inner iterator has.

**`Iterator for Scan<I, St, Closure2<&mut St, I :  : Item, Option<B>>>` methods**

- `fn next(self : &mut Scan<I, St, Closure2<&mut St, I :  : Item, Option<B>>>) -> Option<B>`<br>
  `f` of the state and the inner iterator's next item.

**`Iterator for Fuse<I>` methods**

- `fn next(self : &mut Fuse<I>) -> Option<I :  : Item>`<br>
  The inner iterator's next item, or nothing for good once it has run out.

**`ExactSizeIterator for Fuse<I>` methods**

- `fn len(self : &Fuse<I>) -> i64`<br>
  As many as the inner iterator has, and none once it has run out.

**`Iterator for Peekable<I, I :  : Item>` methods**

- `fn next(self : &mut Peekable<I, I :  : Item>) -> Option<I :  : Item>`<br>
  The item `peek` looked at, if it looked, and otherwise the inner iterator's next.

**`Peekable<I, I :  : Item>` methods**

- `fn peek(self : &mut Peekable<I, I :  : Item>) -> Option<I :  : Item>`<br>
  The item `next` would answer, left where it is.
  Answers a copy where Rust answers a reference into the iterator.
- `fn next_if(self : &mut Peekable<I, I :  : Item>, accept : Closure1<&I :  : Item, bool>) -> Option<I :  : Item>`<br>
  The next item if `accept` takes it, and otherwise nothing, with the item left in place.

**`ExactSizeIterator for Peekable<I, I :  : Item>` methods**

- `fn len(self : &Peekable<I, I :  : Item>) -> i64`<br>
  As many as the inner iterator has, and the one `peek` holds.

**`Iterator for FlatMap<I, Closure1<I :  : Item, U>, U>` methods**

- `fn next(self : &mut FlatMap<I, Closure1<I :  : Item, U>, U>) -> Option<U :  : Item>`<br>
  The current inner iterator's next item, moving to the next one when it runs out.

**`Iterator for Flatten<I, U>` methods**

- `fn next(self : &mut Flatten<I, U>) -> Option<U :  : Item>`<br>
  The current inner iterator's next item, moving to the next one when it runs out.

**`Iterator for Rev<I>` methods**

- `fn next(self : &mut Rev<I>) -> Option<I :  : Item>`<br>
  The inner iterator's last item.

**`DoubleEndedIterator for Rev<I>` methods**

- `fn next_back(self : &mut Rev<I>) -> Option<I :  : Item>`<br>
  The inner iterator's first item.

**`ExactSizeIterator for Rev<I>` methods**

- `fn len(self : &Rev<I>) -> i64`<br>
  As many as the inner iterator has.

**`Iterator for Zip<A, B>` methods**

- `fn next(self : &mut Zip<A, B>) -> Option<(A :  : Item, B :  : Item)>`<br>
  The next item of each, paired, or nothing once either has run out.

**`ExactSizeIterator for Zip<A, B>` methods**

- `fn len(self : &Zip<A, B>) -> i64`<br>
  As many as the shorter of the two has.

**`DoubleEndedIterator for Zip<A, B>` methods**

- `fn next_back(self : &mut Zip<A, B>) -> Option<(A :  : Item, B :  : Item)>`<br>
  The last pair, once the longer iterator's extra items at the back have been dropped.

**`Iterator for Enumerate<I>` methods**

- `fn next(self : &mut Enumerate<I>) -> Option<(i64, I :  : Item)>`<br>
  The inner iterator's next item with its position.

**`ExactSizeIterator for Enumerate<I>` methods**

- `fn len(self : &Enumerate<I>) -> i64`<br>
  As many as the inner iterator has.

**`DoubleEndedIterator for Enumerate<I>` methods**

- `fn next_back(self : &mut Enumerate<I>) -> Option<(i64, I :  : Item)>`<br>
  The inner iterator's last item with its position, which is how many come before it.

## `core::iter::traits`

The `Iterator` trait: one required `next`, and the methods written over it.

**Types**

- `trait Iterator`
- `trait DoubleEndedIterator`<br>
  An iterator that can also answer from its far end, which is what `rev` needs.
  Rust declares it as a subtrait of `Iterator`; Vx has no subtraits, and `Self::Item` here is
  the one the type's `Iterator` impl binds.
- `trait ExactSizeIterator`<br>
  An iterator that knows how many items it has left, which is what `rev` needs from `take`,
  `skip` and `step_by`.
  Rust declares it as a subtrait of `Iterator`; Vx has no subtraits.
- `trait FromIterator<A>`<br>
  A collection that can be built from an iterator's items, which is what `collect` builds.
- `trait Extend<A>`<br>
  A collection that grows by an iterator's items, which is what `partition` and `unzip` fill.
- `trait Sum<A>`<br>
  A type whose values can be added up from an iterator's items, which is what `sum` does.
- `trait Product<A>`<br>
  A type whose values can be multiplied together from an iterator's items, which is what
  `product` does.

**`trait Iterator` methods**

- `fn next(self : &mut Self) -> Option<Self :  : Item>`<br>
  The next item, or nothing once the sequence is finished.
  The only required method. Every other method of this trait is a default written over it,
  so a type becomes iterable by writing this one.
  Calling it again after it has answered nothing is allowed and answers nothing again; an
  iterator that would resume is not something this trait promises either way.
- `fn count(self : &mut Self) -> i64`<br>
  How many items are left, consuming them all to find out.
- `fn last(self : &mut Self) -> Option<Self :  : Item>`<br>
  The final item, consuming the sequence. Nothing when it is already finished.
- `fn nth(self : &mut Self, n : i64) -> Option<Self :  : Item>`<br>
  The item `n` places along, counting the next one as zero, discarding those before it.
  Nothing when the sequence finishes first. The items skipped are consumed either way.
- `fn any(self : &mut Self, f : Closure1<Self :  : Item, bool>) -> bool`<br>
  Does `f` accept any item? Stops at the first it does, leaving the rest unconsumed.
- `fn all(self : &mut Self, f : Closure1<Self :  : Item, bool>) -> bool`<br>
  Does `f` accept every item? Stops at the first it does not.
  True for a sequence that is already finished, which is the usual convention: there is no
  item to disagree.
- `fn find(self : &mut Self, f : Closure1<Self :  : Item, bool>) -> Option<Self :  : Item>`<br>
  The first item `f` accepts, or nothing. Stops there, so the rest is unconsumed.
- `fn position(self : &mut Self, f : Closure1<Self :  : Item, bool>) -> Option<i64>`<br>
  How far along the first item `f` accepts is, counting the next one as zero.
- `fn for_each(self : &mut Self, f : Closure1<Self :  : Item, i32>) -> i32`<br>
  Hand every item to `f`, consuming the sequence.
  `f` answers an `i32` rather than nothing, and this returns the last of them, because no
  closure literal can return void yet (Vx#711). Both signatures become Rust's when it can.
- `fn fold<B>(self : &mut Self, init : B, f : Closure2<B, Self :  : Item, B>) -> B`<br>
  `f` over every item, carrying a value from one to the next: `init` goes in with the
  first item, and what `f` answers goes in with the next. The last answer is the result,
  or `init` for a sequence that is already finished.
- `fn try_fold<Acc, R>(self : &mut Self, init : Acc, f : Closure2<Acc, Self :  : Item, R>) -> R`<br>
  `fold` that can stop early. `f` answers an `Option` or a `Result`: `Some` or `Ok` carries
  its value on to the next item, and the first `None` or `Err` is returned at once, leaving
  the rest of the sequence unconsumed. When every item carries on, the answer is the last
  value wrapped the same way, or `init` wrapped for a sequence that is already finished.
- `fn try_for_each<R>(self : &mut Self, f : Closure1<Self :  : Item, R>) -> R`<br>
  `for_each` that can stop early: the first `None` or `Err` that `f` answers is returned at
  once, and the rest of the sequence is left unconsumed. Otherwise the answer is `Some(0)`
  or `Ok(0)`.
  `f` answers an `Option<i32>` or `Result<i32, E>` whose value is discarded, where Rust's
  answers `()`, because no closure literal can return void yet. It becomes Rust's when it
  can.
- `fn sum(self : &mut Self) -> Self :  : Item`<br>
  Every item added up, consuming the sequence. Zero for one that is already finished.
  The item type says how, by implementing `Sum`. Rust also lets the caller choose a result
  type other than the item's; here the two are the same.
- `fn product(self : &mut Self) -> Self :  : Item`<br>
  Every item multiplied together, consuming the sequence. One for one already finished.
  The item type says how, by implementing `Product`.
- `fn max(self : &mut Self) -> Option<Self :  : Item>`<br>
  The greatest item, or nothing for a sequence already finished. Of several equal greatest
  items, the last.
- `fn min(self : &mut Self) -> Option<Self :  : Item>`<br>
  The least item, or nothing for a sequence already finished. Of several equal least items,
  the first.
- `fn max_by(self : &mut Self, compare : Closure2<&Self :  : Item, &Self :  : Item, Ordering>) -> Option<Self :  : Item>`<br>
  The greatest item as `compare` orders them, `compare(a, b)` saying where `a` sits against
  `b`. Of several equal greatest items, the last.
- `fn min_by(self : &mut Self, compare : Closure2<&Self :  : Item, &Self :  : Item, Ordering>) -> Option<Self :  : Item>`<br>
  The least item as `compare` orders them. Of several equal least items, the first.
- `fn max_by_key<B>(self : &mut Self, key : Closure1<&Self :  : Item, B>) -> Option<Self :  : Item>`<br>
  The item whose `key` is greatest. Of several with equal greatest keys, the last.
- `fn min_by_key<B>(self : &mut Self, key : Closure1<&Self :  : Item, B>) -> Option<Self :  : Item>`<br>
  The item whose `key` is least. Of several with equal least keys, the first.
- `fn map<U>(self : Self, f : Closure1<Self :  : Item, U>) -> Map<Self, Closure1<Self :  : Item, U>>`<br>
  An iterator over these items with `f` applied to each.
- `fn filter(self : Self, keep : Closure1<&Self :  : Item, bool>) -> Filter<Self, Closure1<&Self :  : Item, bool>>`<br>
  An iterator over the items `keep` accepts.
- `fn take(self : Self, n : i64) -> Take<Self>`<br>
  An iterator over at most the first `n` items.
- `fn skip(self : Self, n : i64) -> Skip<Self>`<br>
  An iterator over the items after the first `n`.
- `fn step_by(self : Self, step : i64) -> StepBy<Self>`<br>
  An iterator over the first item and every `step`th one after it.
  # Panics
  When `step` is not positive: a step of zero would hand out the first item forever.
- `fn chain<B>(self : Self, other : B) -> Chain<Self, B>`<br>
  An iterator over these items and then `other`'s, which must be of the same type.
- `fn take_while(self : Self, keep : Closure1<&Self :  : Item, bool>) -> TakeWhile<Self, Closure1<&Self :  : Item, bool>>`<br>
  An iterator over the leading items `keep` accepts, stopping at the first it refuses.
- `fn skip_while(self : Self, skip : Closure1<&Self :  : Item, bool>) -> SkipWhile<Self, Closure1<&Self :  : Item, bool>>`<br>
  An iterator over the items from the first one `skip` refuses onward.
- `fn map_while<U>(self : Self, f : Closure1<Self :  : Item, Option<U>>) -> MapWhile<Self, Closure1<Self :  : Item, Option<U>>>`<br>
  An iterator over what `f` answers for each item, ending where `f` first answers nothing.
- `fn inspect(self : Self, f : Closure1<&Self :  : Item, i32>) -> Inspect<Self, Closure1<&Self :  : Item, i32>>`<br>
  An iterator over these items that hands each to `f` on its way past.
  `f` answers an `i32`, which is discarded, because no closure literal can return void yet.
- `fn scan<St, B>(self : Self, initial : St, f : Closure2<&mut St, Self :  : Item, Option<B>>) -> Scan<Self, St, Closure2<&mut St, Self :  : Item, Option<B>>>`<br>
  An iterator over what `f` answers for each item, with `f` given `initial` to keep and
  change from one item to the next. Ends where `f` first answers nothing.
- `fn fuse(self : Self) -> Fuse<Self>`<br>
  An iterator that answers nothing forever once these items have run out.
- `fn peekable(self : Self) -> Peekable<Self, Self :  : Item>`<br>
  An iterator whose next item can be looked at, by `peek`, without taking it.
- `fn flat_map<U>(self : Self, f : Closure1<Self :  : Item, U>) -> FlatMap<Self, Closure1<Self :  : Item, U>, U>`<br>
  An iterator over the items of each iterator `f` answers, in order.
  `f` answers an iterator, where Rust accepts anything that can become one.
- `fn flatten(self : Self) -> Flatten<Self, Self :  : Item>`<br>
  An iterator over the items of each of these items, which are iterators themselves.
- `fn rev(self : Self) -> Rev<Self>`<br>
  An iterator over these items from the back, for an iterator that can answer from both ends.
- `fn zip<U>(self : Self, other : U) -> Zip<Self, U>`<br>
  An iterator over pairs of these items and `other`'s, in step, as long as both last.
- `fn enumerate(self : Self) -> Enumerate<Self>`<br>
  An iterator over these items, each paired with its position, counting from zero.
  The position is an `i64`, where Rust's is a `usize`.
- `fn reduce(self : &mut Self, f : Closure2<Self :  : Item, Self :  : Item, Self :  : Item>) -> Option<Self :  : Item>`<br>
  `fold` with the first item as the starting value, so nothing for a sequence already
  finished.
- `fn collect<B>(self : &mut Self) -> B`<br>
  Every item, gathered into a new collection of whatever type the result is assigned to:
  `let v : Vec<i64> = it.collect();`. That type says how, by implementing `FromIterator`.
- `fn partition<B>(self : &mut Self, f : Closure1<&Self :  : Item, bool>) -> (B, B)`<br>
  Every item, split in two by `f`: the ones it accepts first, the rest second. Each half
  is a collection of whatever type the result is assigned to, which starts from its
  `Default` and grows through `Extend`.
- `fn unzip<FromA, FromB>(self : &mut Self) -> (FromA, FromB)`<br>
  An iterator of pairs, split into two collections: the first of each pair in one, the
  second in the other.

**`binds. trait DoubleEndedIterator` methods**

- `fn next_back(self : &mut Self) -> Option<Self :  : Item>`<br>
  The last item not yet handed out from either end, or nothing once they meet.
- `fn rfold<B>(self : &mut Self, init : B, f : Closure2<B, Self :  : Item, B>) -> B`<br>
  `fold`, from the back.
- `fn rfind(self : &mut Self, f : Closure1<&Self :  : Item, bool>) -> Option<Self :  : Item>`<br>
  The last item `f` accepts, searching from the back.
- `fn nth_back(self : &mut Self, n : i64) -> Option<Self :  : Item>`<br>
  The item `n` places from the back, counting the last as zero.

**`trait ExactSizeIterator` methods**

- `fn len(self : &Self) -> i64`<br>
  How many items are left.
- `fn is_empty(self : &Self) -> bool`<br>
  Whether no items are left.

**`trait FromIterator<A>` methods**

- `fn from_iter<I : Iterator>(iter : I) -> Self`<br>
  A new collection holding every item `iter` has left, in order.

**`trait Extend<A>` methods**

- `fn extend<I : Iterator>(self : &mut Self, iter : I) -> i32`<br>
  Add every item `iter` has left, in order.
- `fn extend_one(self : &mut Self, item : A) -> i32`<br>
  Add one item.

**`trait Sum<A>` methods**

- `fn sum<I : Iterator>(iter : I) -> Self`<br>
  Every item `iter` has left, added up. Zero when it has none.

**`trait Product<A>` methods**

- `fn product<I : Iterator>(iter : I) -> Self`<br>
  Every item `iter` has left, multiplied together. One when it has none.

**`Sum<T> for T` methods**, stamped for 10 instantiations

- `fn sum<I : Iterator>(iter : I) -> T`<br>
  Every item added up, from zero.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`

**`Product<T> for T` methods**, stamped for 10 instantiations

- `fn product<I : Iterator>(iter : I) -> T`<br>
  Every item multiplied together, from one.

T = `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`

## `core::iter`

`Range` and `range`; importing it brings the trait and the adaptors too.

**Types**

- `struct Range`<br>
  The numbers from `at` up to but not including `end`.
  A half-open span of `i64`, from `at` up to but not including `end`.

**`Iterator for Range` methods**

- `fn next(self : &mut Range) -> Option<i64>`<br>
  The next value in the span, or nothing once `end` is reached.

**`DoubleEndedIterator for Range` methods**

- `fn next_back(self : &mut Range) -> Option<i64>`<br>
  The last value in the span not yet handed out from either end.

**`ExactSizeIterator for Range` methods**

- `fn len(self : &Range) -> i64`<br>
  How many values are left in the span.

**Functions**

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
- `trait Try`<br>
  A value that either carries on, holding an output, or stops early. It is what
  `Iterator::try_fold` reads from its closure's answers, and `Option` and `Result` implement
  it: `Some` and `Ok` carry on, `None` and `Err` stop. Rust's `Try` cut down to what those
  two need, with no residual type and no `?`.

**`trait Try` methods**

- `fn is_continue(self : &Self) -> bool`<br>
  Does this carry on? False means stop here and hand this value back.
- `fn into_output(self : Self) -> Self :  : Output`<br>
  The output of one that carries on.
- `fn from_output(output : Self :  : Output) -> Self`<br>
  One that carries on, holding `output`.

## `core::option`

`Option<T>`, for a value that may be absent.

**Types**

- `enum Option<T>`

**`Try for Option<T>` methods**

- `fn is_continue(self : &Option<T>) -> bool`<br>
  Is there a value?
- `fn into_output(self : Option<T>) -> T`<br>
  The value.
- `fn from_output(output : T) -> Option<T>`<br>
  `Some(output)`.

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
- `fn zip<U>(self : Option<T>, other : Option<U>) -> Option<(T, U)>`<br>
  Both values as a pair when both are there, otherwise nothing.
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

**`Try for Result<T, E>` methods**

- `fn is_continue(self : &Result<T, E>) -> bool`<br>
  Is this `Ok`?
- `fn into_output(self : Result<T, E>) -> T`<br>
  The `Ok` value.
- `fn from_output(output : T) -> Result<T, E>`<br>
  `Ok(output)`.

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

## `core::tuple`

The structs tuple syntax stands for, `Tuple2` to `Tuple6`; imported by any module that writes a tuple.

**Types**

- `struct Tuple2<A, B>`<br>
  A pair.
- `struct Tuple3<A, B, C>`<br>
  Three values.
- `struct Tuple4<A, B, C, D>`<br>
  Four values.
- `struct Tuple5<A, B, C, D, E>`<br>
  Five values.
- `struct Tuple6<A, B, C, D, E, F>`<br>
  Six values.

## `std::alloc`

Raw allocation and deallocation.

**Functions** *(bound directly to C)*

- `fn malloc(size : i64) -> *mut i8`<br>
  C's `malloc`: `size` bytes of uninitialised heap, or null when it cannot.
  Vx cannot test the result against null (Vx#714), so a failed allocation is found by
  writing through it.
- `fn realloc(ptr : *mut i8, size : i64) -> *mut i8`<br>
  C's `realloc`: the block resized, moving it if need be. The old pointer is invalid
  afterwards whether or not it moved.
- `fn free(ptr : *mut i8) -> i32`<br>
  C's `free`. Freeing twice, or freeing what `malloc` did not return, is undefined.

## `std::box`

`Box<T>`, a single-owner heap allocation. Required for recursive types.

**Types**

- `struct Box<T>`<br>
  A single-owner heap allocation, which is what makes a recursive type possible.
  Released by calling `free`, since the language has no `Drop` (Vx#495).

**`Box<T>` methods**

- `fn new(val : T) -> Box<T>`<br>
  Move a value to the heap.
  The allocation is not checked: `malloc` answering null gives a `Box` that writes through
  a null pointer, and Vx cannot test one (Vx#714).
- `fn free(self : &mut Box<T>) -> i32`<br>
  Release the allocation. The pointer is left as it was, so using the box afterwards
  reads freed memory.

## `std::fs`

Files and directories.

**Types**

- `struct File`<br>
  An open file, held by the Rust core behind an opaque pointer.
  Closed by calling `file_drop`, since the language has no `Drop` (Vx#495).

**`File` methods**

- `unsafe fn open(path : *const i8, mode : i32) -> File`<br>
  Open `path`. `mode` is 0 to read, 1 to write, and anything else to read and write.
  Writing creates the file and truncates it; read-and-write creates it and does not
  truncate. Reading does not create it.
  **A failed open cannot be detected.** The returned `File` holds a null pointer, and Vx
  cannot compare a raw pointer against null or cast one to an integer (Vx#714) -- so a
  missing file is indistinguishable from an empty one until that closes. Every later call
  on it answers 0 or -1 rather than doing anything.
  Unsafe because nothing checks that `path` points at a NUL-terminated string.
- `unsafe fn read(self : *mut File, buffer : *mut u8, len : i64) -> i64`<br>
  Read up to `len` bytes into `buffer` and answer how many arrived.
  Zero means end of file, a null file or buffer, or a read error -- the four are not told
  apart. A short read is normal and is not an error.
  Unsafe because `buffer` must have room for `len` bytes; nothing here checks.
- `unsafe fn write(self : *mut File, buffer : *const u8, len : i64) -> i64`<br>
  Write up to `len` bytes from `buffer` and answer how many were taken.
  Zero means a null file or buffer, or a write error, with the same lack of distinction
  `read` has. A short write is not an error and the remainder is not retried here.
- `fn seek(self : *mut File, offset : i64, whence : i32) -> i64`<br>
  Move the read and write position, answering where it ended up.
  `whence` is 0 from the start, 1 from the current position, 2 from the end, as libc's
  `SEEK_SET`, `SEEK_CUR` and `SEEK_END`. Answers -1 for a null file, an unknown `whence`,
  or a seek the operating system refuses.

**Functions**

- `unsafe fn file_drop(file : *mut File) -> void`<br>
  Close the file and release it. Using it afterwards reads freed memory.

**C bindings** *(the native functions this module is built on)*

- `fn vx_file_open(c_path : *const i8, mode : i32) -> *mut i8`<br>
  The Rust core behind `File::open`. Null when the open fails.
- `fn vx_file_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`<br>
  The Rust core behind `File::read`.
- `fn vx_file_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64`<br>
  The Rust core behind `File::write`.
- `fn vx_file_seek(ptr : *mut i8, offset : i64, whence : i32) -> i64`<br>
  The Rust core behind `File::seek`.
- `fn vx_file_drop(ptr : *mut i8) -> i32`<br>
  The Rust core behind `file_drop`.
- `fn fopen(path : *const i8, mode : *const i8) -> *mut i8`<br>
  C's `fopen`, for code that wants a `FILE*` rather than the Rust-backed `File`.
- `fn fread(ptr : *mut u8, size : i64, nmemb : i64, stream : *mut i8) -> i64`<br>
  C's `fread`: the number of whole items read, not the number of bytes.
- `fn fclose(stream : *mut i8) -> i32`<br>
  C's `fclose`.
- `fn fileno(f : *mut i8) -> i32`<br>
  C's `fileno`: the descriptor behind a `FILE*`, for handing to `mmap`.
- `fn fseek(f : *mut i8, offset : i64, whence : i32) -> i32`<br>
  C's `fseek`.
- `fn ftell(f : *mut i8) -> i64`<br>
  C's `ftell`: the current position, which is how the size is found after a seek to the end.
- `fn mmap(addr : *mut i8, len : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8`<br>
  C's `mmap`, declared here so a file can be mapped without importing `std::mmap`.
- `fn munmap(addr : *mut i8, len : i64) -> i32`<br>
  C's `munmap`.

## `std::googletest`

Assertions for tests written in Vx.

**Types**

- `trait GoogletestEq`

**`trait GoogletestEq` methods**

- `fn expect_eq(self : Self, expected : Self) -> i32`<br>
  Report whether this value equals `expected`, and answer non-zero when it does not.
  Records the comparison rather than stopping at it, so a test reports every failure it
  finds rather than only the first.

**`GoogletestEq for f32` methods**

- `fn expect_eq(self : f32, expected : f32) -> i32`<br>
  Compared exactly, so two values a rounding step apart are reported as different.

**`GoogletestEq for i32` methods**

- `fn expect_eq(self : i32, expected : i32) -> i32`<br>
  Compared exactly.

**Functions**

- `fn expect_eq<T : GoogletestEq>(actual : T, expected : T) -> i32`<br>
  `actual.expect_eq(expected)` written the way a test reads: the value under test first.

**C bindings** *(the native functions this module is built on)*

- `fn vx_googletest_expect_eq_f32(actual : f32, expected : f32) -> i32`<br>
  The Rust core behind `f32`'s `expect_eq`.
- `fn vx_googletest_expect_eq_i32(actual : i32, expected : i32) -> i32`<br>
  The Rust core behind `i32`'s `expect_eq`.

## `std::hash_map`

`HashMap<K, V>`.

**Functions** *(bound directly to C)*

- `fn vx_hash_map_new_i32_i32() -> *mut i8`<br>
  An empty map from `i32` to `i32`, owned by the Rust core.
- `fn vx_hash_map_insert_i32_i32(ptr : *mut i8, key : i32, val : i32) -> i32`<br>
  Insert a value under a key, replacing whatever was there.
- `fn vx_hash_map_get_i32_i32(ptr : *mut i8, key : i32) -> *mut i8`<br>
  A pointer to the value under a key, or null when the key is absent.
  Vx cannot test a pointer against null (Vx#714), so `contains_key` is the way to ask
  whether the key is there.
- `fn vx_hash_map_contains_key_i32_i32(ptr : *mut i8, key : i32) -> Bool`<br>
  Is the key present?
- `fn vx_hash_map_len_i32_i32(ptr : *mut i8) -> i32`<br>
  How many entries the map holds.
- `fn vx_hash_map_drop_i32_i32(ptr : *mut i8) -> i32`<br>
  Release the map.
- `fn vx_hash_map_new_i32_f32() -> *mut i8`<br>
  An empty map from `i32` to `f32`, owned by the Rust core.
- `fn vx_hash_map_insert_i32_f32(ptr : *mut i8, key : i32, val : f32) -> i32`<br>
  Insert a value under a key, replacing whatever was there.
- `fn vx_hash_map_get_i32_f32(ptr : *mut i8, key : i32) -> *mut i8`<br>
  A pointer to the value under a key, or null when the key is absent.
  Vx cannot test a pointer against null (Vx#714), so `contains_key` is the way to ask
  whether the key is there.
- `fn vx_hash_map_contains_key_i32_f32(ptr : *mut i8, key : i32) -> Bool`<br>
  Is the key present?
- `fn vx_hash_map_len_i32_f32(ptr : *mut i8) -> i32`<br>
  How many entries the map holds.
- `fn vx_hash_map_drop_i32_f32(ptr : *mut i8) -> i32`<br>
  Release the map.

## `std::hash_set`

`HashSet<T>`.

**Functions** *(bound directly to C)*

- `fn vx_hash_set_new_i32() -> *mut i8`<br>
  An empty set of `i32`, owned by the Rust core.
- `fn vx_hash_set_insert_i32(ptr : *mut i8, val : i32) -> i32`<br>
  Add a value. Adding one already present changes nothing.
- `fn vx_hash_set_contains_i32(ptr : *mut i8, val : i32) -> Bool`<br>
  Is the value in the set?
- `fn vx_hash_set_len_i32(ptr : *mut i8) -> i32`<br>
  How many distinct values the set holds.
- `fn vx_hash_set_drop_i32(ptr : *mut i8) -> i32`<br>
  Release the set.

## `std::io`

Standard input, output and error.

**Functions**

- `unsafe fn stdout_write(buffer : *const u8, len : i64) -> i64`<br>
  Write `len` bytes to standard output and answer how many were taken.
  A short write is possible and is not retried here. Unsafe because `buffer` must have `len`
  bytes to read.
- `unsafe fn stderr_write(buffer : *const u8, len : i64) -> i64`<br>
  Write `len` bytes to standard error, with the same caveats as `stdout_write`.
- `unsafe fn stdin_read(buffer : *mut u8, len : i64) -> i64`<br>
  Read up to `len` bytes from standard input and answer how many arrived.
  Zero means end of input or an error, which are not told apart. Unsafe because `buffer` must
  have room for `len` bytes.

**C bindings** *(the native functions this module is built on)*

- `fn vx_stdout_write(buffer : *const u8, len : i64) -> i64`<br>
  The Rust core behind `stdout_write`.
- `fn vx_stderr_write(buffer : *const u8, len : i64) -> i64`<br>
  The Rust core behind `stderr_write`.
- `fn vx_stdin_read(buffer : *mut u8, len : i64) -> i64`<br>
  The Rust core behind `stdin_read`.

## `std::libc`

Direct bindings to the C library.

**Functions** *(bound directly to C)*

- `fn open(path : *const i8, flags : i32) -> i32`<br>
  C's `open`: a file descriptor, or -1 on failure. `flags` is the platform's, not Vx's.
- `fn close(fd : i32) -> i32`<br>
  C's `close`: 0, or -1 on failure.
- `fn lseek(fd : i32, offset : i64, whence : i32) -> i64`<br>
  C's `lseek`: the new offset, or -1. `whence` is 0 from the start, 1 from the current
  position, 2 from the end.

## `std::llama`

Helpers used by the Llama 2 example.

**Types**

- `struct LlamaConfig`
- `struct TransformerWeightOffsets`
- `struct Tokenizer`

**`LlamaConfig` methods**

- `fn load(filepath : *const i8) -> LlamaConfig`<br>
  Read a checkpoint's header.
  Nothing validates the file: a path that is not a checkpoint gives a config of whatever
  the first seven words happen to be, and the sizes computed from it are then wrong.

**`TransformerWeightOffsets` methods**

- `fn calculate(c : &LlamaConfig) -> TransformerWeightOffsets`<br>
  Where each weight matrix begins, as a float offset into one flat buffer.
  The order is llama2.c's, and the arithmetic assumes the checkpoint was written by it.
- `fn load_all_weights(filepath : *const i8, c : &LlamaConfig) -> Tensor<f32, [?, ?]>`<br>
  Every weight as one 1-by-N tensor, copied out of the mapped checkpoint.
  The length is computed from `c`, so a config that does not match the file reads past its
  end. A copy rather than a view, so the whole model is resident twice while this runs.

**`Tokenizer` methods**

- `fn load(filepath : *const i8, vocab_size : i32) -> Tokenizer`<br>
  Read a tokenizer file. `vocab_size` must match the checkpoint's.
- `fn decode(self : &Tokenizer, prev_token : i32, token : i32) -> String`<br>
  The text for one token, as an owned `String` the caller must `drop`.
  `prev_token` decides whether a leading space is stripped, which is why decoding a token
  in isolation can differ from decoding it in sequence.

**C bindings** *(the native functions this module is built on)*

- `fn vx_load_config(filepath : *const i8) -> *mut i32`<br>
  Read the seven header fields of a llama2.c checkpoint into an array of `i32`.
- `fn vx_load_weights(filepath : *const i8) -> *mut f32`<br>
  Map a checkpoint's weights and hand back a pointer to the first float.
- `fn vx_build_tokenizer(filepath : *const i8, vocab_size : i32) -> *mut i8`<br>
  Read a tokenizer file, answering an opaque handle the Rust core owns.
- `fn vx_decode_token(tokenizer_ptr : *mut i8, prev_token : i32, token : i32) -> *const i8`<br>
  The text for one token, as a NUL-terminated string.
  `prev_token` is needed because llama2's tokenizer strips a leading space after the
  beginning-of-sequence token and not otherwise.
- `fn vx_encode_prompt(tokenizer_ptr : *mut i8, text_ptr : *const i8) -> *mut i32`<br>
  Encode a prompt, answering an array of token ids with its length in the first slot.
- `fn vx_read_prompt_file(filepath : *const i8) -> *const i8`<br>
  Read a whole file as a NUL-terminated string.
- `fn vx_get_llama_config() -> *mut i32`<br>
  The configuration of the checkpoint most recently loaded.

## `std::mmap`

Memory-mapped files.

**Functions** *(bound directly to C)*

- `fn mmap(addr : *mut i8, length : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8`<br>
  C's `mmap`: map `length` bytes of `fd` into memory.
  Answers `MAP_FAILED` rather than null on failure, which is -1 cast to a pointer and
  which Vx cannot test for (Vx#714). `prot` and `flags` are the platform's.
- `fn munmap(addr : *mut i8, length : i64) -> i32`<br>
  C's `munmap`: unmap a region previously mapped. 0, or -1 on failure.

## `std::net`

TCP and UDP sockets.

**Types**

- `struct TcpStream`<br>
  A connected TCP socket, held by the Rust core behind an opaque pointer.
  Closed by calling `tcp_stream_drop`, since the language has no `Drop` (Vx#495).
- `struct UdpSocket`<br>
  A bound UDP socket, held by the Rust core behind an opaque pointer.
- `struct TcpListener`<br>
  A listening TCP socket, held by the Rust core behind an opaque pointer.

**`TcpStream` methods**

- `unsafe fn connect(addr : *const i8) -> TcpStream`<br>
  Connect to `addr`, written as `host:port`. ///
  **A failure cannot be detected.** The value handed back holds a null pointer, and Vx
  cannot compare a raw pointer against null (Vx#714), so a refused connection looks like a
  working one until every later call answers 0.
  Unsafe because nothing checks that `addr` points at a NUL-terminated string.
- `unsafe fn read(self : *mut TcpStream, buffer : *mut u8, len : i64) -> i64`<br>
  Read up to `len` bytes into `buffer` and answer how many arrived.
  Zero means the peer closed, a null socket, or a read error. A short read is normal: TCP
  is a stream and a message may arrive in pieces, so a caller wanting a whole one loops.
  Unsafe because `buffer` must have room for `len` bytes.
- `unsafe fn write(self : *mut TcpStream, buffer : *const u8, len : i64) -> i64`<br>
  Write up to `len` bytes from `buffer` and answer how many were taken.
  A short write is normal and the remainder is not retried here.

**Functions**

- `unsafe fn tcp_stream_drop(stream : *mut TcpStream) -> void`<br>
  Close the connection and release it. Using it afterwards reads freed memory.
- `unsafe fn udp_socket_drop(socket : *mut UdpSocket) -> void`<br>
  Close the socket and release it.
- `unsafe fn tcp_listener_drop(listener : *mut TcpListener) -> void`<br>
  Stop listening and release the socket.

**`UdpSocket` methods**

- `unsafe fn bind(addr : *const i8) -> UdpSocket`<br>
  Bind to `addr`, written as `host:port`. ///
  **A failure cannot be detected.** The value handed back holds a null pointer, and Vx
  cannot compare a raw pointer against null (Vx#714), so a refused connection looks like a
  working one until every later call answers 0.
- `unsafe fn recv(self : *mut UdpSocket, buffer : *mut u8, len : i64) -> i64`<br>
  Receive one datagram into `buffer` and answer its length.
  A datagram longer than `len` is truncated and the rest is lost, which is UDP's behaviour
  and not an error here. The sender's address is not reported.
- `unsafe fn send_to(self : *mut UdpSocket, buffer : *const u8, len : i64, addr : *const i8) -> i64`<br>
  Send one datagram of `len` bytes to `addr`, answering how many were sent.
  Nothing guarantees it arrives, or arrives once, or arrives in order.

**`TcpListener` methods**

- `unsafe fn bind(addr : *const i8) -> TcpListener`<br>
  Listen on `addr`, written as `host:port`.
  **A failure cannot be detected**, for the reason `TcpStream::connect` gives (Vx#714).
- `fn accept(self : *mut TcpListener) -> TcpStream`<br>
  Wait for a connection and answer with it.
  Blocks until one arrives. The stream handed back holds a null pointer when the accept
  failed, which cannot be told from a working one either.

**C bindings** *(the native functions this module is built on)*

- `fn vx_tcp_stream_connect(c_addr : *const i8) -> *mut i8`<br>
  The Rust core behind `TcpStream::connect`. Null when the connection fails.
- `fn vx_tcp_stream_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`<br>
  The Rust core behind `TcpStream::read`.
- `fn vx_tcp_stream_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64`<br>
  The Rust core behind `TcpStream::write`.
- `fn vx_tcp_stream_drop(ptr : *mut i8) -> i32`<br>
  The Rust core behind `tcp_stream_drop`.
- `fn vx_udp_socket_bind(c_addr : *const i8) -> *mut i8`<br>
  The Rust core behind `UdpSocket::bind`. Null when the bind fails.
- `fn vx_udp_socket_recv(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64`<br>
  The Rust core behind `UdpSocket::recv`.
- `fn vx_udp_socket_send_to(ptr : *mut i8, buffer : *const u8, len : i64, c_addr : *const i8) -> i64`<br>
  The Rust core behind `UdpSocket::send_to`.
- `fn vx_udp_socket_drop(ptr : *mut i8) -> i32`<br>
  The Rust core behind `udp_socket_drop`.
- `fn vx_tcp_listener_bind(c_addr : *const i8) -> *mut i8`<br>
  The Rust core behind `TcpListener::bind`. Null when the bind fails.
- `fn vx_tcp_listener_accept(ptr : *mut i8) -> *mut i8`<br>
  The Rust core behind `TcpListener::accept`. Blocks until a connection arrives.
- `fn vx_tcp_listener_drop(ptr : *mut i8) -> i32`<br>
  The Rust core behind `tcp_listener_drop`.

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

- `fn sqrt_f64(x : f64) -> f64`<br>
  The square root, over the `math` dialect so this module needs no libm.
  `core::num` has the same method; this is kept because `normal` runs before an import of it
  would settle, and a local one keeps the dependency to `core::num` alone.
- `fn ln_f64(x : f64) -> f64`<br>
  The natural logarithm, over the `math` dialect.

**`SplitMix64` methods**

- `fn seeded(seed : u64) -> SplitMix64`<br>
  A generator started at `seed`. Every seed is valid, including zero.
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
- `fn next_u16(self : &mut Rng) -> u16`<br>
  The next 16 bits.
- `fn next_u8(self : &mut Rng) -> u8`<br>
  The next 8 bits.
- `fn next_i64(self : &mut Rng) -> i64`<br>
  Uniform over the signed range, negatives included: the bits are reinterpreted, not
  clamped, so half the draws are below zero.
- `fn next_i32(self : &mut Rng) -> i32`<br>
  The next 32 bits read as signed, so negative half the time.
- `fn next_i16(self : &mut Rng) -> i16`<br>
  The next 16 bits read as signed.
- `fn next_i8(self : &mut Rng) -> i8`<br>
  The next 8 bits read as signed.
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
- `fn next_bf16(self : &mut Rng) -> bf16`<br>
  A `bf16` uniform in \[0, 1). Eight bits of mantissa, so the draws are coarse.
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
- `fn range_f32(self : &mut Rng, lo : f32, hi : f32) -> f32`<br>
  Uniform in \[lo, hi), with the caveat `range_f64` gives for a backwards range.
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
- `fn fill_range_f32(self : &mut Rng, out : *mut f32, count : i32, lo : f32, hi : f32) -> i32`<br>
  Fill `count` elements with draws uniform in \[lo, hi).
  `out` must have room for `count` of them; nothing here checks.
- `fn fill_normal_f32(self : &mut Rng, out : *mut f32, count : i32, mean : f32, stddev : f32) -> i32`<br>
  Fill `count` elements with normal draws of the given mean and standard deviation.
  `out` must have room for `count` of them. A negative `stddev` mirrors the distribution
  rather than being refused.
- `fn fill_f16(self : &mut Rng, out : *mut f16, count : i32) -> i32`<br>
  Fill `count` elements with `f16` draws uniform in \[0, 1).
- `fn fill_normal_f16(self : &mut Rng, out : *mut f16, count : i32, mean : f64, stddev : f64) -> i32`<br>
  The normal fill at half precision. The draw and the scaling happen in f64 and narrow
  once at the store, so a tail value is rounded rather than computed twice.

## `std::simd`

SIMD vector types and operations.

**Functions**

- `unsafe fn simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The elementwise sum of two four-lane `f32` vectors, written to `out`.
  All three pointers must address four `f32`s; nothing here checks. `out` may alias `a` or
  `b`.
- `unsafe fn simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The elementwise difference of two four-lane `f32` vectors, written to `out`.
  All three pointers must address four `f32`s; nothing here checks. `out` may alias `a` or
  `b`.
- `unsafe fn simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The elementwise product of two four-lane `f32` vectors, written to `out`.
  All three pointers must address four `f32`s; nothing here checks. `out` may alias `a` or
  `b`.
- `unsafe fn simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The elementwise quotient of two four-lane `f32` vectors, written to `out`.
  All three pointers must address four `f32`s; nothing here checks. `out` may alias `a` or
  `b`.
- `unsafe fn simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32`<br>
  `a * b + c` elementwise over four lanes, written to `out`.
  Fused, so the product is not rounded before the addition: the answer can differ from a
  separate multiply and add in the last bit, and is the more accurate of the two.

**C bindings** *(the native functions this module is built on)*

- `fn vx_simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The Rust core behind `simd_add_f32x4`.
- `fn vx_simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The Rust core behind `simd_sub_f32x4`.
- `fn vx_simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The Rust core behind `simd_mul_f32x4`.
- `fn vx_simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32`<br>
  The Rust core behind `simd_div_f32x4`.
- `fn vx_simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32`<br>
  The Rust core behind `simd_fma_f32x4`.

## `std::string`

`String` and text manipulation.

**Types**

- `struct String`<br>
  A growable, owned string, held by the Rust core behind an opaque pointer.
  Released by calling `drop`, since the language has no `Drop` (Vx#495).

**`String` methods**

- `fn new() -> String`<br>
  An empty string.
- `unsafe fn from_c_str(c_str : *const i8) -> String`<br>
  A copy of a NUL-terminated C string.
  Unsafe because nothing here checks that `c_str` is a valid pointer to a NUL-terminated
  run of bytes. The copy is owned by the new `String`; the argument is not taken.
- `unsafe fn push_c_str(self : *mut String, c_str : *const i8) -> i32`<br>
  Append a NUL-terminated C string, with the same requirement `from_c_str` has.
- `fn len(self : *mut String) -> i32`<br>
  The length in bytes, not in characters: this counts UTF-8 code units.
- `fn as_c_str(self : *mut String) -> *const i8`<br>
  A NUL-terminated copy of the contents.
  **This allocates and leaks.** The Rust core builds a fresh `CString` and hands out its
  raw pointer, so every call costs a copy that is never released: the matching
  `vx_string_free_c_str` is declared in the extern block and is not wrapped as a method
  here. Calling this in a loop grows the process without bound.
- `fn drop(self : *mut String) -> i32`<br>
  Release the string. Reading it afterwards reads freed memory.

**`i32` methods**

- `fn to_string(self : i32) -> String`<br>
  This number in decimal, as an owned `String` the caller must `drop`.

**Functions**

- `unsafe fn string_length(s : *const i8) -> i32`<br>
  The number of bytes before the NUL, found by scanning.
  Unsafe because it walks until it finds one: a pointer to bytes with no NUL runs off the
  end. `core::str` replaces this once a string literal carries its length (Vx#532).
- `unsafe fn string_compare(s1 : *const i8, s2 : *const i8) -> i32`<br>
  C's `strcmp`: negative when `s1` sorts first, positive when `s2` does, zero when equal.
  The magnitude is the difference between the first bytes that differ, which callers should
  not read anything into beyond its sign. Unsafe for the reason `string_length` is.
- `unsafe fn parse_int(s : *const i8) -> i32`<br>
  The leading run of decimal digits as an `i32`.
  Stops at the first byte that is not a digit and answers what it has, so `"12abc"` is 12 and
  `"abc"` is 0 -- there is no way to tell that second case from a genuine zero. A leading `-`
  is not a sign, it is a stop. Nothing checks for overflow: a run longer than `i32` holds
  wraps. `core::str`'s `parse` replaces it (Vx#532).

**C bindings** *(the native functions this module is built on)*

- `fn vx_string_new() -> *mut i8`<br>
  The Rust core behind `String::new`.
- `fn vx_string_from_c_str(ptr : *const i8) -> *mut i8`<br>
  The Rust core behind `String::from_c_str`.
- `fn vx_string_push_c_str(ptr : *mut i8, c_str : *const i8) -> i32`<br>
  The Rust core behind `String::push_c_str`.
- `fn vx_string_len(ptr : *mut i8) -> i32`<br>
  The Rust core behind `String::len`.
- `fn vx_string_as_c_str(ptr : *mut i8) -> *const i8`<br>
  Allocate a NUL-terminated copy and hand out its raw pointer. The caller owns it and
  releases it with `vx_string_free_c_str`; `String::as_c_str` does not, and leaks.
- `fn vx_string_free_c_str(ptr : *const i8) -> i32`<br>
  Release what `vx_string_as_c_str` returned. Nothing in Vx calls this yet.
- `fn vx_string_drop(ptr : *mut i8) -> i32`<br>
  The Rust core behind `String::drop`.
- `fn vx_i32_to_string(val : i32) -> *mut i8`<br>
  The Rust core behind `i32::to_string`.

## `std::tensor`

Operations on `Tensor`, including shape queries and elementwise maths.

**`Tensor<T, [?, ?]>` methods**

- `fn from_ptr_1d(ptr : *mut T, d1 : i32) -> Tensor<T, [?, ?]>`<br>
  A 1-by-`d1` tensor holding a copy of `d1` elements read from `ptr`.
  The elements are copied, so the tensor does not alias the buffer and outlives it. `ptr`
  must address `d1` elements; nothing here checks.
- `fn from_ptr_2d(ptr : *mut T, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`<br>
  A `d1`-by-`d2` tensor holding a copy of `d1 * d2` elements read from `ptr` in row-major
  order. Copied, as `from_ptr_1d` is.
- `fn slice_2d(self : &Tensor<T, [?, ?]>, row : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`<br>
  A `d1`-by-`d2` tensor read from one row of this one, taken row-major from its start.
  **A copy, not a view.** Every `slice_` method here allocates and copies, so writing to
  the result does not touch the original and the cost is the elements moved, not constant.
- `fn slice_2d_from_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>`<br>
  A `d1`-by-`d2` tensor read from row 0 beginning at `start`, reshaped row-major. A copy.
- `fn slice_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32) -> Tensor<T, [?, ?]>`<br>
  A 1-by-`d1` tensor read from row 0 beginning at `start`. A copy.
- `fn fill(self : &mut Tensor<T, [?, ?]>, val : T) -> void`<br>
  Set every element to `val`.
  Lowered through `linalg`, with a separate path chosen at compile time for AVX-512.
- `fn copy(self : &mut Tensor<T, [?, ?]>, src : &Tensor<T, [?, ?]>) -> void`<br>
  Overwrite this tensor's elements with `src`'s.
  Both must have the same shape; nothing here checks, and a mismatch reads or writes past
  an end.
- `fn assign(self : &mut Tensor<T, [?, ?]>, val : T) -> void`<br>
  Set every element to `val`, as `fill` does. The two differ in the `linalg` form they
  emit, not in what they mean.
- `fn compare(self : &Tensor<T, [?, ?]>, other : &Tensor<T, [?, ?]>) -> bool`<br>
  Are the two tensors equal element by element?
  Exact equality, reduced over every element, so two tensors a rounding step apart answer
  false. A NaN anywhere makes the answer false, including against itself.

**`Tensor<T, [N, M]>` methods**

- `fn fill_static(self : &mut Tensor<T, [ N, M ]>, val : T) -> void`<br>
  Set every element of a statically shaped tensor to `val`.
  The shape is known at compile time here, so the emitted loop has constant bounds.

## `std::time`

Clocks and durations.

**Functions**

- `fn now() -> f32`<br>
  Seconds from some fixed point, for measuring how long something took.
  Only differences between two calls mean anything: the origin is unspecified. An `f32`
  holds about seven digits, so a long-running process loses resolution as the value grows
  -- `unix_timestamp` is the `f64` one.
- `fn sleep(seconds : f32) -> i32`<br>
  Pause this thread for at least `seconds`. It may be longer; it is never shorter.
- `fn unix_timestamp() -> f64`<br>
  Seconds since the Unix epoch, as an `f64`. Wall-clock time, so it can move backwards
  when the system clock is adjusted; `now` is the one to measure a duration with.
- `unsafe fn bench_report(name : *const i8, unit : *const i8, value : f32) -> i32`<br>
  Report a benchmark measurement in the form the harness collects.
  Unsafe because `name` and `unit` must point at NUL-terminated strings.

**C bindings** *(the native functions this module is built on)*

- `fn vx_get_time() -> f32`<br>
  The Rust core behind `now`.
- `fn vx_sleep(seconds : f32) -> i32`<br>
  The Rust core behind `sleep`.
- `fn vx_unix_timestamp() -> f64`<br>
  The Rust core behind `unix_timestamp`.
- `fn vx_bench_report(name : *const i8, unit : *const i8, value : f32) -> i32`<br>
  The Rust core behind `bench_report`.

## `std::vec`

`Vec<T>`, a growable array.

**Types**

- `struct Vec<T>`<br>
  A growable array of `T`, held in a buffer the Rust core owns.
  Vx keeps a typed view into that buffer and does its own element loads and stores; growth,
  alignment and the `capacity * elem_size` overflow check belong to Rust. There is no `Drop`
  in the language, so the buffer is released by calling `free` and not before.
- `struct VecIter<T>`<br>
  An iterator over a `Vec<T>`'s elements, holding a pointer to the vector it walks.
  Growing or freeing that vector while this exists leaves the iterator pointing at the old
  buffer.
- `struct VecMap<T, NewItem>`<br>
  The iterator `VecIter::map` builds: the inner walk plus the function applied to each item.

**`Vec<T>` methods**

- `fn new() -> Vec<T>`<br>
  An empty vector with room for two elements.
- `fn with_capacity(capacity : i32) -> Vec<T>`<br>
  An empty vector with room for `capacity` elements before it has to grow.
- `fn free(self : &mut Vec<T>) -> i32`<br>
  Release the buffer and leave the vector empty with no capacity.
  Called by hand, because the language has no `Drop` yet (Vx#495). Reading an element after
  this is reading freed memory; `len` answers 0, so a loop over it is safe.
- `fn as_mut_ptr(self : &Vec<T>) -> *mut T`<br>
  A raw pointer to the first element. Invalidated by anything that grows the vector.
- `fn as_mut_slice(self : &mut Vec<T>) -> &mut T`<br>
  A mutable reference to the first element.
  Named for what it will return once slices exist (Vx#534); today it hands back the first
  element rather than a `(pointer, length)` pair, so the length has to be carried
  separately by whoever reads it.
- `fn as_slice(self : &Vec<T>) -> &T`<br>
  A reference to the first element, with the caveat `as_mut_slice` describes.
- `fn push(self : &mut Vec<T>, val : T) -> i32`<br>
  Append a value, growing the buffer when it is full.
  Capacity doubles, from four upwards, so appending n values reallocates about log2(n)
  times. Any raw pointer or reference taken from this vector is invalidated by a growth.
- `fn get(self : &Vec<T>, index : i32) -> T`<br>
  The element at `index`, by value.
  # Panics
  When `index` is negative or not below `len`. The check is in the Rust core, which prints
  the index and the length and aborts the process -- it does not unwind.
- `fn set(self : &mut Vec<T>, index : i32, val : T) -> i32`<br>
  Overwrite the element at `index`.
  # Panics
  As `get` does, and for the same reason. Setting past the end does not extend the vector;
  `push` is what grows it.
- `fn len(self : &Vec<T>) -> i32`<br>
  How many elements are in the vector, which is not its capacity.
- `fn iter(self : &Vec<T>) -> VecIter<T>`<br>
  An iterator over the elements, borrowing the vector rather than consuming it.

**`Extend<T> for Vec<T>` methods**

- `fn extend<I : Iterator>(self : &mut Vec<T>, iter : I) -> i32`<br>
  Push every item `iter` has left, in order.
- `fn extend_one(self : &mut Vec<T>, item : T) -> i32`<br>
  Push `item`.

**`Default for Vec<T>` methods**

- `fn default() -> Vec<T>`<br>
  An empty `Vec`, as `new` makes.

**`FromIterator<T> for Vec<T>` methods**

- `fn from_iter<I : Iterator>(iter : I) -> Vec<T>`<br>
  A new `Vec` holding every item `iter` has left, which the caller owns and must `free`.

**`Iterator for VecIter<T>` methods**

- `fn next(self : &mut VecIter<T>) -> Option<T>`<br>
  The next element, or nothing once the end is reached.

**`DoubleEndedIterator for VecIter<T>` methods**

- `fn next_back(self : &mut VecIter<T>) -> Option<T>`<br>
  The last element not yet handed out from either end.

**`ExactSizeIterator for VecIter<T>` methods**

- `fn len(self : &VecIter<T>) -> i64`<br>
  How many elements are left between the two ends.

**`VecIter<T>` methods**

- `fn map<NewItem>(self : VecIter<T>, f : Closure1<T, NewItem>) -> VecMap<T, NewItem>`<br>
  An iterator over these elements with `f` applied to each.

**`Iterator for VecMap<T, NewItem>` methods**

- `fn next(self : &mut VecMap<T, NewItem>) -> Option<NewItem>`<br>
  The next element of the inner iterator with `f` applied.

**`ExactSizeIterator for VecMap<T, NewItem>` methods**

- `fn len(self : &VecMap<T, NewItem>) -> i64`<br>
  As many as the inner iterator has.

**`VecMap<T, NewItem>` methods**

- `fn collect(self : &mut VecMap<T, NewItem>) -> Vec<NewItem>`<br>
  Drain the iterator into a fresh `Vec`, which the caller owns and must `free`.

**C bindings** *(the native functions this module is built on)*

- `fn vx_vec_alloc(elem_size : i64, cap : i64) -> *mut i8`<br>
  Allocate a buffer for `cap` elements of `elem_size` bytes. Rust owns the alignment
  and the overflow check on the product.
- `fn vx_vec_grow(ptr : *mut i8, old_cap : i64, new_cap : i64, elem_size : i64) -> *mut i8`<br>
  Reallocate to `new_cap` elements, moving the contents. The old pointer is invalid after.
- `fn vx_vec_free(ptr : *mut i8, cap : i64, elem_size : i64) -> i32`<br>
  Release the buffer.
- `fn vx_vec_bounds_check(index : i64, len : i64) -> i32`<br>
  Abort the process when `index` is outside `0..len`, printing both. Answers 0 otherwise.

______________________________________________________________________

719 functions across 32 modules.
