# Operators and standard traits

The language itself defines `+`, `==` and `<` on numbers. On a struct or an enum of your own,
each operator calls a method of a trait from the standard library. So you give your type an
operator when you implement its trait. This chapter shows those traits, and the traits that copy,
build and convert values. [Generics and traits](generics.md) explains traits and bounds in general.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Arithmetic: `Add`, `Sub`, `Neg` and the others

`a + b` on a struct is `a.add(b)`, and `-a` is `a.neg()`. The traits are in `core::ops`. Each one
has an associated type `Output`, which is the type of the result. Both operands have the same
type: Vx does not yet let the right operand have a different type.

| Operator | Trait | Method |
| -------- | -------- | -------- |
| `a + b` | `Add` | `add` |
| `a - b` | `Sub` | `sub` |
| `a * b` | `Mul` | `mul` |
| `a / b` | `Div` | `div` |
| `a % b` | `Rem` | `rem` |
| `-a` | `Neg` | `neg` |
| `!a` | `Not` | `not` |
| `a & b` | `BitAnd` | `bitand` |
| `a \| b` | `BitOr` | `bitor` |
| `a ^ b` | `BitXor` | `bitxor` |
| `a << b` | `Shl` | `shl` |
| `a >> b` | `Shr` | `shr` |

`a == b` is `a.eq(&b)`, from the trait `PartialEq` in `core::cmp`. The trait gives `ne` for
`a != b`, so you write only `eq`.

This program gives a `Money` type four operators (`book_traits_operators.vx`):

```rust
{{#include ../../../tests/backend/pass/book_traits_operators.vx:program}}
```

It prints:

```text
total 300
change 700
refund -50
equal true, different true
sums 600 6
```

The number types implement the operator traits too. So the generic function `sum3`, with the
bound `T : Add`, accepts `Money` and also `i32`.

The method takes `self` by value, so `price + tip` moves both operands into `add`. `Money`
implements `Copy`, so `price` and `tip` are still usable after the addition. The section on `Copy`
below explains this.

Write `a = a + b` and not `a += b`. A compound assignment on a struct does not compile yet
([Vx#3231](https://github.com/vx-lang/Vx/issues/3231)).

If a type does not implement the trait, the compiler stops with an error:

```text
Error[E3004]: `+` is not defined for Money: it needs an impl of `core::ops::Add`
```

## Order: `Ord` and `PartialOrd`

The trait `Ord` gives a total order: every two values compare as `Less`, `Equal` or `Greater`.
You write one method, `cmp`, which returns an `Ordering`. The trait gives the rest from `cmp`:
the operators `<`, `<=`, `>` and `>=`, and the methods `max`, `min` and `clamp`.

This program orders versions by `major`, and then by `minor` when the two `major` numbers are
equal (`book_traits_ordering.vx`):

```rust
{{#include ../../../tests/backend/pass/book_traits_ordering.vx:program}}
```

It prints:

```text
true false true
max 2.0
min 1.9
clamp 2.0
```

`Ordering::then` gives the first ordering, or the second one when the first is `Equal`. When the
two values compare equal, `max` returns the second value and `min` returns the first one.

`Ord` does not give `==`. The program implements `PartialEq` for that. It also implements `Eq`,
an empty trait that says every value is equal to itself. A generic function can ask for `Eq` in a
bound.

Some types have values that do not compare. A float NaN is not less than, equal to or greater than
any number. Such a type implements `PartialOrd` and not `Ord`. Its one method returns `None` when
the two values do not compare:

<!-- vx-doctest: skip -- a trait from core::cmp, not a program -->

```rust
trait PartialOrd {
  fn partial_cmp(self : &Self, other : &Self) -> Option<Ordering>;
}
```

On a type with only `PartialOrd`, `a < b` calls `partial_cmp`. Each of the four comparisons is
`false` when `partial_cmp` returns `None`.

## Copy, clone and default values: `Copy`, `Clone`, `Default`

An assignment such as `let b = a;` moves a struct. After the move, `a` is not usable
([Ownership and borrowing](ownership.md) explains moves). An assignment copies a type that
implements the empty trait `Copy` from `core::marker`. A struct can be `Copy` only when each of its
fields is `Copy`. The numbers and `bool` are `Copy`, but a `Vec` is not.

`Clone`, from `core::clone`, makes a copy when you ask for it with `x.clone()`. You write the
method yourself, so it can copy what a plain read does not copy, such as the items of a vector.

`Default`, from `core::default`, gives the value that a type starts from. It is a static method,
so you call it through the type: `Size::default()`. The library gives zero for each number type,
`false` for `bool` and `None` for an `Option`. Vx does not derive `Default`, so a struct must
implement it.

This program uses all three traits (`book_traits_copy_clone_default.vx`):

```rust
{{#include ../../../tests/backend/pass/book_traits_copy_clone_default.vx:program}}
```

It prints:

```text
80 24 0
first 1, second 2
```

`Size` is `Copy`, so `a` is still usable after `let b = a;`. Without `impl Copy for Size {}`, the
compiler stops at `a.width` with the error E4001, because the assignment moved `a`.

`second` is a clone of `first`, so `second` has its own vector. A push to `second` does not change
`first`.

A `Copy` impl on `Basket` gets the error E3031, because its field `items` is a `Vec<i64>`.

## Conversions: `From`

`From<T>`, from `core::convert`, builds a value of one type from a value of type `T`. It is a
static method too, so you call it through the target type: `Celsius::from(20)`. A type can
implement `From` once for each source type.

The library implements `From` only where the conversion loses nothing. `i32::from` accepts a
`u8`, because each `u8` value fits in an `i32`. There is no `i32::from` for an `i64`: use
`as` when you accept that a value can change.

This program gives `Celsius` two `From` impls (`book_traits_from.vx`):

```rust
{{#include ../../../tests/backend/pass/book_traits_from.vx:program}}
```

It prints `100 20 300`.

Vx does not have the trait `Into` yet. Rust programmers write `x.into()`, but in Vx you write
`T::from(x)`.

## A struct as a key: `Hash` and `PartialEq`

A generic function can ask for traits in the bounds of its type parameters. To use your struct
there, implement each trait that the bounds name. A key of a hash table usually needs two traits:

- `Hash`, from `core::hash`, so that a hasher can turn the key into a number.
- `PartialEq`, so that two keys with the same hash can be compared.

`Hash` has one method, `hash`, which writes the fields into a `Hasher`. The numbers and `bool`
implement `Hash`, so a struct usually calls `hash` on each of its fields. The library has two
hashers: `FxHasher` is fast, and `SipHasher13` is harder to attack with chosen keys.

In this program, the generic function `hash_of` needs `Hash`, and `add_new` needs `PartialEq`.
`add_new` adds a key to a vector only when the vector does not already hold an equal key
(`book_traits_a_key.vx`):

```rust
{{#include ../../../tests/backend/pass/book_traits_a_key.vx:program}}
```

It prints:

```text
same hash true, other hash false
added true false true
cells 2
```

`a` and `b` have the same fields, so they have the same hash and they are equal. `add_new` adds
`a`, does not add `b`, and adds `c`. `add_new` compares two references with `==`. Two references
compare the values that they point at, so `==` calls `eq`.

If `Cell` does not implement `Hash`, the call `hash_of(&a)` gets an error:

```text
Type 'Cell' does not implement trait 'Hash' required by parameter 'K'
```
