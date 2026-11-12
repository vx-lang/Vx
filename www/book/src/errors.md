# Errors, Option and Result

A function can fail to give an answer. A search can find nothing, and a parse can meet a bad
character. Vx has no exceptions. A function that can fail says so in its return type, with
`Option` or `Result`. This chapter shows how to make these values, how to read them, and how to
pass an error up to the caller.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Option: a value or nothing

`Option<T>` holds either one value of type `T` (`Some`) or no value (`None`). It comes from
`import core::option;`. Write a variant with its full type: `Option<i64>::Some(3)` and
`Option<i64>::None()`. [Vx for Rust programmers](for-rust-programmers.md) lists this and the
other differences from Rust.

Read an `Option` with `match`, as [Control flow](control-flow.md#match) shows for an enum. Each
arm handles one variant. A method is shorter when you need only one fact:

- `o.is_some()` and `o.is_none()` tell which variant `o` is. They borrow `o`, so you can still use
  it after the call.
- `o.unwrap_or(d)` gives the value, or `d` for a `None`.

This program searches for the first square above a limit (`book_errors_find_a_value.vx`):

```rust
{{#include ../../../tests/backend/pass/book_errors_find_a_value.vx:program}}
```

It prints:

```text
first square above 20 is 25
no square above 200
found false
or 0 gives 0
```

## Result: a value or an error

`Result<T, E>` holds either a value of type `T` (`Ok`) or an error of type `E` (`Err`). It comes
from `import core::result;`. Use it when the caller must know why the operation failed. The error
type is often an enum of your own, with one variant for each way to fail.

`r.is_ok()` and `r.is_err()` tell which variant `r` is. They also borrow `r`, so they work for a
payload that is not `Copy`. `r.unwrap_or(d)` gives the value, or `d` for an `Err`.

## The `?` operator

`r?` gives the value of an `Ok`. For an `Err`, it returns that `Err` from the function at once.
The function must return a `Result` with the same error type. So `?` replaces a `match` that only
passes the error up.

`?` does not convert the error. If the error types differ, the compiler reports E3063. Use
`r.map_err(f)` first: it calls the closure `f` on the error of an `Err`, and it keeps an `Ok` as it
is. Write the type that the closure returns, as in `| c : i64 | -> AddError { .. }`. Without it,
the `?` after `map_err` gets a wrong E3063 ([Vx#3233](https://github.com/vx-lang/Vx/issues/3233)).

In this program, `digit` reports a bad character with its code, an `i64`. `two_digits` converts
that code to an `AddError` and passes it up with `?` (`book_errors_question_mark.vx`):

```rust
{{#include ../../../tests/backend/pass/book_errors_question_mark.vx:program}}
```

It prints:

```text
sum 42
error: the character 120 is not a digit
error: the sum 108 is too large
```

A pattern in a `match` cannot test a variant inside a payload. `Err(AddError::TooLarge(n))` gets
the error E3050. So `report` binds the error to `e`, and `report_error` matches on `e`.

`?` takes only a `Result`. A `?` on an `Option` gets the error E3063. Convert the `Option` first:
`o.ok_or(e)` gives `Ok` with the value, or `Err(e)` for a `None`.

```vx
fn quarter(n : i64) -> Result<i64, i64> {
  let h = half(n).ok_or(n)?;
  let q = half(h).ok_or(h)?;
  return Result<i64, i64>::Ok(q);
}
```

A `?` in a task or a `tasks` block passes the error to the block, not to the function.
[Tasks and channels](tasks.md) shows this.

## Take a value out with `take` and `replace`

A field of type `Option` can hold a value for a time. `o.take()` moves the value out and leaves
`None` in `o`. `o.replace(v)` puts `v` in `o` and gives the old contents. Both borrow `o` mutably,
so `o` must be a `let mut` variable or a field of one. Both work for a value that is not `Copy`.

In this program, a desk holds at most one ticket (`book_errors_take_and_replace.vx`). A message
such as `"took"` is a `*const i8`, the type of a string literal:

```rust
{{#include ../../../tests/backend/pass/book_errors_take_and_replace.vx:program}}
```

It prints:

```text
replaced nothing
replaced ticket 1
took ticket 2
empty true
```

## Errors that the program cannot handle

Some errors mean that the program has a bug. Then the best action is to stop the program with a
clear message. [A tour of Vx](tour.md#stopping-a-program) explains `panic`, `abort` and `-> !`.

- `o.unwrap()` and `r.unwrap()` give the value. For a `None` or an `Err`, they stop the program.
- `o.expect(msg)` and `r.expect(msg)` do the same, but they print `msg`.
- `panic(msg)` stops the program with `msg`. `abort()` stops it with no message.

Each of these prints the file and line of the call, not a line in the standard library. Use
`expect` rather than `unwrap`, so that the message tells the reader what went wrong.

This program divides twice. The second division fails (`book_errors_expect_stops_the_program.vx`):

```rust
{{#include ../../../tests/backend/pass/book_errors_expect_stops_the_program.vx:program}}
```

It prints the lines below and ends with an error status. The last line names the line of the
`expect` call.

```text
ok true, err false
10 / 2 = 5
panic: the divisor must not be zero
  at book_errors_expect_stops_the_program.vx:36
```

`unwrap` and `expect` take the `Option` or `Result` by value. So call `is_ok` before them, as this
program does, and not after.

## More methods

`core/option.vx` and `core/result.vx` in the standard library have more methods, each with a
comment. Some of them are `map`, `and_then`, `or`, `filter`, `ok`, `err`, `unwrap_err` and
`unwrap_or_else`.
