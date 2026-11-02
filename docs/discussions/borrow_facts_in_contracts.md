# Borrow facts in contracts

Status: proposed. Issue: #2563. Builds on `contracts_for_extern_and_unsafe_code.md` (#1902), which
has `keeps` for `extern` functions.

Vx has no lifetime syntax. At a call, the borrow checker reads the signature of the called function
and a summary of its body. This note lets a signature state borrow facts as contract clauses:

- `ensures` clauses say what a call does to borrows: what it keeps, and where its result points.
- The summary of a body becomes a set of inferred `ensures` clauses, with the same meaning.
- `requires` and `ensures` clauses on raw pointers say which memory is valid and which memory
  overlaps. The optimizer can use them for alias analysis.

## 1. What the checker cannot know today

`src/hir/provenance.rs` computes a summary of each function before the checker checks any body
(#1095, #2431). For each parameter, the summary says two things:

- whether the result can keep the argument borrowed
- whether the result can hold the borrows that the argument holds

The summary follows a returned path through fields. Through a `&` field, the result holds the
borrows that the argument holds. Through a `&mut` field, the result keeps the argument borrowed.
`VecIter::next` returns `&(*self.vec).data[i]`, through the `&` field `vec`. So two items of one
`v.iter()` can be in use at once, as in Rust.

The summary reads the syntax of bodies and of declarations, such as the types of struct fields. It
has no types of expressions yet. So it cannot follow a method call inside the body that it
summarizes, because it cannot tell which function the call calls. There, and in a few other cases
such as recursion, it says "any parameter".

Nothing computes whether a call keeps a reference argument, for example in a `Vec` that a `&mut`
argument points at. So the checker assumes that a call keeps each reference argument in each `&mut`
argument that can hold a reference. This note calls that assumption the store rule. Two checks use
it:

- E4005 refuses a call that gets a reference to a local variable together with a `&mut` to memory
  that outlives the calling function.
- The fix for #2432 (PR #2541) makes the `&mut` argument keep the borrow while it is in use.

Three functions show the cost:

```vx
import std::vec;

struct Cursor {
  data : &Vec<i64>,
  pos : i32,
}

fn note(out : &mut Vec<&i64>, x : &i64) -> i64 {
  return *x + (out.len() as i64);
}

fn find(v : &Vec<i64>, key : &i64) -> &i64 {
  for i in 0..v.len() {
    if *v.get_ref(i) == *key {
      return v.get_ref(i);
    }
  }
  return v.get_ref(0);
}

fn next(c : &mut Cursor) -> &i64 {
  let r = c.data.get_ref(c.pos);
  c.pos = c.pos + 1;
  return r;
}
```

| Caller | Main at b7cf156b | Why | Rust 1.95 |
| --- | --- | --- | --- |
| `note(out, &local)`, where `out` is a parameter of the caller and `local` is its variable | E4005 at the call | The store rule: `note` can store `x` in `out`. | Accepts |
| `let r = find(&v, &k); k = 7i64;`, then `*r` | E4009 at `k = 7i64` | `find` returns the result of a method call. So `r` keeps `k` borrowed. | Accepts |
| `let a = next(&mut c); let b = next(&mut c);`, then `*a` | E4004 at the second call | `get_ref` is a method call. So `a` keeps `c` borrowed as `&mut`. | Accepts |

A summary has one more limit. It comes from the body, so an edit to a body can change which callers
compile. A clause puts the fact in the signature.

## 2. Three boolean functions for borrows

| Function | It is true when |
| --- | --- |
| `keeps(b)` | After the call, a place outside the call can hold `b`, or a reference that the call got through `b`. For `b = c`, that includes `c.data` and each reference read through it. `b` is a reference parameter. The result does not count here, because the other two functions cover it. |
| `points_into(return, *p)` | A reference in the result can point into the memory that `*p` owns. |
| `reaches(return, *p)` | A reference in the result can point into the memory that `*p` owns, or into memory that `*p` reaches through references, at any depth. |

In `points_into` and `reaches`, `p` is a reference parameter or a path of fields from one, such as
`c.data`.

A clause with `!` narrows what the caller assumes. For example, `ensures !keeps(b)` says that the
call does not keep `b`.

The memory that `*p` owns is `*p` itself and what it owns, such as the buffer of a `Vec`. Memory
that `*p` reaches through a `&` field is not part of it. Memory behind a `&mut` field or a raw
pointer field is part of it. These are the path rules of the summary in section 1. So `*c` owns
`c.pos` and the reference `c.data`, but not the vector that `c.data` points at.

The `&mut` case keeps the checker sound. Take `m : &mut M` with a field `r : &mut i64`. If a result
of `&*m.r` did not keep `m` borrowed, the caller could write `*m.r` while the result is in use. Rust
refuses such a function too (E0621).

Each place in a clause means the place at the start of the call. So `!points_into(return, *c.data)`
is about the vector that `c.data` points at when the call starts. A body can set `c.data` to another
vector and return a reference into it. That vector came from another argument, and the caller keeps
the borrow of that argument.

A later goal is to accept any boolean function in a contract. The functions in this note are
boolean functions that the compiler defines.

## 3. The rules

**No clause means today's rules.** Each function says what a call can do. With no written clause,
the checker uses the inferred clauses of section 4 and the store rule. A clause with `!` removes one
assumption, and the body check proves it. A clause without `!` adds one: the caller assumes the
relation even where an inferred clause rules it out. A library can use that to keep the freedom to
change its body. Neither kind can make the checker unsound. The `extern` note uses the same default
(X1 there).

**The caller reads the clause.**

- `!points_into(return, *p)` removes `*p` from the places that the result can point into. So the
  result does not keep the argument borrowed. It still holds the borrows that `*p` holds.
- `!reaches(return, *p)` removes `*p` and everything that it reaches. So the result does not keep
  the argument borrowed, and it holds none of the borrows that `*p` holds.
- `!points_into(return, *c.data)` removes the vector that `c.data` points at. So the result does not
  hold the borrow that `c.data` makes. It can still hold the other borrows that `*c` holds.
- `!keeps(b)` turns off the store rule for `b`. Then the borrows that `b` gives the call end with
  the call, unless the result holds them.

**The checker checks the body against each clause.** The check has types, so it can follow a
method call. At each call, it uses the clauses of the called function, written and inferred. A
recursive call uses the clauses of the function itself. Where the check cannot follow safe code,
such as a call through a function value, it uses the default for that call, so the clause can fail.
A clause that does not hold is an error.

**Only `unsafe` and `extern` code gets a trusted clause.** An `unsafe` block can hide what the code
does, and an `extern` function has no body. The checker trusts a clause only for those parts, and
it still checks the rest of the body. `vxc --audit` lists each trusted clause (#1902).

**A clause on a trait method binds each impl.** `Iterator::next` can state a clause once, and each
impl must keep it.

**z3 does not read the borrow clauses.** The borrow checker decides `keeps`, `points_into` and
`reaches`. This is also the rule for `atomic` and `writes_only`
(`src/hir/check/atomic_contracts.rs`) and for `reads_only` (#2552). The clauses of section 6 need
both checkers.

## 4. The summary is a set of inferred clauses

The summary of section 1 states the same facts as clauses:

| The summary says, for parameter `p` | The inferred clause |
| --- | --- |
| The result cannot keep the argument borrowed, but it can hold the borrows that the argument holds. | `!points_into(return, *p)` |
| The result can do neither. | `!reaches(return, *p)` |

The summary uses the path rules of section 2, so an inferred clause means the same as a written
one. A caller uses both in the same way. For example, the compiler infers
`!points_into(return, *self)` for `VecIter::next`. That clause is why two items of one `v.iter()`
can be in use at once, and why `Iterator::last` compiles.

An inferred clause needs no body check. The summary says `!` only where the syntax of the body
shows the path, so the clause holds by construction.

**For each place, a written clause decides.** For a place with no written clause, an inferred clause
decides. With neither, the default holds. So a written `!` clause adds a fact that the summary
cannot find, such as one behind a method call. A written clause without `!` overrides an inferred
`!` clause for its place.

**The compiler shows what it inferred.** `vxc` prints the inferred clauses of a function on request.
A programmer can then copy a clause into the signature, to make it part of the contract.

Inference of `!keeps(b)` can follow later. The same walk over the syntax can find a body that stores
`b` nowhere. It also stops at a method call.

## 5. The examples with clauses

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn note(out : &mut Vec<&i64>, x : &i64) -> i64
    ensures !keeps(x)
{ .. }

fn find(v : &Vec<i64>, key : &i64) -> &i64
    ensures !reaches(return, *key)
{ .. }

fn next(c : &mut Cursor) -> &i64
    ensures !points_into(return, *c)
{ .. }
```

- `note(out, &local)` compiles. The body stores `x` nowhere, so the clause holds.
- After `find(&v, &k)`, the result keeps only `v` borrowed, so `k = 7i64` compiles. The body returns
  a result of `v.get_ref`, which points into `*v` and does not reach `*key`.
- `next(&mut c)` gives a result that holds the borrow of `v` that `c` holds, and no borrow of `c`.
  So the second call compiles, and `v` stays borrowed while `a` is in use. The body returns a result
  of `c.data.get_ref`, which points into `*c.data`. `*c` does not own that vector, so the clause
  holds.

## 6. Raw pointers: `valid` and `overlaps`

| Function | It is true when |
| --- | --- |
| `valid(p, n)` | `p` is not null, has the alignment of its type, and points at allocated memory for `n` elements. The elements do not have to hold values yet, so the output buffer of a copy is valid. |
| `overlaps(p, q, n)` | The first `n` elements at `p` and the first `n` elements at `q` share at least one byte. |

`p` and `q` are raw pointers. For references, the types already give these facts: a `&` is valid,
and a `&mut` overlaps nothing else that is in use.

In a `requires`, a clause holds at the start of the call. In an `ensures`, it holds when the call
returns. So a body that frees `p` can still state `requires valid(p, n)`.

Both functions describe only the first `n` elements, and `n` can be 0. So a fact applies only to an
access that the compiler proves is in the first `n` elements.

**`requires` and `ensures` differ in who proves the clause and who relies on it.**

| | Who proves it | Who relies on it | Where the error appears |
| --- | --- | --- | --- |
| `requires` | the caller, at each call | the body | at the call |
| `ensures` | the body | each caller | in the body |

The borrow clauses of section 2 are all `ensures`, because they say what a call does. The caller's
duty comes from them. If a call can keep `x` in `*out`, then `x` must live as long as `*out`. The
E4005 error at `note(out, &local)` is that duty failing. So safe code needs no borrow `requires`.

**A `requires` helps the body.** `requires !overlaps(dst, src, n)` lets a copy assume that its two
buffers are disjoint, as `memcpy` does.

- The code generator can then put each access of the first `n` elements of `dst` and of `src` in
  different alias scopes. The flat code generator already emits `alias_scopes` and `noalias_scopes`
  for disjoint `&mut` fields (#275).
- `requires valid(p, n)` lets the code generator tell LLVM that `p` is not null. LLVM can then load
  an element at index `i < n` before the program needs it, for example out of a loop.

**An `ensures` helps the caller.** `ensures valid(return, n)` and `ensures !overlaps(return, p, n)`
tell the caller that the result is valid and disjoint from `p` in the first `n` elements. The
caller's alias analysis can then keep an element of `p` in a register across a write to the same
index through the result.

**The caller proves a `requires` where it can.** The proof needs both checkers. The borrow checker
can prove that `p` and `q` come from two places that it knows are disjoint, such as two different
variables. For that, it must record where a raw pointer comes from, which it does not do today. z3
proves the bounds on `n`, as it proves other `requires`. Inside an `unsafe` block, the checker
trusts a `requires` that it cannot prove, and `vxc --audit` lists it. The wrapper of a strict
`extern` block also checks it at run time (B3 in #1902).

`overlaps` and `points_into` answer different questions. `points_into` says which borrow a result
keeps, for the borrow checker. `overlaps` says which addresses two pointers share, for the
optimizer. Two fields of one struct do not overlap, but a reference into either one points into
the struct.

## 7. Rust lifetimes and the clauses

| Rust | Vx |
| --- | --- |
| `fn set_r<'a>(h: &mut Holder<'a>, b: &'a i64)` | `fn set_r(h : &mut Holder, b : &i64)`, with no clause |
| `fn note(out: &mut Vec<&i64>, x: &i64) -> i64` | `ensures !keeps(x)` |
| `fn find<'a>(v: &'a Vec<i64>, key: &i64) -> &'a i64` | `ensures !reaches(return, *key)` |
| `fn next<'a>(c: &mut Cursor<'a>) -> &'a i64` | `ensures !points_into(return, *c)` |
| `/// # Safety` text on an `unsafe fn` | `requires valid(p, n)`, `requires !overlaps(p, q, n)` |

The defaults are opposite. In Rust, two lifetimes that a signature does not relate stay unrelated.
So the body must not store or return through them, and `set_r` with no lifetimes is an error in its
body. In Vx, the caller assumes each relation, unless an inferred clause or a written `!` clause
rules it out. So a function can get such a clause later, and each caller that compiled before still
compiles.

Rust can also state a fact about a function value, with `for<'a> Fn(&'a T) -> &'a U`. A closure type
such as `Closure1<..>` has no place for a clause, and it has no body for a summary. Clauses on
function types can come later.

## 8. The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | A written clause is part of the signature, so a caller needs no body for it. An inferred clause comes from the body (open question 3). |
| A body declares nothing global | Clauses are on declarations. The compiler infers clauses from syntax before it checks any body. A body check reads only frozen facts: the written and inferred clauses of the functions that it calls. |

## Open questions

1. The error code for a body that breaks a clause. E8007 covers access clauses today. A code next to
   E4005 would keep the borrow errors in one family.
1. A second argument for `keeps`, such as `!keeps(b, *h)`, to say where a call can keep `b`.
1. Whether a public function of a library must write its clauses, so that an edit to its body
   cannot change which callers compile. Inferred clauses would then serve only private functions.
1. A form of `overlaps` for two lengths or two element types.
1. A way to say that a result overlaps no memory that the caller can reach, as for an allocator.
   C has the `malloc` attribute for this.
1. A separate `initialized(p, n)`, for a body that reads the elements that `valid` gives it.

## The work, in order

1. Parse `keeps` and `points_into` in the `ensures` of a Vx function. z3 skips them, as it skips
   `atomic`.
1. The caller side of `!points_into`: combine the written clauses with the inferred ones where a
   call reads them (`src/hir/check/calls.rs`).
1. The caller side of `!keeps`: the two checks of the store rule.
1. The body side: check each written clause with `ref_provenance_of`. List the trusted clauses in
   `vxc --audit`.
1. Print the inferred clauses of a function on request.
1. `valid` and `overlaps` in `requires` and `ensures` of `unsafe` and `extern` functions, with the
   alias scopes and LLVM attributes that they allow.
1. Clauses on trait methods, and inference of `!keeps(b)`.
1. Later: clauses on function types.
