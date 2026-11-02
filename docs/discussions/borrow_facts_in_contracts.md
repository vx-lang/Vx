# Borrow facts in contracts

Status: proposed. Issue: #2563. Builds on `contracts_for_extern_and_unsafe_code.md` (#1902), which
has `keeps` for `extern` functions.

Vx has no lifetime syntax. At a call, the borrow checker reads the signature of the called function
and a summary of its body. This note lets a signature state borrow facts as `ensures` clauses. A
clause answers where the summary stops. It also states a fact that nothing computes today: whether
a call keeps a reference argument.

## 1. What the checker cannot know today

`src/hir/provenance.rs` computes a summary of each function before the checker checks any body
(#1095, #2431). For each parameter, the summary says two things:

- whether the result can keep the argument borrowed
- whether the result can hold the borrows that the argument holds

The summary follows a returned path through fields. Through a `&` field, the result holds the
borrows that the argument holds. Through a `&mut` field, the result keeps the argument borrowed.
`VecIter::next` returns `&(*self.vec).data[i]`, through the `&` field `vec`. So two items of one
`v.iter()` can be in use at once, as in Rust.

The summary reads only syntax, because the checker has no types yet. So it cannot follow a method
call inside the body that it summarizes. There, and in a few other cases such as recursion, it says
"any parameter".

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

## 2. Two boolean functions

| Function | It is true when |
| --- | --- |
| `keeps(b)` | After the call, a place outside the call can hold `b`, or a reference that the call got through `b`. For `b = c`, that includes `c.data` and each reference read through it. `b` is a reference parameter. The result does not count here, because `points_into` covers it. |
| `points_into(return, *p)` | A reference in the result can point into the memory that `*p` owns. `p` is a reference parameter, or a reference field such as `c.data`. |

A clause with `!` narrows what the caller assumes. For example, `ensures !keeps(b)` says that the
call does not keep `b`.

The memory that `*p` owns is `*p` itself and what it owns, such as the buffer of a `Vec`. Memory
that `*p` reaches through a `&` field is not part of it. Memory behind a `&mut` field or a raw
pointer field is part of it, as in the summary of #2431. So `*c` owns `c.pos` and the reference
`c.data`, but not the vector that `c.data` points at.

The `&mut` case keeps the checker sound. Take `m : &mut M` with a field `r : &mut i64`. If a result
of `&*m.r` did not keep `m` borrowed, the caller could write `*m.r` while the result is in use. Rust
refuses such a function too (E0621).

Each place in a clause means the place at the start of the call. So `!points_into(return, *c.data)`
is about the vector that `c.data` points at when the call starts. A body can set `c.data` to another
vector and return a reference into it. That vector came from another argument, and the caller keeps
the borrow of that argument.

A later goal is to accept any boolean function in a contract. `keeps` and `points_into` are two
boolean functions that the compiler defines, and the borrow checker decides them.

## 3. The rules

**No clause means today's rules.** Both functions say what a call can do. With no clause, the
checker uses the summary for `points_into` and the store rule for `keeps`. A clause with `!` removes
one assumption, and the body check proves it. A clause without `!` adds one: the caller assumes the
relation even where the summary rules it out. A library can use that to keep the freedom to change
its body. Neither kind can make the checker unsound. The `extern` note uses the same default (X1
there).

**The caller reads the clause.**

- `!points_into(return, *p)` removes `*p` from the places that the result can point into. So the
  result does not keep the argument borrowed. It still holds the borrows that `*p` holds.
- `!points_into(return, *c.data)` removes the vector that `c.data` points at. So the result does not
  hold the borrow that `c.data` makes.
- `!keeps(b)` turns off the store rule for `b`. Then the borrows that `b` gives the call end with
  the call, unless the result holds them.

**The checker checks the body against each clause.** The check has types, so it can follow a
method call. At each call, it uses the clauses and the summary of the called function. A recursive call uses the
clauses of the function itself. Where the check cannot follow safe code, such as a call through a
function value, it uses the default for that call, so the clause can fail. A clause that does not
hold is an error.

**Only `unsafe` and `extern` code gets a trusted clause.** An `unsafe` block can hide what the code
does, and an `extern` function has no body. The checker trusts a clause only for those parts, and
it still checks the rest of the body. `vxc --audit` lists each trusted clause (#1902).

**A clause on a trait method binds each impl.** `Iterator::next` can state a clause once, and each
impl must keep it.

**z3 does not read these clauses.** The borrow checker decides them. This is also the rule for
`atomic` and `writes_only` (`src/hir/check/atomic_contracts.rs`) and for `reads_only` (#2552).

## 4. The examples with clauses

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn note(out : &mut Vec<&i64>, x : &i64) -> i64
    ensures !keeps(x)
{ .. }

fn find(v : &Vec<i64>, key : &i64) -> &i64
    ensures !points_into(return, *key)
{ .. }

fn next(c : &mut Cursor) -> &i64
    ensures !points_into(return, *c)
{ .. }
```

- `note(out, &local)` compiles. The body stores `x` nowhere, so the clause holds.
- After `find(&v, &k)`, the result keeps only `v` borrowed, so `k = 7i64` compiles. The body returns
  a result of `v.get_ref`, which points into `*v` and not into `*key`.
- `next(&mut c)` gives a result that holds the borrow of `v` that `c` holds, and no borrow of `c`.
  So the second call compiles, and `v` stays borrowed while `a` is in use. The body returns a result
  of `c.data.get_ref`, which points into `*c.data`. `*c` does not own that vector, so the clause
  holds.

## 5. Rust lifetimes and the clauses

| Rust | Vx |
| --- | --- |
| `fn set_r<'a>(h: &mut Holder<'a>, b: &'a i64)` | `fn set_r(h : &mut Holder, b : &i64)`, with no clause |
| `fn note(out: &mut Vec<&i64>, x: &i64) -> i64` | `ensures !keeps(x)` |
| `fn find<'a>(v: &'a Vec<i64>, key: &i64) -> &'a i64` | `ensures !points_into(return, *key)` |
| `fn next<'a>(c: &mut Cursor<'a>) -> &'a i64` | `ensures !points_into(return, *c)` |

The defaults are opposite. In Rust, two lifetimes that a signature does not relate stay unrelated.
So the body must not store or return through them, and `set_r` with no lifetimes is an error in its
body. In Vx, the caller assumes each relation, unless the summary or a clause with `!` rules it
out. So a function can get such a clause later, and each caller that compiled before still compiles.

Rust can also state a fact about a function value, with `for<'a> Fn(&'a T) -> &'a U`. A closure type
such as `Closure1<..>` has no place for a clause, and it has no body for a summary. Clauses on
function types can come later.

## 6. The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | A clause is part of the signature, so a caller needs no body for it. Without a clause, a caller still reads the summary (open question 3). |
| A body declares nothing global | Clauses are on declarations. A body check reads only frozen facts: the signatures and summaries of the functions that it calls. |

## Open questions

1. The error code for a body that breaks a clause. E8007 covers access clauses today. A code next to
   E4005 would keep the borrow errors in one family.
1. A second argument for `keeps`, such as `!keeps(b, *h)`, to say where a call can keep `b`.
1. Whether a public function of a library must state its clauses, so that an edit to its body
   cannot change which callers compile. The summary would then serve only private functions.

## The work, in order

1. Parse `keeps` and `points_into` in the `ensures` of a Vx function. z3 skips them, as it skips
   `atomic`.
1. The caller side of `!points_into`: change the summary where a call reads it
   (`src/hir/check/calls.rs`).
1. The caller side of `!keeps`: the two checks of the store rule.
1. The body side: check each clause with `ref_provenance_of`. List the trusted clauses in
   `vxc --audit`.
1. Clauses on trait methods.
1. Later: clauses on function types.
