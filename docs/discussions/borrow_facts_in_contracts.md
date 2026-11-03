# Borrow facts in contracts

Status: proposed. Issue: #2563. Builds on `contracts_for_extern_and_unsafe_code.md` (#1902), which
has `keeps` for `extern` functions.

Vx has no lifetime syntax. At a call, the borrow checker reads the signature of the called function
and a summary of its body. This note lets a signature state borrow facts as contract clauses:

- `ensures` clauses say what a call does to borrows: what it keeps, and where its result points.
- The summary of a body becomes a set of inferred `ensures` clauses, with the same meaning.
- `requires` and `ensures` clauses on raw pointers say which memory is valid, which memory holds
  values, and which memory overlaps. The optimizer can use them for alias analysis.

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
| `points_into(return, *p)` | A reference in the result can point into the memory under `*p`. |
| `reaches(return, *p)` | A reference in the result can point into the memory under `*p`, or into memory that `*p` reaches through references, at any depth. |

In `points_into` and `reaches`, `p` is a reference parameter or a path of fields from one, such as
`c.data`.

A clause with `!` narrows what the caller assumes. For example, `ensures !keeps(b)` says that the
call does not keep `b`.

The memory under `*p` is `*p` itself, and the memory behind each `&mut` field and each raw pointer
field of it, at any depth. So the buffer of a `Vec` or of a `Box` is under it. Memory behind a `&`
field is not under `*p`. `*p` reaches that memory. These are the path rules of the summary in
section 1. So `c.pos` and the reference `c.data` are under `*c`, and the vector that `c.data` points
at is not.

A raw pointer does not own its memory. The checker counts that memory under `*p` because no other
borrow protects it. So `v.get_ref(0)` keeps `v` borrowed, and `v.push` cannot move the buffer while
the result is in use.

The `&mut` case keeps the checker sound in the same way. Take `m : &mut M` with a field
`r : &mut i64`. If a result of `&*m.r` did not keep `m` borrowed, the caller could write `*m.r`
while the result is in use. Rust refuses such a function too (E0621).

The rules apply at each level. For `b : &Box<&i64>`, the buffer of the box is under `*b`, and the
`i64` that the buffer points at is not. So a function that returns the `&i64` from the buffer can
state `!points_into(return, *b)`, and a function that returns a reference into the buffer cannot. A
clause on a generic function must hold for each type argument, because a type argument can bring
its own `&mut` and raw pointer fields.

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
  hold the borrow that `c.data` makes. It can still hold the other borrows that `*c` holds. Today
  the checker keeps one set of borrows for a struct value (section 7), so it cannot use this clause
  yet.
- `!keeps(b)` turns off the store rule for `b`. Then the borrows that `b` gives the call end with
  the call, unless the result holds them.

**The checker checks the body against each clause.** The check has types, so it can follow a
method call. At each call, it uses the clauses of the called function, written and inferred. A
recursive call uses the clauses of the function itself. Where the check cannot follow safe code,
such as a call through a function value, it uses the default for that call, so the clause can fail.
A clause that does not hold is an error.

**Only `unsafe` and `extern` code gets a trusted clause.** An `unsafe` block can hide what the code
does, and an `extern` function has no body. The checker trusts a clause only for those parts, and
it still checks the rest of the body. `vxc --audit` lists each trusted clause (#1902). The list
tells a person what to check. It proves nothing.

**A clause on a trait method binds each impl.** `Iterator::next` can state a clause once, and each
impl must keep it.

**Tasks change nothing at a call.** A `Task<T>` cannot leave its `tasks` block (T3 in
`cpu_tasks_send_sync_and_shared_atomics.md`). So each task that a callee starts ends before the
callee returns, and a clause describes the call as a whole. Inside a task body, `!keeps(b)` ends
only the borrow that the call makes. The task's own access to a local still lasts from the `spawn`
to the join (T5 there). The result of `join()` holds the borrows that the result of the task holds.
For a task `spawn { v.get_ref(0) }`, the checker gives E4003 today at a `v.push` after
`let r = h.join();` while `r` is in use. A `spawn on` follows the same rules. Outside a `tasks`
block, it is a task that ends before the next statement (T8 there).

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

**Written clauses about a parameter replace the inferred ones.** If a written clause names a
parameter `p`, or a path from `p`, the inferred clauses about `p` do not apply. For `p`, the caller
then uses only the written clauses and the default. For each other parameter, the inferred clauses
apply. So a programmer who writes one clause about `p` writes each clause about `p` that the callers
need.

- A written `!` clause can add a fact that the summary cannot find, such as one behind a method
  call.
- A written clause without `!` keeps the default for its parameter.
- Two written clauses that contradict each other are an error, for example
  `points_into(return, *p)` and `!reaches(return, *p)`.

**The compiler shows what it inferred.** `vxc` prints the inferred clauses of a function on request.
A programmer can then copy a clause into the signature, to make it part of the contract.

**A private function gets inferred clauses.** Each caller of a private function is in the same
module and in the same compile. So when an edit to its body breaks a caller, the error appears at
once, to the person who made the edit. Most functions are small private helpers, so inference saves
the most clauses there. Vx has no private functions yet: another module can import each function.
`pub fn` only exports a function to C (`docs/lang/abi.md`). Until Vx has private functions, each
function gets inferred clauses. Open question 3 asks what a public function gets after that.

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
  of `c.data.get_ref`, which points into `*c.data`. That vector is not under `*c`, so the clause
  holds.

## 6. Raw pointers: `valid`, `initialized` and `overlaps`

| Function | It is true when |
| --- | --- |
| `valid(p, n)` | `p` is not null, has the alignment of its type, and the first `n` elements at `p` are in one live allocation. `valid(p, 0)` says only that `p` is not null and aligned. |
| `initialized(p, n)` | Each of the first `n` elements at `p` holds a value of its type. A body needs it to read an element. |
| `overlaps(p, q, n)` | The first `n` elements at `p` and the first `n` elements at `q` share at least one byte. |

`p` and `q` are raw pointers. For a negative `n`, all three functions are false. So
`requires valid(p, n)` also requires `n >= 0`. For references, the types already give these facts:
a `&` is valid and initialized, and a `&mut` overlaps nothing else that is in use.

Each function states one fact, and a clause can combine them:

- `valid` is about addresses: the bounds of the allocation and the alignment. The output buffer of
  a copy is valid before the copy writes it.
- `initialized` is about values.
- `!overlaps(p, q, n)` is about `p` and `q` only. It says nothing about other pointers or about
  other tasks.
- Permission to write comes from the pointer type, `*mut T` and not `*const T`. No clause gives
  it.
- No clause says that the elements keep their values. A write can change them.

In a `requires`, a clause holds at the start of the call. In an `ensures`, it holds when the call
returns. So a body that frees `p` can still state `requires valid(p, n)`.

The functions describe only the first `n` elements, and `n` can be 0. So a fact applies only to an
access that the compiler proves is in the first `n` elements.

**`requires` and `ensures` differ in who proves the clause and who relies on it.**

| | Who proves it | Who relies on it | Where the error appears |
| --- | --- | --- | --- |
| `requires` | the caller, at each call | the body | at the call |
| `ensures` | the body | each caller | in the body |

The borrow clauses of section 2 are all `ensures`, because they say what a call does. The caller's
duty comes from them. If a call can keep `x` in `*out`, then `x` must live as long as `*out`. The
E4005 error at `note(out, &local)` is that duty failing. So safe code needs no borrow `requires`.

**A `requires` helps the body, and an `ensures` helps the caller.**

- `requires !overlaps(dst, src, n)` lets a copy assume that its two buffers are disjoint, as
  `memcpy` does.
- `ensures valid(return, n)` and `ensures !overlaps(return, p, n)` tell the caller that the result
  is valid and disjoint from `p` in the first `n` elements. The caller can then keep an element of
  `p` in a register across a write to the same index through the result.

**What the code generator may emit for LLVM:**

| Clause | LLVM output |
| --- | --- |
| `valid(p, n)` | `nonnull` and `align` on `p`. Also an `llvm.assume` with a `"dereferenceable"` bundle of `n` times the size of an element. It goes at the start of the body for a `requires`, and after the call for an `ensures`. The bundle can take a size that the program computes at run time. The `dereferenceable` attribute takes only a constant. |
| `!overlaps(p, q, n)` | Alias scopes on each access through `p` or `q` that the compiler proves is in the first `n` elements. The flat code generator already emits `alias_scopes` and `noalias_scopes` for disjoint `&mut` fields (#275). |
| `initialized(p, n)` | Nothing. It lets the body read the elements. |

LLVM reads a `dereferenceable` fact at one point only, so a body that frees `p` does not break it.
LLVM decides itself whether it can load an element early, for example out of a loop. `nonnull`
alone does not let it do that. The code generator never puts `noalias` on a parameter for
`!overlaps`. LLVM's `noalias` also rules out accesses through each other pointer during the call,
and `!overlaps` does not say that. The
[LLVM reference](https://llvm.org/docs/LangRef.html#parameter-attributes) defines these attributes.

**A fact is about values.** `valid(p, n)` is about the values that `p` and `n` have where the fact
holds. A copy of `p` has the fact too. An assignment to `p` gives `p` a new value, and the fact
stays with the old value.

**Where a fact starts.** The checker gets a raw-pointer fact only from these sources:

- a `requires` of the function, at the start of the body
- an `ensures` of a call, when the call returns
- a raw pointer made from a reference, which is valid and initialized for one element
- a smaller form of a known fact. z3 gets `valid(p, m)` from `valid(p, n)` when `0 <= m <= n`. It
  also gets `valid` for the pointer `k` elements after `p`, with `n - k` elements, when
  `0 <= k <= n`. `initialized` works in the same way. `!overlaps(p, q, n)` gives `!overlaps` for
  the pointers `j` elements after `p` and `k` elements after `q`, with `m` elements, when `j`, `k`
  and `m` are at least 0, `j + m <= n` and `k + m <= n`.

A proof that `n` is in bounds does not show that `p` points at live memory. So a pointer made from
an integer, or read from memory such as an atomic, has no fact. Only a trusted clause of `unsafe` or
`extern` code can give it one.

**Where a fact ends.** A fact ends at the first point where code can free, move or reallocate the
memory. The first version ends it at these points:

- a `&mut` use, a move or a drop of the place that the pointer came from, such as `v.push(1i64)`
  after `v.as_mut_ptr()`. A `&` use keeps the fact, unless the type lets code write through a `&`,
  as `Cell` does.
- a `&mut` use, a move or a drop of each place that can hold a raw pointer, such as a `Box`, when
  the checker does not know which place the pointer came from. A pointer from a parameter or from
  a call result is such a pointer. So in a body with `requires valid(p, 1)`, an assignment
  `*b = Box::new(2i64)` ends the fact, because it frees the old box.
- a call that gets a `&mut`, a raw pointer, or a `&` to a type such as `Cell`
- a call that the checker cannot see into, such as a call of an `extern` function or of a function
  value
- a `spawn` whose task captures such a value. The fact stays ended after the join, because the task
  can run at any time between the two.

A callee that keeps the memory valid states `ensures valid(p, n)`, and the fact starts again when
the call returns. The caller proves a `requires` after it evaluates the arguments. So in
`f(p, &mut v)`, the `&mut v` ends the facts about a `p` that came from `v`, and the
`requires valid(p, n)` of `f` fails.

**What protects the memory during a call.** During a call, two kinds of code run: the callee, and
the tasks that run at the same time as the caller. The body check applies the rules above to the
callee. For tasks, the borrow rules of T5 apply. A task that holds `&mut` to the place that `p` came
from ended the fact at its `spawn`. Until the join, the caller cannot make a new raw pointer from
that place. A task body, also a `spawn on` body, starts with no raw-pointer fact of its parent. So
the safe code of a task can free memory only through a place that it holds, and T5 covers those
places. To free memory through a raw pointer, a task needs `unsafe` code, and the audit lists it.
Tasks that share a raw pointer, for example through an atomic, get no fact from it. An atomic swap
of the pointer does not keep the old memory alive, so only a trusted clause can state a fact about
that memory.

**Three calls.**

<!-- vx-doctest: skip (proposed syntax) -->

```vx
extern "C" {
  fn copy_i64(dst : *mut i64, src : *const i64, n : i64) -> void
      requires valid(dst, n)
      requires valid(src, n)
      requires initialized(src, n)
      requires !overlaps(dst, src, n)
      ensures initialized(dst, n);
}

fn copy_prefix(dst : *mut i64, cap : i64, src : *const i64, n : i64) -> bool
    requires valid(dst, cap)
    requires valid(src, n)
    requires initialized(src, n)
    requires !overlaps(dst, src, n)
{
  if n > cap {
    return false;
  }
  copy_i64(dst, src, n);
  return true;
}
```

- The call in `copy_prefix` compiles. `valid(src, n)` gives `0 <= n`, and the `if` gives
  `n <= cap`. So z3 gets `valid(dst, n)` from `valid(dst, cap)`.
- Without the `if`, the call is an error. The checker cannot prove `n <= cap`.
- In `unsafe { copy_i64(q, src, 4i64) }`, where `q` came from an integer, the checker has no fact
  about `q`. It trusts each clause that it cannot prove, and `vxc --audit` lists those clauses for a
  person to check.

**The caller proves a `requires` with both checkers.** The borrow checker tracks where each raw
pointer comes from and which events end its facts. It does not do that today. z3 proves the bounds
on `n`, as it proves other `requires`. The wrapper of a strict `extern` block also checks a
`requires` at run time (B3 in #1902).

`overlaps` and `points_into` answer different questions. `points_into` says which borrow a result
keeps, for the borrow checker. `overlaps` says which addresses two pointers share, for the
optimizer. Two fields of one struct do not overlap, but a reference into either one points into
the memory under the struct.

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

**What the clauses cannot say yet.** Rust lifetimes can state three facts that this design leaves
for later:

- Two references in one result, each with its own borrow. With `struct Pair<'a, 'b>`, Rust accepts
  `let p = pick(&a, &b); let y = p.y; a = 5;` and a later `*y`. Vx gives E4009 at `a = 5i64`, also
  when `p` is a struct literal. The checker keeps one set of borrows for a struct value. A place
  such as `return.y` in a clause needs a set for each field first.
- Where a call keeps a reference (open question 2).
- A fact about a function value, as in `for<'a> Fn(&'a T) -> &'a U`. A closure type such as
  `Closure1<..>` has no place for a clause, and it has no body for a summary. So a call of a
  function value uses the default.

## 8. The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | A written clause is part of the signature, so a caller needs no body for it. An inferred clause comes from the body. Its freeze lets each body check run in parallel, but an edit to the body can still change which callers compile (open question 3). |
| A body declares nothing global | Clauses are on declarations. The compiler infers clauses from syntax before it checks any body. A body check reads only frozen facts: the written and inferred clauses of the functions that it calls. |

## Open questions

1. The error code for a body that breaks a clause. E8007 covers access clauses today. A code next to
   E4005 would keep the borrow errors in one family.
1. A second argument for `keeps`, such as `!keeps(b, *h)`, to say where a call can keep `b`.
1. What a public function of a library gets. An edit to its body can break callers in other
   packages, and the author does not see those errors. One choice is that the function writes its
   clauses and gets the default where it writes none. Then the standard library must write
   `!points_into(return, *self)` on `VecIter::next`, or `v.iter()` loses what #2431 gave it. The
   other choice keeps the inferred clauses, and a check warns when they change between two versions
   of the package, as `cargo-semver-checks` does for Rust. The question matters once Vx has private
   functions and compiles packages separately. Today the standard library compiles with each
   program.
1. A form of `overlaps` for two lengths or two element types.
1. A way to say that a result overlaps no memory that the caller can reach, as for an allocator.
   C has the `malloc` attribute for this.
1. A fact for access that no other pointer shares during the call. It would let the code generator
   put `noalias` on a parameter.

## The work, in order

1. Parse `keeps`, `points_into` and `reaches` in the `ensures` of a Vx function. z3 skips them, as
   it skips `atomic`.
1. The caller side of `!points_into`: combine the written clauses with the inferred ones where a
   call reads them (`src/hir/check/calls.rs`).
1. The caller side of `!keeps`: the two checks of the store rule.
1. The body side: check each written clause with `ref_provenance_of`. List the trusted clauses in
   `vxc --audit`.
1. Print the inferred clauses of a function on request.
1. Track where each raw pointer comes from, and the events that end its facts.
1. `valid`, `initialized` and `overlaps` in `requires` and `ensures`, with the LLVM output of
   section 6.
1. Clauses on trait methods, and inference of `!keeps(b)`.
1. Later: a set of borrows for each field of a struct value, then places such as `return.y` in
   clauses. Clauses on function types.
