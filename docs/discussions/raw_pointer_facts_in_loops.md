# Raw-pointer facts in loops

Status: proposed, for review. Issues: #2830, #2831, #2837 and #1358. Part of #2829.

`raw_pointer_facts.md` describes v1 of the raw-pointer facts. Its section 11 lists three limits
that stop each loop over a buffer from giving a fact:

- A loop that moves a pointer, as in `q = offset(q, 1i64)`, loses the facts of `q` (limit 2).
- A write through a pointer gives no `Init` fact (limit 5).
- One fact must prove a clause, so two ranges do not join (limit 4).

This note removes the three limits. It also says what the checker knows after a loop (#1358). The
section numbers in the text are those of `raw_pointer_facts.md`, unless the text says "this note".

## 1. The examples

<!-- vx-doctest: skip (proposed syntax) -->

```vx
import core::ptr;

fn first(p : *const i64) -> i64
requires valid(p, 1i64)
requires initialized(p, 1i64)
{
  return unsafe { p[0] };
}

fn sum(p : *const i64, n : i64) -> i64
requires valid(p, n)
requires initialized(p, n)
{
  let mut total = 0i64;
  for i in 0i64..n {
    total = total + unsafe { p[i] };
  }
  return total;
}

// A loop that moves a pointer.
fn walk(p : *const i64, n : i64) -> i64
requires valid(p, n)
requires initialized(p, n)
{
  let mut q = p;
  let mut total = 0i64;
  let mut i = 0i64;
  while i < n invariant 0i64 <= i invariant valid(q, n - i) invariant initialized(q, n - i) {
    total = total + first(q);
    q = offset(q, 1i64);
    i = i + 1i64;
  }
  return total;
}

// A loop that fills a buffer.
fn fill_then_sum(p : *mut i64, n : i64) -> i64
requires valid(p, n)
{
  for i in 0i64..n invariant initialized(p, i) {
    unsafe { p[i] = 1i64; }
  }
  return sum(p as *const i64, n);
}
```

Today each invariant with a pointer clause gets E8008 and E8009, and each call gets E8006. With
this note, both functions compile.

## 2. What comes first

#2842 fixes #2827 and #2838. With it, the checker proves each invariant at the start of each pass:

- When the loop starts. A `for i in a..b` loop proves it with the fact `i == a`.
- At the end of the body and at each `continue`. A `for` loop proves it for `i + 1`.
- Before these two proofs, each variable of the body gets a new name in the facts, as at an
  assignment. So a `let` in the body that hides a variable of the invariant cannot prove it.

The pointer clauses in this note use the same three proofs.

## 3. An assignment keeps the pointer facts about the old value

Today an assignment `i = i + 1i64` drops each pointer fact whose expressions name `i` (section 3).
Since #2800, the prover's facts keep the old value of `i` under a new name, `$old_N`. The pointer
facts do the same:

- Before the assignment, each `lo`, `hi` and `len` of a fact gets the new name of the old value in
  place of `i`. The new name is the one that the prover's facts get, so the two sets of facts
  meet.
- An expression that still names `i` after the change ends its fact, as it does today.
- An assignment inside a call frame still ends the facts, as for the prover's facts.

In `walk`, `Valid(b, 0, n - i)` becomes `Valid(b, 0, n - $old_1)`, and the prover knows
`i == $old_1 + 1`.

The proofs of section 2 give the variables of the body new names in the pointer facts too.

## 4. A write gives an `Init` fact (#2830)

A write `p[k] = x` or `*q = x` in an `unsafe` block adds `Init(b, j + k, j + k + 1)`. Here the
pointer has the base `b` and the offset `j`.

- The checker adds the fact only when the prover models `k`.
- The compiler trusts the write, as it trusts the rest of the `unsafe` block. The write does not
  need a proof of `valid`.
- The write keeps the other facts of the pointer, as it does today. A write does not free memory,
  and it leaves a value of the type in the element.
- For an element type with a drop, the write is an event (section 5). The checker adds the new
  fact after the event.

## 5. Two facts can prove a clause (#2831)

Today the goal of section 7 uses one fact. A goal can now also use two facts of the same kind and
the same base:

| Clause | It also holds when the prover proves the goal for two facts `F1(b, lo1, hi1)` and `F2(b, lo2, hi2)` |
| --- | --- |
| `initialized(p, m)` | `0 <= m && lo1 <= k && lo1 <= lo2 && lo2 <= hi1 && k + m <= hi2` |
| `valid(p, m)` | `0 <= m && lo1 <= k && lo1 <= lo2 && lo2 < hi1 && k + m <= hi2` |

The ranges of two `Init` facts can touch. The ranges of two `Valid` facts must share an element.
Two allocations can sit next to each other in memory, so two `Valid` ranges that only touch can be
in two allocations. Then an `offset` from one into the other has undefined behavior.

The checker tries each pair of facts of the base. It does not join three or more facts. A loop
needs one pair: the invariant `Init(b, 0, i)` and the write `Init(b, i, i + 1)`.

Also, `initialized(p, m)` holds with no fact when the prover proves `m == 0`. Zero elements hold
values. A negative `m` still needs a fact, as today. `fill_then_sum` needs this rule when its loop starts. `valid(p, 0)` still needs a fact,
because it says that `p` is not null and is aligned.

## 6. A pointer clause in an invariant (#2837)

An `invariant` can be a pointer clause: `valid`, `initialized` or `!overlaps`. The checker splits
the invariants of a loop. The prover gets the other invariants, as today. The tracker gets the
pointer clauses.

**When the loop starts**, the checker proves each pointer clause from the pointer facts, as it
proves a `requires` at a call (section 7). If it cannot, it reports E8008. If an event ended a
fact that proves the clause, E8008 gets the note that E8006 gets.

**At the start of the body**, the loop rule of section 6 ends each fact that the body can end. It
also drops the entry of each variable that the body assigns. Then each pointer clause adds its
facts, as a `requires` does at the start of a body (section 4):

- A pointer that the body does not assign keeps its base. The clause adds its facts to that base.
- A pointer that the body assigns gets a new base. The memory of the new base is the memory of the
  base that the pointer had when the loop started.

**At the end of the body and at each `continue`**, the checker proves each pointer clause again.
It proves it from the pointer facts at that point, after the new names of section 2. If it cannot,
it reports E8009. For a pointer that the body assigns, the proof also needs the memory of its base
at that point to be part of the memory of the new base that the pass started with. Otherwise the next pass can use a
fact that a later event does not end. For example, the body can make `q` point into a local
variable.

In `walk`, the end of the body has `q` at the base `b` with the offset 1, and the fact
`Valid(b, 0, n - $old_1)`. The goal for `valid(q, n - i)` is
`0 <= 1 && 0 <= n - i && 1 + (n - i) <= n - $old_1`. It holds because `i == $old_1 + 1` and
`i <= n`.

## 7. After a loop (#1358)

Today, after a loop, the prover knows nothing that the loop learned, and the tracker joins its
state at the exits of the loop. With this note, each invariant holds after a loop when it holds at
each exit:

- A `for i in a..b` loop exits at its head. If the prover proves `a <= b`, the invariant holds
  there with `b` in place of `i`. If not, the checker does not know where the loop ended, and the
  invariant gives no fact after the loop.
- A `while c` loop exits at its head, with `!c`. The invariant holds there. When the loop has no
  other `break`, the prover also knows `!c` after the loop.
- Each `break` is an exit. The checker proves each invariant at each `break`, without an error. An
  invariant that does not hold at a `break` gives no fact after the loop.

This is the same for the prover's facts and the pointer facts. In `fill_then_sum`, `valid(p, n)`
gives `0 <= n` (section 4), so the loop gives `initialized(p, n)` after it. The call `sum(p, n)`
then compiles.

## 8. The parallel compiler

All of this happens inside one body check. It reads no fact of another body.

## 9. Open questions

1. Section 7 of this note proves each invariant at each `break` without an error. The other choice
   is an error, as at a `continue`. Then each invariant holds after each loop, and a programmer
   who wants a `break` that breaks the invariant must remove the invariant.
1. Section 5 of this note tries pairs of facts only. A chain of three facts, as after two writes and
   an invariant, still fails. A loop does not need more than one pair.
1. Section 6 of this note gives the new base of an assigned pointer the memory of its base when the
   loop starts. A loop that makes the pointer point into a local variable gets E8009. A later
   version can let the invariant name the memory.

## 10. The work, in order

1. Section 3 of this note: the pointer facts keep the old value at an assignment.
1. #2830: a write gives an `Init` fact.
1. #2831: two facts prove a clause, and `initialized(p, 0)` needs no fact.
1. #2837: pointer clauses in an invariant.
1. #1358: the facts after a loop.

Each step adds fixtures that pass and fail, as in section 13.
