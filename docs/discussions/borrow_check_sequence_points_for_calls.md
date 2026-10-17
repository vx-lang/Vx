# Evaluation order and sequence points

Status: proposal. Issues: #1800, #1795, #1847, #1848. Tracking: #1501.

This note specifies the order in which Vx evaluates expressions and statements, and the borrow
check rules that follow from that order. Both code generators, the comptime interpreter and the
borrow checker must follow it.

## Why Vx needs a written order

No language document states an order today, and the parts of the compiler disagree. This program
prints `1 2 3 4 3 6 5` with the default generator and `2 1 3 4 3 5 6` with `--legacy-codegen`:

```vx
t[note(1, 0)] = note(2, 5);                     // place or value first?
t[note(3, 1)] += note(4, 7);                    // the index runs twice (#1847)
let p = P { y : note(5, 1), x : note(6, 2) };   // declaration or source order?
```

The flat generator also falls back to the legacy one for a whole program that it cannot compile,
so one program can get either order with no flag (#1848). The borrow checker assumes a third
model: value before place (`src/hir/stmt.rs`).

The borrow checker needs the order too. Each of these programs compiles, and then reads memory that
a `push` freed:

```vx
// #1800: the argument stores a borrow of `a` in `s`.
a.push(if true { s = a.get_ref(0); 0i64 } else { 0i64 });
println!("{}", *s);

// #1795: a method called through a reference variable.
let p = &mut a;
let s = p.get_ref(0);
p.push(10i64);
println!("{}", *s);

// An iterator item, after the iterator is dead.
let mut it = a.iter();
let x = it.next();
a.push(2i64);
// `x` points into the old buffer.
```

## What earlier specifications teach

| Language | Rule | Lesson for Vx |
| --- | --- | --- |
| C, C++ before 17 | Most operands are *unsequenced*. Two unsequenced writes to one object, as in `i = i++ + i`, give undefined behavior. | An open order and side effects in operands together give undefined behavior. Do not leave an order open. |
| C++17 (P0145) | Fixed orders for chosen idioms: `a.b(c)`, `a[b]` and `a << b` go left to right, but `a = b` and `a += b` go right to left. Arguments are *indeterminately sequenced*: one at a time, in any order. `a == b` stays open. | P0431 showed the cost: `a() = b()` and `a().operator=(b())` run in different orders. Rules made idiom by idiom do not fit together. The operator form and the call form of one operation must agree. |
| Java | Strictly left to right. Each operand is fully evaluated before the operation. | One rule that a programmer can remember. Compilers reorder only where nothing can see it. |
| Rust | Operands go left to right, with exceptions. `a = b` evaluates the value before the place. `a += b` goes right to left for primitive types and left to right otherwise, and the Reference warns readers against relying on it. No `++` or `--`, and assignment has the value `()`. | Few side effects inside expressions make order matter less. Exceptions still need a warning in the reference, so avoid them. |
| Rust borrow checker | *Two-phase borrows*: an implied `&mut` receiver, a reborrow in an argument and an overloaded `+=` start as a reservation and become active at the call. An explicit `&mut` is never two-phase. | Reservation and activation are the right model. But `v.push(v.len())` and `Vec::push(&mut v, v.len())` get different answers, which is the P0431 problem again. |
| Swift (SE-0176) | *Law of Exclusivity*: two accesses to one variable may overlap only if both are reads. An access has a start and an end. An `inout` access starts just before the call, after the arguments. The compiler checks locals statically. | An access is a span, not a point, and it starts at the operation, not at the operand. |

Vx is already close to Rust's surface: it has no `++` or `--`, and assignment and compound
assignment are statements, not expressions. A side effect inside an expression comes from a call,
or from a statement inside an `if`, `match` or `unsafe` block used as a value.

## The evaluation rules

**E1. Left to right.** An expression evaluates its operands left to right, as written, and each
operand completely before the next one starts. Then it does its own operation. This holds for
every expression that is not listed in E2:

- a call `f(a, b)`: the callee, then `a`, then `b`, then the call.
- a method call `r.m(a)`: the receiver `r`, then `a`, then the call.
- binary operators, including comparisons: the left operand, then the right one.
- `a[i]`: the base, then the index, then the read.
- `s.f`, `*e`, `-e`, `!e`, `e as T`: the operand, then the operation.
- struct, tuple and array literals: the fields or elements **in the order the source writes
  them**, not in declaration order.
- `return e`, `break e`: the operand, then the jump.
- `print!` and `println!`: all arguments left to right, then the output. The output of one call
  is never split by the evaluation of an argument.

**E2. Conditional evaluation.** These evaluate a condition first and then only the part it
selects: `a && b`, `a || b`, `if`, `match`, `while` and `loop`.

**E3. Assignment.** `place = value` evaluates, in this order:

1. the operands of the place, left to right: the base of a field, the base and index of `a[i]`,
   the operand of `*e`.
1. the value.
1. the write. The program drops the old value after it computes the value, and before the write.

This is E1 applied to the place and then the value. It differs from Rust, where the value comes
first. It agrees with the flat generator and the comptime interpreter today.

**E4. Compound assignment.** `place op= value` evaluates the operands of the place once, then reads
the place, then evaluates the value, then applies the operator, then writes. `a[f()] += 1` calls
`f` once.

**E5. Tuple assignment.** `(a, b) = e` evaluates `e`, then assigns to each place left to right.

**E6. One rule for every form.** A shorthand has the order of what it means:

- `r.m(a)` is `T::m(&mut r, a)` when `m` takes `&mut self`.
- `a[i]` on a type with `Index` is `*a.index(i)`.
- `a += b` on a type with `AddAssign` is `AddAssign::add_assign(&mut a, b)`, with E4 for the
  place.

So `a.push(x)`, `Vec::push(&mut a, x)` and `(&mut a).push(x)` have one evaluation order and one
borrow check.

**E7. Nothing in sequential code has an open order.** A construct that runs work in parallel says
so: the passes of a `spawn` loop, and `spawn on` instances. There, the checker proves that the
parts do not conflict, or it runs them in order (W1033). Vx never gives a program undefined
behavior because of evaluation order.

A compiler may reorder evaluation only where no program can see the difference.

## Accesses and sequence points

An *access* is a read or a write of a place, or a borrow of it. A *sequence point* is the moment
an operation happens: a call, a write, or the end of a statement.

**A1. Exclusivity.** Two accesses to overlapping places may be active at the same time only if
both are reads or shared borrows.

**A2. An access has a start and an end.**

- A read is instant.
- A shared borrow `&x` starts when the program evaluates its operand.
- A `&mut` that an operation needs is **reserved** when the program evaluates its operand, and **starts at
  the operation**. This covers the receiver of a `&mut self` method, a `&mut x` argument (E6),
  and the place of a write. While the reservation lasts, the program may read the place, but may not
  borrow it `&mut`, move it, or write to it.

Unlike Rust, an explicit `&mut x` argument gets a reservation like an implied one. That follows from E6.
Programs such as `push_it(&mut a, a.len())`, which are E4004 today, become valid. It is sound for
the same reason two-phase borrows are: nothing can use the reserved reference before the call.

**A3. Each borrow has one owner**, which the checker writes in the borrow record when it makes the borrow:

- **the operation**, for a borrow that is the operand itself: `&a` in `count(&a, ..)`, or `p`
  passed to a reference parameter.
- **a holder**, for a borrow stored somewhere: `s = a.get_ref(0)` stores it in `s`. The holder is
  a variable, or the variable that a field or element belongs to.

**A4. At the sequence point of a call:**

1. Each reserved `&mut` starts. Every borrow of its place that is still live is an error,
   whatever owns it.
1. The borrows the operation owns end, except the ones the result keeps (A6).
1. The result goes to its variable, which becomes the holder of the borrows the result keeps.
   A result that goes to no variable is an operand of the next operation. That operation owns
   the borrows the result keeps.

A borrow that a holder owns ends at the holder's last use, as today. So `a.push(*a.get_ref(0))` is
valid: `*` copies the element, and the borrow that `get_ref` made ends there, before `push`
starts its `&mut`. `a.push(a.get_ref(0))` with a `Vec<&i64>` is not valid, because the borrow is
still live at the start.

**A5. A receiver that is a reference variable.** For `p.push(..)` with `p : &mut Vec<i64>`, the
place is `*p`. The checker records its borrows on `p`, as for a view through a reference
(`borrow_checker_architecture.md` §4), and A4 checks the borrows of `p`.

**A6. What a result keeps.** A result that can hold a reference keeps:

- the borrows of the arguments it may point into, which the callee's return provenance decides
  (`provenance.rs`). The receiver is argument 0.
- the borrows those arguments hold. If the result may point through the receiver, it may point
  where the receiver points. So `x = it.next()` makes `x` a holder of the borrow of `a` that `it`
  holds.

## The decision this needs

`VecIter::next` has an `unsafe` block, so its return provenance is "any parameter". Under A6 its
result then keeps `self` borrowed `&mut`. `Iterator::last` (`stdlib/core/iter/traits.vx`) uses
the result of one `self.next()` at the next call, so it becomes E4004. This blocked the first
attempt at #1795. In Rust the item has the vector's lifetime, not that of `&mut self`. Vx has no
lifetimes. The options:

1. **The result keeps the receiver and what it holds.** Sound, and one rule for every method.
   `Iterator::last`, and code that keeps an item while it asks for the next one, must change.
1. **A result made through a raw pointer comes from no parameter.** This is Rust's unbounded
   lifetime of `&*ptr`. It is not sound in Vx: `Vec::get_ref` makes its result through a raw
   pointer into the vector's own buffer, so `v.get_ref(0)` would keep nothing borrowed.
1. **The signature says what the result borrows**, for example that `next` returns a reference to
   where `self` points, not into `self`. Precise, but a language change. A fact about one
   signature should be safe under `docs/language_rules_for_a_parallel_compiler.md`, but it must be
   checked against it.

Option 2 is unsound. Option 1 can come first, and option 3 later, to accept more programs.

## What changes

| Rule | Today | Change |
| --- | --- | --- |
| E1 struct literal | flat: declaration order | flat: source order (#1848) |
| E1 `println!` | both: output split by argument evaluation | both: arguments first |
| E3 assignment | legacy: value first. Checker: models value first | legacy and checker: place first (#1848) |
| E4 compound assignment | both: the index runs twice | both: once (#1847) |
| E6, A2 explicit `&mut` argument | E4004 | reserved, valid |
| A3, A4 | ownership guessed after the call (`retain_present`) | owner in the record (#1800) |
| A5 | receiver check skipped | checked (#1795) |
| A6 | held borrows copied only from a reference receiver | from every receiver |

## The work, in order

1. Agree this specification, and copy E1 to E7 into a language document under `docs/lang/`.
1. Make both generators follow E1 to E5 (#1847, #1848, `println!`). Add a test that prints its
   evaluation order and runs on both generators.
1. Add the owner to `BorrowRecord`, and rewrite the call checks against A2 to A4: one function for
   free functions, methods and static methods. Remove `retain_present`. This fixes #1800 and
   replaces #1845.
1. Check a receiver that is a reference variable (A5). This fixes the receiver half of #1795.
1. Make a result hold what the receiver holds (A6). This fixes the iterator program.
1. Make a method's result keep the receiver as the decision above says. This fixes the rest of
   #1795.

Each step adds the programs in this note as tests, in the file for its own step.

## Not covered

A reference stored through a parameter is not tracked:

```vx
keep(&mut refs, a.get_ref(0));   // `keep` pushes the reference into `refs`
a.push(5i64);
println!("{}", *refs.get(0));    // compiles
```

The checker does not know that `keep` stores its second argument in its first. That needs a rule
for what a signature allows a function to do with its reference arguments. It is near #860, and it
is a separate change.

## Sources

- C++17 draft N4659, \[intro.execution\]: https://ece.uvic.ca/~frodo/cppdraft/n4659/html/intro.execution
- P0145R3, Refining Expression Evaluation Order for Idiomatic C++:
  https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0145r3.pdf
- P0400R0, the function-argument wording adopted for C++17:
  https://open-std.org/JTC1/sc22/wg21/docs/papers/2016/p0400r0.html
- P0431R0, Correcting Evaluation Order for C++: https://wg21.link/p0431r0
- Rust Reference, evaluation order of operands: https://doc.rust-lang.org/reference/expressions.html
- Rust Reference, assignment and compound assignment:
  https://doc.rust-lang.org/reference/expressions/operator-expr.html
- rustc dev guide, two-phase borrows: https://rustc-dev-guide.rust-lang.org/borrow-check/two-phase-borrows.html
- Swift SE-0176, Enforce Exclusive Access to Memory:
  https://github.com/swiftlang/swift-evolution/blob/main/proposals/0176-enforce-exclusive-access-to-memory.md
- Java Language Specification §15.7, evaluation order
