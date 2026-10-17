# Sequence points for calls in the borrow checker

Status: proposal. Issues: #1800, #1795. Tracking: #1501.

## The problem

Each of these programs compiles, and then reads memory that a `push` freed:

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

The rule that should refuse them already exists: `&mut a` is an error while the program still uses a
borrow of `a`. A call that takes `&mut self` therefore ends every reference into `a`, and a later use of one
is an error. The rule fails because the checker loses track of who owns each borrow during a call.

## What the checker does today

A call records borrows while it checks its arguments. Afterwards it removes the borrows that were
added since the call started, unless the result can hold a reference. It finds them by comparing
the borrow table with a copy taken before the arguments (`retain_present` in `borrow_cx.rs`). That
copy cannot tell a borrow the call made for itself (`&a` in `count(&a, ..)`) from a borrow an
argument stored in a variable (`s` above). It removes both.

A method and a free function do this in different code (`calls.rs`, the reborrow plan for
functions and `check_methodcall_expr` for methods), in a different order:

| Program | Today |
| --- | --- |
| `a.push(a.len() as i64)` | accepted |
| `push_it(&mut a, a.len() as i64)` | E4004 |
| `(&mut a).push(a.len() as i64)` | E4004 |
| `a.push(*a.get_ref(0))` | accepted |
| `let r = first(&a); a.push(*r); println!("{}", *r)` | E4003 |
| `let p = &mut a; let s = p.get_ref(0); p.push(2i64); use s` | accepted (wrong) |
| `let p = &mut a; let s = first(p); p.push(2i64); use s` | accepted (wrong) |
| `let x = it.next(); a.push(2i64); use x` | accepted (wrong) |

A method whose receiver is a reference variable skips the receiver check completely. That is why
the checker accepts the last three rows.

## The model

A call `recv.m(arg1, arg2)` or `f(arg1, arg2)` runs in this order:

1. **Receiver.** Find the place the receiver names. A `&mut self` borrow of it is *reserved*: the
   place can still be read, but not borrowed `&mut` again.
1. **Arguments**, left to right. Each borrow made here has one owner:
   - **the call**, for a borrow that is the argument itself: `&a` in `count(&a, ..)`, or `p` passed
     to a reference parameter.
   - **a holder**, for a borrow stored somewhere: `s = a.get_ref(0)` stores it in `s`. The holder is
     a variable, or the variable a field or a container belongs to.
1. **Sequence point: the call happens.**
   1. The reserved receiver borrow becomes active. Every borrow of the place that is still live is
      an error, whatever owns it. A borrow owned by this call that ends here is not live.
   1. The call's own borrows end, except the ones the result keeps (see below).
   1. The result goes to its variable. That variable becomes the holder of the borrows the result
      keeps.

A borrow owned by a holder ends at the holder's last use, as today. Nothing compares the table
with a copy. The checker writes the owner in the borrow record when it makes the borrow.

The reservation in step 1 applies to a method's receiver only. An explicit `&mut a` argument is
active at once, so `push_it(&mut a, a.len())` stays an error. This keeps today's behavior for every
accepted row of the table above.

### A receiver that is a reference variable

For `p.push(..)` with `p : &mut Vec<i64>`, the place is `*p`. The checker records its borrows on `p`, as
for a view through a reference (`borrow_checker_architecture.md` §4). Step 3 checks the borrows of
`p`. That refuses `let s = first(p); p.push(2i64)`, because `s` already borrows `p`. In
`let s = p.get_ref(0)`, `s` borrows `p` only after the decision below.

### What the result keeps

The result of a call that can hold a reference keeps two kinds of borrow:

- **the arguments it may point into.** Today the callee's return provenance decides this
  (`provenance.rs`). For a method, the receiver counts as argument 0.
- **what those arguments hold.** If the result may point through the receiver, it may point where
  the receiver points. So `x = it.next()` makes `x` a holder of every borrow `it` holds, which
  includes `a`. Today only a receiver that is a reference variable gets this (`copy_borrows`). With
  this rule `x` keeps `a` borrowed after `it` is dead.

## The decision this needs

`VecIter::next` has an `unsafe` block, so its return provenance is "any parameter". Its result then
borrows `self` mutably. In `Iterator::last` (`stdlib/core/iter/traits.vx`), the result of the first
`self.next()` is still used at the second call, so that is E4004. This blocked the first attempt at
#1795.

In Rust the item has the vector's lifetime, not the lifetime of `&mut self`. Vx has no lifetimes.
The options:

1. **The result keeps the receiver and what it holds.** Sound, and the same rule for every method.
   `Iterator::last` and any code that keeps an item while it asks for the next one must change.
1. **A result made through a raw pointer comes from no parameter.** This is Rust's unbounded
   lifetime of `&*ptr`. It is not sound: `Vec::get_ref` makes its result through a raw pointer
   into the vector's own buffer, so `v.get_ref(0)` would keep nothing borrowed.
1. **The signature says what the result borrows**, for example a marker on `next` saying the result
   points where `self` points, not into `self`. It is precise, but it is a language change. It must
   be checked against `docs/language_rules_for_a_parallel_compiler.md`. It is a fact about one
   signature, so it should be safe.

Option 2 is unsound, so the choice is between 1 and 3. Option 1 can come first and option 3 later,
as a way to accept more programs.

## The work, in order

1. Add the owner to `BorrowRecord`: a holder variable, or the call that made it.
1. Rewrite the call checks against the model, one function for free functions, methods and
   static methods. Remove `retain_present`. This fixes #1800 and replaces #1845.
1. Check a receiver that is a reference variable. This fixes the receiver half of #1795.
1. Make a result hold what the receiver holds. This fixes the iterator program.
1. Make a method's result keep the receiver as option 1 or 3 decides. This fixes the rest of #1795.

Each step adds the programs in this note as tests, each in the file for its own step.

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
