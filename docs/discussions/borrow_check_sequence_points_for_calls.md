# Evaluation order and sequence points

Status: proposal. Decided: E3, and what a result made through a raw pointer keeps (option 4).
Issues: #1800, #1795, #1847, #1848. Tracking: #1850, #1501.

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
so one program can get either order with no flag (#1848). The legacy generator does not agree
with itself either. It evaluates `v[i] = x` on a `Vec`, which is the call `v.set(i, x)`, place
first, and `t[i] = x` on a tensor value first. The borrow checker checks the place first, but its
comment says that the value comes first, as in Rust (`src/hir/stmt.rs`).

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
- struct, tuple and array literals: the fields or elements in the order the source writes them.
  A struct literal must write its fields in declaration order (E3062), so the two orders are the
  same. In C++, an initializer list in another order than the members is a common bug.
- `return e`, `break e`: the operand, then the jump.
- `print!` and `println!`: all arguments left to right, then the output. The output of one call
  is never split by the evaluation of an argument.

**E2. Conditional evaluation.** These evaluate a condition first and then only the part it
selects: `a && b`, `a || b`, `if`, `match`, `while` and `loop`.

**E3. Assignment.** `place = value` evaluates, in this order:

1. the code in the place, left to right: the index of `a[i]`, and a call that the place goes
   through, such as `f()` in `*f() = v`.
1. the value.
1. the write. The program reads a variable that names the place here, at the write. It drops the
   old value after it computes the value, and before the write.

This is E1 applied to the place and then the value. It differs from Rust, where the value comes
first. It agrees with the flat generator and the comptime interpreter today.

Decided: the place first. The legacy generator changes for a tensor element and for `*f()`. The
program reads a variable in the place at the write, as Swift does. If the value gives `t` a new
tensor, `t[1] = v` stores into the new one. Reading `t` first would store into freed memory. So
the place of a write reserves only the borrows that a call in it makes. The borrow checker
already does this: `*a.as_mut_slice() = f(&mut a)` is E4003, and `c.last = c.take()` is valid.

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

**E7. Nothing in sequential code has an open order.** Vx never gives a program undefined
behavior because of evaluation order. Code that runs at the same time follows C1 to C7 below.

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
  and a call in the place of a write (E3). While the reservation lasts, the program may read the place, but may not
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

## Sequence points between parts that run at the same time

Vx does not have threads yet, but it already runs code at the same time in four places:

- the instances of `unroll across`, and the passes of a `spawn` loop that the compiler splits (W1033
  explains a loop that it does not split).
- a `spawn on` region that overlaps the host's later work (`docs/spawn_on.md` §3, not built yet).
- the lanes of a block-and-thread kernel, which meet at `barrier()`.
- `raw::async_copy`, which a copy engine finishes while the lane continues, until `raw::async_wait`.

Rules for these must exist before threads do. Each rule that sequential code uses (E1, A1, A2) then
extends to tasks, and nothing has to be taken back. This is not the parallel compiler of
`docs/language_rules_for_a_parallel_compiler.md`, which is about how the compiler itself runs.

### What earlier memory models teach

| Model | Rule | Lesson for Vx |
| --- | --- | --- |
| C++11 | *Happens-before* is sequenced-before plus *synchronizes-with* (a release write read by an acquire read). A data race is undefined behavior. | Build the order between tasks from the order inside a task. But undefined behavior for a race is not acceptable in safe code. |
| C++ `memory_order_consume` | An ordering that follows data dependencies. No compiler implemented it as specified, and C++26 deprecates it (P3475). | Do not specify an ordering that a compiler cannot check. |
| C++ and Java relaxed accesses | Neither model rules out *out-of-thin-air* values for relaxed or racy accesses, and that problem is still open. | Nothing that the checker relies on may depend on relaxed accesses. |
| Java (JSR-133) | A race is defined behavior, not undefined. Ševčík and Aspinall (2008) showed that the causality rules forbid common compiler optimizations. | It is very hard to give a racy program a meaning. Forbid races when compiling instead. |
| Go | A race on a value larger than one word, such as a slice, can corrupt memory. | No middle ground exists: a language allows memory corruption from races, or prevents races. |
| Rust | `Send`, `Sync` and borrows make safe code free of data races. `thread::scope` lets a task borrow a local, because the scope ends with a join. Atomics follow C++20. | Exclusivity across tasks gives freedom from races. A join that the program structure guarantees lets a task's borrows end like a call's borrows. |
| Cilk, OpenMP tasks | *Serial elision*: a program without races means the same as the program with the parallel keywords removed. | A parallel construct can have a sequential meaning. `spawn on` already has one. |
| CUDA, HSA, Vulkan | Synchronization has a *scope*: a block, a device or the system. Vulkan separates ordering from *availability* and *visibility* in memory that is not coherent. | On a machine with many memories, happens-before needs a scope, and visibility is a separate step. Vx memory spaces and `transfer` already make visibility explicit. |
| CUDA since Volta | Each thread of a warp has its own program counter. Code that assumed the lanes of a warp run in lockstep broke, and `__syncwarp` became necessary. | Do not specify lockstep. Name every point where lanes meet. |
| CUDA, LLVM, Vulkan convergence | A barrier or a warp operation in divergent control flow is undefined behavior in CUDA. LLVM added convergence tokens, and Vulkan added maximal reconvergence, to say where lanes meet again. | Make a barrier in divergent control flow a compile error, not undefined behavior. |
| Swift 6 (SE-0414) | *Region-based isolation*: a value can move to another task when the sender does not use its region after that. | The checker can prove that a value moves into a task, with few annotations. |

### The rules

**C0. Sequence points order only conflicting accesses.** Two accesses *conflict* when they touch
overlapping places and at least one writes. The rules in this note order conflicting accesses, and
nothing else. "Sequential meaning" below is about the result, not about how the program runs.

**C1. Sequential meaning.** Each parallel construct that uses no atomics has a sequential order,
and the program means that order:

- a `spawn on` region runs at its place among the statements.
- the passes of a `spawn` loop run in pass order.
- the instances of `unroll across` run in index order.

The compiler runs parts at the same time only when C3 shows that no program can see the difference.
So a program gives the same result on one thread and on many.

Two cases have more than one valid result:

- **A construct that uses atomics.** Its meaning is the set of results that the memory model of C7
  allows.
- **A reduction that declares reassociation.** A parallel sum of floating-point numbers rounds
  differently from a sequential sum. A reduction may declare that its operator can be
  reassociated. Then the compiler picks a tree order. The order is the same on each run on one
  machine, but the language does not specify it. Without that declaration, C1 holds, and a
  floating-point reduction runs in order.

**C2. Fork and join are sequence points.** At the *fork*, the program evaluates every operand of the
construct by E1: the topology index, the loop bounds and the captured values. Then the parts start.
At the *join*, every access of every part ends. Then the statement after the construct starts.

**C3. Exclusivity between parts.** For A1, each access of a part is active from the fork to the
join. So two parts can access one place only if both read it. A part may write only memory that is
its own, such as its own rows of a tensor. The check behind W1033 already proves this for a `spawn`
loop. When the proof fails, the compiler runs the parts in order and warns (W1033). That is still
right under C1. A program never gets a data race from this.

**C4. Overlap with the host.** When the runtime overlaps a region with the host's later work, the
region's accesses last from the dispatch to the wait. A host access that conflicts with them under
A1 needs the wait before it. That includes:

- a read of a value that the region made or wrote.
- a write to memory that the region reads.
- a write to memory that the region writes.

`docs/spawn_on.md` §3 lists only the first case. The overlap is not built, so no program is wrong
today, but the list must grow before someone builds the overlap.

**C5. Barriers.** In a block-and-thread kernel, the lanes run at the same time. A barrier is a
sequence point for every lane of its *scope*:

- a subgroup: the lanes that the hardware runs together, such as a warp.
- a block: the lanes that share a shared memory.
- a device.

Each access before a barrier, in any lane of its scope, happens before each access after it, in any
lane of the scope. Between two barriers, C3 applies to the lanes: a place that one lane writes may
not be accessed by another lane. The language does not specify that the lanes of a subgroup run in
lockstep. A subgroup operation, such as a shuffle, a vote or a reduction, is a sequence point at
subgroup scope. Today `barrier()` has block scope, and Vx has no other scope.

**C5a. Convergence.** Every lane of a barrier's scope must reach the barrier, and the checker must
prove it. A barrier or a subgroup operation must be in control flow that is the same for every lane
of its scope, or the program gets a compile error. CUDA makes this undefined behavior instead.

**C6. Asynchronous copies.** `raw::async_copy(dst, src, i)` is an access with a span (A2). It starts
at the call and ends at the matching `raw::async_wait`. During the span, `dst[i]` may not be read
and `src` may not be written. `docs/custom_transfer_contract.md` already says this. Here it is one
more instance of A1. Another lane sees the copied value only after a `barrier()` (C5).

**C7. Atomics and shared writes.** Today an atomic write takes `&mut self`, so A1 makes it
exclusive, and atomics are sequential. When Vx gets tasks that share memory:

1. Safe code has no data races. A1 and C3 forbid them.
1. An atomic is the only place that two parts may write at the same time. That needs a type that a
   shared borrow may write through. This is a language change, and it must be checked against
   `docs/language_rules_for_a_parallel_compiler.md`.
1. Memory orderings follow C++20, as in Rust, without `consume`. A release write that an acquire
   read sees is a sequence point between the two tasks.
1. The scope of an atomic, as in PTX, comes from the memory space where the atomic lives: block scope
   in shared memory, device scope in device memory, and system scope in managed memory that the host
   also sees. The programmer cannot pick a scope that is too narrow. This is an open question.
1. A data race in `unsafe` code is undefined behavior, as in Rust. This is the only exception.

### Mapping to hardware

These rules order only what a program can see. So they do not stop the compiler from using weak
hardware fully.

**Weakly ordered CPUs, such as AArch64.** E1 is the order of evaluation, not the order of memory
operations. A1 and C3 make safe code free of data races. A program without data races cannot see
how the hardware reorders plain accesses (the DRF-SC result of C++11 and Rust). So:

- a plain read or write compiles to a plain `LDR` or `STR`, with no `DMB`.
- a fork, a join and the release and acquire points of C7 compile to `STLR` and `LDAR`, or `LDAPR`
  on Armv8.3 and later.
- atomics use the published mappings from C++20 to ARMv8, which have proofs (Pulte et al. 2018,
  Podkopaev et al. 2019).

Three choices would close this door, and the rules avoid all three: sequential consistency for
plain accesses (the cost of Java `volatile`), a meaning for programs with races (Java), and
`consume`.

**GPU streams.** C1 and C4 together give stream semantics. The compiler knows the accesses of each
`spawn on` region. Two regions without conflicting accesses can go to different streams and run at
the same time. Two regions that conflict go to one stream, or an event wait joins them. So the
compiler makes the dependency graph that a CUDA graph would hold, and the program does not write it.
C6 at device level lets a copy overlap compute, as `cudaMemcpyAsync` does.

**GPU warps.** C5 gives barriers a subgroup scope, and subgroup operations become sequence points.
C5a makes divergent barriers a compile error. The language specifies no lockstep, so it stays right
on hardware that schedules each lane alone. The scope of an atomic comes from its memory space (C7),
which maps to the PTX scopes `.cta`, `.gpu` and `.sys`.

## A formal model

The rules can be checked in three layers, from the cheapest to the most expensive.

1. **An axiomatic model in herd7's `cat` language.** The C11, Linux kernel, ARMv8, PTX and Vulkan
   models use this format. E1 gives program order. C2, C5 and C7 give synchronization edges.
   Happens-before is their transitive closure. A1 and C3 become one axiom: happens-before orders
   every two conflicting accesses. Each case of C1 to C7 becomes a litmus test.
   `formal/memory_model/` (#1851) has the model and the tests. memalloy (Wickerson et al. 2017)
   then compares the Vx model with the ARMv8 and PTX models, and checks the mapping above for all
   programs up to a size bound. herd7 models only code that runs at the same time. The sequential
   programs of this note go to the Lean model.

1. **A small core language in Lean 4.** It has calls, the reservation of `&mut`, holders, fork and
   join, and barriers. It proves three theorems:

   - soundness: if the checker accepts a program, no run has two conflicting accesses active at the
     same time.
   - serial elision: if C3 holds, every interleaving gives the result of the sequential order.
   - DRF-SC: a program without data races cannot see weak memory ordering.

   RustBelt, Stacked Borrows and Oxide did this kind of work for Rust. A proof for all of Vx with
   `unsafe` would take years, so the core comes first.

1. **A link to the compiler.** A proof about a model says nothing about the code in `src/hir`.

   - A run mode that checks A1 and A2 while the program runs, as Miri does for Rust. When the static
     checker accepts a program that this mode rejects, the static checker has a bug.
   - A fuzzer that makes programs and compares the two checkers.
   - The test of evaluation order from the work list, on both generators and on the comptime
     interpreter.

## What a method result made through a raw pointer keeps

`VecIter::next` has an `unsafe` block, so its return provenance is "any parameter". Under A6 its
result then keeps `self` borrowed `&mut`. `Iterator::last` (`stdlib/core/iter/traits.vx`) uses
the result of one `self.next()` at the next call, so it becomes E4004. So do `max`, `min` and the
`_by` methods. This blocked the first attempt at #1795. In Rust the item has the vector's
lifetime, not that of `&mut self`. Vx has no lifetimes.

Programs A and B start with `let mut it = a.iter(); let x = it.next().unwrap();`.

| Program | Rust | Option 1 | Option 2 | Options 3 and 4 |
| --- | --- | --- | --- | --- |
| A: `let y = it.next()`, then use `x` and `y` | valid | E4004 | valid | valid |
| B: `a.push(..)`, then use `x` | refused | refused | reads freed memory | refused |
| C: `let r = a.get_ref(0)`, `a.push(..)`, then use `r` | refused | refused | reads freed memory | refused |

Main refuses A and C, and compiles B, which then reads freed memory. The options:

1. **The result keeps the receiver and what it holds.** Sound, and one rule for every method.
   `Iterator::last`, and code that keeps an item while it asks for the next one, must change.
1. **A result made through a raw pointer comes from no parameter.** This is Rust's unbounded
   lifetime of `&*ptr`. It is not sound in Vx: `Vec::get_ref` makes its result through a raw
   pointer into the vector's own buffer, so `v.get_ref(0)` would keep nothing borrowed.
1. **The signature says what the result borrows**, for example that `next` returns a reference to
   where `self` points, not into `self`. Precise, but a language change. A fact about one
   signature should be safe under `docs/language_rules_for_a_parallel_compiler.md`, but it must be
   checked against it.
1. **The summary follows the path of the reference.** `next` returns `&(*self.vec).data[i]`. The
   path goes through `self.vec`, a field of type `&Vec<T>`, so the result points where that field
   points. `x` then keeps the borrow of `a` that `it` holds, and not `it`. The rules:
   - A path through a shared reference field keeps the borrows that the field holds.
   - A path through a `&mut` field keeps the receiver, as option 1 does. Two calls could otherwise
     give two `&mut` to one place.
   - A path through a raw-pointer field stays with the struct that holds the field. So
     `&self.data[i]` in `get_ref` keeps `a`.
   - Any other reference made in `unsafe` code keeps every argument, as today.

Decided: option 4. Option 2 is unsound, and option 1 refuses program A and `v.iter().max()`.
Option 4 needs no syntax. The summary comes from the syntax of each body before the checker
checks any body, as today's summary does, so the checker still checks bodies in parallel. It must
also cover methods and generic functions, which today's summary skips. The cost: the parallel
compiler rules say that the compiler reads nothing about a return type from the body. Today's
summary already breaks that rule, and option 4 makes the exception larger. A change to the body
of `next` can change which callers compile.

## What changes

| Rule | Today | Change |
| --- | --- | --- |
| E1 struct literal | flat: declaration order | flat: source order (#1848) |
| E1 `println!` | both: output split by argument evaluation | both: arguments first (#2416) |
| E3 assignment | legacy: value first for a tensor element and `*e` | legacy: place first (#1848) |
| E4 compound assignment | both: the index runs twice | both: once (#1847) |
| E6, A2 explicit `&mut` argument | E4004 | reserved, valid |
| A3, A4 | ownership guessed after the call (`retain_present`) | owner in the record (#1800) |
| A5 | receiver check skipped | checked (#1795) |
| A6 | held borrows copied only from a reference receiver | from every receiver |
| A6, return provenance | any parameter after `unsafe`. No summary for methods and generic functions | follows the path (option 4). Methods and generic functions too |
| C1 `unroll across` | `docs/lang/semantics.md`: parallel instances with no order | index order. Instances overlap only where C3 proves it |
| C4 | waits planned only before reads of a region's results | also before writes that conflict |
| C5 | lanes between barriers: not measured. Block scope only | checked by C3. Subgroup and device scopes |
| C5a | a barrier in divergent control flow: not measured | compile error |
| C1 reductions | no way to declare reassociation | a reduction that declares it may run as a tree |

## The work, in order

1. Agree this specification, and copy E1 to E7 into a language document under `docs/lang/`.
1. Make both generators follow E1 to E5 (#1847, #1848, `println!`). Add a test that prints its
   evaluation order and runs on both generators.
1. Add the owner to `BorrowRecord`, and rewrite the call checks against A2 to A4: one function for
   free functions, methods and static methods. Remove `retain_present`. This fixes #1800 and
   replaces #1845.
1. Make the provenance summary follow the path of a returned reference (option 4), and summarize
   methods and generic functions.
1. Make a result hold what the receiver holds (A6). With the step before, this fixes the iterator
   program and keeps `Iterator::last` valid.
1. Check a receiver that is a reference variable (A5). This fixes #1795. It needs the two steps
   before it: alone, it makes `Iterator::last` an error.

Each step adds the programs in this note as tests, in the file for its own step.

A later step, after the design of threads: write C1 to C7 into the language document, and
measure which of C3 and C5 the checker enforces today.

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
- Boehm and Adve, Foundations of the C++ Concurrency Memory Model, PLDI 2008
- P3475, Defang and deprecate memory_order::consume: https://wg21.link/p3475
- JSR-133, the Java memory model: https://www.cs.umd.edu/~pugh/java/memoryModel/
- Ševčík and Aspinall, On Validity of Program Transformations in the Java Memory Model, ECOOP 2008
- The Go memory model: https://go.dev/ref/mem
- Rust `std::thread::scope`: https://doc.rust-lang.org/std/thread/fn.scope.html
- Frigo, Leiserson and Randall, The Implementation of the Cilk-5 Multithreaded Language, PLDI 1998
- PTX ISA, memory consistency model:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model
- Vulkan specification, appendix "Memory Model"
- Lustig et al., A Formal Analysis of the NVIDIA PTX Memory Consistency Model, ASPLOS 2019
- Pulte et al., Simplifying ARM Concurrency: Multicopy-Atomic Axiomatic and Operational Models for
  ARMv8, POPL 2018
- Podkopaev, Lahav and Vafeiadis, Bridging the Gap between Programming Languages and Hardware Weak
  Memory Models, POPL 2019
- Wickerson et al., Automatically Comparing Memory Consistency Models, POPL 2017
- Alglave, Maranget and Tautschnig, Herding Cats, TOPLAS 2014
- Jung et al., RustBelt, POPL 2018. Jung et al., Stacked Borrows, POPL 2020
- Weiss et al., Oxide: The Essence of Rust, 2019
- Bocchino et al., A Type and Effect System for Deterministic Parallel Java, OOPSLA 2009
- NVIDIA, Inside Volta (independent thread scheduling), 2017
- LLVM, Convergence and Uniformity: https://llvm.org/docs/ConvergenceAndUniformity.html
- Swift SE-0414, Region based isolation:
  https://github.com/swiftlang/swift-evolution/blob/main/proposals/0414-region-based-isolation.md
