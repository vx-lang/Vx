# Raw-pointer facts: the algorithm

Status: proposed, for review. Issue: #2563.

Section 6 of `borrow_facts_in_contracts.md` (#2564) defines `valid`, `initialized` and
`overlaps`. It also says where a fact about a raw pointer starts and where it ends. This note gives
the algorithm: how the checker tracks those facts through a body, and how it proves a pointer
clause at a call.

## 1. What changes

Today (#2605, #2610):

- Only an `unsafe fn` can have a pointer clause, and only in a `requires`.
- The checker proves no pointer clause. The author of each `unsafe` block that calls the function
  keeps the promise.
- The code generator uses `valid` and `!overlaps` in the body of the function.

This note adds:

- A safe function can have a pointer `requires`. The checker proves it at each call in safe code.
- A function can have a pointer `ensures`. The body proves it, and each caller gets it as a fact.
- A length can be any expression that the prover models, such as `n - k`.
- A boolean function `frees(p)`, so that a fact can stay true across a call (section 5).
- A statement `assume e`, only in an `unsafe` block, for a fact that the types do not show
  (section 7).

In an `unsafe` block, the checker also tries to prove each pointer clause at a call. If it cannot,
it trusts the clause, as it does today. The audit (A1 in #1902) lists each trusted clause.

## 2. Three prover changes come first

The checker gets each length from the facts of the prover. Two gaps in those facts let E8006 accept
a call that breaks a `requires`:

- #2619: the checker makes one pass over a loop body. A fact from before the loop stays true for
  the whole pass, also when the body assigns the variable later.
- #2620: a call that gets `&mut c` can change `c`, but the prover keeps its facts about `c`.

A third gap makes the prover know too little (#2622). It does not read the `ensures` of a method
call, and it does not model a cast such as `v.len() as i64`. A length usually comes from such a
call.

The tracker needs all three. Section 6 uses the loop rule of #2619 for the raw-pointer facts.

## 3. The state

The checker keeps the tracker state next to the borrow state. It saves and joins the tracker state
at the same points as the borrow state.

**A pointer value is a base and an offset.** A base is a name that the checker gives to a pointer
value. The offset is an integer expression, in elements of the pointee type. The checker makes a
new base for:

- each raw pointer parameter, at the start of the body
- each pointer that a call returns, except for the pointer functions below
- each pointer made from a reference
- each pointer read from memory, such as `s.data`, and each pointer made from an integer
- each pointer that an `unsafe` block gives as its value

The pointer functions of `core::ptr` keep the base. `offset(p, k)` and `offset_mut(p, k)` have the
base of `p`, and the offset of `p` plus `k`. `add` and `sub` and their `_mut` forms add `k` or `-k`.
A cast between `*const T` and `*mut T` keeps the base and the offset. A cast to another element
type makes a new base.

The state maps each local variable of a raw pointer type to its base and offset. `let q = p;`
copies the entry of `p`.

**A fact is about a range of elements at a base.** A range is two expressions, `lo` and `hi`. It
holds the elements `lo` to `hi - 1`, counted from the base.

| Fact | It says |
| --- | --- |
| `Valid(b, lo, hi)` | The elements of the range are in one live allocation. The address of element `lo` is not null and is aligned. |
| `Init(b, lo, hi)` | Each element of the range holds a value of its type. |
| `Apart(b1, lo1, b2, lo2, len)` | The `len` elements from `lo1` at `b1` and the `len` elements from `lo2` at `b2` share no byte. |

**A fact is about values.** When the body assigns a pointer variable, the variable gets the entry
of its new value. When a call can change a variable (#2620), the variable loses its entry. In both
cases, the checker also drops each fact and each entry whose expressions name the variable. A
`let` that hides a name does the same, and so does the end of the scope of a variable. An
assignment to an integer variable outside a call keeps the facts: each one names the old value by
the new name that the prover's facts give it, as `$old_N`. A variable
that loses its entry gets a new base with no facts. A base keeps its facts when one of its
variables gets a new value. So a fact about the old value of `p` stays true for each copy of that
value.

The body can also make a raw pointer to a variable `x` or to a place under it, such as `x.f` or
`(*x).f`. Or it can make a `&mut` to such a place and keep it in a variable. Code can then change
`x` through it at any later point. So from that point, `x` has no entry, and neither the tracker
nor the prover keeps a fact that names `x`.

An `unsafe` write through a raw pointer, and a call that gets a `*mut` pointer, can change the
memory at the pointer. So after it, neither the tracker nor the prover keeps a fact that names a
place behind a `&mut` reference, such as `c.n` for `c : &mut Counter`. Memory behind a `&`
reference does not change, except inside a `SharedCell`.

**Each base records its memory.** The memory is a set of two kinds. A fact ends at each event of
each kind in the set (section 5).

- *Local memory of `x`*: the storage of the local variable `x`. A base made from `&x`, `&mut x`, or
  a reference to a field path of `x`, such as `&mut x.a.b`, has the local memory of `x` only. Here
  `x` is a local variable or a parameter that holds a value. The path has only fields, and no `*`.
- *Other memory*: memory that the body does not name. A base from a parameter, from a reference
  parameter, from a read of memory or from an integer has only other memory.

The new base of a call result has other memory. It also has the local memory of each variable that
an argument can point into. Such an argument is a pointer with local memory, a reference to a place
under a local variable, or a value that holds one of these. So a call such as `id(&mut y)` cannot
hide that its result points at `y`.

A pointer made from a reference gets facts only when the checker knows where the reference points.
That is true for a reference to a place under a local variable, as above. It is also true for a
reference parameter that the body does not assign, and for a place under it, such as `&(*r).f`.
Its memory belongs to the caller, so it outlives the body. A pointer made from any other reference,
such as a local reference variable or the result of a call, gets no facts. The reference can point
at a local variable whose scope ends before the body uses the pointer.

**Each ended fact records its event.** When an event ends a fact, the checker keeps the line of the
event and the reason. An error about a missing fact can then say why the fact is gone.

## 4. Where a fact starts

| Source | The facts |
| --- | --- |
| `requires valid(p, n)` of the function | At the start of the body: `Valid(bp, 0, n)`. `bp` is the base of the parameter `p`. |
| `requires initialized(p, n)` of the function | At the start of the body: `Init(bp, 0, n)`. |
| `requires !overlaps(p, q, n)` of the function | At the start of the body: `Apart(bp, 0, bq, 0, n)`. |
| `ensures valid(p, n)` of a call | When the call returns: `Valid(b, k, k + n)`. The argument for `p` has the base `b` and the offset `k`. In `n`, each parameter stands for the value of its argument before the call, and `return` stands for the result. When `p` is `return`, `b` is the new base of the result, and `k` is 0. |
| `ensures initialized(p, n)` and `ensures !overlaps(p, q, n)` of a call | In the same way. |
| A pointer made from a reference, when the checker knows where the reference points (section 3) | `Valid(b, 0, 1)` and `Init(b, 0, 1)`. A reference is valid and points at a value. |
| `assume valid(p, n)` in an `unsafe` block | `Valid(b, k, k + n)` for the base and offset of `p`. The same for the other two functions. |

`valid` is false for a negative length. So each `valid(p, n)` that starts a fact also gives the
prover the fact `n >= 0`.

Each fact has the memory of its base (section 3).

The facts of the `ensures` use each argument that the prover can model, and a fresh name for each
other argument, as `check_call_requires` does. A call that can change a variable drops each fact
that names the variable, after it adds the facts of its `ensures`. So in `grab(k, &mut k)`, the
fact `Valid(b, 0, k)` ends at the call. A `let` of the result gives `return` in a length its
name, as in `let got = read(buf, cap)`.

## 5. Where a fact ends

**No event ends an `Apart` fact.** It is about addresses only, and a free does not change an
address. An assignment to a variable that it names changes it or drops it, as section 3 says.

`Valid` and `Init` facts end at events. Two definitions say which events can free memory:

- *A type with a drop* frees something when the program drops a value of it. `Vec<T>`, `Box<T>`
  and `String` are such types. A type parameter counts as one too. `needs_drop` already answers
  this.
- *An argument that can free memory* has a raw pointer or a type with a drop in its type, at any
  depth, through fields, references and pointers. So `&Vec<i64>`, `*const i64` and a moved `Box`
  can free memory. `i64`, `&i64` and `&mut i64` cannot. The `frees` clause below removes an
  argument from this set.

**The tracker assumes that a call frees memory of its caller only through its arguments.** Vx has
no global variables, so safe Vx code keeps this rule. `unsafe` code and `extern` code must keep it
too. So an `extern` function must not state `ensures valid(return, n)` for memory that a later call
without that pointer can free. An example is `getenv`, whose result a later `setenv` can free.

A fact whose base has other memory ends at these events:

| Event | Example |
| --- | --- |
| A call, or an `mlir!` block, with an argument that can free memory | `v.push(1i64)`, `dealloc(p, 8i64)`, `g(p)` where `g` has no `!frees(p)` |
| An assignment to a place of a type with a drop | `*b = Box::new(2i64)` frees the old box. |
| The drop of a value of a type with a drop | A local `Vec` at the end of its block |
| A `spawn` whose task captures an argument that can free memory | The task can free memory at any point until its `join()` or the end of its `tasks` block. So in that time, the checker keeps no fact about other memory, also a fact that starts in that time. |
| A `spawn on` region that uses a variable from outside it that can free memory | `spawn on(..) { v.push(1i64); }`. The code waits for the region at its end, so the event is at the start of the region. |

A call of a closure also gets the values that the closure captures, as arguments. So a closure
that captures `&mut v` can free memory, also when the call passes only an `i64`. A closure value
whose type does not show what it captures, such as `||->i32`, is an argument that can free memory.
`Closure1` and the other closure structs of `core::ops` keep their captured values behind the raw
pointer `env`, so they are such arguments already.

**A call through an owner ends no fact.** An *owner* is a local variable that owns only memory that
the body made, and that no pointer points into. A call that can free memory only through owners,
and the drop of an owner, free no memory that a fact is about. A local variable `v` is an owner
when:

- The body declares `v` once, with `let v = f(..)`, and no argument of `f` can free memory, as in
  `Vec<i64>::new()`. A call reaches the memory of its caller only through its arguments, so the
  value owns only new memory. A raw pointer, a reference and a closure are no owners.
- The body names `v` only as the receiver of a method call, as an argument of a call, as `&v` or
  `&mut v` there, or as the value that the function returns. No closure, task, `spawn on` region
  or `mlir!` block names it. So only a call can make a pointer into its memory.

`v` stops being an owner at these points:

- A call that gets `v` and another argument that can free memory, such as `v.push(b)` with a `Box`
  of the caller. The call can put memory or a pointer from outside into `v`.
- A call that gets `v` and gives a value that can point into memory, such as `v.as_mut_ptr()`.
- The start of a loop whose body names `v`. The checker checks the body once, and a later statement
  can put memory from outside into `v` before the next pass.
- The end of an `if` or a `match` with a branch that ended it.

The pointer functions of `core::ptr` are no event: `offset`, `add`, `sub`, their `_mut` forms,
`eq`, `is_null` and `null`. They only compute an address. The checker trusts their `mlir!` bodies,
as it trusts the rest of the standard library.

A fact whose base has the local memory of `x` ends at these events:

| Event | What ends |
| --- | --- |
| The end of the scope of the variable | `Valid` and `Init` |
| A move out of the variable, for a type that is not `Copy` | `Init` |
| Each event that ends a fact about other memory, for a variable of a type with a drop | `Init` |

A call that gets `&mut x` can write a new value into `x`. Then `x` keeps its storage and holds a
value of its type after the call. So for a type without a drop, the call ends no fact about the
local memory of `x`. For a type with a drop, a call with a pointer to `x` can move the value out,
with `ptr::read`, and drop it. Then the bytes of `x` hold a value whose memory is free. So the last
row ends `Init` for such a variable.

**At a call, the order is as follows.** The checker evaluates the arguments, with their own
events. Then it proves the `requires` of the callee. Then it applies the events of the call. Then
it adds the facts of the `ensures`. So in `f(p, &mut v)`, the `&mut v` cannot end a fact that `f`
needs, because the proof comes first. The fact ends after the call, as `v` can change in `f`.

**`frees(p)`** is true when the call can free memory that it reaches through `p`, or move that
memory to a new address. `ensures !frees(p)` says that the call does not. At a call, the argument
for `p` is then not an argument that can free memory.

The compiler infers `!frees(p)` before it checks any body, in the way that it infers `!keeps(p)`
(#2603). The inference reads the syntax of the body. It follows the parameter `p` and each value
that the body gets from a value that it follows:

- a copy, or a pointer with the same base
- a value that the body reads through it, when the type of the value has a raw pointer, a
  reference or a type with a drop in it
- a reference that the body takes to a place through it, such as `&mut (*p).v`

It infers `!frees(p)` when the body uses each value that it follows only in these ways:

- It reads an element or a field of a type without a drop through it.
- It takes a reference to an element or a field through it.
- It writes an element or a field of a type without a drop through it.
- It computes `offset(p, k)` or compares two pointers.
- It gives it as an argument for a parameter that has `!frees`.
- It names it in an `assume`.
- It returns it.

A read of a value of a type with a drop is a move. The body can then drop the value and free the
memory that the value owns. So `ptr::read<T>` gets `!frees(p)` only for a `T` without a drop. For
the same reason, a parameter of a type with a drop that the function gets by value never gets
`!frees`. The function drops it, or moves it to another owner.

Each other use stops the inference, for example a store of `p` or a method call that the syntax
cannot resolve. The checker checks a written `!frees(p)` on a Vx function by the same rules, and a
body that breaks it gets E8012. It trusts one on an `extern` function, and the audit lists it. An
`extern` function states it after its return type, as in
`safe fn strlen(s : *const i8) -> i64 ensures !frees(s);`. A written `frees(p)` says that the call
can free that memory, so the compiler infers no `!frees(p)` for it.

A method and an instance of a generic function get `!frees` from their checked body, as they get
`!keeps`. There a method call is a call by name. A method of a trait impl gets only the clauses
that it states, and it cannot add a `!frees` that the trait method does not have (section 7).

## 6. Joins

**`if` and `match`.** Each branch starts from the state before the `if`. After the `if`, a fact
holds when it holds at the end of each branch that reaches the code after the `if`. A branch that
returns, breaks or continues does not count. An entry stays when each such branch has the same
entry. Otherwise its variable gets a new base with no facts.

**Loops.** Before the checker checks a loop body, it finds each event in the body, on any path. It
also finds each variable that the body assigns or that a call in the body can change. It ends each
fact that one of those events ends. It drops each fact and each entry that names one of those
variables. The body and the code after the loop start from that state. So each fact at the start
of the body holds at the start of each pass. A fact that the body makes ends at the end of the
pass.

**Tasks and closures.** A task body, also a `spawn on` body, starts with no raw-pointer fact of its
parent. The parent can free memory while the task runs. A closure body also starts with no
raw-pointer fact of its parent. The parent can free memory before it calls the closure.

## 7. Proving a clause

At a call, the checker proves each pointer `requires` of the callee. In each clause, each parameter
becomes its argument. The argument for `p` has the base `b` and the offset `k`, and `m` is the
length. The checker asks the prover for one goal for each fact about `b`. It uses each fact that the
prover knows at the call, such as the conditions of the path.

| Clause | It holds when the prover proves the goal for one fact |
| --- | --- |
| `valid(p, m)` | For a fact `Valid(b, lo, hi)`: `lo <= k && 0 <= m && k + m <= hi` |
| `initialized(p, m)` | For a fact `Init(b, lo, hi)`: the same goal |
| `!overlaps(p, q, m)`, where `q` has the base `c` and the offset `j` | `m <= 0`. Or `b` is `c`, a fact `Valid(b, lo, hi)` holds both ranges, and `k + m <= j \|\| j + m <= k`. Or, for a fact `Apart(b, lo1, c, lo2, len)` in either order: `lo1 <= k && k + m <= lo1 + len && lo2 <= j && j + m <= lo2 + len` |

`valid(p, 0)` at the end of a range has `k == hi`. The goal accepts it, because the address one
past the end of an allocation is not null and is aligned.

Two facts of the same kind and base also prove `valid` or `initialized`. The first fact holds the
start, and the second the end. For `Init(b, lo1, hi1)` and `Init(b, lo2, hi2)`, the goal is
`0 <= m && lo1 <= k && lo1 <= lo2 && lo2 <= hi1 && k + m <= hi2`. For two `Valid` facts, the goal
has `lo2 < hi1` in place of `lo2 <= hi1`. The two ranges must share an element, because two
allocations can sit next to each other. `initialized(p, m)` also holds when the prover proves
`m == 0`, because zero elements hold values.

When the checker adds an `Init` fact that touches or overlaps another `Init` fact of the base, it
adds one merged fact for both. It keeps the facts that it had, so the end of an `if` still keeps a
fact that both branches have. A later write next to the merged fact grows it. So a chain of writes
gives one fact, and a goal needs at most two facts.

Each goal uses exact integer arithmetic, with no wrap. Otherwise an offset near `i64::MAX` plus a
length could wrap to a small number and pass the bound. An address can still wrap: `offset(p, k)`
multiplies `k` by the size of an element. So the rule for one base needs a `Valid` fact. Inside
one allocation, two different offsets give two different addresses.

The checker does the same at a `return` of a function with a pointer `ensures`. First it applies
the events at the end of the scope of each local variable and each parameter: their drops, and the
end of their local memory. A fact whose expressions name a parameter stays, because an `ensures`
names the parameters too. So a function cannot promise a pointer to its own local memory, or to
memory that it drops.

In an `ensures`, `return` is the returned value, and each expression means its value when the call
returns. The caller adds the facts after the events of the call, so there too each expression has
its value after the call. A parameter is a copy, so the caller sees only its value at the start of
the call. So in v1, a body must not assign a parameter that a pointer `ensures` names.

| Where the proof runs | When the prover cannot prove the clause |
| --- | --- |
| A call in safe code | E8006. The message names the clause. If an event ended the fact, the message also gives the line of the event and its reason. |
| A call in an `unsafe` block | The checker trusts the clause. The audit lists it, and the block gets no W1010. |
| A `return` in a function with a pointer `ensures` | E8012 |

**`assume e`** adds the fact `e`, without a proof. It is a statement, and only an `unsafe` block can
hold it. The audit lists each one. `e` can be a pointer clause or a boolean expression that the
prover models. A safe function can use it for a fact that comes from an invariant of its type,
as `Vec::as_mut_ptr` does in the next section.

**A function value has no clauses.** So a safe function with a pointer `requires` cannot become a
function value in v1 (open question 4). A call through a function value adds no fact, and each
argument that can free memory makes it an event. For the same reason, an impl of a trait method
cannot add a pointer `requires` that the trait method does not have. A call through a trait bound
reads only the clauses of the trait, so it uses only a `!frees` that the trait method states. The
checker checks each impl body against that `!frees` by the rules of section 5. A `pub fn` can be
called from C, which proves nothing. The audit lists each `pub fn` with a pointer `requires`.

## 8. Examples

### A sum over part of a buffer

<!-- vx-doctest: skip (proposed syntax) -->

```vx
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

fn sum_split(p : *const i64, n : i64, k : i64) -> i64
requires valid(p, n)
requires initialized(p, n)
requires 0i64 <= k && k <= n
{
  return sum(p, k) + sum(offset(p, k), n - k);
}
```

1. At the start of `sum_split`, `p` has the base `b` and the offset 0. The facts are
   `Valid(b, 0, n)` and `Init(b, 0, n)`, and the prover knows `n >= 0` and `0 <= k <= n`.
1. `sum(p, k)` needs `valid(p, k)`. The goal is `0 <= 0 && 0 <= k && 0 + k <= n`, and the prover
   proves it. `initialized(p, k)` is the same.
1. The call gets `p`, a raw pointer. The compiler inferred `!frees(p)` for `sum`, because its body
   only reads `p[i]`. So the call ends no fact.
1. `offset(p, k)` has the base `b` and the offset `k`. The goal for `valid(offset(p, k), n - k)`
   is `0 <= k && 0 <= n - k && k + (n - k) <= n`. The prover proves it.

Without `!frees(p)` on `sum`, the first call ends both facts. Then the second call gets E8006, and
the message names the first call as the event.

### A vector

<!-- vx-doctest: skip (proposed syntax) -->

```vx
impl<T> Vec<T> {
  fn as_mut_ptr(self : &Vec<T>) -> *mut T
  ensures valid(return, self.len)
  ensures initialized(return, self.len)
  ensures !frees(self)
  {
    let p = self.data;
    unsafe {
      assume valid(p, self.len);
      assume initialized(p, self.len);
    }
    return p;
  }

  fn len(self : &Vec<T>) -> i32
  ensures return == self.len
  ensures !frees(self)
  {
    return self.len;
  }
}

fn total(v : &Vec<i64>) -> i64 {
  let p = v.as_mut_ptr();
  return sum(p as *const i64, v.len() as i64);
}

fn total_after_push(v : &mut Vec<i64>) -> i64 {
  let p = v.as_mut_ptr();
  v.push(1i64);
  return sum(p as *const i64, v.len() as i64);
}

fn total_in_a_loop(v : &mut Vec<i64>) -> i64 {
  let p = v.as_mut_ptr();
  let n = v.len() as i64;
  for i in 0i64..3i64 {
    sum(p as *const i64, n);
    v.push(i);
  }
  return 0i64;
}
```

- `total` compiles. After `as_mut_ptr`, `p` has a new base `b`, with `Valid(b, 0, v.len)` and
  `Init(b, 0, v.len)`. The `Valid` fact gives `v.len >= 0`. The `ensures` of `len` gives its result
  as `v.len` (#2622). So both goals hold. `len` has `!frees(self)`, so the call of `len` ends no
  fact. The other methods that only read the vector have it too: `get`, `get_ref`, `is_empty`,
  `as_slice` and `as_mut_slice`.
- In `total_after_push`, `v.push(1i64)` gets `&mut` to a vector, which can free memory. So the
  push ends both facts, and `sum` gets E8006 with a note on the line of the push. This is the bug
  that `as_mut_ptr` warns about in its comment.
- In `total_in_a_loop`, the call of `sum` comes before the push in the body. The loop rule finds
  the push before it checks the body. So the facts are gone at the start of the body, and the call
  gets E8006. On the second pass, the push of the first pass can move the buffer. The error has no
  note, because after the push `n` is not the length. With `v.len()` in place of `n`, the note
  gives the line of the push.

The `assume` in `as_mut_ptr` is true only when `data` is not null. So an empty `Vec`, as from
`Vec::with_capacity(0)`, keeps the address 16, which is aligned and not null, as the `Vec` of Rust
does. It allocates no memory for a buffer of 0 bytes and frees none. A type with no fields gives a
buffer of 0 bytes too.

### A pointer to a local variable

<!-- vx-doctest: skip (proposed syntax) -->

```vx
extern "C" {
  safe fn c_fill(out : *mut i64, n : i64) -> void
  requires valid(out, n)
  ensures initialized(out, n)
  ensures !frees(out);
}

fn read_one(v : &mut Vec<i64>) -> i64 {
  let mut x = 0i64;
  let p : *mut i64 = &mut x;
  v.push(1i64);
  c_fill(p, 1i64);
  return x;
}
```

- `p` points at the local memory of `x`, with `Valid(b, 0, 1)`. The push can free other memory
  only, so the fact stays, and the call compiles in safe code.

### A buffer that a read fills

The `Valid` fact of a buffer covers all of the buffer. A read gives an `Init` fact only for the
bytes that it wrote. The result of the read connects the two facts.

<!-- vx-doctest: skip (proposed syntax) -->

```vx
extern "C" {
  // The count of bytes that it read, or 0 at the end of the input or on an error.
  safe fn c_read_some(fd : i32, buf : *mut u8, cap : i64) -> i64
  requires valid(buf, cap)
  ensures 0i64 <= return && return <= cap
  ensures initialized(buf, return)
  ensures !frees(buf);
}

fn read_and_check(fd : i32, buf : *mut u8, cap : i64) -> i64
requires valid(buf, cap)
{
  let got = c_read_some(fd, buf, cap);
  let read_part = checksum(buf as *const u8, got);
  let whole = checksum(buf as *const u8, cap);
  return read_part + whole;
}
```

`checksum` has the clauses of `sum`, for `u8`.

1. At the start, `buf` has the base `b` and the fact `Valid(b, 0, cap)`. It has no `Init` fact,
   so a `checksum` before the read gets E8006.
1. The read proves `valid(buf, cap)`. Its `!frees(buf)` keeps the `Valid` fact. Its `ensures`
   give the prover `0 <= got && got <= cap`, and they give the tracker `Init(b, 0, got)`.
1. `checksum(buf, got)` compiles. The goal for `valid` is `0 <= got && got <= cap`. The goal for
   `initialized` is `0 <= got && got <= got`.
1. `checksum(buf, cap)` gets E8006. The goal for `initialized` needs `cap <= got`, and the prover
   cannot prove it. The bytes after `got` hold no value yet.

The C function `read` returns -1 on an error. `c_read_some` returns 0 instead, because
`initialized` is false for a negative length. The checker trusts the `ensures` of an `extern`
function, and the audit lists it. A second read into `offset_mut(buf, got)` gives a second `Init`
fact. The two `Init` facts touch, so the checker merges them (section 7), and a checksum of both
parts can be one call.

### Allocation and free

The memory from `alloc` holds no values. A call of `dealloc` ends each fact about other memory.

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn alloc_i64(n : i64) -> *mut i64
requires n > 0i64
ensures valid(return, n)
{
  let p : *mut i64 = alloc(vec_bytes(n, 8i64), 16i64);
  if is_null_mut(p) {
    panic("out of memory");
  }
  unsafe {
    assume valid(p, n);
  }
  return p;
}

fn fill_zero(p : *mut i64, n : i64) -> void
requires valid(p, n)
ensures initialized(p, n)
{
  for i in 0i64..n {
    unsafe {
      p[i] = 0i64;
    }
  }
  unsafe {
    assume initialized(p, n);
  }
}

fn alloc_and_free(n : i64) -> i64
requires n > 0i64
{
  let p = alloc_i64(n);
  fill_zero(p, n);
  let s = sum(p as *const i64, n);
  let bytes : *mut i8 = p;
  dealloc(bytes, 16i64);
  return s + sum(p as *const i64, n);
}
```

1. `alloc_i64` checks for null and then states its fact with an `assume`. The `assume` is the
   promise of its author: `alloc` gave `8 * n` bytes at an alignment of 16.
1. In `alloc_and_free`, the call of `alloc_i64` gets only an `i64`, so it is no event. `p` gets a
   new base `b` with `Valid(b, 0, n)`. It has no `Init` fact, because the memory holds no values
   yet. So a `sum(p, n)` at this point gets E8006.
1. `fill_zero` only writes `p[i]` and names `p` in an `assume`. So the compiler infers `!frees(p)`,
   and the call ends no fact. Its `ensures` gives `Init(b, 0, n)`, and the first `sum` compiles.
1. The `let` with the type `*mut i8` is a cast to another element type, so `bytes` has a new base.
   `dealloc(bytes, 16i64)` gets a raw pointer, so it is an event. It ends each fact about other
   memory, also the facts about `p`. The checker does not need to know that `bytes` and `p` hold
   the same address.
1. The second `sum` gets E8006, with a note on the line of the `dealloc`.

`fill_zero` needs an `assume`, because a write gives a fact for one element only, and a loop
cannot join them (section 11).

### A pointer in a field: lists and copies

<!-- vx-doctest: skip (proposed syntax) -->

```vx
struct Node {
  value : i64,
  next : *const Node,
}

fn value_of(n : *const Node) -> i64
requires valid(n, 1i64)
requires initialized(n, 1i64)
{
  return unsafe { (*n).value };
}

fn second_value(head : *const Node) -> i64
requires valid(head, 1i64)
requires initialized(head, 1i64)
{
  let next = unsafe { (*head).next };
  return value_of(next);
}
```

- `value_of(next)` gets E8006. `next` is a pointer read from memory, so it gets a new base with
  no facts (section 3).
- The facts about `head` say that the field `next` holds a value of its type. Any address is such
  a value, also null or the address of a node that the program freed. So no fact about `head`
  passes to `next`.
- A copy or a clone of a struct copies its pointer fields. A read of a field of the copy is a read
  of memory too. So the copy gets no fact from the original, and a free through the original
  leaves no false fact about the copy. The same is true for a pointer that the body stores in a
  field and reads again.
- Only an `ensures` or an `assume` gives a field a fact. For example, a list type can keep the
  invariant that each `next` is null or points at a live node. Then a method can check for null
  and state the fact with an `assume`.

### Tasks and shared memory

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn sum_while_a_task_pushes(v : &mut Vec<i64>) -> i64 {
  let p = v.as_mut_ptr();
  let n = v.len() as i64;
  let mut s = 0i64;
  tasks {
    let h = spawn {
      v.push(1i64);
    };
    s = sum(p as *const i64, n);
    h.join();
  }
  return s;
}
```

The task captures `v`, an argument that can free memory. So the `spawn` ends the facts about `p`,
and `sum` gets E8006. Today, without the clauses of `sum`, this program compiles, because `p`
holds no borrow of `v` (section 9). The push can move the buffer while `sum` reads it. A run with
the elements 5 and 6 printed 7434. The sum of the elements is 11.

What a fact promises, and what the programmer must still do:

- A fact says that the memory is live and holds values at that point of the body. It assumes that
  no other task frees the memory. The `spawn` rule of section 5 keeps that true for Vx tasks.
  When the task captures an argument that can free memory, the parent keeps no fact about other
  memory from the `spawn` to the `join()` or the end of the `tasks` block. A task body starts
  with no fact of its parent (section 6).
- A fact does not order the accesses of two tasks. `Valid` and `Init` do not say that no other
  task writes the memory at the same time. `!overlaps` compares two ranges of one call only.
- So the programmer must still synchronize the tasks. A raw pointer is not `Send`, so a task
  cannot capture one (E3055). To share one, a program puts it in a type with an
  `unsafe impl Send`. That type has a raw pointer in it, so a task that captures it ends the facts
  of its parent about other memory too. The author of the impl, and of each `unsafe` access
  through the pointer, must prevent a data race. A data race is two tasks that access one element
  at the same time, when one of them writes it.
- A thread that Vx did not start, such as a C thread, is outside the `spawn` rule. So an `extern`
  function must not give a fact about memory that such a thread can free. This is the trust rule
  of section 5.

### Calls through a function value or a closure

<!-- vx-doctest: skip (proposed syntax) -->

```vx
fn apply(f : fn(*const i64, i64) -> i64, p : *const i64, n : i64) -> i64
requires valid(p, n)
requires initialized(p, n)
{
  let a = f(p, n);
  return a + sum(p, n);
}

fn apply_closure(c : Closure1<i64, i64>, p : *const i64, n : i64) -> i64
requires valid(p, n)
requires initialized(p, n)
{
  let call = c.func;
  let a = call(c.env, n);
  return a + sum(p, n);
}
```

- In `apply`, the caller picks `f` at run time, so the checker does not know the callee. A
  function value has no clauses, so `p` has no `!frees` in the call `f(p, n)`. So the call ends
  the facts about `p`, and `sum` gets E8006 with a note on the line of `f(p, n)`. A choice such as
  `let g = if fast { first } else { second };` gives a function value too.
- `f` itself can free nothing. A `fn` value names a function and captures no values. A closure
  cannot become a `fn` value: vxc refuses it as an argument, in a `let` and in a cast.
- In `apply_closure`, `call(c.env, n)` gets `c.env`, a raw pointer. The closure can capture
  `&mut v` for the vector that holds the buffer at `p`. So the call is an event, and `sum` gets
  E8006.
- The checker sees what a closure literal in the body captures. So after
  `let times_k = | x : i64 | x * k;`, with `k : i64`, a call `times_k(2i64)` ends no fact.
- `apply(sum, p, n)` is an error in v1, because `sum` has a pointer `requires` (section 7).
- A call through a trait bound uses the clauses of the trait method (section 7).

## 9. The borrow checker and raw pointers today

The tracker does not depend on the borrow checker for raw pointers. Today that checker treats two
kinds of raw pointer in different ways:

- After `let p : *const i64 = &x;`, `p` holds the borrow of `x`. So `x = 6i64` before a use of `p`
  gets E4009. Rust accepts this program.
- After `let p = v.as_mut_ptr();`, `p` holds no borrow. So `v.push(2i64)` and then
  `unsafe { p[0] }` compiles. The read can be a use after a free.

The tracker gives the second program no error. The read is in an `unsafe` block, and no clause
needs a fact there. Open question 6 asks about a warning for it.

## 10. The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | Pointer clauses and `frees` clauses are in signatures. The inferred `!frees` comes from syntax, before the freeze, as `!keeps` does. |
| A body declares nothing global | The tracker state lives in one body check. It reads only the clauses of the callees and the frozen inferences. |

## 11. The limits of v1

- Only a local variable has an entry. A pointer in a field, such as `s.data`, gets a new base with
  no facts at each read. Only an `ensures` or an `assume` gives it a fact.
- A loop that moves a pointer, as in `p = offset(p, 1i64)`, drops the entry of `p` at the start of
  the body. A loop that indexes, as in `p[i]`, keeps it. A loop invariant that is a pointer clause
  about `p` gives `p` its facts again in each pass (`raw_pointer_facts_in_loops.md`).
- A fact about other memory ends at each event that can free memory, except at a call through an
  owner (section 5). So a vector that the body made keeps the facts only while it is an owner. A
  loop whose body names it ends that, and so does a reference to it in a variable. A vector of the
  caller keeps no fact. A clause `requires !points_into(p, *v)` can fix that (#3042).
- At most two facts prove a clause. The checker merges `Init` facts that touch, but a chain of
  three `Valid` facts does not join.
- A write through a pointer makes an `Init` fact for its one element only. So a loop that fills a
  buffer gives no fact for the buffer. The write keeps the facts that the pointer has.
- An `extern` function cannot state a `requires`. Its `ensures` can be a `frees` clause, a
  pointer clause or a clause about numbers. So the `c_read_some` of section 8 cannot be written
  yet.
- A `spawn on` region that uses a variable from outside it that can free memory ends the facts
  about other memory at its start, also when its task frees nothing.
- The inference from the syntax stops at a method call, as the inference of `!keeps` does. Only a
  method or an instance of a generic function gets `!frees` from its checked body.
- The loop rule reads the body before the checker checks it. At a method call, it uses only a
  `!frees` that the method states, as the methods of `Vec` do.

## 12. Open questions

1. Decided (#2674): a body that breaks a pointer `ensures` gets E8012, as a body that breaks
   `!frees(p)` does. A body that breaks a value `ensures` still gets an error with no code.
1. Decided (#2673): the name stays `frees`. The extern note has `ensures freed(p)` for a call that
   ends the life of a handle. `frees(p)` says that it can, and a later `freed(p)` can say that it
   does.
1. Decided (#2674): `assume` is a statement, and only an `unsafe` block can hold it. Outside one
   it gets E5002. The other choice was to trust each clause about a value that an `unsafe` block
   gives as its value.
1. A function value of a safe function with a pointer `requires`. V1 refuses it. Later, the value
   can have an `unsafe fn` type, so that each call through it needs `unsafe`.
1. Safe code that reads `p[i]` when the facts prove `valid`, `initialized` and the bounds. V1 still
   needs `unsafe` for each access.
1. A warning for an access in an `unsafe` block after an event ended its fact, as in section 9.
1. The E4009 rule for `let p : *const i64 = &x`. With facts, Vx can accept this program, as Rust
   does.

## 13. The work, in order

1. Fix #2619, #2620 and #2622.
1. The state, the facts of a `requires`, and the proof at a call of a safe function. Remove the rule
   that only an `unsafe fn` has pointer clauses.
1. The events, with the line and the reason of each ended fact in E8006.
1. The joins at `if`, `match` and loops.
1. `frees`, written and inferred.
1. Pointer `ensures`, with the body check and the facts at the caller. `assume`. The code generator
   emits the `dereferenceable` assume after a call with `ensures valid`.
1. A `Vec` with an empty buffer keeps an aligned pointer that is not null. Then the clauses of
   `Vec::as_mut_ptr` and `Vec::len` in the standard library.
1. The audit lists each trusted pointer clause and each `assume`, with A1 of #1902. `vxc --audit`
   prints one line for each: the file, the line, the column and the fact.

Each step adds fixtures that pass and fail. Each test breaks one rule, for example one event or one
term of a goal, and checks the line of the error.

The code goes in a new `src/hir/check/pointer_facts.rs`. The joins use the points in
`src/hir/check/control.rs` and `src/hir/stmt.rs` where the checker saves the borrow state. The
proof at a call goes in `check_call_requires` (`src/hir/check/calls.rs`).
