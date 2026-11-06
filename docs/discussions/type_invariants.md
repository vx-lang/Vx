# Type invariants

Status: proposed, for review. Issues: #2832 and #2840. Part of #2829.

## 1. The problem

The `unsafe` code of a type often trusts a fact about its fields. `Vec<T>` trusts that `len` is at
most `capacity`, that `data` points at `capacity` elements, and that the first `len` of them hold
values. Today no code states these facts, and nothing keeps them:

- Safe code can write a field. After `v.len = 100000000`, `v.get(99999999)` reads outside the
  buffer, and the program stops with a segmentation fault (#2840).
- The raw-pointer facts cannot use the facts. A pointer read from a field gets a new base with no
  facts (limit 1 in section 11 of `raw_pointer_facts.md`). So `Vec::as_mut_ptr` states its
  `ensures` with two `assume`s.

A type invariant states such a fact once, in the type. The checker keeps it true, and each use of
the type gets it as a fact.

This gives contracts a new place. Until now, each contract of Vx is about code: `requires` and
`ensures` describe a function, and `invariant` describes a loop. A type invariant is about data. It
describes each value of the type, in each function that gets the value.

## 2. The syntax

An `invariant` clause comes after the name of the struct and before its fields, as a `requires`
comes before the body of a function:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
struct Vec<T>
invariant 0i32 <= self.len && self.len <= self.capacity
invariant valid(self.data, self.capacity as i64)
invariant initialized(self.data, self.len as i64)
{
  data : *mut T,
  len : i32,
  capacity : i32,
}
```

In an invariant, `self` is the value of the type. An invariant can name the fields of `self` at any
depth. It is an expression that the prover models, or a pointer clause: `valid`, `initialized` or
`!overlaps`. It calls no function.

## 3. Who can write the fields

Only the module that declares the type can write its fields. A module is one `.vx` file. Code in
another module can read each field of a type with an invariant. It cannot do these things:

- write a field, at any depth, as in `v.len = 0` or `v.len += 1`
- take a `&mut` to a field, or to a place under one
- move a field out of the value
- build a value of the type with a struct literal

Each one gets a new error. Such code calls a function of the module, such as
`Vec::new` or `push`. An `unsafe` block does not lift this rule.

Code in another module can still write a whole value. `v = Vec::new()` and a swap of two `Vec`s
are legal. Each one puts a value whose invariant holds in the place of another such value.

So the invariant holds for each value of the type in code outside the module. Unsafe code can
still change a field through a raw pointer. It must keep the invariant, as it keeps each other
promise of unsafe code.

This rule does not make the invariant sound. Section 4 does that: the checker proves the invariant
at each `return` of each function that can write a field. This rule decides only where the
invariant can be false for a short time:

- Inside the module, a function can break the invariant. It must give it back before a call or a
  `return`.
- Outside the module, the invariant is never false.

**Why the module, and not the impl blocks of the type.** In C++, only the member functions and the
friends of a class can write its private fields. The class body declares each of them, so the class
body is the full list of writers. In Vx, an impl block is a separate item, and a trait impl can be
in any module. So the impl blocks of a type are a closed set only if the type also lists its
friends.

`VecIntoIter::drop` needs this access, because it sets `vec.len` to 0. Without a list of friends,
`Vec` must give a function that sets `len`, such as the `unsafe fn set_len` of Rust's `Vec`. That
function becomes part of the API of `Vec`. With the module as the boundary, `VecIntoIter` gets the
access because it is in the same file. `Vec` adds nothing to its API.

C++ needs a small boundary, because there the boundary is the only thing that keeps an invariant.
In Vx, the checker keeps it. A larger boundary gives the checker more proofs to do. It does not
make the invariant weaker.

**A struct literal outside the module.** It is an error, for two reasons. A literal writes each
field, so it needs the right to write each field. A literal also ties the code to the full list of
fields. Then a new field in a `pub` type breaks each such literal, and that is a semver break. Rust
adds `#[non_exhaustive]` to a struct to prevent this.

**Later: visibility (#489).** `visibility-and-semver-package-management.md` gives each field a read
level and a write level. Each module can read the field `pub get len`, and only the module that
declares it can write it. The rule of this section is the same as `pub get` on each field of a type
with an invariant. When Vx gets these levels, the rule becomes a rule about the write level:

- Code that can write a field of the type has the duties of section 4.
- Code that can only read the fields gets the facts of section 5.
- A field of a type with an invariant cannot have a write level wider than its module.

A type with an invariant can also make a field private. Then other modules cannot read it either.

**Later: `const` fields.** After the struct literal, no code writes a `const` field. This includes
code in the module. So `const` is the narrowest write level: `const`, then the module, `pkg`, `ws`
and `pub`. For an invariant, a `const` field has two effects:

- A clause that names only `const` fields needs a proof only at the struct literal. Section 4 asks
  for no other proof of it.
- A `const` field stops a write of the field, not a write of the whole value.
  `*r = FixedBuf::new(8)` through a `&mut` is still legal. So a call that gets a `&mut` to the value
  still ends the facts about a `const` field.

In C++, a class with a `const` member has no copy assignment. So C++ also stops a write of the
whole value. `Vec` cannot use `const`, because `push` gives `data` and `capacity` new values. A
fixed buffer can:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
struct FixedBuf<T>
invariant valid(self.data, self.cap as i64)
invariant 0i32 <= self.len && self.len <= self.cap
{
  const data : *mut T,
  const cap : i32,
  len : i32,
}
```

The first invariant needs a proof only where a struct literal builds a `FixedBuf`.

## 4. Inside the module

Code in the module can break the invariant for a short time, for example to change `len` and then
`capacity`. A function of the module gets the invariant and must give it back:

- **When the body starts**, the invariant holds for each value of the type that the function gets:
  a parameter, a field of a parameter at any depth, and a value behind a `&` or `&mut` parameter.
- **After a call**, the invariant holds for each value of the type that the call returns or can
  change.
- **At each `return`**, the invariant must hold for each value of the type that the function
  returns, and for each one behind a `&mut` parameter. The checker proves it, or it reports a new
  error.
- **Before each call**, the invariant must hold for each value of the type that the call gets. The
  callee assumes it when its body starts.
- **At a struct literal** of the type, the invariant must hold for the new value.

A value behind a `&` parameter cannot change, so the function proves nothing for it at a `return`.
A `SharedCell` field is the exception, and a type with such a field cannot have an invariant in v1.

In `VecIntoIter::drop`, the body takes the elements out of `self.vec` and breaks
`initialized(self.data, self.len)`. Then it sets `self.vec.len` to 0, and the invariant holds
again at the `return`.

## 5. Facts from an invariant

Outside the module, the invariant holds at each read of a field. Inside the module, it holds
where section 4 says, until code writes a field. At each read of a field of a value whose
invariant holds, the checker adds the invariant as facts. `self` becomes the place of the value:

- The prover gets each invariant that it models, such as `v.len <= v.capacity`.
- For a pointer clause about the field that the code reads, the tracker gives the new base of the
  pointer the facts of the clause. After `let p = v.data;`, `p` has
  `Valid(b, 0, v.capacity as i64)` and `Init(b, 0, v.len as i64)`.

These facts follow the rules of `raw_pointer_facts.md`:

- The base has other memory (section 3), so an event that can free memory ends its facts
  (section 5). Another read of `v.data` gets them again, while the invariant holds.
- A call that can change `v` ends each fact that names `v.len` or `v.capacity` (section 3).

So `as_mut_ptr` proves its `ensures` with no `assume`. Outside the module, safe code that reads
`v.data` gets facts too.

## 6. Generic types and traits

- The invariant of `Vec<T>` holds for each `T`. A generic function outside the module gets the
  facts of a `Vec<T>`.
- An impl of a trait for the type belongs to the module that holds the impl. An impl in the
  module of the type can write its fields. An impl in another module cannot.
- A value reached through a raw pointer, such as `*mut Vec<T>`, gets no facts. A read through a raw
  pointer is `unsafe`, and the pointer can point at memory that holds no value.

## 7. The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | The invariant is part of the declaration of the type, so it is frozen with the type. |
| A body declares nothing global | A body check reads the invariant. The rule of section 3 needs only the module of the type and the module of the body. |

## 8. Open questions

1. Section 4 needs the invariant before each call. So a helper function of the module cannot get a
   value whose invariant is broken. The helper can take the fields one by one instead. A later
   version can add a clause that says that a function does not assume the invariant.
1. A type with a `SharedCell` field cannot have an invariant in v1.
1. A write level wider than the module, such as `pkg`, for a field of a type with an invariant. It
   is sound, because each function that writes a field must give the invariant back. But then the
   invariant can be false in more than one file.
1. The keyword for a `const` field. In Vx, `const` already names a value known at compile time, as
   in `const N : i32` and in a `const` generic parameter.
1. An invariant about memory behind a pointer. `initialized(self.data, self.len)` is a fact about
   the buffer, not about a field, so the rule of section 3 does not keep it. It holds because safe
   code cannot write through a `*mut T`, and unsafe code must keep the invariant. This is enough
   while one value owns the buffer. It is not enough for a type whose invariant uses the invariant
   of a value behind a pointer. Spec# (owners and `expose`) and Dafny (`Repr` sets) solve that case
   with the ownership of memory. A later version of Vx can do the same. This note uses "own" only
   in this sense, as `drop_semantics.md` does.

## 9. The work, in order

1. The syntax, and the rule of section 3. Then `Vec` states its invariant, and that fixes #2840.
   The contracts page of the book gets a section on type invariants.
1. The proofs of section 4.
1. The facts of section 5: first the prover's facts, then the pointer facts. Remove the `assume`s of
   `Vec::as_mut_ptr`.
1. The invariants of `VecIter`, `VecIterMut` and `VecIntoIter`.
