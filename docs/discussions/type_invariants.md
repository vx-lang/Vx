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

## 3. Who can change the fields

The module that declares the type owns it. A module is one `.vx` file. Code in another module can
read each field of a type with an invariant. It cannot do these things:

- write a field, at any depth, as in `v.len = 0` or `v.len += 1`
- take a `&mut` to a field, or to a place under one
- build a value of the type with a struct literal

Each one gets a new error. Such code calls a function of the module, such as
`Vec::new` or `push`. An `unsafe` block does not lift this rule.

So the invariant holds for each value of the type in code outside the module. Unsafe code can
still change a field through a raw pointer. It must keep the invariant, as it keeps each other
promise of unsafe code.

#489 is about visibility. When Vx gets private fields, a type with an invariant can make its fields
private. Until then, this rule makes them read-only outside the module.

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

1. The owner of the fields is the module. The other choice is the impl blocks of the type. Then
   `VecIntoIter::drop` needs a function of `Vec` to set `len`.
1. Section 4 needs the invariant before each call. So a helper function of the module cannot get a
   value whose invariant is broken. The helper can take the fields one by one instead. A later
   version can add a clause that says that a function does not assume the invariant.
1. A struct literal outside the module is an error here. The other choice is to allow it with a
   proof of the invariant at the literal.
1. A type with a `SharedCell` field cannot have an invariant in v1.

## 9. The work, in order

1. The syntax, and the rule of section 3. Then `Vec` states its invariant, and that fixes #2840.
   The contracts page of the book gets a section on type invariants.
1. The proofs of section 4.
1. The facts of section 5: first the prover's facts, then the pointer facts. Remove the `assume`s of
   `Vec::as_mut_ptr`.
1. The invariants of `VecIter`, `VecIterMut` and `VecIntoIter`.
