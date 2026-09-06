# Tensors inside structs

**Status:** unimplemented, and partly built. The representation question has an answer for
statically-shaped tensors and an open one for dynamically-shaped tensors. This doc records what
exists so [#356](https://github.com/hiraditya/Vx/issues/356) does not have to be re-derived.

A struct holding a tensor-typed field type-checks and passes vx-dialect verification, then fails
LLVM lowering:

```console
$ vxc field.vx --action emit-mlir
'llvm.insertvalue' op operand #1 must be primitive LLVM type, but got 'memref<?x?xf32>'
MLIR verification failed
```

## 1. Two things called "Ref", and which one this is

`Ref` has been proposed for this twice, meaning two different things. Only the second is live.

**The source-level wrapper is settled, and settled against.** `Ref<Tensor<..>, Memory::X>` was
removed for tensors under [#429](https://github.com/hiraditya/Vx/issues/429) — see
[`docs/lang/types.md`](../../lang/types.md) and the spelling table in
[`docs/memory_algebra.md`](../../memory_algebra.md). It and `Tensor<.., Memory::X>` said the same
thing, every consumer had to peel the wrapper, and the AST lowering dropped the space the wrapper
carried while honouring that same space written as a placement. The type survives for non-tensor
values. Nothing here proposes bringing it back for tensors.

**The lowering representation is the live question, and it is the one to build.** A tensor field
holds a *reference to* the tensor rather than the tensor by value. This is not a new direction to
argue for: the codebase has already taken it twice, and simply never extended it to tensors.

## 2. What is already built

Three sites decide this, on two codegen paths.

| Site | State |
| ---- | ----- |
| AST struct **type** assembly — `MeliorGenerator::lower_type` | **Built.** A field whose lowered type starts with `memref<` is rewritten to `!llvm.ptr`. |
| AST struct **construction** — `StructInitExpr::lower` | **Missing.** The field's own type stays `memref<..>` and is handed to `llvm.insertvalue` for a `!llvm.ptr` slot. This mismatch is the verifier error. |
| AST field **read** — the `llvm.extractvalue` arm | **Missing**, and currently unreachable: construction fails first, so no program gets here. |
| Flat path — `LayoutCtx::field_info` | **Declines.** A tensor field makes the enclosing layout incomputable, so the whole function declines with `a struct layout that is not modelled yet` and falls back to the AST path. |

Two precedents matter:

`field_info` already maps `Type::Pointer`, `Type::Ref`, `Type::Borrow`, `Type::Function` and
`Type::Closure` to a pointer-sized `FieldTy::Opaque`, which the flat emitter lowers to `!llvm.ptr`.
Its doc comment names `tensor` explicitly among the types that return `None`. The slot this
proposal needs already exists; tensors are the one pointer-like thing not placed in it.

Reference-typed struct fields landed on both paths — `struct H { r : &i32 }` constructs, reads and
derefs, with a differential behind it. See §18.1 of
[`scalar_references_flat.md`](scalar_references_flat.md), which is a section of that document rather
than a separate file. Its conclusion is worth carrying over: the blocker there was an escape-analysis
gap, not a representational one, and once the base was materialized the existing raw-pointer-field
machinery from [#242](https://github.com/hiraditya/Vx/issues/242) carried the field with no further
change.

## 3. The open question: pointer, or descriptor

`&i32` needs no shape. A tensor reference does, and that is the whole of the remaining difficulty.

A `memref` value is a descriptor — allocated pointer, aligned pointer, offset, and a size and stride
per rank — so storing "a pointer" discards the sizes and strides. Whether that loss matters depends
on the field's declared type:

- **`Tensor<T, [static dims]>`** — the shape is in the type. A bare `!llvm.ptr` field is sufficient,
  and the read side rebuilds the memref from the pointer plus the static extents.
- **`DynTensor<T>`** — the shape exists only in the descriptor. A bare pointer loses it, and nothing
  in the type recovers it. Either the field stores the descriptor, or it stores a pointer alongside
  the extents.

The fixture that motivated #356's headline, `tests/middle_end/pass/implicit_transfer.vx`, declares
`weights: DynTensor<f32>` — the harder of the two.

Three ways to resolve it, in the order I would consider them:

1. **Pointer field, static shapes only.** Extend `field_info` with a `Type::Tensor` arm returning
   `(8, 8, FieldTy::Opaque)`, and refuse a `DynTensor` field in the checker with a diagnostic naming
   the field. Smallest change, and it makes the refusal a Vx diagnostic instead of an MLIR verifier
   message — which is what #354 chose for array literals of placed tensors, as E3018.
1. **Descriptor field.** Store the memref by value. Correct for both spellings, and the largest
   change: the struct stops being an `!llvm.struct` of primitives, and every consumer that assumes
   insertable fields has to follow.
1. **Pointer plus extents.** A `DynTensor` field lowers to a pointer and a rank-sized run of `i64`
   extents. Keeps fields primitive, at the cost of a field layout that no longer matches the source
   field count one-to-one.

Option 1 is the recommendation. It unblocks the reachable surface, keeps the flat path in play
rather than declining, and converts the remaining case from a verifier crash into a diagnostic. The
`DynTensor` field can then be taken on evidence, when a program needs it.

## 4. Order of work

1. `field_info` grows a `Type::Tensor` arm — the flat path stops declining these structs.
1. AST construction converts a `memref` field value to `!llvm.ptr` before `llvm.insertvalue`. There
   is an exact precedent three lines above it: the loop already special-cases an `index`/`i32`
   mismatch between the field's type and the struct's slot.
1. AST read rebuilds the memref from the pointer and the static extents.
1. The checker refuses a `DynTensor` field with a diagnostic naming the field.

Steps 1 and 2 are what `implicit_transfer.vx` needs to stop being quarantined. Step 3 has no caller
until step 2 lands.

## 5. Tests

`implicit_transfer.vx` carries an `XFAIL-LOWER:` marker today. The middle-end harness reads that
marker in both directions, so the marker fails the suite once the fixture lowers, and cannot outlive
the bug.

Note that the fixture's own `RUN` line does not currently pass: `vxc --action emit-mlir` exits 1 on
it, and the harness never runs the `RUN` line because `run_middle_end_test` drives `MeliorGenerator`
directly. That is a harness gap of its own, not specific to this feature.

What to add: a flat-vs-AST differential on construct-then-read of a statically-shaped tensor field,
matching the shape of `flat_runs_a_reference_typed_struct_field`; and a checker test pinning the
`DynTensor` field diagnostic.

## 6. Related

- [#356](https://github.com/hiraditya/Vx/issues/356) — the defect this doc scopes.
- [#354](https://github.com/hiraditya/Vx/issues/354) — array literal of placed tensors; the same
  refuse-or-lower choice, resolved as a diagnostic.
- [#242](https://github.com/hiraditya/Vx/issues/242) — the raw-pointer-field machinery this reuses.
- [#455](https://github.com/hiraditya/Vx/issues/455) — the AST path drops the memref for a rank-0
  tensor. Unrelated cause, adjacent symptom.
- [`scalar_references_flat.md`](scalar_references_flat.md) §18.1 — reference-typed struct fields.
