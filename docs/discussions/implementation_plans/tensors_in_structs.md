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

## 3. Pointer, or descriptor: descriptor

`&i32` needs no shape. A tensor reference does, and that was the whole of the remaining difficulty.

A `memref` value is a descriptor: allocated pointer, aligned pointer, offset, and a size and stride
per rank. Storing "a pointer" discards the sizes and strides, which a `Tensor<T, [static dims]>`
field could rebuild from its type and a `Tensor<T, [?, ?]>` field could not. The fixture that
motivated #356's headline, `tests/middle_end/pass/implicit_transfer.vx`, declares `weights: Tensor<f32, [?, ?]>`, the harder of the two.

An earlier draft of this document recommended a pointer field for static shapes and a diagnostic
for the rest, on the premise that a memref cannot sit inside an `!llvm.struct`. The premise was
wrong: the verifier error in #356 is about the operand being a builtin `memref`, not about
aggregates. `llvm.insertvalue` accepts a nested descriptor struct, and a
`builtin.unrealized_conversion_cast` from the memref to
`!llvm.struct<(ptr, ptr, i64, array<Nxi64>, array<Nxi64>)>` resolves to zero leftover casts under
the passes `src/codegen/mod.rs` already runs (`expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts`), checked with mlir-opt. Rank is static, so
there is one descriptor shape per rank.

So the field stores the descriptor by value. It is correct for both spellings, the struct stays an
`!llvm.struct` whose fields are stored and loaded whole, and nothing is refused.

## 4. What landed, and what is left

The flat path holds the descriptor:

1. `FieldTy::Tensor(elem, rank)`: `field_info` sizes a tensor field as its descriptor (24 bytes
   plus 16 per rank, 8-aligned). A struct with a tensor field has a layout, so the flat path no
   longer declines it.
1. The flat lowering of a field read takes the result type from the declared field type: the
   layout carries the element and rank, the declaration the extents.
1. The flat emitter spells the field as the descriptor struct, casts a memref value to it on a
   `FieldStore`, and casts a loaded descriptor back to the memref the result names on a
   `FieldLoad`. The casts reconcile after memref lowering.

The AST path still inserts the memref itself into the struct and fails verification; it needs the
same cast at construction and the reverse at a field read. Until then the differential tests for
this shape are flat-only.

Of the corpus programs that construct such a struct, `pinned_annotation_struct_field.vx` compiles
through the flat path. The other three (`implicit_transfer.vx`, `w1024_implicit_transfer.vx`,
`memory_algebra_implicit.vx`) initialize a `Tensor<f32, [?, ?]>` field with `Tensor<f32>()`, a
rank-0 tensor the checker admits into a rank-2 field. A rank-0 descriptor is not a rank-2 one, and
no pass folds a cast between them, so the flat emitter declines the store and the programs stay
where they were until the checker refuses the mismatch.

## 5. Tests

`implicit_transfer.vx` carries an `XFAIL-LOWER:` marker today. The middle-end harness reads that
marker in both directions, so the marker fails the suite once the fixture lowers, and cannot outlive
the bug.

Note that the fixture's own `RUN` line does not currently pass: `vxc --action emit-mlir` exits 1 on
it, and the harness never runs the `RUN` line because `run_middle_end_test` drives `MeliorGenerator`
directly. That is a harness gap of its own, not specific to this feature.

`flat_runs_a_tensor_typed_struct_field` covers construct-then-read of a static and of a `[?, ?]`
field, flat-only until the AST path carries the descriptor too.

## 6. Related

- [#356](https://github.com/hiraditya/Vx/issues/356) — the defect this doc scopes.
- [#354](https://github.com/hiraditya/Vx/issues/354) — array literal of placed tensors; the same
  refuse-or-lower choice, resolved as a diagnostic.
- [#242](https://github.com/hiraditya/Vx/issues/242) — the raw-pointer-field machinery this reuses.
- [#455](https://github.com/hiraditya/Vx/issues/455) — the AST path drops the memref for a rank-0
  tensor. Unrelated cause, adjacent symptom.
- [`scalar_references_flat.md`](scalar_references_flat.md) §18.1 — reference-typed struct fields.
