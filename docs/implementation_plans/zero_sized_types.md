# [RFC] Zero-sized types: layout, ABI and serialization

*Status: draft, revised after the [review on #2797](https://github.com/vx-lang/Vx/pull/2797#issuecomment-6666240425). Related: [#2195](https://github.com/vx-lang/Vx/issues/2195), [#2313](https://github.com/vx-lang/Vx/issues/2313), [#2315](https://github.com/vx-lang/Vx/pull/2315).*

## Summary

A zero-sized type (ZST) is a type whose values occupy no bytes. In Vx today, that means an empty struct, a struct made only of empty structs, `PhantomData<T>`, and the environment of a closure that captures nothing. Vx gives an empty struct size 0 and alignment 1, as Rust does.

[#2313](https://github.com/vx-lang/Vx/issues/2313) showed a bug in field identity. The compiler identified a field by its byte offset, two fields shared an offset, and the emitter picked the wrong one. [#2315](https://github.com/vx-lang/Vx/pull/2315) fixes that with explicit field indices. This RFC states the rest of the rules.

The central claim is that **a ZST leaves the bytes but stays in the schema.** Machine code can remove it. Anything that records structure must keep it: a compiled artifact, a debugger, a serializer, an API diff. Most of the risk is in that second group.

The RFC proposes nine rules, a work plan and six open questions.

## Changes after review

The review on #2797 made six points. Each item below answers the review point with the same number:

1. R7 no longer calls a new private field source-compatible. It is a major change when every field was public before.
1. R4 limits field omission at the C boundary to ZSTs with alignment 1.
1. R9 no longer tracks the identity of a ZST.
1. R9 requires a plain-data bound for a byte restore. A restore API must establish the facts that its proofs claim.
1. R6 now records a fingerprint of each aggregate that a body uses. The fingerprint covers fields, drops and thread traits. Import refuses a body when a fingerprint does not match.
1. R5 and R2 add requirements and tests for captures, drops and borrows. "What Vx does today" records a bug: `PhantomData<T>` does not pass `T` to `Send`, `Sync` and `Owned`.

## Why ZSTs matter for Vx

A ZST carries a fact through runtime code at no cost. Code checks the fact once, at compile time or at runtime. The payoff is a check or bytes that the hot path no longer has.

| Use | Where the check happens | Payoff | Possible Vx use |
|---|---|---|---|
| Proof value | compile time | removed from the code | carry an admission verdict or a placement fact into a kernel |
| Proof of work (`Init<Tag>`) | once, at runtime | no check on each access | a device context or stream initialized once |
| Branded index | once, when code makes the index | no bounds check, no panic | kernel indexing without bounds checks |
| Guard with `Drop` | none | acquire and release, zero bytes | exclusive use of an accelerator queue |
| Marker (`PhantomData<T>`) | compile time | names a type parameter, zero bytes | already in `core::marker` |
| 1-ZST field in a transparent wrapper | layout | casts cost nothing | wrappers around device pointers |

Memory-space types say which kind of memory a pointer points into. A branded ZST can also say which instance, so a pointer from GPU 0 cannot reach GPU 1's stream.

Every one of these depends on controlling who can build the value. Rule R8 covers that. Each one also depends on the compiler keeping the ZST's other obligations: its drop, its thread traits and its construction rules. Rules R5 and R6 cover those.

## What Vx does today

- **Layout.** [`fields_layout`](../../src/layout.rs) starts at offset 0 and alignment 1, so an empty struct is size 0, alignment 1. `align_bytes == 0` marks an unmodelled stub, so a ZST never has alignment 0.
- **MLIR type.** [`agg_struct_ty_of`](../../src/codegen/flat.rs) lowers an empty struct to `!llvm.struct<()>`. A ZST field keeps its own member in the enclosing struct, so a field's index is its GEP index.
- **Enum slots.** [`enum_slot_storage`](../../src/layout.rs) gives a zero-sized payload slot the type `!llvm.array<0 x iN>`, which keeps its alignment.
- **Field access.** After #2315, `FieldLoad`, `FieldStore` and `FieldAddr` carry a field index in `imm`, marked by bit 63. The HIR verifier asserts the flag.
- **Layout check.** `CheckAggregate` compares type IDs. A [`TypeId`](../../src/gid.rs) is a module hash, a symbol hash, a generic hash and flags. It identifies the declaration, not its shape.
- **Construction and drops.** A struct literal lowers to stores of the fields that the compiler knew at compile time. The compiler also decides at compile time which values need a drop call, and writes each call into the body as a `DropStmt`. So an imported body builds and drops each value as its compile-time types required.
- **Artifacts.** [`write_type_definition`](../../src/metadata.rs) records each type's ID, name, size and alignment, and each field's name, offset, size and type. The format tag is `vxlib-interface-v19`.
- **API diff.** [`compare_structs`](../../src/api_diff.rs) matches fields by name. An added, removed or retyped field is a major bump. A reordered field is not reported.
- **Allocation.** [`vec_alloc`](../../stdlib/std/vec.vx) returns null for a zero-byte buffer, and `vec_grow` keeps it null. So `get_ref` on a `Vec<Empty>` returns a null `&T`. [`Box::new`](../../stdlib/std/box.vx) calls `alloc(sizeof<T>(), 16)` even when the size is 0, and panics on null.
- **C boundary.** [`check_extern_signatures`](../../src/hir/decl_check/mod.rs) rejects a tensor (E3022). It says nothing about ZSTs.
- **Visibility.** A struct field is a name and a type. Any module can write a struct literal, so any module can build any ZST.
- **Thread traits.** [`held_types`](../../src/hir/check/thread_safety.rs) derives `Send`, `Sync` and `Owned` from the declared fields of a type. `PhantomData<T>` declares no field. So `PhantomData<*const u8>` is `Send` and `Sync`, though a raw pointer is neither. `PhantomData<&T>` is `Owned`, though a reference is not. This is a bug, and its fix does not depend on this RFC.

## Measurements

Clang 18.1.3 and LLVM 18.1.3. The sources are in the appendix.

| Case | Result |
|---|---|
| GNU C `struct E {}` | Size 0. `struct F { struct E m; const int *p; }` is 8 bytes, with `p` at offset 0. Same as Vx. |
| C++, same source | `E` is 1 byte. `F` is 16 bytes, with `p` at offset 8. |
| `int f(struct E, int)`, clang C | Clang removes the empty argument on x86-64 Linux, aarch64 Linux and arm64 macOS. |
| Same, clang C++ | Clang removes it on x86-64 Linux and arm64 macOS. aarch64 Linux passes an `i8`, so the `int` moves from `w0` to `w1`. |
| LLVM `define i32 @g({} %e, i32 %x)` | `llc` removes `%e` on x86-64 and aarch64. `%x` arrives in `edi` or `w0`. This is the type Vx gives a by-value empty struct. |
| C `struct { char a; long z[0]; char b; }` | `b` is at offset 8 and the size is 16. Without `z`, `b` is at offset 1 and the size is 2. |
| C `struct { char c; double d[]; }` | Size 8, alignment 8. A flexible array member has the same effect. |
| GNU C `struct { struct E first; struct E second; }` | Both fields have the same address. In C++ they have different addresses. |

Two conclusions follow. For C, a 1-ZST disappears completely, in memory and in calls. A ZST with alignment above 1 does not: it still adds padding and raises the struct's alignment.

## Proposed rules

### R1. Layout

1. A ZST has size 0 and alignment of at least 1.
1. A struct's alignment is the largest alignment of its fields, ZSTs included. A ZST with alignment above 1 still pads the fields after it.
1. A sequence of ZSTs has stride 0.
1. Every ZST field keeps its own physical member. A field's index is its declaration position and its GEP index.

Every Vx ZST has alignment 1 today. Rule 2 matters as soon as Vx gains `[T; 0]`, `align_of` or an alignment attribute.

### R2. Identity

1. The compiler identifies a field by index, never by offset or address. This is #2315.
1. A ZST may have the same address as any other value. Pointer equality says nothing about a ZST.
1. An analysis of aliasing, such as borrow facts or raw-pointer facts, goes by field path, not by address.
1. Code can borrow two distinct ZST fields mutably at the same time, also from separate tasks where the task rules allow it.
1. Two mutable borrows of the same ZST field conflict. Equal addresses decide neither case.

### R3. Allocation

1. Allocating a ZST never calls the allocator. The pointer is non-null and aligned. Rust's `NonNull::dangling` uses the alignment itself as the address. Freeing it does nothing.
1. A `Vec` of ZSTs never allocates. Its capacity is the largest value its capacity field holds. `get_ref` returns the dangling pointer, not null.
1. `Box::new` of a ZST does not call `malloc`. C lets `malloc(0)` return null, so today `Box::new(Empty {})` can panic with "out of memory" on a conforming platform. glibc and macOS return a non-null pointer, which is why this did not show up.

The null `&T` matters as soon as references get `llvm.nonnull`, which Vx already emits for `requires valid` pointers.

### R4. The C boundary

1. A ZST by value in an `extern` signature, as a parameter or as the return type, is an error with a new diagnostic code. The message says to remove the ZST from the signature.
1. A struct that C reaches through a pointer can hold a ZST field with alignment 1. The field takes no bytes and adds no padding. So the C struct without that field has the same layout.
1. A ZST field with alignment above 1 in such a struct is an error. Zero size alone does not keep the layout, because the field still adds padding.
1. A struct whose size is 0 cannot cross the boundary, by value or as a pointee. ISO C has no empty struct.
1. [`abi.md`](../lang/abi.md) gets a section that records these rules and the C++ difference. Vx does not follow the C++ ABI, so `abi.md` records the difference and the compiler does not handle it.

Rejection is safer than removal for rule 1. With removal, the C declaration must omit the parameter by hand, and nothing checks it.

A later alignment feature can lift rule 3. It must then generate the matching C padding, and check that the offsets, size and alignment agree on both sides. It must reject any foreign layout that it cannot match.

### R5. Kernels and tasks

1. A ZST captured by a `spawn on` body is not a kernel parameter. It carries no data, so the kernel has nothing to receive. This keeps the parameter list the same on every backend.
1. The compiler removes the parameter during code generation, after the ownership, borrow and thread checks see the capture. A moved guard stays moved, and the caller cannot use it again.
1. A ZST with `Drop` captured by a device `spawn on` body is an error until Vx defines where its drop runs.
1. The drop of a captured guard follows the task's ownership rules, on normal completion and on cancellation.

This RFC did not check the current behavior on the NVIDIA backend.

### R6. Artifacts

This is the rule with the most risk today. An imported body depends on the aggregates it uses in three ways. All three can change while the bytes and the `TypeId` stay the same, so `CheckAggregate` passes.

- **Field position.** A struct in library A loses a ZST field, or A reorders its fields. Library B's `.vxlib` holds a body that reads a field of that struct by position. The old index now names a different field. When the types differ, the emitter declines or emits invalid MLIR. When they match, the body reads a wrong value with no error. This is #2313 again, across a version boundary.
- **Construction and drops.** A's version 2 adds a public field `permit : Permit`. `Permit` is a ZST with a private field (once fields have visibility), and only `obtain_permit` builds one after an authorization check. The size, the alignment and the other fields stay the same. B's old body still builds the struct without a `Permit`, so it skips the authorization. If `Permit` also gets a `Drop` impl, B's old body never calls it, and the resource stays held.
- **Thread traits.** A's version 2 adds a marker that makes the struct not `Send`. B's old body still moves the struct into a task.

The rules:

1. In memory, field instructions keep physical indices, as #2315 does.
1. Each imported body records a fingerprint of every aggregate that it uses. The fingerprint covers:
   - the size and the alignment,
   - each field's name, type and visibility, in declaration order,
   - the `Drop` impls of the type and of its fields,
   - the `Send`, `Sync` and `Owned` verdicts.
1. Import refuses a body when a fingerprint does not match. The error names the aggregate and the library to rebuild. Rust does the same with [E0460](https://doc.rust-lang.org/error_codes/E0460.html).
1. The format tag becomes `vxlib-interface-v20`.

These rules close the three risks above. Any change to an aggregate makes the downstream library rebuild, even when the change adds no bytes.

A later version can keep some old bodies. A body that only reads named fields does not depend on how code builds, drops or sends the aggregate. A body that writes a field is not in this group, because the write drops the old value of the field. For a body that only reads fields, import can resolve each field reference by name and type, and check only the size and alignment. Java resolves field references this way: JLS §13.1 compiles a field reference to its class, the field's name and its declared type. Java callers that create objects also stay valid, because `new` runs the class's own constructor, which ships with the class. A Vx struct literal copies the construction into the caller. So every other body must keep the full fingerprint.

The two-library tests in the work plan must exist before this relaxation lands.

### R7. API diff

`api_diff` gives one verdict today. The [visibility discussion](../discussions/visibility-and-semver-package-management.md) already proposes separate verdicts for source and binary compatibility. ZSTs are the clearest case for the split:

| Change | Source | Memory layout | Artifacts today | Artifacts with R6 | With the later relaxation |
|---|---|---|---|---|---|
| Add a public 1-ZST field | major, unless the struct already had a private field | unchanged | wrong field, if indices shift | refused, rebuild | kept only for bodies that only read fields |
| Add a private 1-ZST field (once fields have visibility) | major, unless the struct already had a private field | unchanged | wrong field, if indices shift | refused, rebuild | kept only for bodies that only read fields |
| Remove a ZST field that a body reads | major | unchanged | wrong field | refused, rebuild | refused, and the error names the field |
| Reorder fields | compatible: literals use names | offsets change | wrong field | refused, rebuild | kept only for bodies that only read fields |
| Add a ZST with alignment above 1 | major | alignment changes | wrong field | refused, rebuild | refused |
| Add a `Drop` impl to a field's type | not covered here | unchanged | the new drop never runs | refused, rebuild | kept only for bodies that only read fields |
| Make the struct not `Send` | major for code that sends it | unchanged | old bodies still send it | refused, rebuild | kept only for bodies that only read fields |

1. Adding a private field to a struct whose fields were all public is a major change. [Cargo's semver rules](https://doc.rust-lang.org/cargo/reference/semver.html#major-adding-a-private-struct-field-when-all-current-fields-are-public) say the same. A type that starts with a private field can add more fields later without a source break.
1. `api_diff` reports a field reorder. It is a binary break for any struct that crosses the C boundary.
1. The layout record keeps whether a type is a 1-ZST. Once fields can be private, downstream code can depend on that property without the source showing it.

### R8. Construction control

Proof types need a way to limit who can build them. Two options:

- **A. A private field.** `struct Init<Tag> { pkg seal : Seal }`, using the field visibility that the visibility discussion proposes. Only the package can write the literal. No new concept.
- **B. A construction marker on the struct.** It says who may write the literal. It works before field visibility exists, but it is a second mechanism for the same thing.

This RFC prefers A. Option A also lets a library add fields later without a source break, as R7 explains.

### R9. Value serialization

Vx has no value serializer yet. These rules apply when it gets one:

1. A ZST encodes as nothing, and a sequence of ZSTs as its length. A decoder caps that length. A hostile length of 2^63 allocates nothing and makes the decoder loop forever.
1. Decoding is construction, so it obeys R8. A ZST is decodable only if it opts in. Markers such as `PhantomData` opt in. Proof, witness and guard types never do, or a decoder could create a proof or a guard from nothing.
1. A serializer does not track the identity of a ZST. It writes each occurrence as a value. A ZST's address means nothing (R2), and all values of one ZST type hold the same data. Guards and proofs are the only ZSTs whose identity matters. Rules 2 and 5 keep them out of decoding and byte restore.
1. A format that keeps identity for other values uses an object ID plus a field path or an element index. It never uses an address alone.
1. Restoring from bytes is also decoding. A byte restore, such as reading an mmap'd checkpoint, requires a bound that says the type is plain data. Rust's `bytemuck::Pod` is an example. A proof type and a guard type never have the bound.
1. A byte move keeps a proof valid only when the fact still holds at the destination. A proof that a device is ready can stay valid after a move inside one process. It is not valid after a restore in another process.
1. A restore API that returns a proof must establish the fact again, for example by running the initialization again. It then builds the proof with the proof type's own constructor. It never builds the proof from the restored bytes.

## Work plan

Each item can be one sub-issue of #2195 or of a new tracking issue.

1. Merge #2315.
1. **Thread traits.** Treat `PhantomData<T>` as holding a `T` in `held_types`. Test markers whose type argument is not `Send`, not `Sync` or not `Owned`. This item does not depend on the rest of the RFC.
1. **Allocation (R3).** Dangling pointers for size 0 in `alloc`, `Box` and `Vec`, and a capacity rule for a `Vec` of ZSTs, with tests.
1. **C boundary (R4).** The diagnostic, the `abi.md` section, and tests. Test a parameter, a return value, a struct of only ZSTs, and a ZST field behind a pointer. When ZSTs with alignment above 1 exist, test that the boundary rejects one.
1. **Kernels and tasks (R5).** Check ZST captures in `spawn on`, and remove them from the kernel's parameters. Test a captured guard on normal completion and on cancellation.
1. **Borrows (R2).** Test mutable borrows of two distinct ZST fields, and two borrows of the same field.
1. **Artifacts (R6).** The fingerprint, the import check and format v20. Write two-library tests for each of these cases:
   - add a ZST field, remove one, and reorder the fields,
   - an old body that builds the struct,
   - an old body that drops the struct after its field type gets a `Drop` impl,
   - an old body that sends a struct that is no longer `Send`.
1. **API diff (R7).** Separate source and binary verdicts. Report reordering and the private-field rule.
1. **Construction (R8).** Blocked on field visibility.
1. **Serialization (R9).** Add the rules to [`std_library_design.md`](../std_library_design.md) as part of the value serializer design.
1. **Later relaxation (R6).** Keep bodies that only read fields. This needs the two-library tests first.

R1 needs documentation and tests, not code. Its first real test is the first ZST with alignment above 1.

## Open questions

1. Should an enum with no variants exist? It is the uninhabited type, the `False` of the proof calculus.
1. Should a payload-free enum with one variant be a ZST, as in Rust, instead of an `i32` tag? It would change the ABI of existing enums.
1. Should references get `llvm.nonnull`? If they do, R3 becomes required for soundness, not just tidiness.
1. Where should alignment above 1 come from: `align_of`, `[T; 0]` or an attribute? R1 should be settled before any of them lands.
1. Should debug information list ZST fields? A debugger that hides them shows a struct that does not match its source.
1. How does a type get the plain-data bound of R9? The compiler can derive it from the fields, as it derives `Send`. Or the type can opt in with `unsafe impl`. A type with a private field probably needs to opt in, because its fields can hold an invariant.

## Appendix: measurement sources

```c
// layout.c: clang -x c layout.c && ./a.out; clang++ -x c++ layout.c && ./a.out
#include <stdio.h>
#include <stddef.h>
struct E {};
struct F { struct E m; const int *p; };
struct FAM { char c; double d[]; };
struct Z { char a; long z[0]; char b; };
struct NoZ { char a; char b; };
struct Pair { struct E first; struct E second; };
int main(void) {
  struct Pair pair;
  printf("%zu %zu %zu\n", sizeof(struct E), sizeof(struct F), offsetof(struct F, p));
  printf("%zu %zu\n", sizeof(struct FAM), _Alignof(struct FAM));
  printf("%zu %zu\n", sizeof(struct Z), offsetof(struct Z, b));
  printf("%zu %zu\n", sizeof(struct NoZ), offsetof(struct NoZ, b));
  printf("%d\n", (void *)&pair.first == (void *)&pair.second);
}
```

```c
// pass.c: clang -x {c,c++} --target={x86_64-linux-gnu,aarch64-linux-gnu,arm64-apple-macos} -O1 -S -emit-llvm
struct E {};
int f(struct E e, int x);
int g(int x) { struct E e; return f(e, x); }
```

```llvm
; zst.ll: llc -O2 -mtriple={x86_64-linux-gnu,aarch64-linux-gnu}
define i32 @g({} %e, i32 %x) {
  ret i32 %x
}
```

## Links

- [Review of this RFC on #2797](https://github.com/vx-lang/Vx/pull/2797#issuecomment-6666240425)
- [PR #2315: lower: preserve field identity when offsets coincide](https://github.com/vx-lang/Vx/pull/2315)
- [Issue #2313: preserve explicit field identity in flat loads, stores, and addresses](https://github.com/vx-lang/Vx/issues/2313)
- [Epic #2195: complete enum lowering and nested layout support](https://github.com/vx-lang/Vx/issues/2195)
- [Zero-sized types as proofs (hardmo.de)](https://www.hardmo.de/article/2021-03-14-zst-proof-types.md)
- [The Rustonomicon: Exotically Sized Types](https://doc.rust-lang.org/nomicon/exotic-sizes.html)
- [Cargo SemVer compatibility: adding a private struct field](https://doc.rust-lang.org/cargo/reference/semver.html#major-adding-a-private-struct-field-when-all-current-fields-are-public)
- [Rust error E0460: a dependency built against another version](https://doc.rust-lang.org/error_codes/E0460.html)
- [Java Language Specification, Chapter 13: Binary Compatibility](https://docs.oracle.com/javase/specs/jls/se23/html/jls-13.html)
- [Swift Library Evolution](https://apple-swift.readthedocs.io/en/latest/LibraryEvolution.html)
