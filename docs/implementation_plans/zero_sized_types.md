# [RFC] Zero-sized types: layout, ABI and serialization

*Status: draft. Related: [#2195](https://github.com/vx-lang/Vx/issues/2195), [#2313](https://github.com/vx-lang/Vx/issues/2313), [#2315](https://github.com/vx-lang/Vx/pull/2315).*

## Summary

A zero-sized type (ZST) is a type whose values occupy no bytes. In Vx today, that means an empty struct, a struct made only of empty structs, `PhantomData<T>`, and the environment of a closure that captures nothing. Vx gives an empty struct size 0 and alignment 1, as Rust does.

[#2313](https://github.com/vx-lang/Vx/issues/2313) showed what goes wrong when the compiler identifies a field by its byte offset: two fields can share an offset, and the emitter picked the wrong one. [#2315](https://github.com/vx-lang/Vx/pull/2315) fixes that with explicit field indices. This RFC states the rest of the rules.

The central claim is that **a ZST leaves the bytes but stays in the schema.** Machine code can erase it. Anything that records structure must keep it: a compiled artifact, a debugger, a serializer, an API diff. Most of the risk is in that second group.

The RFC proposes nine rules, a work plan and five open questions.

## Why ZSTs matter for Vx

A ZST carries a fact that was checked once through runtime code at no cost. The check happens at compile time or once at runtime. The payoff is a check or bytes that the hot path no longer has.

| Use | Where the check happens | Payoff | Possible Vx use |
|---|---|---|---|
| Proof value | compile time | erased | carry an admission verdict or a placement fact into a kernel |
| Proof of work (`Init<Tag>`) | once, at runtime | no check on each access | a device context or stream initialized once |
| Branded index | once, when the index is made | no bounds check, no panic | kernel indexing without bounds checks |
| Guard with `Drop` | none | acquire and release, zero bytes | exclusive use of an accelerator queue |
| Marker (`PhantomData<T>`) | compile time | a type parameter is named, zero bytes | already in `core::marker` |
| 1-ZST field in a transparent wrapper | layout | casts cost nothing | wrappers around device pointers |

Memory-space types say which kind of memory a pointer points into. A branded ZST can also say which instance, so a pointer from GPU 0 cannot reach GPU 1's stream.

Every one of these depends on controlling who can build the value. Rule R8 covers that.

## What Vx does today

- **Layout.** [`fields_layout`](../../src/layout.rs) starts at offset 0 and alignment 1, so an empty struct is size 0, alignment 1. `align_bytes == 0` marks an unmodelled stub, so a ZST never has alignment 0.
- **MLIR type.** [`agg_struct_ty_of`](../../src/codegen/flat.rs) lowers an empty struct to `!llvm.struct<()>`. A ZST field keeps its own member in the enclosing struct, so a field's index is its GEP index.
- **Enum slots.** [`enum_slot_storage`](../../src/layout.rs) gives a zero-sized payload slot the type `!llvm.array<0 x iN>`, which keeps its alignment.
- **Field access.** After #2315, `FieldLoad`, `FieldStore` and `FieldAddr` carry a field index in `imm`, marked by bit 63. The HIR verifier asserts the flag.
- **Layout check.** `CheckAggregate` compares type IDs. A [`TypeId`](../../src/gid.rs) is a module hash, a symbol hash, a generic hash and flags. It identifies the declaration, not its shape.
- **Artifacts.** [`write_type_definition`](../../src/metadata.rs) records each type's ID, name, size and alignment, and each field's name, offset, size and type. The format tag is `vxlib-interface-v19`.
- **API diff.** [`compare_structs`](../../src/api_diff.rs) matches fields by name. An added, removed or retyped field is a major bump. A reordered field is not reported.
- **Allocation.** [`vec_alloc`](../../stdlib/std/vec.vx) returns null for a zero-byte buffer, and `vec_grow` keeps it null. So `get_ref` on a `Vec<Empty>` returns a null `&T`. [`Box::new`](../../stdlib/std/box.vx) calls `alloc(sizeof<T>(), 16)` even when the size is 0, and panics on null.
- **C boundary.** [`check_extern_signatures`](../../src/hir/decl_check/mod.rs) rejects a tensor (E3022). It says nothing about ZSTs.
- **Visibility.** A struct field is a name and a type. Any module can write a struct literal, so any module can build any ZST.

## Measurements

Clang 18.1.3 and LLVM 18.1.3. The sources are in the appendix.

| Case | Result |
|---|---|
| GNU C `struct E {}` | Size 0. `struct F { struct E m; const int *p; }` is 8 bytes, with `p` at offset 0. Same as Vx. |
| C++, same source | `E` is 1 byte. `F` is 16 bytes, with `p` at offset 8. |
| `int f(struct E, int)`, clang C | The empty argument is dropped on x86-64 Linux, aarch64 Linux and arm64 macOS. |
| Same, clang C++ | Dropped on x86-64 Linux and arm64 macOS. aarch64 Linux passes an `i8`, so the `int` moves from `w0` to `w1`. |
| LLVM `define i32 @g({} %e, i32 %x)` | `llc` drops `%e` on x86-64 and aarch64. `%x` arrives in `edi` or `w0`. This is the type Vx gives a by-value empty struct. |
| C `struct { char a; long z[0]; char b; }` | `b` is at offset 8 and the size is 16. The member has no bytes but has alignment 8. |
| C `struct { char c; double d[]; }` | Size 8, alignment 8. A flexible array member has the same effect. |

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

### R3. Allocation

1. Allocating a ZST never calls the allocator. The pointer is non-null and aligned. Rust's `NonNull::dangling` uses the alignment itself as the address. Freeing it does nothing.
1. A `Vec` of ZSTs never allocates. Its capacity is the largest value its capacity field holds. `get_ref` returns the dangling pointer, not null.
1. `Box::new` of a ZST does not call `malloc`. C lets `malloc(0)` return null, so today `Box::new(Empty {})` can panic with "out of memory" on a conforming platform. glibc and macOS return a non-null pointer, which is why this has not shown up.
   The null `&T` matters as soon as references get `llvm.nonnull`, which Vx already emits for `requires valid` pointers.

### R4. The C boundary

1. A ZST by value in an `extern` signature, as a parameter or as the return type, is an error with a new diagnostic code. The message says to remove the parameter.
1. A ZST field is allowed inside a struct that C reaches through a pointer. It takes no bytes, so the C struct without that field has the same layout.
1. A struct whose size is 0 cannot cross the boundary, by value or as a pointee. ISO C has no empty struct.
1. [`abi.md`](../lang/abi.md) gets a section that records these rules and the C++ difference. Vx does not follow the C++ ABI, so the difference is documented, not handled.
   Rejection is safer than erasure for rule 1. With erasure, the C declaration has to omit the parameter by hand, and nothing checks that it did.

### R5. Kernels

1. A ZST captured by a `spawn on` body is not a kernel parameter. It carries no data, so there is nothing to pass. Dropping it keeps the parameter list the same on every backend, whatever each backend would do with an empty parameter.
   The current behavior needs checking on the NVIDIA backend.

### R6. Artifacts

This is the rule with the most risk today. A struct in library A loses a ZST field, or its fields are reordered. Library B's `.vxlib` holds a body that reads a field of that struct by position. The bytes and the `TypeId` stay the same, so `CheckAggregate` passes, and the old index names a different field. When the types differ, the emitter declines or emits invalid MLIR. When they match, it reads a wrong value with no error. This is #2313 again, across a version boundary.

1. In memory, field instructions keep physical indices, as #2315 does.
1. In a `.vxlib`, a field reference is written as the aggregate's `TypeId`, the field's name and the field's type. Import resolves it to the current index. A missing name or a different type is an import error that names the field. Java does the same: JLS §13.1 compiles a field reference to its class, the field's name and its declared type, and the chapter's introduction lists adding and reordering fields among the changes that keep old binaries working.
1. Each imported body records the size and alignment it assumed for each aggregate it touches. Import refuses a mismatch. This catches what a field reference cannot, such as a folded `sizeof<T>()`.
1. The format tag becomes `vxlib-interface-v20`.
   With R6, adding a 1-ZST field or reordering fields no longer invalidates artifacts built downstream. Removing a field that a body reads becomes a clear error instead of a wrong load.

### R7. API diff

`api_diff` gives one verdict today. The [visibility discussion](../discussions/visibility-and-semver-package-management.md) already proposes separate verdicts for source and binary compatibility. ZSTs are the clearest case for the split:

| Change | Source | Memory layout | Artifacts today | Artifacts with R6 |
|---|---|---|---|---|
| Add a public 1-ZST field | major (a literal must give it) | unchanged | wrong field, if indices shift | compatible |
| Add a private 1-ZST field (once fields have visibility) | compatible | unchanged | wrong field, if indices shift | compatible |
| Remove a ZST field that a body reads | major | unchanged | wrong field | import error |
| Reorder fields | compatible (literals use names) | offsets change | wrong field | compatible |
| Add a ZST with alignment above 1 | major | alignment changes | wrong field | refused on import |

1. `api_diff` reports a field reorder. It is a binary break for any struct that crosses the C boundary.
1. The layout record keeps whether a type is a 1-ZST. Once fields can be private, downstream code can depend on that property without the source showing it.

### R8. Construction control

Proof types need a way to limit who can build them. Two options:

- **A. A private field.** `struct Init<Tag> { pkg seal : Seal }`, using the field visibility that the visibility discussion proposes. Only the package can write the literal. No new concept.
- **B. A construction marker on the struct.** It says who may write the literal. It works before field visibility exists, but it is a second mechanism for the same thing.
  This RFC prefers A.

### R9. Value serialization

Vx has no value serializer yet. These rules apply when it gets one:

1. A ZST encodes as nothing, and a sequence of ZSTs as its length. A decoder caps that length. A hostile length of 2^63 allocates nothing and loops forever.
1. Decoding is construction, so it obeys R8. A ZST is decodable only if it opts in. Markers such as `PhantomData` opt in. Proof and witness types never do, or a decoder could create a proof from nothing.
1. A serializer that tracks object identity keys it by address and type, never by address alone. By R2, a ZST field shares its address with its neighbor.
1. Moving bytes, as `transfer` and an mmap'd checkpoint do, needs only matching layouts. A ZST contributes nothing there.

## Work plan

Each item can be one sub-issue of #2195 or of a new tracking issue.

1. Merge #2315.
1. **Allocation (R3).** Dangling pointers for size 0 in `alloc`, `Box` and `Vec`, and a capacity rule for a `Vec` of ZSTs, with tests.
1. **C boundary (R4).** The diagnostic, the `abi.md` section, and tests for a parameter, a return value, a struct of only ZSTs, and a ZST field behind a pointer.
1. **Kernels (R5).** Check ZST captures in `spawn on`, and drop them from the kernel's parameters.
1. **Artifacts (R6).** Symbolic field references, the size and alignment record, format v20, and two-library tests: add a ZST field, remove one, reorder.
1. **API diff (R7).** Separate source and binary verdicts, and report reordering.
1. **Construction (R8).** Blocked on field visibility.
1. **Serialization (R9).** Record the rules in [`std_library_design.md`](../std_library_design.md) when value serialization is designed.
   R1 and R2 need documentation and tests, not code. Their first real test is the first ZST with alignment above 1.

## Open questions

1. Should an enum with no variants exist? It is the uninhabited type, the `False` of the proof calculus.
1. Should a payload-free enum with one variant be a ZST, as in Rust, instead of an `i32` tag? It would change the ABI of existing enums.
1. Should references get `llvm.nonnull`? If they do, R3 becomes required for soundness, not just tidiness.
1. Where should alignment above 1 come from: `align_of`, `[T; 0]` or an attribute? R1 should be settled before any of them lands.
1. Should debug information list ZST fields? A debugger that hides them shows a struct that does not match its source.

## Appendix: measurement sources

```c
// layout.c: clang -x c layout.c && ./a.out; clang++ -x c++ layout.c && ./a.out
#include <stdio.h>
#include <stddef.h>
struct E {};
struct F { struct E m; const int *p; };
struct FAM { char c; double d[]; };
struct Z { char a; long z[0]; char b; };
int main(void) {
  printf("%zu %zu %zu\n", sizeof(struct E), sizeof(struct F), offsetof(struct F, p));
  printf("%zu %zu\n", sizeof(struct FAM), _Alignof(struct FAM));
  printf("%zu %zu\n", sizeof(struct Z), offsetof(struct Z, b));
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

- [PR #2315: lower: preserve field identity when offsets coincide](https://github.com/vx-lang/Vx/pull/2315)
- [Issue #2313: preserve explicit field identity in flat loads, stores, and addresses](https://github.com/vx-lang/Vx/issues/2313)
- [Epic #2195: complete enum lowering and nested layout support](https://github.com/vx-lang/Vx/issues/2195)
- [Zero-sized types as proofs (hardmo.de)](https://www.hardmo.de/article/2021-03-14-zst-proof-types.md)
- [The Rustonomicon: Exotically Sized Types](https://doc.rust-lang.org/nomicon/exotic-sizes.html)
- [Java Language Specification, Chapter 13: Binary Compatibility](https://docs.oracle.com/javase/specs/jls/se23/html/jls-13.html)
- [Swift Library Evolution](https://apple-swift.readthedocs.io/en/latest/LibraryEvolution.html)
