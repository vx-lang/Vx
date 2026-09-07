# Design: Vector types and vector loads through pointers

**Status:** Proposed
**Scope:** A first-class SIMD vector type `<N x T>`, the rules for loading and storing it through raw pointers, its safe producers from tensors, and the `unsafe` boundary. No changes to `Tensor`, the memory algebra, or the borrow checker.
**Relation to other work:** independent of the tensor-spelling RFC; the tensor-side producers in Section 6 use `Tensor<T, [?<=N]>` from the bounded-extents RFC where noted, and work on static tensors without it.

______________________________________________________________________

## 0. For the implementing agent

Syntax is illustrative. `<4 x f32>` is the notation for the concept; map it onto the grammar, and if the tree already has a vector type or intrinsic path, report it before proposing anything. Phase 0 in Section 9 lists what to find first.

The decisions in this document are settled unless the tree contradicts them; a contradiction is a stop-and-report.

______________________________________________________________________

## 1. The problem

The program someone wants to write:

```
fn test_simd(p1: *mut f32, p2: *mut f32, p3: *mut f32) {
    let v1: <4 x f32> = *p1;
    let v2: <4 x f32> = *p2;
    *p3 = v1 + v2;
}
```

`*p1` on a `*mut f32` is one `f32`. Binding it to `<4 x f32>` hides three separate questions:

1. **Width** — reading four elements through a pointer to one is a reinterpretation of what is behind the pointer.
1. **Alignment** — a 16-byte vector load from a 4-byte-aligned address is a different instruction, or a fault, on some hardware.
1. **Count** — nothing in `*mut f32` says four elements exist.

Every existing language answers these with something implicit: C casts to `float4*` and relies on UB rules; intrinsics like `_mm_loadu_ps(float const*)` put the width in the function name; Swift's `UnsafeRawPointer.load(as:)` names the type at the load. The closest to type-safe is Zig, where a many-item pointer sliced with comptime bounds yields a pointer-to-array with the length in the type, which coerces to `@Vector`, and alignment is part of the pointer type. Rust reaches the same place through `[f32; 4]`, with the pointer step `unsafe`.

The honest model of the operation itself is MLIR's vector dialect, which is also the lowering target: `vector.load %base[%i] : memref<?xf32>, vector<4xf32>` — the source is a memref, the result type states the width, alignment is an attribute, out-of-bounds is handled by `maskedload` or `transfer_read` with `in_bounds`.

______________________________________________________________________

## 2. Decisions

1. **A vector is N lanes of T.** `<N x T>` is a POD value type, distinct from `Tensor`. It lowers to `vector<NxT>` and lives in registers. `Tensor` is linear with `NEEDS_DROP` and lowers to a memref. This is MLIR's own split — `Tensor` ↔ memref, vector ↔ vector — and it means the borrow checker ignores vectors (`TYPE_POD`).
1. **`*` never changes width.** Deref yields the pointee type. `let v: <4 x f32> = *p1` with `p1: *mut f32` is a type error.
1. **A vector load is a deref of a pointer to the vector type.** The program that wants a vector load takes `*mut <4 x f32>`.
1. **Reinterpretation is a pointer cast, and the cast is safe.** `p.cast::<<4 x f32>>()` produces a pointer value and asserts nothing — Rust's rule. The contract is discharged at the access.
1. **Access through a raw pointer is `unsafe`**, exactly as for every other raw pointer. This feature adds no safety regime of its own; it inherits the existing one.
1. **Scalar-to-vector conversion is splat**, spelled explicitly. It is never a load.
1. **The type has one alignment — the target datalayout's — and pointers carry the rest.** `<4 x f32>` is 16-aligned wherever the datalayout says so: struct fields, allocations, `align_of`. The frontend never computes a vector alignment by a rule of its own. Under-alignment is a property of a *pointer* (`*mut align(4) <4 x f32>`), and a cast from `*mut f32` produces exactly that.
1. **The safe producers of vectors are tensors.** A tensor knows its extent; a Vx-allocated tensor knows its alignment from the machine file. No assertion is needed.
1. **`unsafe` suspends the borrow checker's vouching, not the memory algebra's.** A raw pointer into a device space dereferenced on the host is still a placement error inside `unsafe`.
1. **Type-directed deref is rejected.** The annotation never decides how many bytes a load reads.
1. **Vectors are allowed as fields of ordinary structs**, at natural alignment. They are refused in `extern "C"` structs and by value across `extern "C"` calls in the first cut.
1. **`unsafe fn` for user functions is a prerequisite**, not something this design works around. See Section 7a.

______________________________________________________________________

## 3. The three operations, spelled apart

```
// 1. Vector load — the pointer type says what is behind it.
//    `unsafe fn`: the contract is on the caller's pointers, so the caller is the one
//    vouching (Rust's rule). The body still marks its own unsafe operations
//    (Rust-2024 rule, Section 7a). Parameters are under-aligned pointers because
//    that is what a cast from *mut f32 produces; loads emit alignment 4.
unsafe fn test_simd(p1: *mut align(4) <4 x f32>,
                    p2: *mut align(4) <4 x f32>,
                    p3: *mut align(4) <4 x f32>) {
    unsafe { *p3 = *p1 + *p2; }
}

// 2. Splat — "widening" a *value* is broadcast, not a load.
let s: <4 x f32> = splat(x);          // x: f32

// 3. Reinterpret — a safe cast on the pointer value; it preserves the source
//    alignment (4), so the result is an under-aligned vector pointer. The access
//    is where the assertion is discharged.
let q1: *mut align(4) <4 x f32> = p1.cast::<<4 x f32>>();   // p1: *mut f32
unsafe { let v = *q1; }
```

The caller that owns the buffer writes the cast and the `unsafe`:

```
let buf: *mut f32 = ...;                     // known: 4 floats behind it
unsafe { test_simd(buf.cast(), other.cast(), out.cast()); }
```

If the caller's buffer came from a Vx-allocated tensor whose granule proves 16-byte alignment, `t.as_vec_ptr::<4>()` yields a naturally aligned `*mut <4 x f32>` with no assertion, and a `test_simd` declared over `*mut <4 x f32>` emits aligned loads. Both signatures are legitimate; the pointer type records which contract the function asks for.

**Reinterpret where the fact is known.** Pushing the cast to the producer of the pointer means the assertion is made once, by the code that allocated the buffer. A callee that takes `*mut f32` and reinterprets internally re-asserts something it cannot check at every call.

______________________________________________________________________

## 4. The type `<N x T>`

- `N` is a comptime constant and, in the first cut, a power of two. (`<3 x f32>` has a padded size and its own alignment rule in the datalayout; handle it as a follow-up rather than guess.) `T` is a scalar element type (`f32`, `f16`, `i32`, ...; the fp8 types if the target has vector support for them, otherwise refused). `Tensor<<N x T>, ...>` is a type error — tensor elements are scalars.
- POD: no drop, copied by value, ignored by the borrow checker.
- Operations: elementwise arithmetic and comparison on same-typed vectors; `splat(x)`; lane access `v[i]` with `i` a comptime constant less than `N` (compile error otherwise); shuffles, reductions, and masked forms are follow-ups.
- Alignment and size are the target datalayout's, in every position. See Section 5.
- Lowering: `vector<NxT>`; arithmetic to `arith.*` on vectors.
- Identity: a generic instantiation over `T` and `N`, content-hashed like any other; `N` participates in the digest as an integer argument.

`<N x T>` is not `Tensor<T, [N]>`. Conversions between them are explicit and cheap (Section 6). Keeping them distinct is what lets a vector be a register value with no descriptor and no drop.

______________________________________________________________________

## 5. Alignment

**One source of truth: the target datalayout.** `<N x T>` has the alignment the datalayout assigns to `vector<NxT>` — 16 for `<4 x f32>` on x86-64 — in every position: struct fields, allocations, `size_of` / `align_of`. The frontend never computes a vector alignment by a rule of its own; it asks the datalayout (or the host machine file, with a test asserting the two agree). This is the invariant that keeps the frontend's layout and LLVM's layout identical, and it is the invariant an "element-aligned vector type" would have broken: a field the frontend places at offset 4 and LLVM places at offset 16 is the layout-versus-datalayout split, reopened.

Under-alignment is a property of pointers, Zig-style:

- `*mut <4 x f32>` — naturally aligned (16). Loads and stores emit `alignment = 16`.
- `*mut align(4) <4 x f32>` — under-aligned. Loads and stores emit `alignment = 4`, an unaligned vector load: correct on every target, free on modern hardware when the address happens to be aligned.
- **A cast preserves the source pointer's alignment and can never claim more.** `p.cast::<<4 x f32>>()` with `p: *mut f32` (alignment 4) yields `*mut align(4) <4 x f32>`. With `p: *mut align(16) f32` it yields `*mut <4 x f32>`. The alignment flows through the cast as a type-level fact, the same way the pointer's memory space does.
- `q.assume_aligned::<16>()` — upgrades an under-aligned pointer; `unsafe`.
- `t.as_vec_ptr::<4>()` on a Vx-allocated tensor whose granule proves ≥ 16 — naturally aligned, **safe**: the allocation's granule is declared in the machine file. One more place "hardware is declared, never assumed" pays off.
- `&s.v` where `s.v: <4 x f32>` is a struct field — naturally aligned, because the field is.
- Over-aligned pointers (`*mut align(64) <4 x f32>`) are permitted and emit the stated alignment.

### 5a. Vectors in structs and across FFI

- **Ordinary structs: allowed.** The field takes the datalayout's natural alignment and the struct's layout is whatever the datalayout produces. No frontend-side layout rule is added; the existing one, which already defers to the datalayout, covers it.
- **`extern "C"` structs: refused in the first cut** with a diagnostic. The C ABI for vector members is target-specific (SysV classifies `__m128` as SSE class; other targets differ) and not worth guessing before there is a user.
- **By-value vector parameters or returns across `extern "C"`: refused in the first cut.** Pass by pointer.
- These two refusals are the same policy as tensor fields in `extern "C"` structs: refuse rather than silently pick a layout.

______________________________________________________________________

## 6. Safe producers from tensors

These need no `unsafe`, because the tensor's type carries the count and (for Vx-allocated tensors) the alignment.

```
// Static tensor → vector value (copy)
let t: Tensor<f32, [4]> = ...;
let v: <4 x f32> = t.to_vec();

// Bounded tensor → full chunks + typed remainder (requires the bounded-extents RFC)
let a: Tensor<f32, [?<=N]> = ...;
for (x, y, o) in zip(a.chunks::<4>(), b.chunks::<4>(), out.chunks::<4>()) {
    o = x + y;                                // x, y, o : <4 x f32>
}
let tail: Tensor<f32, [?<=3]> = a.remainder::<4>();   // the tail has a type
```

`chunks::<W>()` yields `⌊extent / W⌋` vectors; every load is in bounds by construction, so it lowers to `vector.load` with no mask. `remainder::<W>()` is a bounded tensor with bound `W - 1`; the tail loop is ordinary typed code, not a magic epilogue. Borrow facts on `a`, `b`, `out` (word 2) establish no-alias, which is what lets the loads hoist — something raw pointers can never provide.

On a static tensor, `chunks` works without the bounded-extents RFC: the chunk count and remainder are comptime constants.

______________________________________________________________________

## 7. The `unsafe` surface

Vx has `unsafe` blocks and `unsafe fn` with Rust's semantics. This feature slots in without extending the regime.

**Safe**

- `p.cast::<<N x T>>()` — a pointer value; asserts nothing
- `splat(x)`; vector arithmetic; lane access with a comptime index
- `Tensor<T, [N]>::to_vec()`
- `chunks::<W>()`, `remainder::<W>()` on a tensor
- `Tensor::as_ptr()` — produces a raw pointer; using it is unsafe
- `Tensor::as_vec_ptr::<W>()` on a tensor with a provable base alignment

**Unsafe**

- `*q` for any raw pointer, including `*mut <N x T>`
- `q.add(i)` / `q[i]`
- `q.assume_aligned::<A>()`
- `Tensor::from_ptr(p, extents)` — asserts a count; if the existing `from_ptr_2d` is not `unsafe`, it should become so under this document

**Not unlocked by `unsafe`**

Placement. A raw pointer's memory space is part of its type (or, if it is not today, this is the Phase 0 question that matters most). Dereferencing a pointer into `Memory::HBM` from host code is an E60xx error inside `unsafe`, because the address space does not exist where the code runs. Rust has no analogue since it has one address space; Vx has several, and `unsafe` cannot argue with a machine file. Document this sentence prominently: Rust users will assume `unsafe` unlocks everything.

### 7a. Prerequisite: `unsafe fn` for user functions

Today the unsafe-call flag on a function is hard-coded `false` for anything that is not `extern`. The caller-vouches rule in Section 3 needs the declaration form, so this lands **before** the vector work, as its own small change:

- **Parse** `unsafe fn` on user function declarations.
- **Store** the flag in the signature's word-3 flags (`UNSAFE_FN`, next to visibility and the inline attributes). Not in identity: marking a function `unsafe` does not change what it is. Because it is in the signature, callers in other modules read it from the frozen registry without touching the body — bodies depend on signatures.
- **Call-site rule:** reuse the check that already fires for `extern` calls outside an `unsafe` context. Nothing new; the flag just stops being constant.
- **Body rule — adopt Rust 2024 from day one:** the body of an `unsafe fn` is *not* itself an unsafe context. Unsafe operations inside it still need `unsafe { }` blocks. `unsafe fn` states a contract on the caller; `unsafe { }` marks an operation whose proof the type system does not have. Keeping them orthogonal is what makes "where the proofs stop" readable at the operation rather than smeared over the function, and there is no edition to migrate later.

**Why this is a prerequisite and not a follow-up.** The alternative first cut — a safe-looking `fn test_simd(p1: *mut <4 x f32>, ...)` that vouches for its own pointers inside an `unsafe` block — is a safe function with an unchecked precondition. Rust forbids that by convention because it is unsound: any safe code can call it with garbage. For a language whose thesis is that guarantees are provable from the source text, it is also a lie in the signature. And it is a migration hazard: adding `unsafe` to a signature later breaks every caller, so doing it right first is cheaper than doing it twice. Given that the flag and the call-site check both exist, the change is parse plus one bit.

**If it must slip anyway:** callee-vouches is tolerable only for non-public functions, each with a `// SAFETY:` comment stating the precondition and a tracking issue. Never in the stdlib's public surface. `Tensor::from_ptr` in particular must be `unsafe fn`, not a safe wrapper.

______________________________________________________________________

## 8. Machine-declared vector width

The host machine file (`--host`) can declare the vector register width. Then:

- `<16 x f32>` on a host that declares 256-bit vectors is a compile error, not a silent split.
- A portable `<native x f32>` resolves its lane count from the declaration at compile time — Rust's `Simd<f32, LANES>` and Zig's `suggestVectorLength`, decided by the same file that decides everything else.

This is a follow-up, not part of the first cut, but the type should be designed so `N` can later come from a machine-file figure (which means `N` must go through the same "evaluated from frozen declarations, provenance kept for diagnostics" path as tensor bounds).

______________________________________________________________________

## 9. Implementation

### Phase 0 — investigate and report

1. Does any vector type, SIMD intrinsic, or `vector` dialect emission exist in the tree today? If so, what is its surface and lowering?
1. How are raw pointer types represented, and do they carry a memory space? (Section 7's placement rule depends on this.)
1. Which raw-pointer operations are `unsafe` today: deref, `add`, casts, `from_ptr_2d`? Is `from_ptr_2d` marked?
1. How does `p.cast::<U>()` (or its equivalent) work today, and is it safe?
1. Where does the flat path lower loads and stores through raw pointers; what opcode, what alignment handling?
1. How would `N` in `<N x T>` be digested into a GID — same path as `const N` generic parameters?
1. Whether `TYPE_POD` (word 3) is set by structural rules the vector type can reuse.
1. Where the unsafe-call flag lives (the one hard-coded `false` for non-`extern`), and where the call-site check for `extern` calls is implemented. Is there a free bit in word 3 for `UNSAFE_FN`?
1. How the frontend obtains type sizes and alignments today — does it query the datalayout, or compute them itself? (Section 5 requires the former for vectors; if the current path computes, report where the two are reconciled.)
1. How struct layout is computed, and whether a field type's alignment is taken from the same source as its size.

**Checkpoint:** written report; confirm the decisions in Section 2 against it.

### Phase 1 — `unsafe fn` (Section 7a)

Parse; `UNSAFE_FN` in word 3; call-site check unlocked; Rust-2024 body rule. Tests: calling an `unsafe fn` outside `unsafe` is an error; a raw deref inside an `unsafe fn` body still needs an `unsafe` block; the flag is readable from another module's frozen signature.

### Phase 2 — the type

`<N x T>`: parse, represent, identity, POD flag, alignment and size from the datalayout, lowering to `vector<NxT>`. Test: `align_of::<<4 x f32>>()` equals what LLVM reports for `vector<4xf32>` on the current target.

### Phase 3 — operations

Elementwise arithmetic, `splat`, comptime lane access. Differential tests flat vs oracle.

### Phase 4 — pointers

`*mut <N x T>` deref and store (unsafe, `llvm.load` / `llvm.store` with the pointer type's alignment); `cast` preserving source alignment (safe); `align(A)` in pointer types; `assume_aligned` (unsafe).

### Phase 5 — structs and FFI

Vector fields in ordinary structs; test that the frontend's field offset equals LLVM's for a struct `{ f32, <4 x f32> }` (expect offset 16, size 32). Refusals for `extern "C"` structs and by-value FFI.

### Phase 6 — tensor producers

`to_vec()` on static tensors; `as_vec_ptr::<W>()` with alignment derived from the allocation granule; `chunks` / `remainder` on static tensors. Bounded-tensor `chunks` lands with the bounded-extents RFC.

### Phase 7 — placement under `unsafe`

Test that a device-space raw pointer dereferenced on the host is an error inside `unsafe`. If raw pointers do not carry a space today, this is the moment to decide; report rather than implement.

### Phase 8 — docs

The three operations, the alignment rule, the `unsafe` surface, `unsafe fn` and the body rule, and the placement sentence.

______________________________________________________________________

## 10. Tests

- `let v: <4 x f32> = *p` with `p: *mut f32` is a type error.
- `*q` with `q: *mut align(4) <4 x f32>` outside `unsafe` is an error; inside, lowers to a 4-aligned vector load.
- `*q` with `q: *mut <4 x f32>` (natural) lowers to a 16-aligned load.
- `p.cast::<<4 x f32>>()` with `p: *mut f32` yields `*mut align(4) <4 x f32>`, compiles outside `unsafe`, and cannot be assigned to `*mut <4 x f32>` without `assume_aligned`.
- `align_of::<<4 x f32>>()` equals LLVM's alignment for `vector<4xf32>` on the current target; `struct { a: f32, v: <4 x f32> }` has `v` at the offset LLVM places it.
- `extern "C" struct` with a vector field is refused; by-value vector parameter on an `extern "C"` fn is refused.
- `Tensor<<4 x f32>, [2]>` is a type error.
- `unsafe fn f()` called outside `unsafe` is an error; a raw deref inside `f`'s body without an `unsafe` block is an error (Rust-2024 rule).
- `splat(x)` produces a broadcast, never a load (inspect the emitted IR).
- Lane access with an out-of-range comptime index is an error; with a runtime index is an error.
- `to_vec()` on `Tensor<f32, [4]>` compiles without `unsafe`; on `Tensor<f32, [5]>` is a type error.
- `chunks::<4>()` on `Tensor<f32, [10]>` yields two vectors and a `Tensor<f32, [2]>` remainder, all comptime.
- Device-space raw pointer deref on host inside `unsafe` is a placement error.
- Differential (flat vs oracle) for every new opcode; determinism gates unchanged.

______________________________________________________________________

## 11. Phase 0 report

Carried out against the tree at the phase 0 plan commit (`cd3fbd06`). No code changed. Every
claim below was read or measured. Phase references use this document's current numbering.

### 11.1 The seven original questions

1. **Existing vector surface.** `Type::Simd(ElementType, usize)` parses
   (`src/syntax/types.rs:626`) and the oracle maps it to `vector<NxT>`
   (`src/codegen/generator.rs:1585`). That mapping is the whole of it: no operations, no loads,
   no stores, and the flat path has no model at all (a `<4 x f32>` parameter declines with "a
   parameter type"). Two facts make the target easier than expected: the pass pipeline already
   runs `convert-vector-to-llvm` (`src/codegen/mod.rs:145`), and the slice reductions already
   emit `vector.load` and `vector.reduction` internally (`src/hir/check/calls.rs:1523`). The
   dialect is proven through the pipeline; only the surface is missing.

1. **Raw pointers and memory spaces.** The slot exists,
   `Pointer(Box<Type>, Option<MemorySpace>, bool)` (`src/syntax/types.rs:617`), and nothing ever
   fills it. The parser passes `None` (`src/parser/types.rs:169`), and no other site in the tree
   constructs a pointer with `Some(..)`. The deref check ignores it, binding
   `Type::Pointer(t, _, _)` (`src/hir/check/access.rs:906`). Section 7's placement rule has a
   slot, no data, and no rule, so the placement phase (Vx#480) is a spelling decision before it
   is an implementation.

1. **What is `unsafe` today.** Per-operation: a raw-pointer deref outside `unsafe` is refused
   (`src/hir/check/access.rs:907`), an integer-to-pointer `as` needs `unsafe`
   (`src/hir/check/operators.rs:165`), and `tensor_view_2d` needs `unsafe` for the reason this
   document gives for `from_ptr` (`src/hir/check/calls.rs:1474`). Per-function: `extern` blocks
   are unsafe by default and opt out with a `safe` keyword (`is_safe`, `src/syntax/decl.rs:88`,
   `src/parser/decl.rs:812`), registered as `!ext.is_safe` (`src/hir/env.rs:176`). A user
   function is registered `false` unconditionally (`src/hir/env.rs:195`) and the parser has no
   `unsafe fn` form, so the call-site check (`src/hir/check/calls.rs:610`) is live for externs
   and unreachable for user functions. This is what Section 7a is for.

1. **Pointer casts.** `check_ascast_expr` admits scalar to scalar, integer to pointer (under
   `unsafe`), and the closure-to-fat-pointer cast. There is no pointer-to-pointer arm, so the
   reinterpreting cast is new surface however it is spelled.

1. **Flat loads and stores through raw pointers.** `PtrIndex` and `PtrStore` lower to
   `llvm.getelementptr` plus `llvm.load`/`llvm.store` on the pointee scalar, with no alignment
   attribute anywhere. The pointer phase adds the attribute rather than changing the shape.

1. **Digesting `N` into a GID.** `src/layout.rs`'s `field_info` has no `Simd` arm
   (`src/layout.rs:180-195`), so a vector has no size, alignment, or `FieldTy` today and a struct
   containing one declines. The flat emitter's per-register type table is `ElementType`-based; a
   vector needs a parallel table, exactly as pointers needed `ptr_of`.

1. **POD.** `Type::is_linear` lists `Ref`, `Tensor`, `Matrix`, `Verified`, `Pinned`
   (`src/syntax/types.rs:643`); `Simd` is absent, so a vector is already non-linear. The GID bit
   `TYPE_NEEDS_DROP` (`src/gid.rs:51`) is opt-in. POD is therefore the default once `Simd` joins
   the layout and identity paths; no rule has to be relaxed for it.

### 11.2 The five checks the plan left open

1. **The deref check does not read the pointer's space.** Answered in 11.1 (2): no producer, no
   consumer.

1. **`*mut <4 x f32>` parses**, as `Pointer(Simd(F32, 4), None, true)`, read directly off the
   debug print described in 11.4. Both paths emit `!llvm.ptr` for such a parameter, since MLIR
   pointers are opaque. So the pointer-to-vector spelling costs nothing in the parser, and the
   reinterpreting cast costs one arm in `check_ascast_expr`.

1. **User `unsafe fn` does not exist**, per 11.1 (3). Size of adding it: a keyword before `fn`, a
   flag on `Function`, one changed line at `src/hir/env.rs:195`, and the call check at
   `src/hir/check/calls.rs:610` already does the rest. The `safe`/`is_safe` handling on externs
   is the shape to copy, inverted. Where the flag is stored is an open point; see 11.5 (1).

1. **A vector struct field sits at offset 16.** `!llvm.struct<(f32, vector<4xf32>)>` measures 32
   bytes with its vector field at offset 16 (zero-base `getelementptr` through
   `mlir-translate --mlir-to-llvmir`). An element-aligned vector would have put it at 4. This is
   the measurement behind decision 7 taking the datalayout's alignment in every position.

1. **A host allocation guarantees only `malloc`'s alignment.** `Tensor<f32, [8]>::new()` is a
   `memref.alloc`, which this tree's pipeline lowers to a plain `llvm.call @malloc(size)` with no
   alignment attribute (confirmed end to end with `--emit-llvm`). See 11.5 (5).

### 11.3 The original ten decisions, confirmed

Decisions 1 through 10 as first written were all confirmed against the tree; no contradiction was
found, and the checkpoint's stop condition did not fire. Two notes that survive the revision:

- **Decisions 2 and 10 are contradicted by current code, not by the design.** `is_assignable`
  admits a scalar where a vector is expected and a vector where a scalar is expected, with the
  comments "for loading from pointer" and "for storing to pointer" (`src/hir/expr.rs:354-370`).
  That leniency *is* the type-directed deref decision 10 rejects, and it is what lets today's
  `backend/pass/simd_test.vx` bind `*p1` to a `<4 x f32>`. The type phase removes it and rewrites
  that fixture as Section 3 shows. This is the one place the tree must change to match the
  design.
- **Decision 9's memory-algebra half has nothing to suspend yet**, per 11.1 (2). The sentence
  stays as the intended rule; the placement phase gives it an implementation.

### 11.4 Incidental defect found

`check_dereference_expr` prints `DEREF ERROR! inner_ty is ...` and a forced
`Backtrace::force_capture()` to **stdout** on the ordinary "deref outside `unsafe`" error path
(`src/hir/check/access.rs:908-910`): 52 lines measured for a three-line program. stdout is the IR
stream under `--action emit-mlir`. Filed as Vx#483, in the code the pointer phase extends. It
also served as the probe for check 2 above.

### 11.5 Open points

Raised reviewing the revision; none is a stop-and-report, and each wants an answer before the
phase that depends on it.

1. **"Word-3 flags, not in identity" (Section 7a) is self-contradictory as written.** `TypeId` is
   `words: [module_hash, symbol_hash, generic_hash, flags]` with `#[derive(PartialEq, Eq, Hash)]`
   over all four words (`src/gid.rs:99-112`), so setting a word-3 bit changes both equality and
   the hash. There is no precedent to lean on: `ATTR_INLINE` and `ATTR_MUST_USE` are declared in
   `gid.rs` and set nowhere in the tree, so no attribute bit is ever set on a real entity today.
   Either `UNSAFE_FN` lives outside the id, or identity comparison masks the attribute bits,
   which changes how every id compares. Phase 0 question 8 asks exactly this; answer it before
   placing the bit.

1. **"The frontend never computes a vector alignment by a rule of its own; it asks the
   datalayout" (Section 5) is not how this tree works.** `src/layout.rs` computes every layout
   itself: `scalar_size_align` derives size and alignment from `ElementType::bits()`, a pointer
   field is a hardcoded `(8, 8)`, and a tensor field's descriptor size is arithmetic. The only
   datalayout anywhere is a *string* attached to the MLIR module for the target
   (`src/driver.rs:969-988`, `arch_triple_and_datalayout`); there is no queryable datalayout API
   on the frontend side. The invariant decision 7 states is right; the mechanism has to be this
   section's own parenthetical, compute it and pin it with a test asserting agreement with LLVM.
   Phase 0 question 9 asks this, and the answer is "it computes". Question 10's answer is that a
   struct field's size and alignment both come from the same `field_info` call, so they cannot
   disagree with each other.

1. **The alignment subtyping direction is unstated.** Section 10 says an under-aligned pointer
   cannot be assigned to a naturally aligned one without `assume_aligned`, which is right. The
   reverse should be stated as allowed: `*mut align(16) <4 x f32>` must be usable where
   `*mut align(4) <4 x f32>` is expected. Section 3's own example depends on it, since a caller
   holding an `as_vec_ptr` result passes it to a `test_simd` declared over the under-aligned
   form.

1. **The cast spelling is still open.** `p.cast::<<4 x f32>>()` (decision 4, Section 3) puts a
   `<` immediately after `::<`. Spelling it `p as *mut <4 x f32>` avoids that, matches how this
   tree already casts pointers, and needs no new type grammar: `*mut <4 x f32>` parses today, per
   11.2 (2). Recorded as an alternative, not a decision.

1. **"A Vx-allocated tensor knows its alignment from the machine file" (decision 8, Section 5)
   holds for declared sub-spaces, not for host allocations.** Measured, a host tensor is a plain
   `malloc` with no alignment attribute, and no granule is declared for `CPU_DRAM`, so the
   alignment `as_vec_ptr` may state there is the target C ABI's `malloc` guarantee, a platform
   fact rather than a declared one. A declared sub-space with a granule is the case the sentence
   actually covers. Anything stronger is requested with `memref.alloc`'s `alignment` attribute
   rather than assumed.
