# Design: Vector types and vector loads through pointers

**Status:** Adopted (phase 0 complete; see Section 11)
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
1. **Vectors are element-aligned by default.** `<4 x f32>` has alignment 4, not 16. Higher alignment is stated in the pointer type.
1. **The safe producers of vectors are tensors.** A tensor knows its extent; a Vx-allocated tensor knows its alignment from the machine file. No assertion is needed.
1. **`unsafe` suspends the borrow checker's vouching, not the memory algebra's.** A raw pointer into a device space dereferenced on the host is still a placement error inside `unsafe`.
1. **Type-directed deref is rejected.** The annotation never decides how many bytes a load reads.
1. **The reinterpreting cast is spelled `p as *mut <N x T>`.** Safe, asserting nothing, as decision 4 says. `p.cast::<<N x T>>()` puts a `<` immediately after `::<`, and the tree already casts pointers with `as`.
1. **A vector inside a struct takes natural alignment**, `N * align_of(T)` capped at the target's vector register width; element alignment (decision 7) governs pointers only. LLVM lays a `vector<4xf32>` field out at offset 16, so element alignment in a struct would give the layout table one answer and the data layout another.
1. **The alignment a tensor-derived vector pointer may state comes from the allocation's own guarantee**: the target C ABI's `malloc` alignment for host spaces (a Vx tensor is a plain `malloc`), or a declared granule for a declared sub-space. Anything stronger is requested with `memref.alloc`'s `alignment` attribute, not assumed.
1. **A user `unsafe fn` is a prerequisite of phase 3, or the callee vouches with a block until it exists.** Only `extern` blocks carry a function-level safety flag today.

______________________________________________________________________

## 3. The three operations, spelled apart

```
// 1. Vector load — the pointer type says what is behind it.
//    `unsafe fn`, not a function containing an unsafe block: the contract is on the
//    caller's pointers, so the caller is the one vouching (Rust's rule).
unsafe fn test_simd(p1: *mut <4 x f32>, p2: *mut <4 x f32>, p3: *mut <4 x f32>) {
    *p3 = *p1 + *p2;
}

// 2. Splat — "widening" a *value* is broadcast, not a load.
let s: <4 x f32> = splat(x);          // x: f32

// 3. Reinterpret — a safe cast on the pointer value; the access is where the
//    assertion is discharged.
let q1: *mut <4 x f32> = p1.cast::<<4 x f32>>();
unsafe { let v = *q1; }
```

The caller that owns the buffer writes the cast and the `unsafe`:

```
let buf: *mut f32 = ...;                     // known: 4 floats, 4-aligned
unsafe { test_simd(buf.cast(), other.cast(), out.cast()); }
```

**Reinterpret where the fact is known.** Pushing the cast to the producer of the pointer means the assertion is made once, by the code that allocated the buffer. A callee that takes `*mut f32` and reinterprets internally re-asserts something it cannot check at every call.

______________________________________________________________________

## 4. The type `<N x T>`

- `N` is a comptime constant. `T` is a scalar element type (`f32`, `f16`, `i32`, ...; the fp8 types if the target has vector support for them, otherwise refused).
- POD: no drop, copied by value, ignored by the borrow checker.
- Operations: elementwise arithmetic and comparison on same-typed vectors; `splat(x)`; lane access `v[i]` with `i` a comptime constant less than `N` (compile error otherwise); shuffles, reductions, and masked forms are follow-ups.
- Natural alignment is `align_of::<T>()`. See Section 5.
- Lowering: `vector<NxT>`; arithmetic to `arith.*` on vectors.
- Identity: a generic instantiation over `T` and `N`, content-hashed like any other; `N` participates in the digest as an integer argument.

`<N x T>` is not `Tensor<T, [N]>`. Conversions between them are explicit and cheap (Section 6). Keeping them distinct is what lets a vector be a register value with no descriptor and no drop.

______________________________________________________________________

## 5. Alignment

- `*mut <4 x f32>` obtained by `cast` from a `*mut f32` has alignment 4. A load through it lowers to `llvm.load` with `alignment = 4` — an unaligned vector load, correct on every target and free on modern hardware when the address happens to be aligned.
- Higher alignment is part of the pointer type, Zig-style: `*mut align(16) <4 x f32>`. A load through it lowers with `alignment = 16`.
- Producing an aligned pointer from an unaligned one is an assertion: `q.assume_aligned::<16>()` is `unsafe`.
- Producing an aligned pointer from a Vx-allocated tensor is **not** an assertion: the allocation's granule is declared in the machine file, so `t.as_vec_ptr::<4>()` on a tensor whose base is provably 16-aligned returns `*mut align(16) <4 x f32>` safely. This is one more place "hardware is declared, never assumed" pays off.

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

**Checkpoint:** written report; confirm the decisions in Section 2 against it.

### Phase 1 — the type

`<N x T>`: parse, represent, identity, POD flag, lowering to `vector<NxT>`.

### Phase 2 — operations

Elementwise arithmetic, `splat`, comptime lane access. Differential tests flat vs oracle.

### Phase 3 — pointers

`*mut <N x T>` deref and store (unsafe, `llvm.load` / `llvm.store` with alignment attribute); `cast` (safe); `align(A)` in pointer types; `assume_aligned` (unsafe).

### Phase 4 — tensor producers

`to_vec()` on static tensors; `as_vec_ptr::<W>()` with alignment derived from the allocation granule; `chunks` / `remainder` on static tensors. Bounded-tensor `chunks` lands with the bounded-extents RFC.

### Phase 5 — placement under `unsafe`

Test that a device-space raw pointer dereferenced on the host is an error inside `unsafe`. If raw pointers do not carry a space today, this is the moment to decide; report rather than implement.

### Phase 6 — docs

The three operations, the alignment rule, the `unsafe` surface, and the placement sentence.

______________________________________________________________________

## 10. Tests

- `let v: <4 x f32> = *p` with `p: *mut f32` is a type error.
- `*q` with `q: *mut <4 x f32>` outside `unsafe` is an error; inside, lowers to a 4-aligned vector load.
- `*q` with `q: *mut align(16) <4 x f32>` lowers to a 16-aligned load.
- `p.cast::<<4 x f32>>()` outside `unsafe` compiles.
- `splat(x)` produces a broadcast, never a load (inspect the emitted IR).
- Lane access with an out-of-range comptime index is an error; with a runtime index is an error.
- `to_vec()` on `Tensor<f32, [4]>` compiles without `unsafe`; on `Tensor<f32, [5]>` is a type error.
- `chunks::<4>()` on `Tensor<f32, [10]>` yields two vectors and a `Tensor<f32, [2]>` remainder, all comptime.
- Device-space raw pointer deref on host inside `unsafe` is a placement error.
- Differential (flat vs oracle) for every new opcode; determinism gates unchanged.

______________________________________________________________________

## 11. Phase 0 report

Completed against the tree at `7ea26962`. No code changed. Every claim below was read or
measured; file:line references were verified at that commit.

### 11.1 The seven questions

1. **Existing vector surface.** `Type::Simd(ElementType, usize)` parses
   (`src/syntax/types.rs:626`) and the oracle maps it to `vector<NxT>`
   (`src/codegen/generator.rs:1585`). That mapping is the whole of it: no operations, no
   loads, no stores, and the flat path has no model at all (a `<4 x f32>` parameter declines
   with "a parameter type"). Two facts make the target easier than expected: the pass
   pipeline already runs `convert-vector-to-llvm` (`src/codegen/mod.rs:145`), and the slice
   reductions already emit `vector.load` and `vector.reduction` internally
   (`src/hir/check/calls.rs:1523`). The dialect is proven through the pipeline; only the
   surface is missing.

1. **Raw pointers and memory spaces.** The slot exists,
   `Pointer(Box<Type>, Option<MemorySpace>, bool)` (`src/syntax/types.rs:617`), and nothing
   ever fills it. The parser passes `None` (`src/parser/types.rs:169`), and no other site in
   the tree constructs a pointer with `Some(..)`. The deref check ignores it, binding
   `Type::Pointer(t, _, _)` (`src/hir/check/access.rs:906`). Section 7's placement rule has a
   slot, no data, and no rule, so phase 5 (Vx#480) is a spelling decision before it is an
   implementation.

1. **What is `unsafe` today.** Per-operation: a raw-pointer deref outside `unsafe` is refused
   (`src/hir/check/access.rs:907`), an integer-to-pointer `as` needs `unsafe`
   (`src/hir/check/operators.rs:165`), and `tensor_view_2d` needs `unsafe` for the reason this
   document gives for `from_ptr` (`src/hir/check/calls.rs:1474`). Per-function: `extern`
   blocks are unsafe by default and opt out with a `safe` keyword (`is_safe`,
   `src/syntax/decl.rs:88`, `src/parser/decl.rs:812`), registered as `!ext.is_safe`
   (`src/hir/env.rs:176`). A user function is registered `false` unconditionally
   (`src/hir/env.rs:195`) and the parser has no `unsafe fn` form, so the call-site check
   (`src/hir/check/calls.rs:610`) is live for externs and unreachable for user functions.

1. **Pointer casts.** `check_ascast_expr` admits scalar to scalar, integer to pointer (under
   `unsafe`), and the closure-to-fat-pointer cast. There is no pointer-to-pointer arm, so the
   cast of decision 4 is new surface either way.

1. **Flat loads and stores through raw pointers.** `PtrIndex` and `PtrStore` lower to
   `llvm.getelementptr` plus `llvm.load`/`llvm.store` on the pointee scalar, with no alignment
   attribute anywhere. Phase 3 adds the attribute rather than changing the shape.

1. **Digesting `N` into a GID.** `src/layout.rs`'s `field_info` has no `Simd` arm
   (`src/layout.rs:180-195`), so a vector has no size, alignment, or `FieldTy` today and a
   struct containing one declines. The flat emitter's per-register type table is
   `ElementType`-based; a vector needs a parallel table, exactly as pointers needed `ptr_of`.

1. **POD.** `Type::is_linear` lists `Ref`, `Tensor`, `Matrix`, `Verified`, `Pinned`
   (`src/syntax/types.rs:643`); `Simd` is absent, so a vector is already non-linear. The GID
   bit `TYPE_NEEDS_DROP` (`src/gid.rs:51`) is opt-in. POD is therefore the default once `Simd`
   joins the layout and identity paths; no rule has to be relaxed for it.

### 11.2 The five open checks

1. **The deref check does not read the pointer's space.** Answered in 11.1 (2): no producer,
   no consumer.

1. **`*mut <4 x f32>` parses**, as `Pointer(Simd(F32, 4), None, true)` (read directly off the
   debug print described in 11.4). Both paths emit `!llvm.ptr` for such a parameter, since
   MLIR pointers are opaque. Amendment A therefore costs one arm in `check_ascast_expr` and
   nothing in the parser.

1. **User `unsafe fn` does not exist**, per 11.1 (3). The size of adding it: a keyword before
   `fn`, a bool on `Function`, one changed line at `src/hir/env.rs:195`, and the call check at
   `src/hir/check/calls.rs:610` already does the rest. The `safe`/`is_safe` handling on
   externs is the shape to copy, inverted.

1. **A vector struct field sits at offset 16.** `!llvm.struct<(f32, vector<4xf32>)>` measures
   32 bytes with its vector field at offset 16 (zero-base `getelementptr` through
   `mlir-translate --mlir-to-llvmir`). Element alignment would put it at 4. Amendment B
   stands.

1. **A host allocation guarantees only `malloc`'s alignment.** `Tensor<f32, [8]>::new()` is a
   `memref.alloc`, which this tree's pipeline lowers to a plain `llvm.call @malloc(size)` with
   no alignment attribute (confirmed end to end with `--emit-llvm`). So the number
   `as_vec_ptr` may state for a host space is the target C ABI's malloc guarantee, 16 bytes on
   both supported hosts, and it is a platform fact rather than a declared one. Anything
   stronger needs `memref.alloc`'s `alignment` attribute. Amendment C is worded accordingly
   as decision 13.

### 11.3 Section 2, confirmed

Decisions 1 through 10 are all confirmed against the tree; no contradiction was found, and
the checkpoint's stop condition did not fire. Two notes:

- **Decisions 2 and 10 are contradicted by current code, not by the design.** `is_assignable`
  admits a scalar where a vector is expected and a vector where a scalar is expected, with the
  comments "for loading from pointer" and "for storing to pointer"
  (`src/hir/expr.rs:354-370`). That leniency *is* the type-directed deref decision 10 rejects,
  and it is what lets today's `backend/pass/simd_test.vx` bind `*p1` to a `<4 x f32>`. Phase 1
  removes it and rewrites that fixture as Section 3 shows. Recorded here as the one place the
  tree must change to match the design.
- **Decision 9's memory-algebra half has nothing to suspend yet**, per 11.1 (2). The sentence
  stays as the intended rule; phase 5 gives it an implementation.

### 11.4 Incidental defect found

`check_dereference_expr` prints `DEREF ERROR! inner_ty is ...` and a forced
`Backtrace::force_capture()` to **stdout** on the ordinary "deref outside `unsafe`" error path
(`src/hir/check/access.rs:908-910`): 52 lines measured for a three-line program. stdout is the
IR stream under `--action emit-mlir`. Filed as Vx#483, in the code phase 3 extends. It also
served as the probe for check 2 above.
