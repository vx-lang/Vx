# Vector types, phase 0: confirm the design against the tree

**Tracking:** Vx#475 (this phase), Vx#482 (meta), design in `docs/discussions/design-vector-types.md`.
**Outcome:** a written report, four amendments recorded in the design, no code changes.
**Working rules:** read `agents/AGENTS.md` first. Every `cargo` and `mlir-*` command needs
`export PATH="/opt/homebrew/opt/llvm/bin:$PATH"` and `-j 8`. Do not edit the tree while a
cargo run is in flight. Do not commit; the report is the deliverable.

## 1. What the phase is for

The design's Section 9 lists seven questions to answer before any code, with a checkpoint:
"written report; confirm the decisions in Section 2 against it." Most of the seven were
answered during review (Vx#472). This plan records those answers so they are not re-derived,
names the three that still need a look, and says exactly what to write down.

A contradiction between the tree and a Section 2 decision is a stop-and-report, per the
design's Section 0. None has been found so far; the amendments below are additions, not
contradictions.

## 2. Answered already (verify the line numbers still hold, then cite them)

| # | Question | Answer | Where |
|---|----------|--------|-------|
| 1 | Existing vector surface | `Type::Simd(ElementType, usize)` parses. The oracle maps it to `vector<NxT>` for signatures and nothing else; no operations, no loads. The flat path has no model (a `<4 x f32>` parameter declines with "a parameter type"). The slice reductions emit `vector.load` + `vector.reduction` internally, so the vector dialect already passes the lowering pipeline. | `src/syntax/types.rs:626`, `src/codegen/generator.rs:1585`, `src/hir/check/calls.rs:1523`, `src/hir/flatten.rs` (`lower_ty_synth` has no `Simd` arm) |
| 2 | Raw pointers and memory spaces | `Type::Pointer(Box<Type>, Option<MemorySpace>, bool)`: the slot exists. The parser never fills it: every parsed pointer is `Pointer(.., None, ..)`. | `src/syntax/types.rs:617`, `src/parser/types.rs:169` |
| 3 | What is `unsafe` today | Deref of a raw pointer outside `unsafe` is refused. Integer-to-pointer `as` needs `unsafe`. `tensor_view_2d` (the tree's `from_ptr_2d`) needs `unsafe`, with the design's reason in its comment. Only `extern` functions carry `is_unsafe`; a user function is registered with the flag hard-coded `false`. | `src/hir/check/access.rs:907`, `src/hir/check/operators.rs:165`, `src/hir/check/calls.rs:1462-1480` and `:610`, `src/hir/env.rs:195` |
| 4 | Pointer casts | `check_ascast_expr` admits scalar→scalar, integer→pointer (unsafe), and the closure→fat-pointer cast. There is no pointer-to-pointer cast. | `src/hir/check/operators.rs` (`check_ascast_expr`) |
| 5 | Flat loads/stores through raw pointers | `PtrIndex`/`PtrStore` opcodes: `llvm.getelementptr` + `llvm.load`/`llvm.store` on the pointee scalar, no alignment attribute. | `src/hir/flatten.rs` (search `Opcode::PtrIndex`), `src/codegen/flat/emit/` (the `op_ptr_index`/`op_ptr_store` emitters) |
| 6 | Digesting `N` into a GID | `src/layout.rs` has no `Simd` case (`field_info` and the type-identity path), so a vector has no GID today. The flat emitter's per-register type table is `ElementType`-based (`etypes`); pointers already needed a parallel table (`ptr_of`), and a vector will need one too. | `src/layout.rs`, `src/codegen/flat.rs:1579-1610` |
| 7 | POD flag | Linearity is `Type::is_linear` (tensors are linear); the GID word-3 bit is `TYPE_NEEDS_DROP`. A vector is neither linear nor needs drop, so both rules give POD by default once `Simd` is added to them. | `src/syntax/types.rs:643`, `src/gid.rs:51` |

Also relevant, found during review: the checker admits a scalar where a vector is expected
and back ("for loading from pointer" / "for storing to pointer"), which is exactly the
type-directed deref decision 10 rejects. That leniency is what phase 1 removes.
`src/hir/expr.rs:354-370`.

## 3. Still to check

Each is a read plus, where noted, one program compiled with the built `target/debug/vxc`
(`--action emit-mlir`, add `--legacy-codegen` for the oracle). Record the answer with a
file:line.

1. **Does the deref check read the pointer's space?** Read `src/hir/check/access.rs` around
   line 907 and the `Type::Pointer` arms near it. Since the parser never fills the space, also
   answer: is there any path that produces a `Pointer(.., Some(space), ..)` today (search
   `Some(MemorySpace` and `Pointer(` across `src/hir`)? Expected answer: no, and the check
   does not consult it. That makes phase 5 (Vx#480) a spelling decision first.

1. **The cast spelling.** Amendment A replaces `p.cast::<<N x T>>()` with `p as *mut <N x T>`.
   Confirm the type grammar already parses `*mut <4 x f32>` as a pointer to a `Simd` (write
   `fn f(p: *mut <4 x f32>) -> i32 { return 0; }` and compile it on the oracle). If it parses,
   the amendment costs one arm in `check_ascast_expr`; if it does not, note what the parser
   does instead.

1. **`unsafe fn` for user functions.** Amendment D. Confirm `src/hir/env.rs:195` is the only
   place the flag is set for user functions and that the parser has no `unsafe fn` form
   (search `Unsafe` in `src/parser/decl.rs`; today it appears only for blocks). State the
   size of adding it: a keyword before `fn`, a field on `Function`, the flag threaded to
   `env.functions`, and the call check at `calls.rs:610` already does the rest.

1. **A vector as a struct field (amendment B).** Read `src/layout.rs`'s `field_info` and note
   what an unknown type does today (declines the struct, or panics). Confirm the data-layout
   fact the amendment rests on: emit a module with `!llvm.struct<(f32, vector<4xf32>)>` through
   `mlir-opt --convert-to-llvm` and `mlir-translate --mlir-to-llvmir`, and read the offset LLVM
   gives the vector field (expected 16, not 4). That number is the argument for natural
   alignment in structs.

1. **Alignment a host allocation guarantees (amendment C).** Find the runtime allocation the
   flat path's `memref.alloc` becomes (search `aligned_alloc\|posix_memalign\|malloc` in
   `stdlib/rust_core/src` and the MLIR lowering options in `src/pipeline.rs`). Record the
   alignment it promises; that is the number `as_vec_ptr` may state for `CPU_DRAM`.

## 4. Deliverable

Append a section `## 10. Phase 0 report` to `docs/discussions/design-vector-types.md`:

- the seven numbered answers, one paragraph each, every claim with a file:line;
- the answers to Section 3 above, same form;
- a line per Section 2 decision saying "confirmed" or naming the contradiction (none is
  expected);
- the four amendments A–D, in the wording of the meta issue Vx#482, each with the tree fact
  that motivates it.

Then edit the design's Section 2 in place: add the amendments as decisions 11–14, so the
document is the single source of truth going forward. Set its status line to "Adopted".

Post the report's summary (the decision lines and the amendments) as a comment on Vx#475 and
tick the phase in Vx#482. Phase 1 is Vx#476; do not start it under this plan.

## 5. Stop conditions

- A Section 2 decision the tree contradicts: stop, write the contradiction into the report,
  and say so on Vx#475 before anything else.
- `*mut <4 x f32>` does not parse: report; the spelling of amendment A is then open.
- The LLVM offset for a vector field is 4, not 16: amendment B is moot; report and drop it.
