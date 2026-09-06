# Review: tensor-spelling RFC (pre-launch) and bounded-extents RFC (post-launch)

**Reviews:** `rfc-tensor-spelling.md`, `rfc-bounded-extents.md`
**Reviewed against:** `main` at `85342711`
**Follows:** `rfc-unified-tensor-vx-review.md`

The split is the right call. The spelling RFC is small, the bounded RFC is additive on
top of it, and both took the first review's findings on board (implicit widening stays,
receivers are covered, the solver wording now matches the tree, codes continue after
E6018). What is left is below, ordered by how much each changes the plan.

## 1. Tensor-spelling RFC

### 1.1 The representation decision is missing, and it is the largest cost

Dims are `Vec<Expr>` (`Type::Tensor(ElementType, Vec<Expr>, Option<Placement>)`). `?` is
not an expression. The RFC never says how a `?` dimension is represented, and the two
options have very different costs:

- **`Expr::DynDim`.** Small diff, but it leaks a non-expression into every `Expr` match,
  `substitute`, `stamp_dim_literals`, the mangler, and the const evaluator, each of which
  then has to refuse it.
- **`Vec<Dim>` with `Dim::{Expr(Expr), Dyn}`** (what Vx#245 proposed as
  `Dim::{Exact, Bounded}`). The honest shape, and the one the bounded RFC needs a year
  later for `?<=B`. It touches every `Type::Tensor(` pattern: 90 sites across 25 files,
  plus 37 `Type::DynTensor(` sites that go away.

Choose the second. Doing it once, before the tag, is the point of the spelling RFC; the
bounded RFC then adds a variant instead of reopening 90 sites.

### 1.2 The dims-list requirement is already the tree's behavior

§3.1's second bullet — `Tensor<T>` with no list does not parse as a type — has been true
since Vx#399 (`DIMS_REQUIRED`, `src/parser/types.rs:15-18`, and the tracking at
`:356-392`). Phase 1 shrinks to: the `?` token, `?` inside a dims list, and the
`DynTensor` fix-it. Note that each dim is parsed by `parse_expr()` today
(`src/parser/types.rs:365`); `?` has to be intercepted before that call, and `?<=B` later
is a dim-level production, not an expression.

### 1.3 §2's "correctness gap" in the flat identity is asserted, not shown

`tensor_gid` hashes element and shape only, and the mangler excludes placement on
purpose (`src/syntax/types.rs`, "Topology stays excluded"). On the flat path the
placement rides on the **alloc instruction**, not the type: `TensorAlloc`'s spare operand
carries `memory_space_dispatch_id(&p.space)` (`src/hir/flatten.rs:2572-2584`), and the
memref type the emitter writes has no space either way. Two tensors that differ only in
placement lower to the same memref, so sharing a GID is a representation choice that
matches the target, and adding placement to the hash would split `TensorMap` entries
without changing a byte of output.

Phase 0 question 2 asked for the test. It exists now —
`flat_matches_ast_two_tensors_differing_only_in_placement` in
`flat_codegen_differential.rs` — and it passes: two tensors that differ only in placement,
in one program, come out the same on both paths. §3.4's identity change is a no-op and
§2's third paragraph is withdrawn. The memory space stays out of `tensor_gid`.

### 1.4 Two impls can match one receiver, and nothing says which wins

Today `fill` lives on `impl<T: Float> DynTensor<T>` and `fill_static` on
`impl<T: Float, const N: i32, const M: i32> Tensor<T, [N, M]>`
(`stdlib/std/tensor.vx:9, :219`). Different names, so no ambiguity. Under §3.3 a `[2, 3]`
receiver matches both `[?, ?]` and `[N, M]`. The bounded RFC merges `fill` and
`fill_static` (§5.8 there), so the rule is needed by then, and it belongs in the spelling
RFC because that is where `?` enters patterns. Propose: most-specific pattern wins, where
a static or `const` dimension is more specific than `?`, and a tie is an error.

The converse also needs stating: a `[?, 4]` receiver against a `[N, M]` pattern must
**fail** to bind `N` (a `const` parameter cannot bind to `?`) rather than binding it to a
non-constant. `unify_types_internal` (`src/hir/env.rs:661-700`) binds `N` to the concrete
dim `Expr` today with no such check.

### 1.5 Phase 0 answers available now

1. **Constructor result type.** All dims fold → `Type::Tensor(el, dims, None)`; any dim
   that does not → `Type::DynTensor(el, None)` (`src/hir/check/calls.rs:1303-1319`; the
   same rule for `from_ptr_2d`-style views at `:1470-1487`). It is all-or-nothing today,
   so §3.6's per-position derivation (`[?, 3]` from `[n, 3]`) is new behavior, not a
   generalization. It is the right behavior.
1. **Placement-only difference on the flat path.** See §1.3. The differential test at
   `flat_codegen_differential.rs:896-901` already compiles a placed `uninit`; extend it
   with an unplaced twin in the same program.
1. **Rank at each `DynTensor` site.** Every `DynTensor` type position in the corpus is
   rank 2, and every constructor call whose argument list is not two elements is a nested
   literal initializer (`Tensor<f32>([[..], [..]])`), which is also rank 2. The rank-1
   case from Vx#404 (`Tensor<f32>([n])` into a `DynTensor` parameter) exists in the
   corpus only in a comment. Expect **zero** rank errors from migration step 2, and write
   the Vx#404 regression by hand — the corpus will not produce it.
1. **`.shape` sites.** Five, in two files (`tests/frontend/pass/custom_matmul.vx`,
   `tests/frontend/pass/macro_custom_tensor.vx`). Cheap enough to migrate outright
   rather than warn.

The stdlib's `DynTensor<T>` methods are rank 2 by their bodies, not only by their
patterns: 20 inline-MLIR lines in `stdlib/std/tensor.vx` and `llama.vx` spell
`memref<?x?xT>`. `[?, ?]` is the honest rewrite and no method becomes rank-generic by
the rename.

### 1.6 `extent(i)` needs an oracle lowering too

§3.7 says `extent(i)` lowers to `TensorDim` (opcode 44) on the flat path. The oracle path
lowers `.shape` at `src/codegen/lower/expr.rs:459`; `extent(i)` needs the same
`memref.dim` there, which is a rename of that arm.

### 1.7 §6 is a separate project and should not gate the spelling change

`scripts/campaigns/admission_matrix.sh` exists and is one config against every SKU in
`fleet/` (11 SKU files after excluding `admit.vx` and `host-*`); the "15 configs" and
"6 SKU files" in §6 do not match the tree, and there is no llama2.c token comparison
harness under `tests/` or `scripts/`. Each row of §6 is real work with hardware
dependencies. Land the spelling RFC behind the gates that exist (its §0 list is right),
and file §6 as its own ticket so the tag is not held on a 2×H100 test.

### 1.8 §7 cites numbers the tree does not contain

"95 of 130 corpus programs" appears nowhere in `flat_corpus_sweep.rs` or its output
format. `KNOWN_DECLINES` has roughly 68 entries. Whatever the README ends up saying
should be read off the sweep's actual output on the day, not carried in from a document.

## 2. Bounded-extents RFC

### 2.1 "Tensors are linear, so a value cannot be reassigned between a guard and a use" is wrong

Linear means consumed at most once per path. Assignment to a tensor binding is legal:
`c = a @ b` (`tests/frontend/pass/matmul_assign_in_place.vx:37`,
`benchmarks/flash_attention_ane/flash_attention_split.vx:170`). A matmul assignment
fills the existing buffer in place (Vx#391) and keeps its shape, but `t = transfer(..)`
or `t = Tensor<..>::uninit(..)` re-binds `t` to a fresh allocation whose extents the
guard never saw. So inside `if t.extent(0) <= 2048 { t = ..; narrow<..>(t) }` the guard
proves nothing about the `t` at the narrow. §5.5's lexical-guard rule needs one more
condition: no assignment to the guarded binding between the guard and the `narrow`. That
is a syntactic scan of the then-branch, still no flow analysis, and it has to be stated.

### 2.2 "Sized by the bound" is not what `memref.alloc` does

§5.9 says bounded allocation "reuses the alloc path with size operands from the bound and
extents written to the descriptor." `memref.alloc(%e0, %e1) : memref<?x?xT>` allocates
`e0 × e1` and sets sizes to the extents; there is no way to hand it a larger buffer. The
shape §5.7 describes — buffer of `B0 × B1`, descriptor sizes `[e0, e1]`, strides
`[e1, 1]` — is `memref.alloc() : memref<B0*B1 x T>` followed by
`memref.reinterpret_cast` to `memref<?x?xT>` with the extents as sizes and `[e1, 1]` as
strides. That is a new emission shape on both paths, not a reuse. It also means every
bounded tensor's memref carries an explicit strided layout, which some downstream
consumers (`linalg.fill` in the stdlib's inline MLIR, the FA-2 / cuDNN providers) may
refuse where they accept the identity layout today. Phase 0 should check what each
provider accepts.

### 2.3 Bound inference at calls into bounded const generics (Phase 0 Q1)

The existing rule binds `N` to the concrete dim (`unify_types_internal`). Extending it
per state: `[512, 4096]` against `[?<=M, 4096]` binds `M = 512`; `[?<=2048, 4096]` binds
`M = 2048`; `[?, 4096]` fails. This is derivable from `⊑` and needs no new inference —
but it interacts with §1.4 above (a `const` parameter bound to a static extent vs. to a
bound is two different instantiations). State it.

### 2.4 Two solvers, one obligation kind

§5.5 picks `SmtProver` (fresh process per proof) for `narrow`, and §5.6 makes **every
bounded allocation** a proof obligation. In `matmul` that is one z3 spawn per `uninit`,
per instantiation. The "persistent per-worker QF_LIA solver" §5.5 calls an optimization
will be needed on day one; plan it in Phase 3 rather than after.

### 2.5 Diagnostic family

E-A (a static extent exceeds the target bound) is a type mismatch. The E6xxx family is
memory algebra and topology (`src/diagnostic.rs:269-340`); type errors live in E3xxx
(E3001 "type mismatch in variable declaration", through E3023). Put E-A there; E-B/E-C/E-D
are placement and admission and belong in E6xxx.

### 2.6 Smaller items

- §4.7 now describes the tree correctly. Keep the sentence "per-worker state is fine";
  it is what the first RFC got wrong.
- §5.3's `?<=` "lexed as `?` followed by `<=`" works only if `?` is a dim-level token
  (§1.2 above); `<=` inside `parse_expr` would try to parse a comparison.
- The `narrow_checked` trap has a landing spot: `Opcode::Abort = 38`.
- §5.10's warning-first staging is right. The 18 `transfer`/`spawn` files are the
  migration list; `examples/llama.vx` is the one the README shows.

## 3. Recommended order for the spelling RFC

1. Decide §1.1 (`Vec<Dim>`), §1.4 (impl specificity), and whether §1.3's identity change
   is conditional. All three are one-line decisions that block Phase 1.
1. Phase 0 is already answered above except for the placement test; write that test
   first.
1. Phases 1–5 as written, with §1.2's smaller Phase 1.
1. File §6 as its own ticket. Do not attach it to the tag.
