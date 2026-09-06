# Review: RFC unified `Tensor` with per-dimension extent states

**Reviews:** `rfc-unified-tensor-vx.md`
**Reviewed against:** `main` at `85342711`
**Status:** findings for the RFC author; nothing here is implemented

The RFC was written without access to the tree and says so. This review checks each
assumption against the code and reports what holds, what does not, and what the RFC
does not address. File references are to the tree at the commit above.

## 1. Conflicts with the tree

The RFC's §0.4 says any conflict between it and the tree is a stop-and-report. These are
the conflicts.

### 1.1 Widening is already implicit, and method receivers depend on it

`is_assignable` (`src/hir/expr.rs:496-530`) accepts a shaped `Tensor` at a `DynTensor`
position with no cast written. The flat path emits the `memref.cast` for it itself
(`forget_extents_for_param` / `forget_extents_for_return`, `src/hir/flatten.rs:2992-3011`;
`cast_memref`, `src/codegen/flat/emit/arith.rs:315`). Method resolution treats a dims-less
pattern as a shape wildcard (`src/hir/env.rs:661-675`), which is how a shaped receiver
resolves a method declared on `DynTensor<T>`.

§4.4 makes every widening explicit. Applied as written, `a.fill(x)` on a shaped `a`
becomes `(a as Tensor<[?, ?], f32>).fill(x)`. The RFC does not mention receiver
position. The choice is between an exception to §4.4 for receivers and shape-generic
methods (§5.10, which the RFC defers). This decision shapes the rest of the design and
has to come first.

### 1.2 The memory space is in the type but not in the flat identity

`Type::Tensor(ElementType, Vec<Expr>, Option<Placement>)` carries placement.
`tensor_gid` (`src/hir/flatten.rs:48`) hashes only the element and the shape:

```rust
DefPath::Named(&format!("$tensor::{elem:?}::[{}]", shape.join(",")))
```

§4.1 requires the memory space in the digest. The flat path drops it today. `Placement`
also records which projection the source *stated* (`stated: Stated`,
`src/syntax/types.rs:206-218`); that field must stay out of any content hash, or the same
type spelled two ways gets two identities.

### 1.3 The flat path never evaluates a dimension expression

`tensor_dim_string` (`src/hir/flatten.rs:162`) admits a literal or a bare identifier and
declines everything else. `Tensor<f16, [BATCH * CTX, HEADS * HDIM]>` in the corpus declines
on the flat path today, and all three `const_generics*.vx` fixtures are in
`KNOWN_DECLINES` (`tests/integration_test/flat_corpus_sweep.rs`). The checker folds dims
(`src/hir/check/calls.rs:1241`); the flattener does not.

A bound such as `?<=Serving.max_ctx` needs "fold to an integer before minting" on the
flat path. That is new machinery, not a tag in the digest.

### 1.4 The solver model is not what §4.7 describes

A persistent z3 process is held per checker on `check_state.rs:91`
(`solver: Option<crate::hir::seam::Solver>`), spawned lazily at
`src/hir/check/transfer.rs:747` and `src/hir/decl_check/topology.rs:185`. It is
per-worker and lock-free, so it passes the CI grep, and it is also a lazily-initialized
availability cache, which §4.7's wording forbids. Either the wording changes or the tree
does.

There are two solver front-ends, and the RFC assumes one:

| Front-end | Process model | Logic | Driven by |
| --- | --- | --- | --- |
| `hir::seam::Solver` | persistent, `(push)`/`(pop)` per obligation | QF_BV preamble | transfer seams |
| `hir::prover::SmtProver` | fresh process per proof | QF_LIA | `comptime` checks (`src/hir/stmt.rs:581`) |

`narrow` proofs are QF_LIA. The RFC has to say which front-end discharges them.

### 1.5 Rank is assumed to be 2

`DynTensor` lowers to `[?, ?]` on the flat path (`src/hir/flatten.rs:87-95`) and to
`memref<?x?xT>` on the oracle (`src/codegen/lower/expr.rs:2116`). Vx#404 is open on this.
Static rank is the right fix. The migration is not mechanical: 40 `.vx` files spell
`DynTensor`, each needs its rank written in, and a rank-1 value flowing into a rank-2
annotation becomes a type error where today it is a miscompile.

## 2. Gaps in the design

### 2.1 The stdlib cannot be migrated under the RFC's own rules

`stdlib/std/tensor.vx` allocates results with extents taken from scalar arguments:
`from_ptr_2d(ptr, d1, d2)`, `slice_2d(self, row, d1, d2)`,
`DynTensor<f32>::uninit([a.shape[0], b.shape[1]])`. None has a bound, so each is
`Unbounded`, so under E-D each is host-only. Every device matmul goes through them.

Fixing this means every stdlib signature becomes bounded-generic, which is §5.10. The RFC
defers §5.10 and depends on it in the same document.

### 2.2 `narrow`'s fact sources do not exist

§5.5 lists enclosing guards, loop bounds, and dominating checked narrowings. The checker is
an AST walk with no dominator or path-sensitive fact collection. `prover.rs` takes explicit
`comptime` assertions only. The nearest existing thing is `fold_raw_extent`
(`src/hir/check/raw.rs:432`), which folds loop bounds for traffic accounting. Phase 3 is
one bullet in the RFC and the largest piece of new analysis in it.

Bounds on scalars were dropped from Vx#245 (`fn stage(n: i32) where n <= 512`), and
`Tensor::alloc(extents: [b, s])` with scalar `b` needs exactly that fact. `where` parses
today only for `impl Transfer<A, B>` (`src/parser/decl.rs:126`).

### 2.3 No `config` declaration, no top-level `const`, no `?` token

`struct Config` in the parser tests is an ordinary struct. The lexer has no `?` token.
There is no file-scope `const`; only `const N` generic parameters exist. §5.3's preferred
bound source and its fallback are both absent. Open question 3 has to be answered before
Phase 1, and the answer is probably "add top-level `const` first."

### 2.4 E-D is a warning-to-error promotion with real blast radius

A runtime dim skips the capacity check with W1029 today
(`src/hir/check/transfer.rs:278`); `static_extent_of_dims` (`src/hir/check/raw.rs:318`)
sizes literal dims only. 18 `.vx` files combine `DynTensor` with `transfer` or `spawn`,
`examples/llama.vx` among them. Each is the "latent admission hole" §6.9 names, and each
becomes an error on migration.

### 2.5 §5.11's ownership assumption is wrong

`Tensor` is in `Type::is_linear` (`src/syntax/types.rs:608-618`): it moves, it is not a
view. `TYPE_NEEDS_DROP` is a word-3 flag (`src/gid.rs:51`). The owning/view split the RFC
recommends does not describe the current type.

### 2.6 §5.6 has a cost it does not state

"Allocation is sized by the bound" means a `Tensor<[?<=8192, 4096], f16>` holding extent
10 allocates 64 MiB, on the host too. That is the right call for admission, and it should
be said out loud. It is also another reason stdlib helpers cannot just take the widest
bound.

### 2.7 Receiver-position `.shape` is untyped

`.shape` on any non-struct value returns `Tensor<i32, []>` without checking the base is a
tensor (`src/hir/check/access.rs:458`), and a `DynTensor`'s shape is unreadable by design
(Vx#399). `TensorDim` (opcode 44) exists on the flat path, so `extent(i)` has a landing
spot. The RFC should say `.shape` is retired.

## 3. Invariants that are not gates

§8 lists things to check at every checkpoint. In the tree:

| Invariant | Where it lives today |
| --- | --- |
| 67 / 23 / 0 admission matrix | walkthrough docs only; no test |
| 64/64 token ids | walkthrough docs only; no test |
| 442,368 B KV transfer | walkthrough docs only; no test |
| TSan zero races | no CI step, no script |
| byte-identical at 1/8/32/48 threads | `codegen_determinism.rs` runs 3 times at default threads; `pipeline_scale_test.rs` compares 1 vs 4 |
| `llama-cpp.vx` / `flashattention.vx` do not decline | neither file exists by that name; `backend/pass/llama2_v2.vx` is in `KNOWN_DECLINES` |

Invariant 8 is false before the RFC starts. An implementer following §0.3 literally stops
at checkpoint 1. Either the invariants become gates first, or §8 is rewritten around the
gates that exist.

## 4. What the RFC gets right, with evidence

- §2.1's duplication is real. `tensor.vx` has `fill` on `DynTensor<T>` and `fill_static`
  on `Tensor<T, [N, M]>`.
- `impl<T: Float, const N: i32, const M: i32> Tensor<T, [N, M]>` already works
  (`stdlib/std/tensor.vx:219`). The §5.10 mechanism exists for static dims; the RFC can
  build on it rather than defer it.
- The `⊑` relation's placement rule matches the host-default rule already in
  `is_assignable`.
- §5.12: `llvm.insertvalue` accepts a nested memref descriptor (verified with mlir-opt), so
  Vx#356 is independent of this RFC, as it says.
- The 24-byte `HirInstruction` (`src/bytecode.rs:307`) has `type_idx`; the cast opcodes
  fit as described in §5.7.
- The lock lint exists (`.github/workflows/ci.yml:38`) and would catch a `OnceLock`.

## 5. Phase 0 answers available now

1. Type syntax is `Tensor<f32, [2, 3]>`, `Tensor<f32, []>` for rank 0, and
   `Tensor<f32, [2, 3], Memory::X>` with placement third (`src/parser/types.rs:338-392`).
   Element first, dims second. The RFC's `Tensor<[dims], elem, space>` is reversed.
   `DynTensor<f32>` is the dynamic form. `Tensor<f16>([8, 64])` from the deck is the
   constructor, not a type. `Ref<T, Memory::X>` no longer parses; it survives as
   `Type::Ref` internally and in comments.
1. Identity: `TypeId::new(0, sym, 0, 0)` where `sym` is a content hash of a formatted
   string. A tag for the extent state can go in that string; the string is the digest.
1. Lowering: `Tensor<T, [n, m]>` to `memref<nxmxT>`; `DynTensor` to `memref<?x?xT>`
   at rank 2, assumed. No bound-like attribute exists.
1. `DynTensor` allocates through `DynTensor<T>::new([..])` / `::uninit([..])` /
   `Tensor<T>([..])`, taking the shape as a runtime argument
   (`src/hir/check/calls.rs:1202-1256`).
1. Solver: see §1.4 above.
1. Admission sizes literal dims only; runtime dims warn W1029 and skip.
1. No config construct; no top-level `const`.
1. Machine files reach the session by `--machine` (`src/driver.rs:93-99`), merged as
   declarations before the program's own. Their figures are not reachable from
   type-level expressions.
1. E6001–E6018 are in `src/diagnostic.rs:270-340`; the next free code is after E6018,
   not E6014.
1. `as` is `Expr::AsCast` (`src/syntax/expr.rs:194`), checked in
   `check_ascast_expr` (`src/hir/check/operators.rs:19`), and `Opcode::Cast = 12` on the
   flat path. A tensor-to-tensor `as` has no arm today; a `slice as f16` element cast
   does.
1. Declines: see `KNOWN_DECLINES` in `flat_corpus_sweep.rs`; differential tests are
   `flat_codegen_differential.rs`.
1. Generator: `--memalg` in `src/bin/corpus/mod.rs:106`, content test at line 577. It
   emits one static `Tensor<f32>([2, 2])` and nothing dynamic.
1. 40 files spell `DynTensor`; 18 of them also `transfer` or `spawn`.
1. Word 2 is a lifetime/variance bitfield or an arena index (`src/gid.rs:148-170`);
   a tensor parameter is unaffected by this RFC.

Bounded dynamism is Vx#245, open and unimplemented. §1's "Vx's existing bounded
dynamism" overstates it.

## 6. Recommended order

1. Decide §1.1 (receivers). Everything else follows from it.
1. Add top-level `const`, or a `config` block, so a bound has somewhere to come from.
1. Put the memory space in the flat identity and fold dims on the flat path, both
   independent of this RFC and both needed by it.
1. Make §8's invariants into gates, or rewrite §8 around the gates that exist.
1. Then Phase 1.
