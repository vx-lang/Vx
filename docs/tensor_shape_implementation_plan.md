# Tensor Shapes: Why `DynTensor`

Rationale and findings behind splitting `Tensor` into a statically shaped `Tensor<T, [d0, d1]>`
and a runtime-shaped `DynTensor<T>` (Vx#399). Written from the investigation that produced it, so
the evidence stays attached to the decision.

Status of the work this document covers:

| Piece | Issue | State |
| --- | --- | --- |
| Matmul result shape computed by the checker | Vx#397 | open, blocked on the split |
| `c = a @ b` into an existing buffer | Vx#391 | open, blocked on Vx#397 |
| The `Tensor` / `DynTensor` split | Vx#399 | open, design settled here |
| Monomorph name collision on tensor shapes | Vx#401 | fixed |
| Bounded dynamic shapes (`[<=N, ..]`) | Vx#245 | open, the home for meaning 2 below |

## How we got here

Vx#397 recorded two bugs that compile cleanly today. A matmul with disagreeing inner dimensions:

```
let mut a : Tensor<f32> = Tensor<f32>([2, 3]);
let mut b : Tensor<f32> = Tensor<f32>([4, 5]);
let c = a @ b;                        // k is 3 on one side and 4 on the other
```

and a declared result shape that lies:

```
let c : Tensor<f32, [9, 9]> = a @ b;  // the result is [2, 4]
```

Both exit 0. The fix looked local: have `check_binary`'s matmul arm compute `[m, n]` from
`[m, k] @ [k, n]` and reject a `k` mismatch. That is about thirty lines, and against these two
programs it changed nothing.

The shape is erased one step earlier. `check_let_decl_stmt` (`hir/stmt.rs`) binds the
annotation, so a dims-less annotation over a statically shaped initializer discards the shape the
initializer states one token later. The matmul arm then sees two rank-unknown operands and has
nothing to compute with. The flat lowerer, which recovers shapes from initializers rather than from
the checked type, emits
`memref<2x4xf32>` for a value the checker calls `[9, 9]` — three answers to one question, which is
Vx#388 in a single program.

That spelling dominates the corpus: 41 dims-less local bindings against 20 that carry shapes,
including every program behind Vx#390 and Vx#391.

## The four meanings of a dims-less `Tensor`

Auditing where `Type::Tensor(el, vec![], top)` comes from turned up four unrelated uses of the one
spelling. This is the core finding.

**1. Accidental erasure.** The 41 local bindings above. The programmer wrote a shape and the type
system dropped it. Direct cause of both Vx#397 bugs.

**2. Genuine dynamism.** Four corpus programs need a dimension that is not known until run time.
`llama2_v2.vx` reads `dim`, `hidden_dim`, `kv_dim` and `vocab_size` from a model config;
`custom_matmul` sizes its result `Tensor<f32>([a.shape[0], b.shape[1]])` to work at any shape. A
compiler that demands static shapes cannot compile a model loader, so this meaning has to survive.

**3. Error recovery.** Eleven sites in the checker return a dims-less tensor as the "I do not know"
value, immediately after reporting an error:

```rust
self.errors.push(format!("Cannot call expression of type {:?}", callee_ty));
...
Type::Tensor(ElementType::F32, vec![], None)      // hir/check/calls.rs:155

self.errors.push(format!("Undefined static method '{}'.", resolved_name));
Type::Tensor(ElementType::F32, vec![], None)      // hir/check/calls.rs:927
```

A type error produces a value that keeps type-checking downstream. `Type::Unknown` already exists
for this and is already handled ("Already-reported failure upstream; stay quiet", Vx#294).

**4. A scalar in disguise.** `len()` returns `Type::Tensor(I64, vec![], None)` — a plain count
wearing the tensor spelling (the `len` arm of `check_methodcall_expr`).

## Why a separate type rather than a rule

The first instinct was to reject the erasing `let` with a diagnostic. A separate type is better
because it removes the case rather than detecting it: with `Tensor<f32>` no longer a complete type,
`let mut a : Tensor<f32> = Tensor<f32>([2, 3])` fails because the type does not exist. There is no
rule to write, and no way to regress it.

Each meaning then gets its own home:

- Meaning 1 becomes unspellable.
- Meaning 2 becomes `DynTensor<T>`, and Vx#245's bounded form (`DynTensor<f32, <=4096>`) restores
  the capacity checks that W1029 currently warns are missing.
- Meaning 3 has to become `Type::Unknown`.
- Meaning 4 becomes `Scalar(I64)`.

The constructor spelling `Tensor<f32>([2, 3])` is unaffected: its type argument is the element and
the dims come from the argument.

## What dynamic shapes are for, and what they cost

Worth stating plainly, because the split keeps them rather than removing them.

They earn their place in three ways. Model-config-driven sizes, as in the llama2 port, cannot be
literals without recompiling per model. Shape-polymorphic library code (`custom_matmul`) wants one
function over many shapes. Serving-time variability — batch size, sequence length — is the case
Vx#245 frames best: the guarantee needs byte bounds rather than exact shapes, because proving a KV
tile fits VMEM requires `seq_len <= 4096` and never that it is 1723.

The cost is that static verification switches off. W1029 exists because a dynamic shape silently
skipped the capacity check, so placement verification — the language's central claim — does not
apply to dynamic tensors. Vx#397's two bugs are the same hole seen from the type system's side. The
flat path declines most dynamic-shape work, leaving those programs on the AST oracle.

Keeping both meanings under one spelling means paying that cost on the 41 programs that never asked
for it.

## Where the array/vector analogy holds

`Tensor<T, [d0, d1]>` is `std::array` and `DynTensor<T>` is `std::vector`. Rust makes the same
split and enforces it for locals: `let a: [i32] = [1, 2, 3]` does not compile, because an owned
local carries its size in its type. That maps exactly onto meaning 1.

At boundaries Rust stops using fixed arrays and offers two explicit spellings instead — `&[T]` for
runtime-sized and `const N: usize` for static-polymorphic. Vx already has the second (const
generics). The split supplies the first.

## The mangling detour (Vx#401)

Making the checker shape-aware broke every stdlib tensor method, which exposed a second defect worth
recording, because the first diagnosis of it was wrong.

`Type::mangle` encoded a tensor as `Tensor$<elem>$<rank>`, and that name serves two sites with
opposite needs.

*Monomorph identity* (`instantiate_function`, `hir/env.rs`) names an instantiation from its
substitution map. Rank does not separate `[2, 3]` from `[4, 5]`, so a generic instantiated at both
got one body:

```mlir
%v15 = func.call @ident$Tensor$f32$2(%v0) : (memref<2x3xf32>) -> memref<2x3xf32>
%v17 = func.call @ident$Tensor$f32$2(%v1) : (memref<4x5xf32>) -> memref<2x3xf32>
func.func @ident$Tensor$f32$2(%arg0: memref<2x3xf32>) -> memref<2x3xf32> { ... }
```

The second call hands a `[4, 5]` buffer to a body declaring `[2, 3]`. Only the debug assertion
caught it; a release build falls back silently.

*Method dispatch* has to find what an `impl Tensor<T>` block defines, and those bodies are written
rank-generically against `memref<?x?xT>`. This site wants the shape gone.

Rank served neither — too coarse for the first, still shape-bearing enough to miss for the second.
The first attempt removed rank from the name, on the theory that it was what broke dispatch. It was
not: a shaped receiver still failed, because `unify_types_internal`'s tensor arm required equal
rank and rejected the impl before any name was built.

The fix gives each site what it needs. The mangled name carries the extents
(`Tensor$f32$2x3`), so monomorphs are distinct; unification treats a dims-less pattern as a shape
wildcard, so `impl Tensor<T>` matches any shape. That reading is consistent for every caller, since
all of them pass pattern-then-concrete — a `Tensor<f32>` parameter accepting a shaped argument is
the same rule. Under the split, the wildcard belongs to `DynTensor` and the pattern side becomes
explicit.

Landed as `Fixes: #401`, with both halves negative-controlled: removing the extents fails the
two-shape regression test, disabling the wildcard brings back "Method 'fill' not found".

## Cost of the split, measured

- 101 `Type::Tensor` pattern sites to triage (static, dynamic, or both).
- 45 dims-less constructions to reclassify into the four meanings.
- Roughly 80 corpus type positions to conform: 41 local bindings, 20 parameters, 19 returns.

Parameters and returns are where the two meanings still blur once locals are fixed. Each is either
static-polymorphic (const generics, already in the language) or genuinely runtime (`DynTensor`).

## Sequencing

1. **Done.** Extents in the mangled name, dims-less patterns as wildcards (Vx#401). Shaped tensors
   now dispatch and monomorphize correctly, which removes the wall the split would otherwise hit.
1. **Next, design-independent.** The matmul arm computes `[m, n]` and rejects a `k` mismatch
   (Vx#397 step 1). It fires today for code that spells its shapes — 20 bindings and 17 parameters
   in the corpus — and gains reach as shapes become the norm.
1. **The split** (Vx#399): introduce `DynTensor`, triage the 101 match sites, reclassify the 45
   constructions, conform the corpus.
1. **Then** Vx#391's assignment form, and Vx#245's bounded shapes as `DynTensor`'s verified variant.

## Open question

What is rank-0? The dims-less spelling currently doubles as the rank-0 scalar wrap
(`let t : Tensor<f32> = 1.0`, Vx#396). Under the split it becomes `Tensor<f32, []>` — explicit,
and distinct from `DynTensor<f32>`.

## References

Vx#245 (bounded dynamic shapes), Vx#330 (tensor type audit), Vx#388 (one shape, computed once),
Vx#390, Vx#391, Vx#396, Vx#397, Vx#399, Vx#401, W1029 (dynamic shape unverified).
