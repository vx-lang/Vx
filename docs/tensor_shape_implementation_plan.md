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
| Flat path declines every `DynTensor` parameter | Vx#409 | open, blocked on the split; largest flat gap |

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

## Doing the split, in slices

Each slice below is separately committable and gated. The counts are measured, not estimated.

1. **`len()` answers a scalar** — one line. Removes meaning 4 outright. *(Done, e956d905.)*
1. **Error recovery becomes `Type::Unknown`** — 24 dims-less `F32` constructions live in the
   checker, of which 5 clearly sit within a few lines of an error report and the rest need reading
   one at a time. Blind replacement would be wrong here: some are genuine tensor results and some
   are fall-through returns. Expect diagnostic churn, since `Unknown` suppresses the cascade a
   plausible-looking tensor currently produces, and several `fail` tests pin error counts.
1. **Introduce `DynTensor<T>`** — the type, its parser spelling, and its lowering. Nothing
   migrates yet, so the corpus stays green.
1. **Migrate meaning 2** — the four genuinely dynamic programs, plus the parameters and returns
   that are dynamic rather than erased. *(Done. 27 signature positions across 9 files and the 3
   locals whose extents are run-time values now spell `DynTensor`. See "What the migration
   found" below.)*
1. **Make a dims-less `Tensor` unspellable** and conform what remains. *(Done. The parser
   refuses it and names the three replacements; the last program that spells it is the test
   asserting the refusal.)*

Slice 2 is the one to sequence carefully. The remaining 9 dims-less constructions outside the
checker are the parser's default for a bare `Tensor` spelling and a few type-level defaults; those
belong with slice 5, where the spelling itself changes.

## What the migration found

Migrating the shape-polymorphic signatures was meant to be a spelling change: a dims-less
`Tensor<f32>` parameter and a `DynTensor<f32>` parameter both lower to `memref<?x?xf32>`, which a
reduced program confirms. One program disagreed.

`middle_end/pass/topology.vx` had been compiling through the flat path, and the two compilers were
giving its one function two different signatures:

```mlir
func.func @process(%arg0: memref<f32>) -> memref<f32>            // flat path, ships by default
func.func @process(%arg0: memref<?x?xf32>) -> memref<?x?xf32>    // AST oracle
```

A rank-0 descriptor is `{ptr, ptr, offset}`; a rank-2 one carries two more sizes and two more
strides. The flat path also dropped the `vx.transfer` that `t_host.to_device()` emits on the oracle.
Neither shows up in a single compilation, which is why nothing caught it: the file lives in
`middle_end/pass`, whose CHECK lines run against the oracle.

The cause is meaning 1 and meaning 2 sharing a spelling, seen from the lowerer's side.
`tensor_elem_shape` maps a dims-less `Tensor<f32>` to an *empty* shape, which the flat lowerer reads
as rank-0, while the oracle reads the same spelling as rank-2 dynamic. Spelled `Tensor<f32, [4, 4]>`
the two paths agree exactly, transfer included.

Spelling it `DynTensor<f32>` makes the flat path decline instead of guessing, so the program moved to
`KNOWN_DECLINES` and coverage went 235 -> 234. That number is worth reading the right way: one
program stopped being compiled two different ways.

The decline reason is named rather than folded into the generic parameter bucket, and the histogram
now measures the gap:

```
flat decline:  10 type-not-modelled(a dynamic tensor parameter)   <- largest bucket
flat decline:   8 unresolved-callee
flat decline:   5 type-not-modelled(a callee return type)
```

Carrying run-time extents through the flat lowerer is Vx#409, and it is the single largest flat-path
gap left.

## Rank 0, settled

The dims-less spelling doubled as the rank-0 scalar wrap (`let t : Tensor<f32> = 1.0`, Vx#396).
It is `Tensor<f32, []>` now, and the three spellings lower to three different things:

| spelling | memref |
| --- | --- |
| `Tensor<f32, [2, 3]>` | `memref<2x3xf32>` |
| `Tensor<f32, []>` | `memref<f32>` |
| `DynTensor<f32>` | `memref<?x?xf32>` |

The middle row did not hold when slice 5 started. `lower_tensor_type` read an empty dimension
list as "unknown" and emitted `memref<?x?xf32>`, which was right while the dims-less spelling was
the only way to say "unknown" and wrong the moment `DynTensor` existed: a parameter that stated
rank 0 silently became rank 2. Only `DynTensor` reaches that branch now.

Two middle-end tests show what the correction buys. `generics.vx` used to pass a scalar to a
generic by casting it into a rank-2 memref
(`builtin.unrealized_conversion_cast %cst : f32 to memref<?x?xf32>`); it now allocates a real
`memref<f32>` and fills it. That cast was the Vx#396 fiction, and the split removed it rather
than patching it.

## What slice 5 found

Conforming the corpus was mostly deletion. Of 123 dims-less local bindings, none needed a shape
written by hand: 33 were the scalar wrap and say `Tensor<T, []>`, and the other 90 dropped an
annotation that had erased what the initializer already carried. That is meaning 1 measured — the
type system was discarding a shape the programmer had written one token later, 90 times.

Three checker sites still produced the spelling and now answer for what they hold. `tensor_view_2d`
takes the constructor's rule, so a view over a caller's buffer keeps the shape its call site states
and only a run-time extent makes it dynamic. A raw primitive that returns nothing says `void`. And
`matmul_into` asks whether its operands are tensors rather than statically shaped ones, since
filling a caller's buffer is the reason it exists.

Making it unspellable then found the gaps that only appear once the two types are really distinct:
`Type::substitute` had no `DynTensor` arm, so a generic element survived monomorphization and
reached codegen; `as_ptr`, `len` and relational comparison accepted only `Type::Tensor`, though
none of them reads a shape. Each is a place the old spelling had been doing double duty.

Coverage moved both ways and ended level. `middle_end/pass/loops.vx` joined the flat path once its
parameters stated `[10, 10]` instead of nothing; `topology.vx` and `traits.vx` left it, because
their shape-generic signatures are honestly `DynTensor` and the flat lowerer does not carry
run-time extents yet (Vx#409, now 12 programs and the largest measured gap).

## References

Vx#245 (bounded dynamic shapes), Vx#330 (tensor type audit), Vx#388 (one shape, computed once),
Vx#390, Vx#391, Vx#396, Vx#397, Vx#399, Vx#401, W1029 (dynamic shape unverified).
