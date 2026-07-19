# Implementation Plan: Standard Library Design

> Design + roadmap for the Vx standard library. Prompted by the flat-pipeline convergence work
> (#200/#217): the attention corpus's `.exp()` comes from `import std::math`, which surfaced the need
> for a deliberate stdlib design. Two driving questions:
>
> 1. Which library functions do we implement in Vx, and which do we short-circuit to native (Rust/C)?
> 1. Do operations live as **methods** on a type (`f32::exp`) or **free functions** (`exp<T>(n: T)`)?
>
> Related: the flat codegen epic [#197](https://github.com/hiraditya/Vx/issues/197), corpus/method
> calls [#217](https://github.com/hiraditya/Vx/issues/217).

## 0. TL;DR (the decisions)

- **Layering.** Keep the existing three layers: **native** (libm + `libvx_std_core` + MLIR runner
  utils) ← **`extern` decls** (typed FFI boundary in `.vx`) ← **Vx stdlib** (`stdlib/std/*.vx`). Push
  as much as possible *up* into Vx; drop to native only at irreducible primitives.
- **Q1 — native vs Vx.** Short-circuit to native **only** for the four irreducible categories:
  (a) transcendental/hardware math, (b) OS / syscalls, (c) raw allocation, (d) opaque
  performance/unsafety-critical data-structure cores. Everything composable from those is Vx.
- **Q2 — methods vs free functions.** **Methods (via traits)** are the default for "an operation *on
  a* value of a type" (`x.exp()`, `v.push(e)`, `s.len()`); **free functions** for operations *over
  several* values or constructors (`min(a, b)`, `dot(a, b)`, `zeros<T>(n)`). Traits give both call
  forms for free — `x.exp()` **and** `Math::exp(x)` — so this is a *default*, not a restriction.
- **The forcing function.** The stdlib is the ultimate dogfood for the flat pipeline: converging
  (#197) means the stdlib must eventually lower through the flat codegen, not just the AST path. That
  goal — more real code through the flat path — is *why* the "prefer Vx" bias in Q1 matters.

## 1. What exists today (the ground truth)

The stdlib is **not** greenfield. There is already a working three-layer design:

```
  ┌─ Vx stdlib  (stdlib/std/*.vx)                     ← import std::math; x.exp()
  │    traits + impls + free fns; wraps extern decls
  ├─ extern decls  (extern { fn expf(x:f32)->f32; })  ← the typed FFI boundary, in .vx
  └─ native
       - libm         (C math: expf/sqrtf/…, linked with `-lm`, src/jit.rs)
       - libvx_std_core.dylib  (Rust: io/env/net/simd/collections, stdlib/rust_core/)
       - libmlir_runner_utils  (printMemref*, memref alloc/copy)
```

- **Module loader** (`src/module_loader.rs`): `import std::math` searches `stdlib/std/math.vx`, then
  `stdlib/` (so `import graph::traversal` → `stdlib/graph/traversal.vx`).
- **`stdlib/std/` (21 modules today):** `math`, `vec`, `string`, `hash_map`, `hash_set`, `option`,
  `result`, `iter`, `box`, `closure`, `alloc`, `tensor`, `io`, `net`, `fs`, `mmap`, `time`, `simd`,
  `libc`, `googletest`, `llama`.
- **`stdlib/rust_core/` (the Rust runtime, `libvx_std_core`):** `io` (print_i32/f32/str/println),
  `env`, `net`, `simd`, `googletest`, `collections/` (hash_map/set, vec, string, option, result),
  `ffi/` (rt, macros, llama).
- **Method dispatch.** `math.vx` declares `trait Math { fn exp(self: Self) -> Self; … }` and
  `impl Math for f32 { fn exp(self: f32) -> f32 { return unsafe { expf(self) }; } }`. A call
  `x.exp()` desugars (codegen) to the mangled monomorphic function `f32$exp`, which calls the extern
  `expf`. This is the existing, coherent pattern for "primitive methods."

**Takeaway:** the architecture is sound. This document *names the rules* the existing code follows,
resolves the open questions, and plans the work to make the stdlib first-class under the flat pipeline.

## 2. Q1 — Which functions live in Vx, which short-circuit to native?

### The principle

> **Drop to native only at an *irreducible primitive* — something Vx cannot express, cannot express
> *safely*, or cannot express *without unacceptable cost*. Implement everything else in Vx.**

A function is an *irreducible primitive* if it falls into one of exactly four categories:

| Category | Why native | Boundary | Examples |
|---|---|---|---|
| **A. Transcendental / hardware math** | Correct rounding + hardware intrinsics live in libm / the FPU; re-deriving `exp` in Vx is wrong and slow | `extern` → libm (`-lm`) | `expf`, `sqrtf`, `sinf`, `logf`, `fabsf`, … |
| **B. OS / syscalls** | Only the OS can do IO / time / memory-mapping / networking | `extern` → `libvx_std_core` (Rust) | `print_*`, `read`/`write`, `time`, `mmap`, sockets, `env` |
| **C. Raw allocation** | The heap is owned by the allocator; a bump/GC in Vx is a research project, not a stdlib entry | `extern` → `libvx_std_core` / libc | `malloc`/`free` (behind `alloc`/`box`) |
| **D. Opaque perf/unsafety-critical cores** | A hash table or growable buffer needs raw pointer arithmetic + realloc; a naive Vx version would be unsafe or O(n) | `extern` → `libvx_std_core` collections | `HashMap`/`HashSet` internals, `String`/`Vec` growth |

**Everything else is Vx**, built by composing A–D:

- **Numeric helpers** — `min`/`max`/`clamp`/`abs` (integers), `pow` by squaring, `lerp`, degrees↔
  radians, `sum`/`mean` over a slice. (Composable from arithmetic + libm.)
- **Iterators / combinators** — `map`/`filter`/`fold`/`enumerate`/`zip`, `Option`/`Result`
  combinators (`map`, `unwrap_or`, `and_then`). (Pure control flow.)
- **Tensor ops** — reductions, elementwise, softmax, attention. (These are the *product*; they must
  be Vx — that is the language's reason to exist.)
- **String algorithms** — `split`, `trim`, `starts_with`, formatting — *above* the raw
  growable-buffer core in D.

### Rationale, and the convergence tie-in

- **Correctness & performance are the only reasons to leave Vx.** A/B/C/D are exactly the places where
  a Vx implementation would be *wrong* (A: rounding), *impossible* (B: syscalls), or *unsafe/slow*
  (C/D: raw memory). Nothing else qualifies.
- **Dogfooding drives the bias.** #197's goal is to make the flat-array pipeline the production path.
  The stdlib is the largest, most realistic body of Vx code we have — every stdlib function written in
  Vx (instead of shipped as a Rust builtin) is another test of the real compiler. So when a function
  is *borderline* (could be a compiler builtin or Vx), **write it in Vx.** `dot`/`sum`/`max`/`min` are
  the cautionary tale: they are currently compiler-recognized builtins in `hir/flatten.rs`, which is
  why they lower but also why they are *special*. New reductions should be Vx `trait`/`impl` methods
  over slices, not new builtins.
- **The `extern` block is the contract.** Every native dependency is a typed `extern fn` in a `.vx`
  file. This keeps the FFI surface auditable (grep for `extern`), typed, and small. New native
  entry points require a new `extern` decl + a `#[no_mangle] pub extern "C"` in `rust_core` (or a libm
  symbol) — a deliberate, reviewable step.

## 3. Q2 — Methods (`f32::exp`) or free functions (`exp<T>(n)`)?

### The rule

> **Method (trait) when the operation is fundamentally *about one value of a type*. Free function when
> it is *about several values*, or *constructs* a value, or has *no natural receiver*.**

Because Vx traits already give *both* call syntaxes (`x.exp()` and `Math::exp(x)`) and *generic
dispatch* (`fn f<T: Math>(x: T)`), this is a question of the **primary, idiomatic** form — not an
either/or.

| Form | Use when | Examples |
|---|---|---|
| **Trait method** `x.op()` | the receiver is *the* subject: unary math, container mutation/query, conversions | `x.exp()`, `x.sqrt()`, `v.push(e)`, `v.len()`, `s.trim()`, `x.abs()` |
| **Free function** `op(a, b, …)` | multiple co-equal args, constructors, or reductions over a slice | `min(a, b)`, `max(a, b)`, `dot(a, b)`, `zeros<T>(n)`, `range(lo, hi)` |

**Why methods are the default for unary ops (`.exp()`):**

- **Discoverability** — `x.` surfaces every operation on `x`; a flat namespace of free functions does
  not.
- **Generic bounds read well** — `fn softmax<T: Math>(row: &[T])` says exactly what it needs.
- **It matches the existing design** — `trait Math` + `impl Math for f32` is already how `.exp()`
  works. Do not fork the convention.

**Why some things stay free functions:**

- `min(a, b)` has *two* co-equal arguments — `a.min(b)` privileges `a` arbitrarily. (Provide the method
  *too* if it reads well, but the free function is the primary.)
- Constructors (`zeros`, `range`, `Tensor<T>(...)`) have no receiver to be a method *of*.
- `dot(a, b)`, `matmul(a, b)` are binary and symmetric-ish; free functions match the math notation.

**Anti-patterns to avoid:**

- **Compiler builtins pretending to be functions.** `dot`/`sum`/`max`/`min` are recognized by name in
  `hir/flatten.rs` today. That is a shortcut, not the design. The target is: they are ordinary Vx
  trait methods / free functions the compiler lowers like any other call — no name is special.
- **Duplicating an op as both a method and a free function with divergent behavior.** If both exist,
  one delegates to the other (usually the free function delegates to the method, or vice-versa) so
  there is a single source of truth.
- **`unsafe` leaking past the wrapper.** `extern` calls are `unsafe`; the Vx `impl` wraps them so
  callers never write `unsafe { expf(x) }` — they write `x.exp()`.

## 4. Roadmap — make the stdlib first-class under the flat pipeline

The stdlib compiles today through the **AST** path. Convergence (#197) needs it to compile through the
**flat** path too. That is the real work item behind #217, in dependency order:

1. **Flat HIR: method calls.** `hir/flatten.rs` has no `Expr::MethodCall` arm. Desugar
   `recv.method(args)` → the resolved (monomorphized/mangled, e.g. `f32$exp`) function call and route
   through `lower_call` (the mangled fn is in the frozen registry's `fn_sigs` once its module is
   compiled). Emitter: nothing new — it is a `func.call`.
1. **Flat HIR: `extern` decls + `unsafe` blocks.** `extern fn expf` is a declaration with no body; the
   flat path must (a) surface it in `fn_sigs` so calls resolve, and (b) emit a `func.func private @expf(f32) -> f32` at the module top (same mechanism as the print-helper decls the emitter already
   prepends). `unsafe { … }` is transparent — lower the inner expression.
1. **Differential harness: multi-module / imports.** `tests/integration_test/flat_codegen_differential.rs`
   compiles a single `src` string. To differentially test anything that `import`s the stdlib (the
   softmax corpus), it must resolve imports (via `module_loader`) and lower **all** modules — the
   `std::math` blocker in #217. This unblocks the AST oracle *and* the flat path for the softmax
   corpus in one step.
1. **Retire builtin reductions.** Once trait-method calls lower through the flat path, migrate
   `dot`/`sum`/`max`/`min` from name-recognized builtins in `flatten.rs` to ordinary Vx stdlib methods
   (a `trait Reduce for &[T]` or free functions in `std::tensor`). This shrinks the compiler's special
   surface and is the clean proof that the design holds.

### Testing strategy

- **Per-module differential**: once (3) lands, run each `stdlib/std/*.vx`-using program through
  `assert_output_parity` / `assert_parity`. The softmax corpus (`full_softmax`, `multi_query`,
  `grouped_query`, `sparse_local`) is the headline target.
- **The stdlib itself as a corpus**: compile the whole stdlib through the flat path and diff against
  the AST path — the strongest convergence signal.

## 5. Open questions (need a human decision)

- **Generic numeric trait hierarchy.** Do we want a `Num`/`Float`/`Int` trait tower (like Rust's
  `num-traits`) so `fn mean<T: Float>(xs: &[T])` works across `f32`/`f64`, or keep per-type impls
  (`impl Math for f32`, `impl Math for f64`) and duplicate? (Leaning: a small `Float` trait, because
  the tensor ops are the point and they want to be element-generic.)
- **Error model.** `Result`/`Option` exist; do stdlib fallible ops (`parse`, `open`) return `Result`
  uniformly, or panic? (Leaning: `Result` for anything touching the OS, panic for contract violations
  — but pin it down.)
- **`no_std`-like split.** Is there a core subset (math, option/result, iter) usable without the OS
  (B/C native deps), vs a full subset (io/fs/net/collections)? Relevant if Vx targets accelerators
  where there is no libc.
- **How much of `dot`/`sum`/… migrates now vs after C3.** Retiring builtins (step 4) is cleanest after
  the flat path is the production codegen; doing it earlier means maintaining both.

## Key files

- Module loading: `src/module_loader.rs`. Import resolution: `src/syntax/resolve.rs` (`ImportIndex`).
- Vx stdlib: `stdlib/std/*.vx`. Rust runtime: `stdlib/rust_core/src/` (`libvx_std_core`).
- FFI/link: `src/jit.rs` (`-lm`, `libvx_std_core`, `libmlir_runner_utils`). Extern handling:
  `src/hir/env.rs` (`module.externs`).
- Flat-path work: `src/hir/flatten.rs` (method calls / extern), `src/codegen/flat.rs` (decls),
  `tests/integration_test/flat_codegen_differential.rs` (multi-module harness).
