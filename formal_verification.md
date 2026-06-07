# Formal Verification and SMT Integration in Vx

## Goal

To evolve Vx from a language with strict structural type-checking into a formally verifiable language (approaching the capabilities of Lean or Coq), by introducing an SMT solver backend to computationally prove memory safety, hardware deadlocks, and array bounds at compile time.

## Proposed Architecture

### Phase 1: Dependent Type System Expansion

Currently, Vx supports limited dependent typing for Tensor shapes (`Tensor<f32, [M, K]>`). We will expand the AST and Semantic Analyzer (`src/sema`) to support generalized dependent types:

- **Refinement Types**: Allow types to be bounded by logical predicates (e.g., `let x: i32 where x > 0;`).
- **Propositional Typing**: Elevate `comptime { assert(P) }` into the type signatures, so a function mathematically guarantees its preconditions and postconditions.

### Phase 2: SMT Solver Integration (Z3)

Instead of building a custom theorem prover, we will integrate an external SMT solver (e.g., Z3) into the Vx compilation pipeline.

- Add `z3` (via the `z3-rs` crate) as a dependency to the compiler.
- Create a new `src/sema/prover.rs` module that translates Vx AST boolean expressions and `comptime` assertions into Z3 `Bool` contexts.

### Phase 3: Constraint Generation & Verification

During the `Sema` pass, the Type Checker will lower programs into mathematical constraints:

- **Spatial Bounds**: Prove that any index used inside a `spawn on(Topology::NPU[i])` is strictly within the physical bounds of the hardware.
- **Array/Tensor Bounds**: Generate Z3 constraints for all array accesses to formally prove the absence of Out-Of-Bounds errors statically.
- **Data Linearity**: Map the `transfer(ref, MemorySpace)` operations into a state-transition system verifiable by the SMT solver to guarantee no double-frees or dangling references across heterogeneous memory spaces.

## Open Questions for the Community

- Should we bundle the Z3 solver with the `vxc` compiler, or expect users to have it installed in their system `PATH` and communicate via the SMT-LIBv2 standard over stdin/stdout?
- What syntax should we adopt for formal loop invariants? (e.g., `for i in 0..N invariant(i < bounds) { ... }`)
- How expressive should the theorem prover be? Should we stick to decidable theories (Linear Integer Arithmetic) or allow non-linear arithmetic (which might hang the compiler)?

This proposal ties in perfectly with our earlier discussion about `Verified<T>` and `Deferred<T>`. Here is how the SMT integration (Issue #108) directly answers how we structure those topology function signatures.

### 1. The Topology Function Signatures (Phase 1 & 3 in action)

If we introduce refinement types and proposition typing, the function signature becomes the exact contract the SMT solver verifies.

Instead of just `fn dummy(x: i32) -> i32`, we can use a `where` clause (or `requires`/`ensures` contracts) that the SMT solver ingests.

**The Individual Function Level:**

```rust
// A function that strictly runs on an NPU and guarantees its memory interactions
fn tile_matmul(a: Tensor<f32, [128, 128]>, b: Tensor<f32, [128, 128]>) -> Tensor<f32, [128, 128]>
where
    topology == Topology::NPU, // SMT checks that caller's spawn_on matches
    a.loc == b.loc // SMT proves both tensors are in the same physical memory space before execution
{
    // ...
}

```

**The Namespace/Module Level (Your idea):**
Having to write `where topology == Topology::NPU` on every function is tedious. Your idea of a blanket namespace is perfect. It essentially acts as a lexical scope that implicitly injects SMT propositions into every function it contains.

```rust
namespace npu_kernels where topology: Topology::NPU {
    // SMT solver implicitly injects `requires topology: NPU` here.
    // If a user tries to call this directly from the Host thread without `spawn_on`,
    // the SMT solver flags a precondition violation at compile time.
    fn dummy(x: i32) -> i32 { ... }

    fn fast_gelu(x: Tensor<f32, [N]>) -> Tensor<f32, [N]> { ... }
}

```

In this model, a `Verified<T>` is literally just a type whose `where` clauses have successfully resolved to `Sat` (Satisfiable) in the Z3 context!

### 2. Brainstorming the Open Questions

Here are my thoughts on the open design questions you posed to the community:

**Bundling Z3 vs. SMT-LIBv2:**
I strongly recommend **not** bundling Z3 statically and instead communicating via the **SMT-LIBv2 standard over stdin/stdout** (or a lightweight IPC channel).

- *Why?* Bundling Z3 creates a massive compilation dependency, bloats the `vxc` binary, and tightly couples you to one solver's release cycle. Emitting SMT-LIBv2 makes the Vx compiler solver-agnostic. If a user prefers CVC5 or Alt-Ergo because it handles a specific theory better, they can just swap it out. You can ship `vxc` and simply tell users, "Vx requires a compliant SMT solver in your PATH."

**Syntax for Formal Loop Invariants:**
For a systems language, keeping it close to the loop header is crucial so the developer understands *why* the loop is sound before reading the body.
Borrowing slightly from languages like Dafny or Ada SPARK, a `where` or `invariant` clause attached directly to the block feels natural:

```rust
for i in 0..N
where invariant(i < bounds), invariant(memory_safe(ptr + i))
{
    // ...
}

```

Alternatively, if you want to reuse the `comptime` keyword to clearly delineate what is executed by the compiler vs the hardware:

```rust
for i in 0..N {
    comptime { assert(i < bounds); }
    // ...
}

```

**Decidable vs. Non-linear Arithmetic:**
**Stick strictly to decidable theories** (specifically QF_LIA - Quantifier-Free Linear Integer Arithmetic, bitvectors, and arrays).
Do not allow unbounded non-linear arithmetic (like multiplying two symbolic variables together, `x * y > z`). If you allow full non-linear arithmetic, the SMT solver will inevitably hang on edge cases, resulting in infinite compile times. The fastest way to kill adoption of a new language is an unpredictable, hanging compiler.

If a user *must* do something non-linear, force them to provide an explicit computational bound, or introduce a strict `timeout` (fuel limit) for the solver in the compiler flags (e.g., `vxc --smt-timeout=500ms`).

### Tying it to the Plugin Architecture

This completely validates the plugin architecture we discussed earlier. If a vendor adds a new NPU target via a plugin, they aren't just providing MLIR lowering rules; they are providing **SMT axioms**.

For example, a vendor plugin could inject an axiom into the solver like: `Axiom: NPU_V3_Max_Memory = 64GB`. Then, when your `Sema` pass translates the AST to Z3, it uses the vendor's axioms to formally prove that the user's `spawn_on` block won't OOM the device.
