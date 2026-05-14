# Akar Design Document: Tensor Layout Conversions and Reshaping

## Motivation
In the Akar language, tensor shapes and topologies are fully encoded in the static type system (e.g., `Tensor<f32, [128, 128], Topology::ANE>`).
While the `as` keyword enables developers to cast element types and memory spaces, using it to convert between distinct shapes (e.g., `Tensor<f32, [128, 128]>` to `Tensor<f32, [4, 64, 64]>`) introduces dangerous ambiguity. The compiler implicitly enforces a C-style contiguous row-major boundary. This hides the order of the dimensions during the split, potentially resulting in silent performance degradation or logical bugs if the user actually intended to permute axes or view blocks differently.

Akar is built for ML/Systems developers who desire the familiarity of Python/PyTorch with the zero-cost guarantees of a compiled language. We need an intuitive, unambiguous syntax for reshaping and transposing multidimensional tensors.

## Explored Alternatives

### 1. Einops-Style Pattern Matching
**Concept:** Introduce a declarative string-based macro to describe dimension splits and recombinations natively in the compiler.
```rust
let matrix_3d_blocks = rearrange!(
    matrix_2d,
    "(b1 h) (b2 w) -> (b1 b2) h w",
    b1 = 2, b2 = 2
);
```
**Pros:** Highly readable for complex rearrangements, strictly avoids "magic numbers."
**Cons:** Requires embedding a mini-parser for the string DSL inside the frontend. It is heavily biased toward ML tasks and may feel non-idiomatic for general systems programming.

### 2. Explicit Layout Bounds via `as`
**Concept:** Force developers to provide an affine map or permutation map inside the generic layout whenever they use the `as` keyword for a shape cast.
```rust
let b = a as Tensor<f32, [4, 64, 64], Layout::Permuted<[0, 2, 1, 3]>>;
```
**Pros:** Reuses existing language syntax entirely.
**Cons:** Very verbose. It couples logical shape views tightly with the physical memory allocator boundary. Successive transpositions become extremely hard to read.

### 3. Shape Destructuring
**Concept:** Split and combine dimensions dynamically using Rust-like tuple destructuring.
```rust
let [[b1, b2], [h, w]] = a.split([2, 64], [2, 64]);
let b = Tensor::concat([b1, b2, h, w]);
```
**Pros:** Deeply idiomatic to systems programming and strict pattern-matching algorithms.
**Cons:** Visually noisy for ML and limits the compiler's ability to easily map the operations to `memref` view collapsing/expanding without extensive control-flow analysis.

## Chosen Solution: PyTorch-style Method Chaining

Instead of overloading `as` or forcing string-based DSLs, Akar will implement built-in intrinsic methods directly on the `Tensor` type: `.reshape()` and `.transpose()`.

```rust
let matrix_2d: Tensor<f32, [128, 128]> = ...;

let matrix_3d_blocks: Tensor<f32, [4, 64, 64]> = matrix_2d
    .reshape([2, 64, 2, 64])
    .transpose([0, 2, 1, 3])
    .reshape([4, 64, 64]);
```

### Why We Chose This
1. **Familiarity**: It perfectly matches the mental model of ML practitioners transitioning from Python (NumPy/PyTorch) to Akar.
2. **Zero-Cost**: Unlike Python, these methods are statically evaluated intrinsics. They do not execute at runtime; they instruct the compiler to output LLVM MLIR `memref.collapse_shape`, `memref.expand_shape`, and `memref.transpose` views.
3. **Safety**: Because they return new static types, subsequent type-checking ensures that the transformations align precisely with expected function signatures.
4. **Compile-time Checks**: The methods hook seamlessly into our newly built `comptime` evaluator, ensuring all shape arithmetic (e.g., verifying `128 * 128 == 4 * 64 * 64`) is mathematically sound during compilation.

## Conclusion
The `as` keyword will be strictly reserved for ElementType coercion and topology shifting. Logical shape rearrangements will be performed safely and explicitly via intrinsic method chaining, yielding a clean, performant, and ML-friendly language.
