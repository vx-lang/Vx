# Tensor operations

The module `std::tensor` gives a tensor the operations of numerical code. It has sums, statistics,
element-wise changes, reductions along an axis and a matrix product. [A tour of
Vx](tour.md#arrays-and-tensors) shows how to make a tensor, read its shape, and `reshape` or `pad`
it. This chapter starts from there.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Sums of a 1-D tensor

A reduction turns many elements into one value. A sum is the most common one. Add
`import std::tensor;` to get these methods on a 1-D tensor of `f16`, `bf16`, `f32` or `f64`:

- `a.sum()` adds the elements, and `a.mean()` divides that sum by the length.
- `a.dot(&b)` multiplies the elements of `a` and `b` in pairs and adds the products.
- `a.norm()` is the length of `a` as a vector: the square root of `a.dot(&a)`.
- `a.max()` and `a.min()` give the largest and the smallest element. They skip a NaN.

This program calls each one (`book_tensor_ops_sums.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tensor_ops_sums.vx:program}}
```

It prints `sum 4 dot -0.5`, then `max 3 min -2`, then `mean 1 norm 4.2426405`.

The length can be fixed in the type, as in `Tensor<f32, [4]>`, or known only at run time, as in
`Tensor<f32, [?]>`. A row of a 2-D tensor is a 1-D tensor too, so `m[1].sum()` adds row 1.

These methods add the elements in an order that they choose, so that they can use vector
instructions. The answer can differ in the last bits from a plain loop that adds one element at a
time. The methods add `f16` or `bf16` elements in `f32`, and they give an `f32`. An `f16` total
above 65504 does not fit in an `f16`.

In earlier versions of Vx, the compiler had `dot`, `sum`, `max` and `min` as built-in functions.
Now they are methods. So `dot(q[i], k[j])` is now `q[i].dot(&k[j])`, and the program needs `import std::tensor;`.

## Statistics and softmax

`std::tensor` has these methods on a 1-D tensor too:

- `a.argmax()` and `a.argmin()` give the index of the largest or smallest element. They give an
  `Option<i32>`: `None` for an empty tensor. When two elements are equal, they give the first one.
- `a.variance()` is the mean of the squared distances from the mean. It divides by the length `n`,
  not `n - 1`. `a.std_dev()` is its square root.
- `a.softmax_inplace()` replaces each element `x` with `exp(x)` divided by the sum of `exp` over
  all the elements. After it, the elements are positive and their sum is 1.

`softmax_inplace` subtracts the largest element before it calls `exp`. The answer is the same, but
`exp(1000.0)` does not overflow. This program uses each method (`book_tensor_ops_statistics.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tensor_ops_statistics.vx:program}}
```

It prints `argmax 4`, `argmin 3`, `variance 7.6 std_dev 2.756809750418044`, then
`softmax 0.09003057317038046 0.24472847105479764 0.6652409557748218` and
`empty argmax is none: true`. The last digits of the softmax line can differ on another machine,
because `exp` comes from the system's math library.

`softmax_inplace` changes the tensor, so the tensor must be `mut`. The other methods of this
section do not change the tensor.

## Element-wise operations in place

An element-wise operation does the same thing to each element. The methods of `std::tensor`
change the tensor in place, so they make no new tensor. A chain of them allocates no memory.

| Method | What it does to each element `y[i]` |
| --- | --- |
| `y.add_assign(&x)` | `y[i] + x[i]`. `sub_assign`, `mul_assign` and `div_assign` do the same with `-`, `*` and `/`. |
| `y.add_scalar(a)` | `y[i] + a` |
| `y.scale(a)` | `y[i] * a` |
| `y.scaled_add(a, &x)` | `y[i] + a * x[i]` |
| `y.clamp_inplace(lo, hi)` | holds it between `lo` and `hi` |
| `y.abs_inplace()`, `y.sqrt_inplace()` | its absolute value, its square root |
| `y.map_inplace(f)` | `f(y[i])`, for a closure `f` |

The `_assign` methods take the place of `+=` and the other compound operators, because Vx cannot
overload them on a tensor yet. Both tensors must have the same length. A different length stops
the program with an error message. This program uses them one after the other
(`book_tensor_ops_in_place.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tensor_ops_in_place.vx:program}}
```

It prints one line for each step, from `add_assign 11 22 33 44` to `sqrt_inplace 4.5 9 10 10`.

`map_inplace` calls the closure once for each element. When one of the named methods does the same
work, use it. It is faster.

## Your own reduction with `reduce`

`t.reduce(init, |acc, x| ..)` is a reduction that you write. It starts with `acc` equal to `init`.
For each element `x`, it calls the closure and makes the result the new `acc`. At the end, it gives
the last `acc`. `reduce` is part of the language, so it needs no import. It works on a tensor of
any rank and of any element type, integers too. The closure can read a variable from outside it.

This program finds a total, a largest value, a total of reduced prices, and a product
(`book_tensor_ops_reduce.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tensor_ops_reduce.vx:program}}
```

It prints `total 18 most 7.5 sale 9`, then `product 720`.

`init` must have the type of the elements, and the closure must give that type too. The compiler
refuses a program that breaks either rule. `reduce` can combine the elements in any order, as `sum` does. So use a closure
for which the order does not change the answer, such as `+`, `*` or a maximum.

## Matrices: reductions along an axis and `matmul`

A 2-D tensor of a float type has `sum`, `mean`, `max` and `min` over all its elements. It has the
element-wise methods of the table above, except `scaled_add` and `map_inplace`. It also has
reductions along an axis, as numpy's `axis=` argument does:

- `m.sum_axis(0)` reduces each column, so it gives one value for each column.
- `m.sum_axis(1)` reduces each row, so it gives one value for each row.
- `mean_axis`, `max_axis` and `min_axis` work in the same way.

Each one gives a new 1-D tensor. An axis other than 0 or 1 stops the program.

`a.matmul(&b)` is the matrix product for `f32` and `f64`. It makes a new tensor with the rows of
`a` and the columns of `b`. Element `[i][j]` is the sum over `k` of `a[i][k] * b[k][j]`. This
program uses both (`book_tensor_ops_matrices.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tensor_ops_matrices.vx:program}}
```

It prints `sum 21 max 6`, `sum_axis(0) 5 7 9`, `sum_axis(1) 6 15` and `mean_axis(1) 2 5`. Then it
prints `matmul is 2 by 2` and the two rows `4 5` and `10 11`.

The number of columns of `a` must be equal to the number of rows of `b`. `matmul` checks this when
the program runs, not when it compiles, even when the two shapes are in the types. A wrong shape
stops the program with the message
`matmul needs as many columns on the left as rows on the right`.

For code on a GPU, see [Parallel code on a GPU](gpu.md).
