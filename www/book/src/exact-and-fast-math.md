# Exact and fast math

A float method such as `x.exp()`, `x.ln()`, `x.sin()` or `x.tanh()` gives an exact result: for
`f32`, the float nearest to the true value. `x.fast_exp()` and `x.fast_tanh()` are faster on a GPU,
but their results are less accurate. This chapter explains when to use which.

The methods are in `core::num`, so a program that calls them starts with `import core::num;`.

## Exact by default

Vx implements the float methods in Vx itself, ported from LLVM libc. The `f32` methods are correctly
rounded: each result is the float nearest to the true value. Most `f64` methods are correctly
rounded too. The comment above each method in `stdlib/core/num.vx` says when one is not, as for the
`f64` `tanh`, which is almost always correctly rounded.

Because the methods are Vx code, the CPU, an NVIDIA GPU and the CPU copy of a GPU kernel run the
same code and give the same bits. Vx does not promise the same bits on every target. A kernel on an
Intel GPU (`arch: spirv64`) calls the device's own function for each method, because the methods
compute in `f64`, and an Arc GPU has no `f64`. Its last bits can differ from the CPU.

## The cost on a GPU

To round correctly, `f32` methods such as `exp` and `tanh` compute in `f64`, with a table of values.
A GPU does `f64` arithmetic more slowly than `f32`: an A100 at half the rate, and an L4 at 1/64 of
it. In a kernel that calls `tanh` or `exp` for each element, that cost can be a large part of the
time of the kernel.

## `fast_exp` and `fast_tanh`

`x.fast_exp()` and `x.fast_tanh()` use the target's own function: a few `f32` instructions on an
NVIDIA GPU, and the C library on a CPU. They are not correctly rounded, and their results can
differ from target to target, and between a kernel on a GPU and its CPU copy.

On an NVIDIA GPU of `sm_75` (Turing) or newer, `fast_tanh` on `f32` is one instruction,
`tanh.approx.f32`, as `tanhf` is in CUDA with `--use_fast_math`. Its relative error can be as
large as 2^-11, about 0.0005. So only about the first three digits of the result are correct. An
older GPU computes `fast_tanh` with a few more instructions and a smaller error.

`fast_exp` and `fast_tanh` exist for `f32`, `f64`, `f16` and `bf16`. An `across` loop can call
them on `f32` and `f64`. It cannot call any math method on `f16` or `bf16` yet
([#3353](https://github.com/vx-lang/Vx/issues/3353)).

This program computes the GELU of 1024 numbers in two kernels, one with `tanh` and one with
`fast_tanh`, and compares the results (`book_gpu_exact_and_fast_tanh.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_exact_and_fast_tanh.vx:program}}
```

It prints:

```text
gelu(1) = 0.841192
the two kernels differ by less than 0.002: true
```

The GPU code of `gelu_exact` has `f64` instructions, and the GPU code of `gelu_fast` has none. The
error of `tanh.approx.f32` is at most 0.0005, and the size of `0.5 * v` is at most 2 here, so the
two kernels differ by at most 0.001 on a GPU. On an A100, the GELU kernel of llm.c ran about 1.25
times as fast with `fast_tanh` as with `tanh`. That number came from the code before `fast_tanh`
used `tanh.approx.f32`.

## Which one to use

- Use the exact methods by default.
- Use `fast_exp` or `fast_tanh` in a GPU kernel when the function is a large part of the time of
  the kernel, and a small error does not matter. Activation functions and softmax
  in machine learning are examples.
- Test a program that uses them with a tolerance, as the program above does, not by comparing bits.
  Test a program that runs kernels on an Intel GPU with a tolerance too.
