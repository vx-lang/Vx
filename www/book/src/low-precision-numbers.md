# Low-precision numbers

Machine-learning hardware stores numbers in fewer bits than `f32`, so that more of them fit in
memory and more of them move per second. Vx has these types as part of the language. A machine
file lists the ones each device supports (`dtypes:`), and the compiler checks a program against
that list.

## The types

| Type | Bits | Values | Infinity | NaN | Status |
| :--- | ---: | :--- | :---: | :---: | :--- |
| `f16` | 16 | up to 65504, about 3 decimal digits | yes | yes | works |
| `bf16` | 16 | the range of `f32`, about 2 decimal digits | yes | yes | works |
| `f8e4m3` | 8 | up to 448 | no | yes | declaration only ([#1874]) |
| `f8e5m2` | 8 | up to 57344 | yes | yes | declaration only ([#1874]) |
| `f4e2m1` | 4 | 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives | no | no | declaration only ([#1874]) |
| `f8e8m0` | 8 | powers of two, 2⁻¹²⁷ to 2¹²⁷ | no | yes | planned ([#1874]), block scales only |
| `i4` | 4 | -8 to 7 | | | works |
| `u4` | 4 | 0 to 15 | | | works |

In a float type's name, `e` is the number of exponent bits and `m` the number of mantissa bits.
More exponent bits give a wider range. More mantissa bits give more precision.

"Declaration only" means the type checker and the memory capacity checks accept the type, but the
compiler cannot generate code for it yet. You can use it in a machine file's `dtypes:` and in the
type of a tensor that the capacity checks count. A program that computes with it does not compile.

## Conversions

Nothing converts by itself. You write `as`:

```rust
let x : f32 = 2.7;
let h = x as bf16;
let back = h as f32;
```

A conversion to a narrower float rounds to the nearest value it can hold. On a tie it picks the
value whose last bit is 0.

A value too large for the type gives infinity in `f16` and `bf16`. `f8e4m3` and `f4e2m1` have no
infinity, so a value too large for them will give the largest value they hold: 448 or 6. `f8e5m2`
will do the same, and give 57344. NVIDIA's conversion instructions also work this way when
saturation is on.

Integer arithmetic wraps around. In `i4`, `3 * 2 + 3` is 9, which does not fit, so the answer is
-7. In `u4`, `15 + 1` is 0.

## Arithmetic on fp8 and fp4

The compiler does not support this part yet ([#1875], [#1876]).

`f16`, `bf16`, `i4` and `u4` compute like the other number types. An operation on two `bf16`
values gives a `bf16`.

An operation on `f8e4m3`, `f8e5m2` or `f4e2m1` values needs a result type that you write:

```rust
let c : f32 = a * b + d;     // a, b and d are f4e2m1
let e : f8e4m3 = a * b;
let f = a * b;               // error: the result type is not written
```

The compiler computes the whole expression in `f32`, and rounds once, to the type you wrote. This
is what the hardware does. The tensor cores of an NVIDIA Blackwell GPU multiply fp4 values, add the
products in `f32`, and convert the sum in a separate step.

This rule does not break the rule that nothing converts by itself. The result type is in the
program text, so the reader sees where the precision changes.

A matmul works the same way:

```rust
let c : Tensor<f32, [64, 64]> = a @ b;      // a and b are f4e2m1 tensors
```

The result type can be `f32`, `f16`, `bf16`, `f8e4m3`, `f8e5m2` or `f4e2m1`. The operands can be of
two different narrow types, for example an `f4e2m1` tensor times an `f8e4m3` tensor.

## Block-scaled fp4: NVFP4 and MXFP4

The compiler does not support this part yet ([#1877]).

An `f4e2m1` value is a number from -6 to 6. To hold real weights, NVFP4 and MXFP4
put the values in blocks, and give each block a scale. A value is its `f4e2m1` number times the
scale of its block.

| Format | Values per block | Scale per block | Scale per tensor |
| :--- | ---: | :--- | :--- |
| NVFP4 | 16 | `f8e4m3` | one `f32` |
| MXFP4 | 32 | `f8e8m0`, a power of two | none |

The scale is part of the tensor type:

```rust
let w : Tensor<f4e2m1, [4096, 4096], Scale<f8e4m3, block: 16>>;   // NVFP4
let m : Tensor<f4e2m1, [4096, 4096], Scale<f8e8m0, block: 32>>;   // MXFP4
```

`quantize` makes such a tensor from an `f32` tensor, and chooses each block's scale. `dequantize`
turns it back into `f32`. `@` applies the scales and adds the products in `f32`.

The memory capacity checks count the scales too. A 4096 by 4096 NVFP4 tensor is 8 MiB of values
and 1 MiB of scales, so it needs 9 MiB.

[#1874]: https://github.com/vx-lang/Vx/issues/1874
[#1875]: https://github.com/vx-lang/Vx/issues/1875
[#1876]: https://github.com/vx-lang/Vx/issues/1876
[#1877]: https://github.com/vx-lang/Vx/issues/1877
