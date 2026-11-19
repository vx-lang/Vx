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
| `f8e4m3` | 8 | up to 448 | no | yes | works, except matmul ([#1876]) |
| `f8e5m2` | 8 | up to 57344 | yes | yes | works, except matmul ([#1876]) |
| `f4e2m1` | 4 | 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives | no | no | works, except matmul ([#1876]) |
| `f8e8m0` | 8 | powers of two, 2⁻¹²⁷ to 2¹²⁷ | no | yes | works, for block scales |
| `i4` | 4 | -8 to 7 | | | works |
| `u4` | 4 | 0 to 15 | | | works |

In a float type's name, `e` is the number of exponent bits and `m` the number of mantissa bits.
More exponent bits give a wider range. More mantissa bits give more precision.

## Conversions

Nothing converts by itself. You write `as`:

```rust
let x : f32 = 2.7;
let h = x as bf16;
let back = h as f32;
```

`as` also converts each element of a float tensor, and gives a new tensor of the same shape. The
cast only reads the tensor, so you can use the tensor again after it:

```rust
let w = Tensor<bf16, [64, 64]>::new();
let wide = w as f32;     // a Tensor<f32, [64, 64]>
let narrow = wide as f16;
```

A conversion to a narrower float rounds to the nearest value it can hold. On a tie it picks the
value whose last bit is 0.

A value too large for the type gives infinity in `f16` and `bf16`. `f8e4m3` and `f4e2m1` have no
infinity, so a value too large for them gives the largest value they hold: 448 or 6. `f8e5m2`
does the same, and gives 57344. NVIDIA's conversion instructions also work this way when
saturation is on.

`f8e8m0` holds only powers of two, and it is for the scales of MXFP4 blocks. A conversion to it
rounds down: `3.0 as f8e8m0` is 2, as the OCP MX rule for a block scale wants. A value below 2⁻¹²⁷,
0 included, gives 2⁻¹²⁷, and a value above 2¹²⁷ gives 2¹²⁷. `f8e8m0` has no sign, so a value
below 0 gives NaN.

Integer arithmetic wraps around. In `i4`, `3 * 2 + 3` is 9, which does not fit, so the answer is
-7. In `u4`, `15 + 1` is 0.

## Arithmetic on fp8 and fp4

`f16`, `bf16`, `i4` and `u4` compute like the other number types. An operation on two `bf16`
values gives a `bf16`.

An operation on `f8e4m3`, `f8e5m2`, `f4e2m1` or `f8e8m0` values needs a result type that you write:

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

A matmul will work the same way. The compiler does not support it yet ([#1876]):

```rust
let c : Tensor<f32, [64, 64]> = a @ b;      // a and b are f4e2m1 tensors
```

The result type can be `f32`, `f16`, `bf16`, `f8e4m3`, `f8e5m2` or `f4e2m1`. The operands can be of
two different narrow types, for example an `f4e2m1` tensor times an `f8e4m3` tensor.

A matmul of tiles in the code of a subgroup loop already works with fp8 tiles, when the result is
`f32`:

```rust
acc[w] = acc[w] + (sa[w] as f32) @ (sb[w] as f32);   // sa and sb are f8e4m3 tiles
```

On an NVIDIA GPU from `sm_89` on, it runs on the tensor cores in two cases:

- Both tiles have the same fp8 type, and the sizes are multiples of 16 x 8 x 32.
- One tile is fp8 and the other is `f16`, and the sizes are multiples of 16 x 8 x 16. Each lane
  converts its fp8 values to `f16`, which holds them exactly.

The tensor cores add the products in an order of their own, so the last bits of a sum can differ
from the sum on the CPU. When both tiles are fp8, the tensor cores of `sm_89` multiply the fp8
values. On `sm_90`, the assembler of NVIDIA (`ptxas`) converts the values to `f16` first, and the
tensor cores multiply them as `f16`.

## Block-scaled fp4: NVFP4 and MXFP4

The compiler does not support the scaled tensor type yet ([#1877]). In the code of a GPU block or
subgroup, `lanes::matmul_scaled` already multiplies two NVFP4 or MXFP4 tiles whose scales are in
tensors of their own ([#3645]).

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

[#1876]: https://github.com/vx-lang/Vx/issues/1876
[#1877]: https://github.com/vx-lang/Vx/issues/1877
[#3645]: https://github.com/vx-lang/Vx/issues/3645
