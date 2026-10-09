# Addressability and scaled number formats

**Status:** proposed 2026-10-09. Tracking issue: Vx#1437. Replaces the one-byte-per-element
rule of Vx#1426 once a code generator stores elements packed.

______________________________________________________________________

## 0. Summary

C and C++ say the smallest unit of memory is a byte, on every machine. Vx targets machines where
that is false:

- a TPU's VMEM is addressed in tiles of 32-bit words;
- a DSP is addressed in 16-bit or 32-bit words;
- B200's TMEM can be reached only by matrix instructions;
- `i4` and `f4e2m1` are stored two to a byte.

Scaled number formats go further. An `f4e2m1` element is one of 15 values from -6 to 6. Its real
value is `element x scale`, and the scale is stored separately, one for each block of 16 or 32
elements. The smallest unit with a meaning is the block, not the element.

Vx copies C today. #1426 makes a `bool`, `i4`, `u4` or `f4e2m1` tensor element take a whole byte,
because both code generators store it that way. `f8e4m3` and `f4e2m1` are plain element types
with no scale, so a program can add two fp4 tensors and get numbers that mean nothing.

This plan makes the address unit a property of each memory space, stores narrow elements packed,
and adds a scaled tensor type that keeps elements and scales together.

The decisions:

1. **The language says what a program can observe; the memory space says how it is stored.** A
   program can ask for its own layout (#246). Changing a machine file can change how fast a
   program runs, never whether it compiles.
1. **The address unit is a number of bits, declared by each memory space.** A space whose memory
   has no linear address (TMEM, a TPU's VMEM) declares `addressable: false`. Tiles stay in
   `tile:` (#246); they decide layout, not addressing.
1. **A reference needs an element that starts on an address unit.** Otherwise the program reads
   and writes the element by value, and the compiler generates the masking.
1. **A scaled tensor is one value with two parts**: the elements and the scales. Each part is a
   tensor with its own memory space.
1. **An fp8 or fp4 tensor without a scale is for storage only.** It can be loaded, moved and
   converted. Arithmetic on it is an error.

## 1. Five things C treats as one

| Idea | Question it answers | Who answers it in Vx |
|---|---|---|
| Storage size | How many bits does an element take here? | the (element type, memory space) pair |
| Access size | What is the smallest unit written without touching its neighbours? | the memory space |
| Addressability | Is `&x` allowed? | the memory space and the element's position |
| Who may address it | Which operations reach this space? | the memory space (#254) |
| Races | Do two tasks writing two elements conflict? | the access size, or the scale block |

Separating them is what lets one program describe a CPU, a GPU and a TPU without lying about any
of them.

## 2. Addressability

### 2.1 The machine file

`Memory` gains three fields:

<!-- vx-doctest: skip (proposed syntax, not parsed yet) -->

```rust
Memory CPU_DRAM { address_unit: 8 bits, access: 8 bits }
Memory VMEM     { within: Memory::HBM, capacity: 64 MiB, addressable: false, tile: [8, 128] }
Memory TMEM     { within: Memory::L2, capacity: 256 KiB, addressable: false }
```

- `address_unit:` is the smallest unit a reference can point to. The default is `8 bits`, so
  every existing machine file keeps its meaning.
- `access:` is the smallest unit the hardware writes without reading its neighbours first. The
  default is `address_unit`.
- `addressable: false` means no reference into the space exists. Its data is reached only through
  whole-tensor operations (`transfer`, matmul, the operations #254 lists).

### 2.2 Packed storage

A tensor of `bool`, `i4`, `u4` or `f4e2m1` is stored packed: 8 `bool`s or 2 four-bit elements to a
byte. A program cannot tell, except through `size_of` and capacity, so the language does not
promise one byte per element.

The memory space can choose another packing; a TPU packs along sublanes of a 32-bit word. That is
the `tile:` work in #246. Until it lands, packed means "dense in row order".

Struct fields are not packed. A `bool` field takes a byte, as in C, so a struct passed to C code
has the layout C expects. Packing struct fields is a later question (§6).

### 2.3 References and element access

`&t[i]` compiles only when element `i` starts on an address unit of the space `t` is in:

- an `i32` tensor in host memory: always;
- an `i4` tensor in host memory: never, because two elements share each byte;
- any tensor in a space with `addressable: false`: never.

When `&t[i]` is refused, `t[i]` (a read) and `t[i] = v` (a write) still work by value. The
compiler generates the load, the shift and the mask. The error says so:

```
error[E3xxx]: cannot take a reference to an element of Tensor<i4, [64]>
  note: two i4 elements share one byte, and host memory is addressed in bytes
  help: read it with `t[i]` or write it with `t[i] = v`
```

A row view (`t[i]` on a 2-D tensor) is allowed when the row starts on an address unit. An `i4`
tensor with an odd number of columns has rows that do not, so those row views are refused.

### 2.4 Races

Two tasks that write different elements inside one `access` unit conflict. They both read the
unit, change their own bits and write the unit back, so one write is lost.

When a `parallel for` or a split gives two tasks elements that share an access unit, that is a
compile error, not a silent race. The same rule sets the split boundaries: the compiler rounds
each task's range to whole access units when the shape allows it.

A cache line is not part of the rule. Sharing one is slow, not wrong.

### 2.5 The boundary with C

The C ABI belongs to host memory. A packed tensor cannot be passed through `extern "C"` as an
element pointer. It is passed as bytes (`Tensor<u8, ..>`, through an explicit conversion), so
the C side sees exactly what is stored.

## 3. Scaled formats

### 3.1 The formats

| Format | Element | Scale | One scale for |
|---|---|---|---|
| MXFP8 / MXFP6 / MXFP4 (OCP MX) | e4m3, e5m2, e2m3, e3m2, e2m1 | `f8e8m0` (a power of two) | 32 elements along one axis |
| NVFP4 (Blackwell) | e2m1 | `f8e4m3`, plus one `f32` for the tensor | 16 elements, then the whole tensor |
| FP8 on Hopper | e4m3 / e5m2 | `f32` | the whole tensor, or each row or column |
| DeepSeek-V3 FP8 | e4m3 | `f32` | 1 x 128 for activations, 128 x 128 for weights |

Every row is "an element type, a scale type, and a block shape". The block shape is a shape, so
one type covers all of them.

### 3.2 The type

```rust
// MXFP4: one f8e8m0 scale for each 1 x 32 block of elements.
let w : ScaledTensor<f4e2m1, [4096, 4096], f8e8m0, [1, 32]>;
```

A `ScaledTensor` has two parts, each an ordinary tensor:

- `w.data : Tensor<f4e2m1, [4096, 4096]>`
- `w.scales : Tensor<f8e8m0, [4096, 128]>`, the data's shape divided by the block shape.

NVFP4's second level is a scaled tensor whose scales are themselves scaled:

```rust
let a : ScaledTensor<f4e2m1, [M, K], ScaledTensor<f8e4m3, [M, K / 16], f32, [M, K / 16]>, [1, 16]>;
```

The nested form is honest but hard to read. A standard-library alias (`NvFp4<[M, K]>`) is the
intended way to write it.

This needs one new element type, `f8e8m0` (8 exponent bits, no sign, no mantissa), which MLIR
already has as `f8E8M0FNU`.

### 3.3 Building and taking apart

Quantized weights arrive as two separate arrays, for example in a checkpoint file. A program
joins them with one call:

```rust
let w = ScaledTensor::from_parts(data, scales);   // checks scales' shape == data's shape / block
let (data, scales) = w.into_parts();
```

`from_parts` is checked while compiling when the shapes are known, and at run time otherwise.

### 3.4 What a program may do with one

- **Read an element by value.** `w[i][j]` is the real value, `data x scale`, as an `f32`. It is
  never a reference: an element has no meaning without its block's scale.
- **Write whole blocks only.** Changing one element can change its block's best scale, so a
  program writes through `quantize`, which takes a plain tensor and returns a scaled one.
- **Use the operations the hardware supports.** That is mainly block-scaled matmul. Anything else
  needs `dequantize` first, which returns a plain `f32` or `bf16` tensor.
- **Move it.** `transfer(w, space)` moves both parts. Each part can also be placed on its own,
  because hardware wants them in different spaces: B200's block-scaled `tcgen05.mma` reads the
  elements from shared memory and the scales from TMEM.

### 3.5 Shape changes

A block is never cut. These are compile errors:

- a slice or view whose start or size along a blocked axis is not a multiple of the block;
- a `reshape` that splits a blocked axis;
- a `pad` that leaves a partial block. `pad` to a whole number of blocks is allowed, and the new
  blocks get the scale that represents zero.

`transpose` moves the blocked axis to the other side, so it cannot be a view. It re-quantizes, and
says so in its name (`transpose_requantize`), because the numbers change.

### 3.6 Capacity

Capacity counts both parts. MXFP4 takes 4.25 bits per element (4, plus 8 bits shared by 32).
NVFP4 takes 4.5 bits per element plus 4 bytes for the tensor.

### 3.7 Storage-only element types

`Tensor<f8e4m3, ..>`, `Tensor<f8e5m2, ..>` and `Tensor<f4e2m1, ..>` without a scale stay in the
language for storage. They can be created, loaded, transferred, printed (as their values) and
converted, but arithmetic on them is an error:

```
error[E3xxx]: cannot add two Tensor<f4e2m1, [64]>
  note: an f4e2m1 element means nothing without its scale
  help: join it with its scales using ScaledTensor::from_parts, or convert it with .to_f32()
```

`.to_f32()` treats every scale as 1. It exists for reading files and for tests.

## 4. The machine model

`dtypes:` lists formats as well as element types:

```rust
dtypes: [f32, f16, bf16, f8e4m3,
         f4e2m1 scaled f8e8m0 per [1, 32],
         f4e2m1 scaled f8e4m3 per [1, 16]],
```

E6026 then refuses a format the part cannot take, with the formats it can:

```
error[E6026]: Device cannot take f4e2m1 scaled f32 per [128, 128]
  note: it takes f4e2m1 scaled f8e8m0 per [1, 32], or f4e2m1 scaled f8e4m3 per [1, 16]
```

Where the scales are stored, such as CUTLASS's 128 x 4 interleaved order or B200's copy into
TMEM, is the compiler's choice, through `tile:` (#246). The program sees `w.scales` in row order.

## 5. Phases

| Phase | What lands | Needs |
|---|---|---|
| 0 | This plan, agreed | nothing |
| 1 | Packed storage for `bool`, `i4`, `u4` and `f4e2m1` tensors on both code generators. Capacity counts packed bits again, replacing #1426's rule. Element reads and writes by value. | nothing |
| 2 | `address_unit:`, `access:` and `addressable:` on `Memory`. `&t[i]` refused where §2.3 says. | phase 1 |
| 3 | The race check of §2.4 on `parallel for` and splits. | phase 2 |
| 4 | `f8e8m0`, `ScaledTensor`, `from_parts`, `into_parts`, element reads, capacity, `transfer`, shape checks (§3.2 to §3.6). | phase 1 |
| 5 | Arithmetic refused on storage-only types (§3.7), with `quantize` and `dequantize` in the standard library. | phase 4 |
| 6 | Formats in `dtypes:` and the extended E6026 (§4). | phase 4 |
| 7 | Block-scaled matmul lowered to the hardware. | phase 6 and the NVPTX backend (#251) |

Phase 1 uses MLIR's narrow-type emulation for `memref` and `arith`
(`populateMemRefNarrowTypeEmulationPatterns`), which turns a `memref<64xi4>` into a
`memref<32xi8>` and generates the shifts and masks, instead of writing them in Vx. Phases 2 and 4
can be built at the same time.

## 6. Open questions

1. **Quantizing.** `quantize` has to choose each scale (usually from the block's largest absolute
   value), a rounding mode (nearest-even or stochastic), and what an overflow does (saturate or
   infinity). These change the numbers a program computes, so the defaults must be written down
   and be the same on every target. Which defaults, and which can a program choose?
1. **Block-scaled matmul's result.** The accumulator type (`f32` on all current hardware), and
   whether the result can be returned scaled directly (fused quantize) or always comes back plain.
1. **Converting between formats**, for example MXFP4 to NVFP4. Each conversion re-quantizes. Is it
   one function per pair, or `quantize(dequantize(x))` with the compiler fusing the two?
1. **Races on scaled tensors.** Is the unit the block, or the larger of the block and the access
   size? Writes are whole blocks already (§3.4), so the block is likely enough.
1. **Packed struct fields.** Should a program be able to ask for `i4` fields to share a byte, as
   a checked C bitfield? Not needed for any current target.
1. **Placement syntax.** How is a scaled tensor with its parts in two spaces written as a type?
   One option is `ScaledTensor<.., data: Memory::SMEM, scales: Memory::TMEM>`.
1. **Address units that are not a whole number of bytes on a host**, such as the Cortex-M
   bit-band region. Declarable with `address_unit: 1 bits`, but nothing generates code for it yet.
