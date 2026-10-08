# Padding tensors

> **Status: proposal, for review.** A plan for #1183: `t.reshape(shape, PadMode::Pad)` to a larger
> shape. #1423 makes that case an error until this plan is built.

## What exists today

- `reshape` is always a **view**: both code generators make one `memref.reinterpret_cast` of the
  source buffer, and the checker records the result as a borrow of the source (`view_of` in
  `src/hir/check/views.rs`). That is right for an exact reshape and for `PadMode::Trim`, which keeps
  the first elements.
- A view cannot hold more elements than its buffer, so `Pad` to a larger shape read past the buffer
  (#1183). #1423 makes it an error.
- `PadMode` is not part of the language. Every program that uses it declares
  `enum PadMode { Pad, Trim }` itself, and the checker matches the names `PadMode`, `Pad` and `Trim`
  as strings.
- There is no other padding operation. `docs/implementation_plans/tensor_library.md` lists `pad` and
  `concat` as planned.
- `transpose` is the model to follow. It is written like `reshape`, but both code generators make a
  new buffer and copy into it, and the checker treats the result as an owned tensor that is dropped
  like any other.

## Two kinds of padding

There are two different operations people call "pad", and they put the zeros in different places.
For `t : Tensor<f32, [2, 2]>` holding `a b / c d`, padded to `[3, 3]`:

| Flat padding (`reshape` with `Pad`) | Padding each axis | | --- | --- | | `a b c` | `a b 0` | |
`d 0 0` | `c d 0` | | `0 0 0` | `0 0 0` |

- **Flat padding** keeps the elements in row order and adds zeros at the end. It is the inverse of
  `Trim`. It is useful for rounding a flat buffer up to a multiple of a vector width or a tile size.
- **Padding each axis** keeps every element at its own index, `big[i][j] == t[i][j]`. This is what
  convolution borders, aligning a matrix to a tile size, and padding a batch of sequences to one
  length need. It is `numpy.pad`, `torch.nn.functional.pad` and `tensor.pad` in MLIR.

For model code, padding each axis is the one that matters more. Flat padding is what
`reshape(.., PadMode::Pad)` already promises.

**Recommendation:** build both, as two operations, in two phases:

1. Phase 1: `t.reshape(shape, PadMode::Pad)` does flat padding (this closes #1183).
1. Phase 2: a new `t.pad(shape)` pads each axis with zeros at its end. Its source and target must
   have the same rank.

Phase 2 reuses most of phase 1: the new buffer, the zero fill, the copy and the drop. Only where the
copy writes differs.

## Phase 1: flat padding with `reshape`

### Meaning

`let big = t.reshape([m0, m1, ..], PadMode::Pad);` where `t` has `n` elements and the new shape has
`m >= n`:

- `big` is a **new tensor**, not a view. Its first `n` elements in row order are `t`'s, and the rest
  are zero.
- `t` is not borrowed, so it can be changed, moved or dropped while `big` is alive.
- `big` is dropped at the end of its scope, like the result of `t.clone()`.
- When `m == n`, `Pad` is an exact reshape and stays a view, as today.

The pad value is always zero. A `PadMode::Pad(value)` can come later without changing anything
above.

### Steps, one PR each

1. **Make `PadMode` part of the standard library.** Declare `enum PadMode { Pad, Trim }` in
   `stdlib/std/tensor.vx`. The checker then accepts the mode only when it is that enum, not any enum
   named `PadMode`. Update the tests that declare their own. *Open question:* the standard library
   has no prelude, so every program would need `import std::tensor;`. The other choice is to make
   `PadMode` built in, as `Memory` and `Topology` are.

1. **Checker and the AST code generator.**

   - In `check_reshape`, accept `Pad` to a larger shape. Rewrite the call to a separate internal
     operation, so that later passes see a different operation, as they do for `transpose`. Then
     `view_of` does not treat it as a view, and the drop pass gives a `let` bound to it a drop.
   - In the AST code generator, lower it in four steps:
     1. Allocate the new tensor, or use the caller's buffer when the result is returned.
     1. Fill it with zeros.
     1. Reinterpret both buffers as flat, and take the first `n` elements of the new one.
     1. `memref.copy` the source into that prefix.
   - The flat code generator declines the new operation, so these programs fall back to the AST code
     generator. Add them to `KNOWN_DECLINES`.

1. **The flat code generator.** Add a `TensorPad` opcode to `src/bytecode.rs` and an emitter in
   `src/codegen/flat/emit/tensor.rs` modelled on `op_tensor_transpose`. Add the opcode to
   `makes_a_fresh_tensor`, so a padded tensor that is never bound to a name is still dropped. Remove
   the `KNOWN_DECLINES` entries.

1. **Documentation.** Describe `reshape`, `PadMode::Trim` and `PadMode::Pad` in the language
   reference, which mentions neither today, and update §9 of `docs/discussions/view_type.md`.

### Limits in phase 1, each an error rather than wrong code

- **Sizes not known while compiling:** these are already refused for every reshape.
- **A tensor placed in GPU memory or shared memory:** where should the new buffer go? `clone`
  ignores placement today in both code generators, so copying `clone` would put a GPU tensor's copy
  in host memory. Refuse this until it is decided. It needs the same answer as `clone`.

### Tests

Each run test runs on both code generators:

- **Values:** a `[2, 2]` tensor padded to `[3, 3]` and to `[9]`. Print every element: the first four
  are the source's and the rest are zero.
- **Not a view:**
  - change the source after padding, and the padded tensor does not change;
  - move the source into a function while the padded tensor is alive. This must compile, while the
    same program with a view must not.
- **Return:** return a padded tensor from a function. That is error E4005 for a view.
- **Drops:** a padded tensor bound with `let`, a padded tensor never bound, and one padded in a
  loop. An MLIR check shows one `vx.drop` for each new buffer.
- **Errors:** a placed source, and `Pad` with sizes known only at run time.

## Phase 2: padding each axis with `t.pad(shape)`

- **Checker:** the new shape has the same rank as `t`, and is at least as large on every axis. The
  result is a new tensor, as in phase 1.
- **Code:** allocate and fill with zeros, as in phase 1. Then make a strided view of the new buffer
  with `t`'s sizes and the new buffer's strides, and `memref.copy` `t` into it.
- **Flat code generator:** its views refuse strided memrefs today (`view_operands` in
  `src/codegen/flat/emit/tensor.rs`). This copy writes into one, so the emitter has to build it
  itself, as `op_tensor_transpose` does for the copy it reads from.
- **Later, each its own issue:**
  - padding at the start of an axis (`before` and `after` amounts);
  - a pad value other than zero;
  - edge and reflect modes;
  - sizes known only at run time;
  - `concat`, which is the same copy into a part of a larger buffer.

## Decisions needed

1. Build both kinds of padding, as above, or only one?
1. Should `PadMode` live in `std::tensor`, with an import, or be built in?
1. Where should a padded copy of a GPU tensor live? This is the same question for `clone`.
