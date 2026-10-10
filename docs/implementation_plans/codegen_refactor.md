# Code generator refactor

**Status:** proposed 2026-10-10. Tracking issue: Vx#1531. Measured at main e5ebfc72.

______________________________________________________________________

## 0. Summary

Code generation is about 23,000 lines in three parts:

| Part | Files | Lines |
|---|---|---|
| The AST generator | `src/codegen/generator.rs`, `src/codegen/lower/*.rs` | 12,400 |
| The flat generator | `src/codegen/flat.rs`, `src/codegen/flat/emit/*.rs` | 6,500 |
| The MLIR lowering | `src/dialect/VxLowering.cpp` | 4,000 |

The flat generator is the default. When it cannot compile a program, `vxc` falls back to the
AST generator, and `--legacy-codegen` selects the AST generator by hand. **Both generators
stay.** Compiling the same program two independent ways is how many codegen bugs get found, so
the goal is parity between them, not removing one.

The code grew quickly (225 commits to these files in two months), and it shows in four ways:

1. **Tables written twice.** Operator names, comparison predicates, casts, integer widths, the
   memref descriptor type and the print helpers each exist once per generator. Some copies
   already disagree (section 2).
1. **Gaps in each generator.** The flat generator falls back on 75 of 773 programs. The AST
   generator has about ten open bugs the flat one does not have.
1. **Very large functions.** Eight functions are over 300 lines. The largest is 675.
1. **Comments that tell history.** About 900 comment lines describe how a bug was found, which
   issue a line came from, or a milestone name, rather than what the code does.

Order: cleanups first, then shared tables, so that later parity work is written once. Breaking
up large functions and splitting `VxLowering.cpp` can go alongside the parity work.

______________________________________________________________________

## 1. Cleanups

Small changes with no change in behavior, except the print fallback.

- Delete code nothing calls:
  - `parse_command_line_options` (`codegen/mod.rs`);
  - `is_llvm_ptr` (`generator.rs`);
  - the `addVxLoweringPass` and `addVxToLLVMPass` declarations (`codegen/mod.rs`) and their C++
    bodies;
  - `registerVxLoweringPass`, which has an empty body;
  - the unused `_region` in `generate_function` and the stray doc line above `placed_attr`.
- Make the AST generator's print fallback (`lower/expr.rs`, the `Warning: unsupported print arg type` arm) a compile error. Today it writes a warning to stdout and calls `print_i32` on a
  value of any type.
- `VxLowering.cpp`:
  - one helper for "look up a runtime function, or declare it", which is written out six times;
  - merge `abiTagForType` and `elemDtypeCode`, the same table except that one also has f16 and
    bf16;
  - move the doc comment of `diagnoseUnrunnableSpawns` back above that function.

______________________________________________________________________

## 2. Shared tables

Move each table into `src/mlir_ty.rs`, with both generators calling it:

| Table | AST generator | Flat generator |
|---|---|---|
| Operator to MLIR op name | `lower/mod.rs` (`MeliorOpInfo`) | `flat.rs` (`arith_op`) |
| Comparison predicates | `lower/mod.rs` | `flat.rs` |
| `as` cast to conversion op | `lower/expr.rs`, `generator.rs` (`coerce_type`) | `flat.rs` (`cast_op`) |
| Integer width and signedness | `generator.rs` (`scalar_type_bits`) | `flat.rs` (`is_signed`, `int_bits`) |
| Memref descriptor type | `generator.rs` (`tensor_descriptor_str`) | `flat.rs` (`memref_descriptor_ty`) |
| Element type to MLIR type | `lower/mod.rs` (`extract_mlir_element_type`) | `flat.rs` |
| Which `printMemref*` and scalar print helper | `lower/mod.rs` | `flat/emit/io.rs` |
| Runtime helper declarations | declared on first use | `flat.rs` (`RUNTIME_HELPERS`) |

Then decide each place where the two disagree, with one rule and a test that compiles the
program both ways:

- **Closure value type.** AST: a pair of pointers. Flat: one pointer.
- **A borrow into a memory space.** AST: `!llvm.ptr<N>`. Flat: always `!llvm.ptr`.
- **Enum payload slots.** AST: the widest type by bits, and a struct and a scalar may not share
  a slot. Flat: by size and alignment, and they may.
- **The `vx.placed` attribute.** AST: worked out again from the topology. Flat: read from the
  placement.

Named structs (AST) and unnamed structs (flat) are a difference in spelling only, so that one can
stay.

______________________________________________________________________

## 3. Parity

### 3.1 What the flat generator does not compile

From a sweep of `tests/{backend,frontend,middle_end,optimizations}/pass`, `tests/modules` and
`examples` with `--action emit-mlir`. Each program is counted under the first function the flat
generator declines, so closing a gap can uncover another behind it.

| Gap | Programs | Issues |
|---|---|---|
| Closures as values, mostly iterator chains (`range().map(..).take()`) | 22 | #473, #242 |
| Enums that carry data, and a `match` that returns one | 19 | #233 |
| Borrows of non-tensors (`&x`, `&v[i]`), iterating a `Vec` by reference | 10 | #242, #1084 |
| Reassigning a tensor: a matmul whose result may alias an input, or inside a branch | 8 | #216 |
| Small gaps in the emitter (a load of a non-scalar through a pointer, and others) | 6 | #633 |
| A user `impl transfer` body | 5 | #474 |
| Storing to a field of a field | 2 | #212 |
| Fixtures that expect an error, which the flat generator must give too | 3 | |

`tests/integration_test/flat_corpus_sweep.rs` lists these programs in `KNOWN_DECLINES`. The list
only shrinks.

### 3.2 What the AST generator gets wrong

#675 (unsigned `>>` and `%`), #936, #1162, #1342, #1524, #469, #634, #646, #812, #926 and the
scalar half of #1084.

### 3.3 Checking parity

Today the `EXPECT` lines in `tests/backend/pass` run on both generators. Programs in the other
directories are compiled but their output is not compared. Extend the comparison: every program
both generators compile must print the same thing and exit with the same code. Differences go in
a list that, like `KNOWN_DECLINES`, only shrinks.

______________________________________________________________________

## 4. Large functions

| Function | File | Lines |
|---|---|---|
| `FunctionCallExpr::lower` | `lower/expr.rs` | 675 |
| `BinaryOpExpr::lower` | `lower/expr.rs` | 561 |
| `lower_for_loop` | `lower/control_flow.rs` | 552 |
| `AssignStmt::lower` | `lower/stmt.rs` | 479 |
| `materializeGpuKernels` | `VxLowering.cpp` | 396 |
| `SpawnOpLowering` | `VxLowering.cpp` | 316 |
| `TransferExpr::lower` | `lower/tensors.rs` | 314 |
| `IndexAccessExpr::lower` | `lower/expr.rs` | 305 |
| `coerce_type` | `generator.rs` | 298 |
| `emit_module_mlir` | `flat.rs` | 294 |

Copied code to merge while breaking these up:

- `lower_for_loop` has two loop forms written out twice.
- The Grad, Vjp and Jvp lowerings in `lower/tensors.rs` are three copies.
- "Declare the helper if it is not declared yet" appears four times in `lower/`.
- Three call paths in `lower/expr.rs` handle the result the same way.
- `flat/emit/call.rs` and `flat/emit/tensor.rs` each have two copied blocks.
- `build_agg_map` in `flat.rs` repeats the field mapping of `agg_struct_ty_of`.
- `generator.rs` has two functions named `placed_attr`.

Each function is one pull request, with no change to the generated MLIR. The test suite and a
diff of `--action emit-mlir` over the corpus, before and after, show that.

______________________________________________________________________

## 5. Splitting `VxLowering.cpp`

| New file | What it holds | Lines |
|---|---|---|
| `VxKernelClassify.cpp` | kernel kinds, matmul roles, device math helpers | 580 |
| `VxToStandard.cpp` | `SpawnOpLowering`, `TransferOpLowering`, the standard pass | 600 |
| `VxOwnership.cpp` | buffer origins, placing frees and drops, their diagnostics | 890 |
| `VxGpuOutline.cpp` | `materializeGpuKernels`, `deviceImageOf` | 500 |
| `VxToLLVM.cpp` | the plugin ABI patterns and the LLVM pass | 1,000 |
| `VxBufferPasses.cpp` | the four small optimization passes | 380 |

NVPTX assumptions move into `VxGpuOutline.cpp`: shared memory as address space `3` (three
places), the `"nvptx64"` comparison, the GPU topology range, `sm_80` and `+ptx76`. This is also
the first step of shared-work items 1 and 3 in `docs/gpu_backends.md`.

The split moves code without changing it, one file per pull request.

______________________________________________________________________

## 6. Comments

Do this per file, after that file's changes in sections 4 and 5, so each file is changed once.

| File | Comment lines | In blocks of 6+ lines | Estimated removal |
|---|---|---|---|
| `VxLowering.cpp` | 1,140 | 770 | 400-450 |
| `flat.rs` | 800 | 290 | 170, plus about 50 issue tags at line ends |
| `lower/expr.rs` | 420 | 160 | 85 |
| `generator.rs` | 360 | 120 | 70 |
| `lower/tensors.rs` | 130 | 90 | 45 |
| `lower/mod.rs`, `lower/stmt.rs`, `lower/control_flow.rs` | 300 | 110 | 60 |

What goes: how a bug was found, benchmark figures, issue numbers and milestone names ("M2b-2",
"stage C", "R3"). What stays: the rule the code follows, said once.

______________________________________________________________________

## 7. Order

1. Section 1, in two or three pull requests.
1. Section 2, before any parity work, so a fix lands once.
1. Section 3, gap by gap, with the parity check from 3.3 first.
1. Sections 4 and 5 alongside section 3. They touch different code.
1. Section 6 last for each file.
