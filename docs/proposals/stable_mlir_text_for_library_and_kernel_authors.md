# [RFC] Stable MLIR text for library and kernel authors

*A proposal to the MLIR community.*

## Summary

Vx is a programming language for tensor programs. Its standard library contains 164 blocks of upstream MLIR that people wrote by hand. MLIR makes no promise that this text will still parse after the next LLVM release. Three forms that the library uses today did not exist in 2021.

I think the problem is larger than one language. Programmers need a layer between the language and the hardware that the compiler can still read and optimize. Upstream MLIR dialects are a good vocabulary for that layer. A small stability promise could make MLIR the standard home for it.

This RFC describes the situation, the problem and the opportunity. It ends with four questions.

## The situation: MLIR as inline assembly

C and Rust give a library author inline assembly. Vx gives them an `mlir!` block. The header binds Vx values to SSA names and declares effects. The body is a list of upstream MLIR operations. This block is from `stdlib/core/num.vx`:

```
fn count_ones(self : usize) -> usize {
    return mlir!(inputs : (%a = self : i64), returns : usize, dialects : ["math"]) {
        %r = math.ctpop %a : i64
        macro.yield %r : i64
    };
}
```

The compiler checks the header against the Vx types. It wraps the body in a private `func.func` and parses it with the MLIR parser. At `-O1` and above, it inlines the wrapper before loop fusion. The block then takes part in the same passes as the code around it.

The Vx standard library uses 164 of these blocks. They implement bit operations, atomics, SIMD reductions, tensor fills and a port of libm. They use `arith`, `math`, `vector`, `memref`, `scf`, `linalg` and the `llvm` dialect. None of them needs a compiler intrinsic or a call into libc.

Three properties make this design attractive:

- The optimizer sees inside the block. Inline assembly is a black box to it.
- One block can lower to several targets, wherever its dialects have a lowering.
- A library author can add a new operation without a change to the compiler.

## The problem: the text is not stable

MLIR's custom assembly formats change as the dialects improve. That is good for MLIR. For text inside a library, each change is a break. The table shows four real changes. The first three changed syntax that the Vx standard library uses today.

| Operation | Old text | New text | Commit |
|---|---|---|---|
| `linalg.fill` | `linalg.fill(%v, %buf) : f32, memref<?xf32>` | `linalg.fill ins(%v : f32) outs(%buf : memref<?xf32>)` | [7294be2b8e9a](https://github.com/llvm/llvm-project/commit/7294be2b8e9a), March 2022 |
| `vector.reduction` | `vector.reduction "add", %x : vector<16xf32> into f32` | `vector.reduction <add>, %x : vector<16xf32> into f32` | [fe0bf7d4694e](https://github.com/llvm/llvm-project/commit/fe0bf7d4694e), February 2022 |
| `llvm.atomicrmw` | `llvm.atomicrmw fadd %p, %v monotonic : f32` | `llvm.atomicrmw fadd %p, %v monotonic : !llvm.ptr, f32` | [d8153ae29959](https://github.com/llvm/llvm-project/commit/d8153ae29959), February 2023 |
| `nvvm.vote` | `nvvm.vote.ballot.sync %m, %p : i32` | `nvvm.vote.sync ballot %m, %p -> i32` | [cd2f85a24b55](https://github.com/llvm/llvm-project/commit/cd2f85a24b55), April 2025 |

Each change was reasonable. The `vector.reduction` commit carried the NFC tag, yet it changed the text that people write.

Vx copes in two ways today. It pins one LLVM release, currently LLVM 22. It also parses every block with the MLIR parser, so a break usually appears as a clear parse error. But each LLVM upgrade can force edits to library source, and library authors must make them. If third-party libraries contain `mlir!` blocks, a compiler upgrade can break code that the Vx project does not control.

Other layers handle this question in different ways:

- **Inline assembly.** The hardware vendor defines the assembly syntax. Changes that break old code are rare.
- **PTX.** NVIDIA versions the PTX ISA, and newer drivers accept older PTX. Many CUDA libraries contain inline PTX.
- **LLVM IR.** The Developer Policy says: "The textual format is not backwards compatible." But the current LLVM loads any bitcode since version 3.0.
- **MLIR bytecode.** The format is "versioned and stable." Since 2023, a dialect can also version its own ops and upgrade old IR in `upgradeFromVersion`. Nobody writes bytecode by hand.
- **StableHLO.** Portable artifacts get 5 years of backward and 2 years of forward compatibility. The promise excludes the pretty-printed text.

Every stable form of MLIR today is a serialized form. No form that a person writes by hand has a promise.

The problem is not unique to Vx. Mojo exposes MLIR operations in source through `__mlir_op`. Its docs say that its internal dialects "may change without notice." Julia's `Base.llvmcall` takes LLVM IR text, so it has the same exposure one level lower.

A 2022 thread, "MLIR generic IR stability and upgradability", asked a related question. It discussed bytecode and the generic form. It ended without a policy, and one reply suggested that each dialect should decide for itself. It did not discuss hand-written text.

People raised this problem after my talk on `mlir!` blocks and in a Reddit thread. They are right. I also think the problem points to an opportunity.

## The opportunity: a stable layer for kernel code

ML programmers now work at two levels. They write the model in a language or framework, such as PyTorch or JAX. They write the hot loops in a kernel language, such as Triton or Pallas. In a kernel language, the author controls tiling, memory placement and synchronization. The compiler still reads the code and optimizes it.

An `mlir!` block fills the same role inside one language. It is lower level than Vx and higher level than assembly. The compiler sees all of it.

Many kernel languages already lower through MLIR, Triton and Pallas among them. The upstream dialects that Vx uses describe scalar math, SIMD, memory access, loops and atomics. They do this without naming a target. That makes them a good vocabulary for the kernel layer. The missing part is a promise.

PTX shows what a promise buys. It is a stable, versioned layer that people write by hand, for one vendor's GPUs. StableHLO gave whole ML graphs a stable layer through a versioned dialect, VHLO. Between them sits a level with no stable form: target-neutral scalar, vector, memory and atomic operations inside one kernel. MLIR is the natural home for that level. A stable text there would serve every frontend that embeds MLIR, and every kernel language that emits it.

## Options

The options go from cheapest to most expensive. A project could take more than one.

1. **Write the policy down.** The MLIR docs state the policy for bytecode and for the C API. I found none for custom assembly formats. One paragraph, like the one in the LLVM Developer Policy, would help frontends plan.
1. **Mark format changes.** A commit tag or a release-note section for assembly format changes would let frontends find breaks without a bisect. Today such a change can carry the NFC tag.
1. **Test hand-written text upstream.** Vx can contribute its blocks as round-trip tests. LLVM keeps `compatibility.ll` for bitcode in the same spirit. A change that breaks the corpus then becomes visible in review.
1. **Keep old forms for one release.** For a few core dialects, the parser could accept the old form for one release and emit a warning. The cost is old parser code that stays in the tree for that release.
1. **Define a stable kernel profile.** Specify a small set of ops with a versioned text form, as StableHLO and VHLO did for graphs. This is the largest option. It only makes sense if several frontends need it.

## Questions for the community

1. Does a written policy exist for the stability of custom assembly formats? If not, would the community accept one?
1. Would upstream accept a small corpus of MLIR text from frontends as round-trip tests?
1. Would the maintainers of `arith`, `math`, `vector`, `memref`, `scf` and the `llvm` dialect consider a one-release window for format changes?
1. Which other projects embed MLIR text in source, or plan to? Is there enough interest for a stable kernel profile?

## Links

- Vx: https://github.com/vx-lang/Vx
- How Vx uses MLIR: https://vxlang.org/blog/vx-fully-embraces-ml-ir-mlir.html
- The 2022 thread: https://discourse.llvm.org/t/mlir-generic-ir-stability-and-upgradability/65371
- LLVM IR backwards compatibility: https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility
- MLIR bytecode format: https://mlir.llvm.org/docs/BytecodeFormat/
- StableHLO compatibility: https://openxla.org/stablehlo/compatibility
- Mojo inline MLIR: https://mojolang.org/docs/reference/inline-mlir/
