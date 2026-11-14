# [RFC] Stable MLIR text for library and kernel authors

## Summary

Vx is a programming language for tensor programs. Its standard library contains about 180 small pieces of MLIR code that people wrote by hand. MLIR does not promise that this code will still compile with the next LLVM release. The text of three ops that the library uses today changed after 2021.

I think this problem is bigger than one language. Programmers need a level between the language and the hardware: lower than the language, but still code that the compiler can read and optimize. MLIR is a good fit for that level. A small promise of stability could make MLIR the standard tool for it.

This RFC explains how Vx uses MLIR, what breaks, and why a fix could help other projects too. It ends with four questions.

## Some terms

- **MLIR** is a toolkit for building compilers. It is part of the LLVM project. A compiler uses it to represent a program and to optimize it in steps.
- A **dialect** is a set of related operations. For example, `arith` has integer and float arithmetic, `math` has functions like `sqrt`, and `vector` has SIMD operations. Other dialects used below are `memref` (memory buffers), `scf` (loops and if statements), `linalg` (operations on whole arrays), `llvm` (LLVM IR instructions) and `nvvm` (NVIDIA GPU instructions). The dialects that ship with LLVM are the **upstream** dialects.
- An **op** (operation) is one instruction, such as `arith.addi` (add two integers). An op takes values, called **operands**. It can also carry fixed settings, called **attributes**, such as which comparison a compare op does. Some ops also contain a nested block of ops, called a **region**, like the body of a loop.
- MLIR code has a text form. Each op can define its own short syntax, called its **custom assembly format**. Each op also has a long **generic form** that spells out every operand and attribute. MLIR also has a binary form, called **bytecode**.
- **Lowering** turns ops of one dialect into ops of a lower dialect, step by step, until the result is machine code.

## How Vx uses MLIR: like inline assembly

In C and Rust, a library author can write **inline assembly**: a few machine instructions inside a normal function. Vx has a similar feature, the `mlir!` block. Instead of machine instructions, the block contains MLIR ops. This example is a simpler form of a block in `stdlib/core/num.vx`. It counts the 1 bits in an integer:

```
fn count_ones(x : i64) -> i64 {
    return mlir!(inputs : (%a = x : i64), returns : i64, dialects : ["math"]) {
        %r = math.ctpop %a : i64
        macro.yield %r : i64
    };
}
```

The first line of the block is its header. It gives the Vx value `x` the MLIR name `%a`, and it says what type the block returns. The body is MLIR text. `math.ctpop` is the upstream op that counts the 1 bits. `macro.yield` is a Vx op, not an upstream one. It gives the result back to the Vx code.

The compiler checks the header against the Vx types. Then it puts the body in a small MLIR function and gives that function to the MLIR parser. When optimization is on (`-O1` and above), the compiler copies small functions like this one into the code that calls them. Compilers call this **inlining**. After that, the MLIR optimizer treats the block like any other code around it.

The Vx standard library source contains about 180 of these blocks. Some of them are inside Vx macros, so one block in the source becomes one copy for each integer type, for example. The blocks implement bit operations, atomic operations, SIMD sums, tensor fills, and a port of the C math library (libm). They use the upstream dialects `arith`, `math`, `vector`, `memref`, `scf`, `linalg` and `llvm`. Only three blocks use an op from Vx's own dialect. These are the heap allocation functions. Their op lets the compiler pick the right C allocator for each target, so the library does not have to name one. No other block needs a special compiler built-in or a call to a C library.

This design has three advantages over inline assembly:

- The optimizer can see inside the block. To the optimizer, inline assembly is a black box.
- One block works on several kinds of hardware, because MLIR can lower its dialects to each of them.
- A library author can add a new operation without changing the compiler.

## The problem: the text changes

MLIR developers often improve a dialect, and sometimes they change the custom assembly format of its ops. That is good for MLIR. But for MLIR text inside a library, each change breaks the library. The table shows four real changes. The Vx standard library uses the first three ops today.

| Operation | Old text | New text | Commit |
|---|---|---|---|
| `linalg.fill` | `linalg.fill(%v, %buf) : f32, memref<?xf32>` | `linalg.fill ins(%v : f32) outs(%buf : memref<?xf32>)` | [7294be2b8e9a](https://github.com/llvm/llvm-project/commit/7294be2b8e9a), March 2022 |
| `vector.reduction` | `vector.reduction "add", %x : vector<16xf32> into f32` | `vector.reduction <add>, %x : vector<16xf32> into f32` | [fe0bf7d4694e](https://github.com/llvm/llvm-project/commit/fe0bf7d4694e), February 2022 |
| `llvm.atomicrmw` | `llvm.atomicrmw fadd %p, %v monotonic : f32` | `llvm.atomicrmw fadd %p, %v monotonic : !llvm.ptr, f32` | [d8153ae29959](https://github.com/llvm/llvm-project/commit/d8153ae29959), February 2023 |
| `nvvm.vote` | `nvvm.vote.ballot.sync %m, %p : i32` | `nvvm.vote.sync ballot %m, %p -> i32` | [cd2f85a24b55](https://github.com/llvm/llvm-project/commit/cd2f85a24b55), April 2025 |

Each change had a good reason. LLVM developers tag a commit NFC ("no functional change") when it does not change what the compiler does. The `vector.reduction` commit had this tag, but it still changed the text that people must write.

Would the generic form help? No. All four changes also break the generic form:

- `vector.reduction`: the reduction kind changed from a string to a new attribute type, `#vector.kind<add>`.
- `linalg.fill`: the op got two groups of operands (`ins` and `outs`) and a region. The short text in the table hides the region, but the generic form must spell it out.
- `llvm.atomicrmw`: the operand types changed when LLVM moved to pointers without an element type ("opaque pointers").
- `nvvm.vote`: the op got a new name.

The generic form keeps the syntax the same for all ops. It does not keep an op's name, attributes or operands the same.

Vx copes in two ways today. First, it uses one fixed LLVM release, LLVM 22. Second, the MLIR parser reads every block, so a change usually shows up as a clear error at compile time, not as wrong code. But each LLVM upgrade can still force library authors to edit their code. And if other people's libraries contain `mlir!` blocks, a compiler upgrade can break code that the Vx project does not control.

## What other tools promise

Other low-level formats handle this question in different ways:

- **Inline assembly.** The chip vendor defines the assembly syntax, and old assembly code keeps working on new tools.
- **PTX.** PTX is NVIDIA's low-level GPU language. NVIDIA gives each PTX version a number, and newer drivers still accept older PTX. Many CUDA libraries contain PTX that people wrote by hand.
- **LLVM IR.** The LLVM Developer Policy says that "the textual format is not backwards compatible". But the binary form, called bitcode, is stable. The current LLVM reads any bitcode since version 3.0.
- **MLIR bytecode.** The MLIR docs call this binary format "versioned and stable". Since 2023, a dialect can also give its ops a version number. It can then update old code when it reads it (`upgradeFromVersion`). But nobody writes bytecode by hand.
- **StableHLO.** StableHLO is a dialect for whole ML models. Tools released up to 5 years later can read a saved model. Tools released up to 2 years earlier can read it too. The promise does not cover the text form.

So MLIR's stable forms are all binary files that tools write. No text form that a person writes by hand has a promise.

Other projects have this problem too. Mojo lets programmers write MLIR ops in source code with `__mlir_op`. Its docs say that its internal dialects "may change without notice". Julia's `Base.llvmcall` takes LLVM IR text, so it has the same problem one level lower.

A 2022 thread on the LLVM forum, "MLIR generic IR stability and upgradability", asked a similar question. It talked about bytecode and the generic form. It did not end with a policy, and one reply said that each dialect should decide for itself. It did not talk about text that people write by hand.

People raised this problem after my talk on `mlir!` blocks at the MLIR workshop in Santa Clara, and in a [Reddit thread](https://www.reddit.com/r/Compilers/comments/1xet0e1/). They are right. I also think the problem shows an opportunity.

## The opportunity: a stable level for kernel code

Today, ML programmers often write code at two levels. They write the model in a framework, such as PyTorch or JAX. They write the most time-critical loops, called **kernels**, in a kernel language, such as Triton or Pallas. In a kernel language, the programmer splits the data into small blocks (tiles) and decides where each block goes in memory. The programmer also decides when threads wait for each other. The compiler still reads this code and optimizes it.

An `mlir!` block has the same role inside one language. It is lower level than Vx, but higher level than assembly. And the compiler can see all of it.

Many kernel languages already compile through MLIR, Triton and Pallas among them. The upstream dialects that Vx uses describe scalar math, SIMD, memory access, loops and atomic operations. They do this without naming a specific chip. That makes them a good language for kernel code. The missing part is a promise of stability.

PTX shows what such a promise gives. It is a stable, versioned format that people write by hand, but only for NVIDIA GPUs. StableHLO gives whole ML models a stable format. It does this with VHLO, a copy of the StableHLO dialect in which every op has a version number. Between these two levels, nothing has a stable form: the math, vector, memory and atomic operations inside one kernel, for any chip. MLIR is the natural place for that level. A stable text there would help every language that embeds MLIR, and every kernel language that generates it.

## Options

The options go from the cheapest to the most expensive. MLIR could pick more than one.

1. **Write the policy down.** The MLIR docs state a policy for bytecode (it is stable) and for the C API (it has no guarantees yet). I found no policy for custom assembly formats. One paragraph, like the one in the LLVM Developer Policy, would help projects plan.
1. **Mark format changes.** A tag in the commit title, or a section in the release notes, could list changes to assembly formats. Projects could then find a break without searching the commit history for it. Today, such a change can be marked NFC.
1. **Test hand-written text in LLVM.** Vx can give its blocks to LLVM as tests that parse each block and print it again. LLVM already keeps a similar test for bitcode: a text file of LLVM IR, `compatibility.ll`, with copies saved as bitcode by older releases. A change that breaks one of these tests is then visible in code review.
1. **Accept the old form for one release.** In a few core dialects, the parser could accept the old text and print a warning. The cost is old parser code in LLVM for that release.
1. **Define a stable kernel subset.** Choose a small set of ops, and give their text form a version number. StableHLO and VHLO did this for whole models. This is the biggest option. It only makes sense if several projects need it.

## Questions for the community

1. Does MLIR have a written policy for the stability of custom assembly formats? If not, would the community accept one?
1. Would LLVM accept a small set of MLIR text from other projects as tests?
1. Would the maintainers of `arith`, `math`, `vector`, `memref`, `scf` and the `llvm` dialect consider accepting the old form for one release after a format change?
1. Which other projects put MLIR text in source code, or plan to? Do enough projects need a stable kernel subset?

## Links

- Vx: https://github.com/vx-lang/Vx
- How Vx uses MLIR: https://vxlang.org/blog/vx-fully-embraces-ml-ir-mlir.html
- The 2022 thread: https://discourse.llvm.org/t/mlir-generic-ir-stability-and-upgradability/65371
- The Reddit thread: https://www.reddit.com/r/Compilers/comments/1xet0e1/
- LLVM IR backwards compatibility: https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility
- MLIR bytecode format: https://mlir.llvm.org/docs/BytecodeFormat/
- StableHLO compatibility: https://openxla.org/stablehlo/compatibility
- Mojo inline MLIR: https://mojolang.org/docs/reference/inline-mlir/
