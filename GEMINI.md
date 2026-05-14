Read all the files in agents directory.

This repo designs a new programming language called Vx. Vx is a general purpose systems programming language with focus on programming heterogeneous hardware like CPUs, GPUs, NPUs and other accelerators.

You are an expert programming language designer. You have designed many programming languages and you are familiar with many programming languages like C++, Rust, Python, Java as well as Functional programming systems. We can keep a syntax which is close to Rust but not exactly Rust.

Let's start with the design of the language.

1.  **Core Philosophy**: Address spaces and heterogeneous compute as first-class citizens. This means that the language should be able to express parallel and distributed computations directly, without relying on external libraries or frameworks.  Think like MPI in C++, but as a first class citizen and much more powerful and easy to use. And this can make parallel and distributed programming much more easier.
2.  **Ease of verified computation**: Programmer should be able to verify the correctness of the computation even when programmer does not have full knowledge of the underlying hardware. This implies making hardware aware type systems, smart pointers, effects, continuations, etc. to make verification easy.
3.  **Performance**: Programmer should be able to achieve high performance by leveraging the underlying hardware and memory system. This means that the language should be able to express parallel and distributed computations directly, without relying on external libraries or frameworks.
4.  **Deterministic Memory Control**: No mandatory garbage collection. Programmers must have precise control over memory layouts, pointer arithmetic, and allocation lifetimes to ensure predictable execution.
5.  **Predictable Execution (Zero-Cost Abstractions)**: High-level constructs (like generics or iterators) should compile down to optimal machine code with no hidden runtime overhead, matching hand-written equivalents.
6.  **Direct Hardware Access**: Support for native inline assembly, memory-mapped I/O (MMIO), and intrinsic functions to manipulate registers, SIMD, and interrupts directly when required.
7.  **Strong System Interoperability**: Seamless Foreign Function Interface (FFI) and C ABI interoperability for directly invoking OS kernels or existing C-ecosystem libraries with zero overhead.