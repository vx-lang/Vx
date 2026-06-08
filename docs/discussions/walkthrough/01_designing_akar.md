# Akar Compiler Bootstrapping Walkthrough

We have successfully completed the first major phases of bootstrapping the **Akar** programming language. This walkthrough covers the new language specifications and the newly built compiler frontend.

## 1. Formal Language Specification

Based on the `GEMINI.md` and `Proposal.md` documents, I have drafted the formal language specifications in the `docs/` directory:

- \[docs/syntax.md\](file://\<WORKSPACE_ROOT>/docs/syntax.md): Defines the lexical and syntactic structure, including the novel `spawn on` blocks and `transfer` primitives for managing heterogeneous compute.
- \[docs/types.md\](file://\<WORKSPACE_ROOT>/docs/types.md): Formalizes the topology-aware type system. It specifies how `Ref<T, Memory>` isolates address spaces and how `Verified<T>` bridges the hardware state monad to `Pinned<T, Topology>`.

## 2. Compiler Scaffolding (C++)

To guarantee frictionless integration with MLIR and LLVM down the road (leveraging your previous experience with LLVM passes), the compiler is implemented in **C++** using CMake.

### 2.1 The Lexer

I have built a robust Lexer capable of parsing Akar's custom tokens.

Files created:

- \[CMakeLists.txt\](file://\<WORKSPACE_ROOT>/CMakeLists.txt)
- \[include/akar/Lexer.h\](file://\<WORKSPACE_ROOT>/include/akar/Lexer.h)
- \[src/Lexer.cpp\](file://\<WORKSPACE_ROOT>/src/Lexer.cpp)
- \[src/main.cpp\](file://\<WORKSPACE_ROOT>/src/main.cpp)

### 2.2 Validation

I wrote a sample Akar script \[test.ak\](file://\<WORKSPACE_ROOT>/test.ak) that tests our new primitives:

```rust
fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_data = transfer(a, Memory::NPU_HBM);
        let result = custom_matmul(local_data);
        return transfer(result, Memory::Host_DRAM);
    }
}
```

Running the lexer over this code correctly parses our custom tokens:

```text
Akar Compiler (akarc) - Bootstrap Phase
=======================================
Successfully lexed 64 tokens.
Token: [fn] at line 1:1
Token: [distributed_matmul] at line 1:4
...
Token: [Verified] at line 1:61
Token: [spawn] at line 2:5
Token: [on] at line 2:11
Token: [Topology] at line 2:14
Token: [::] at line 2:22
...
```

## Next Steps

We are now ready to move to **Phase 3: Parsing and AST Generation**. This will involve defining our Abstract Syntax Tree in C++ and converting this flat token stream into a nested hierarchical representation of the program before we proceed to Semantic Analysis.
