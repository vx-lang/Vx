# Implementation Plan: Pointers, Structs, and Safe Borrowing

The goal is to implement deterministically controlled memory representations using Custom Structs, safe Borrows (`&`), and raw Pointers (`*mut`), alongside a strict Rust-like borrow checker with an `unsafe` escape hatch.

## The "Stack Frame" Concept in Vx

You raised an excellent question: *How do we store a pointer to vmem on a call-stack?*

Because Vx compiles to MLIR/LLVM IR, the concept of a pointer is inextricably linked to an **Address Space**. On the host CPU call stack (which lives in Address Space 0, `Host_DRAM`), allocating space for a pointer to `NPU_HBM` (Address Space 1) simply allocates a 64-bit integer slot on the CPU's local stack.

LLVM represents this natively as `!llvm.ptr<1>`. The hardware safely stores the 64-bit address on the CPU stack, but the compiler statically guarantees that the CPU cannot emit a normal `load` instruction against a `ptr<1>`, preventing hardware faults! Therefore, storing remote pointers is trivial and mathematically safe, provided our type system correctly tracks the `MemorySpace` attached to every reference/pointer.

## Proposed Changes

### 1. Lexer (`src/lexer.rs`)

- [NEW] Add tokens: `Struct`, `Unsafe`, `Ampersand` (`&`), `Dot` (`.`).
- Ensure `Star` (`*`) is context-aware for unary dereferencing.

### 2. AST (`src/ast.rs`)

- [NEW] Add `StructDecl` struct. Expand `Program` to hold `Vec<StructDecl>`.
- [MODIFY] Expand `Type`:
  - `Borrow(Box<Type>, Option<MemorySpace>, bool /* is_mut */)`: Represents safe `&` and `&mut`.
  - `Pointer(Box<Type>, Option<MemorySpace>, bool /* is_mut */)`: Represents raw `*const` and `*mut`.
  - `Struct(String)`: Custom user types.
- [MODIFY] Expand `Expr`:
  - `Borrow(Box<Expr>, bool /* is_mut */)`
  - `Dereference(Box<Expr>)`
  - `UnsafeBlock(Vec<Statement>, Option<Box<Expr>>)`: Evaluates an unsafe block.
  - `StructInit(String, Vec<(String, Expr)>)`

### 3. Parser (`src/parser.rs`)

- [MODIFY] Add root-level struct parsing.
- [MODIFY] Add Pratt unary parsing for `&` and `*`.
- [MODIFY] Add `unsafe { ... }` block expression parsing.
- [MODIFY] Support chained `.` struct field access.

### 4. Semantic Analysis (`src/sema.rs`)

- [NEW] **Borrow Checker Pass**:
  - Enforce Rust-like rules: One mutable borrow OR multiple immutable borrows in a safe context.
  - Prevent mutable aliasing dynamically within safe logic.
- [MODIFY] Verify that `Dereference` (`*ptr`) and raw pointer arithmetic ONLY occur inside an `unsafe { ... }` block.

### 5. MLIR Codegen (`src/codegen.rs`)

- [MODIFY] Lower `StructDecl` to MLIR `!llvm.struct<(...)>`.
- [MODIFY] Lower `Borrow` and `Pointer` to `!llvm.ptr<address_space>` directly utilizing the AST's `MemorySpace`.
- [MODIFY] Lower `Dereference` to `llvm.load`.

## User Review Required

> [!IMPORTANT]
>
> 1. **Structs holding Tensors**: If a struct holds a `Tensor`, and we pass a pointer to that struct, does the struct pointer inherit the memory space, or do the tensors inside retain their own? I propose the pointer to the struct has its own memory space (where the struct itself lives), while fields retain their own nested mappings.
> 1. **Unsafe syntax**: I propose `unsafe { ... }` acts as an Expression (like Rust) yielding the last statement, rather than just a Statement block.

Please approve this plan or provide feedback so we can begin execution!
