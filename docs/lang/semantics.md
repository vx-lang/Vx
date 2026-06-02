# Vx Operational Semantics & Memory Model

This document specifies the runtime evaluation, spatial computing execution model, and memory rules of the Vx language.

## 1. Spatial Computing Model

Vx does not assume a single, flat instruction stream executing on a monolithic CPU. The language intrinsically models distributed, heterogeneous topologies (e.g., CPU, GPU, NPU, AccCore).

### 1.1 The `spawn on(Topology)` Operation

The `spawn on(Topology)` statement transitions the execution context from the current hardware unit to the specified hardware topology.

**Operational Rule:**
Let $E[ \\text{spawn on}(\\tau) { B } ]$ be a program state evaluated on an active hardware unit $\\rho$.

1. The runtime checks if the topology $\\tau$ is reachable from $\\rho$.
1. The runtime reserves resources on $\\tau$.
1. Execution of the block $B$ is enqueued on $\\tau$.
1. The spawning unit $\\rho$ continues asynchronously unless a data dependency explicitly synchronizes the contexts.
1. All variables captured within $B$ that are not in a shared or transferred memory space relative to $\\tau$ will result in a compiler error.

### 1.2 The `unroll across(Topology)` Operation

`unroll across` is a spatial loop parallelizer. It takes a physical or logical topology slice (e.g., `Topology::NPU[0..4]`) and spatially duplicates the inner block $B$.

**Operational Rule:**
`unroll across(Topology::Unit[A..B]) { |id| B }` evaluates to parallel `spawn on` instances:
`spawn on(Topology::Unit[A]) { let id = A; B } || ... || spawn on(Topology::Unit[B-1]) { let id = B-1; B }`.

______________________________________________________________________

## 2. Memory Model & Transfers

Standard languages employ a flat memory model (`*mut T`). Vx uses a partitioned address space memory model via `Ref<T, MemorySpace>`.

### 2.1 Spatial Isolation

If variable $v$ is of type `Ref<T, Memory::HostDRAM>` and execution context is `Topology::NPU[0]`, attempting to dereference $v$ triggers a **spatial fault** (caught statically at compile-time by the Semantic Analyzer).

### 2.2 The `transfer` Primitive

Data movement across distinct memory hierarchies is explicit.
`let local_data = transfer(host_data, Memory::NPUHBM);`

**Operational Rule:**

1. Given source reference $S$ residing in space $M_1$.
1. Evaluates to a target reference $T$ in space $M_2$.
1. The runtime schedules an asynchronous or synchronous DMA transfer from $M_1$ to $M_2$.
1. Linearity: If $S$ represents uniquely owned data, the `transfer` consumes $S$. Future use of $S$ in space $M_1$ is invalid.

______________________________________________________________________

## 3. Typestates and Hardware Fallibility

Because computations map to physical hardware, availability is non-deterministic.

### 3.1 `Verified<T>` vs `Pinned<T, Topology>`

- `Verified<T>`: A lazily evaluated computation graph. The compiler has verified memory and type safety, but the physical location is deferred to the runtime cost model.
- `Pinned<T, Topology>`: An eagerly bound computation locked to a specific physical unit.

### 3.2 Monadic Transitions (`HardwareState`)

The transition `Verified<T> -> try_pin(T) -> HardwareState<Pinned<T>, Verified<T>>` acts as a failible monad. The runtime attempts to acquire the lock on the Topology. If successful, it returns `Available(Pinned)`, else `Saturated(Verified)`.

______________________________________________________________________

## 4. Compile-Time Evaluation (`comptime`)

Vx supports deterministic ahead-of-time evaluation via `comptime` blocks.

- Execution within a `comptime` block happens during the Semantic Analysis compiler pass.
- Operations inside `comptime` are guaranteed to have zero runtime overhead.
- Used predominantly for layout calculation, array sizing, and topological assertion (`assert(...)`).

**Operational Rule:**
If $E[\\text{comptime} { B }]$ evaluates to $v$, the AST is strictly replaced by the literal or reduced expression $v$ before lowering to the intermediate representation (MLIR).

______________________________________________________________________

## 5. Automatic Differentiation

Vx provides semantic primitives for program transformations related to calculus.

- `grad(f)`: Yields a new function $f'$ which computes the gradients of $f$.
- `vjp(f)`: Vector-Jacobian Product. Propagates gradients backwards (Reverse-Mode AD).
- `jvp(f)`: Jacobian-Vector Product. Propagates tangents forwards (Forward-Mode AD).

**Operational Rule:**
When $g = \\text{grad}(f)$ is evaluated, the compiler generates an adjoint computation graph for $f$, tracing linear data consumption (as verified by the borrow checker). Linear variables are consumed exactly once in the forward pass and uniquely referenced in the reverse pass.

______________________________________________________________________

## 6. Closure and Lambda Semantics

Vx closures (lambdas) support capturing variables from their enclosing environment. The semantics of lambda capture are heavily influenced by the type being captured, enforcing memory safety and borrow-checking rules at compile time.

### 6.1 Implicit Borrow for Struct Captures

When a struct is captured within a lambda, Vx implicitly captures it by reference (`Type::Borrow`) rather than copying it by value.

**Operational Rule:**

1. Let $S$ be a variable of struct type residing in the outer scope.
1. When evaluating a lambda `|args| -> ret { ... S ... }` that captures $S$, the semantic analyzer automatically coerces the capture type to a borrow reference.
1. During code generation, a `BorrowExpr` is evaluated to retrieve the pointer to $S$, and this pointer is stored within the closure's internal environment structure.
1. Any member access (`S.x`) within the closure automatically dereferences the pointer, maintaining syntactic transparency.

### 6.2 By-Value Capture for Scalar Types

Primitive and scalar types (e.g., integers, floats) are captured by value (copied) into the closure environment, since they are inherently `Copy`.

### 6.3 Lifetime, Mutability, and Coercion Constraints

Because structs are implicitly borrowed:

- **Immutability:** Borrowed captures are immutable by default. Modifying them within the closure will trigger a mutability error.
- **Function Pointer Coercion:** A closure that actively captures environment variables has an internal state. Thus, it cannot be coerced into a raw function pointer (`fn(...) -> ...`). Raw function pointers strictly represent stateless code, and any coercion attempt will result in a static type mismatch error.
