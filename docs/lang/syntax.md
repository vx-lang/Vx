# Vx Language Syntax

This document outlines the core syntax of the **Vx** programming language. Vx uses a C-family, Rust-like syntax but introduces novel constructs for spatial execution and distributed state.

## 1. Basic Structure

Vx programs are structured into modules, functions, and scopes.

```rust
// A basic function
fn compute_metrics(data: Tensor<f32>) -> Tensor<f32> {
    // Variable declaration
    let intermediate = data.map(|x| x * 2.0);
    intermediate
}
```

> [!NOTE]
> **Implicit Returns**: Vx supports Rust-style implicit returns. The final expression in a block (such as a function body, `comptime` block, or `unsafe` block) can omit the trailing semicolon, causing the block to evaluate to the value of that expression. This avoids boilerplate `return` statements.

## 2. Spatial Execution Scopes: `spawn on`

To explicitly route computation to a specific physical topology, Vx uses the `spawn on` block. This overrides the compiler's default cost-model inferred routing.

```rust
fn distributed_matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    comptime {
        assert(a.shape[1] == b.shape[0], "Inner dimensions must match for matmul!");
    }
    // Spawn computation on a specific accelerator core
    spawn on(Topology::AccCore[0]) {
        let result = custom_matmul(a, b);
        // Compute happens entirely on AccCore[0]
        result
    }
}
```

### 2.1 NPU Slice: Spawn Across a Range of Devices

When a range expression is used inside the NPU index brackets, the parser constructs a `Topology::Slice` — representing a logical group of NPU devices. This enables spawning work across multiple NPUs without explicit looping.

```rust
fn parallel_forward_pass() -> i32 {
    // Spawn computation across NPUs 0 through 3
    spawn on(Topology::NPU[0..4]) {
        let chunk : Tensor<f32> = 1.0;
        // All four NPUs execute this block
    }
    return 0;
}
```

> [!NOTE]
> `Topology::NPU[0..4]` desugars to `Topology::Slice(NPU(0), 0, 4)` in the AST. The default memory for a Slice is `NPU_HBM`, same as a single NPU device.

### 2.2 Dynamic Topology Index

Topology indices can be runtime expressions, including loop variables. This enables patterns where different loop iterations target different hardware units:

```rust
fn scatter_to_npus() -> i32 {
    for i in 0..4 {
        spawn on(Topology::NPU[i]) {
            let local_data : Tensor<f32> = 1.0;
            // Each iteration spawns on a different NPU
        }
    }
    return 0;
}
```

> [!IMPORTANT]
> The topology index expression (e.g., `i` in `NPU[i]`) is evaluated in the **outer** scope, not the spawned scope. The variable `i` belongs to the CPU topology where the `for` loop executes; it is resolved before the context switches to the NPU.

## 3. Topology-Aware Function Signatures

Functions in Vx can explicitly declare the hardware topology they are designed to run on as part of their signature using the `on` keyword. This allows the compiler to enforce correctness at the language level and ensures that functions compiled for specific accelerators (like an NPU or GPU) are only called within valid execution scopes.

```rust
// This function is strictly compiled for and bound to NPU[0]
fn dummy_kernel(x: i32) on Topology::NPU[0] -> i32 {
    return x;
}

fn process() {
    // Valid: calling the function from a matching topology scope
    spawn on (Topology::NPU[0]) {
        dummy_kernel(0);
    }
}
```

### 3.1 User-Defined Topologies

The set of topologies is **open**: a program can declare its own with a `Topology` block,
registered at parse time. `memory:` (the topology's default memory space) is required;
`visible:` defaults to just that space when omitted; and each `transfer` clause adds an edge
— with an integer cost and a `relaxed`/`sync` consistency marker (default `sync`) — to the
compiler's transfer cost graph. The memory space may itself be a novel, custom name.

```rust
Topology AcmeTPU {
    memory: Memory::Local_SRAM,
    visible: [Memory::Local_SRAM, Memory::CPU_DRAM],
    transfer Memory::CPU_DRAM -> Memory::Local_SRAM : 40 sync,
}

// A topology backed by an entirely custom memory space.
Topology Island {
    memory: Memory::IslandRAM
}
```

Declared topologies are checked for *coherence* (e.g. the default space must be visible and
reachable from the host); see [`seam_obligations.md`](./seam_obligations.md) and the
[`hardware_monad.md`](./hardware_monad.md) design note.

### 3.2 Topology-Polymorphic Functions

A function can be generic over a topology with a `D: Topology` bound. Inside the signature
and body the variable is written `Topology::D`, and a `Pinned<T, Topology::D>` result carries
the binding. Monomorphization specializes the function per concrete topology at the call site.

```rust
fn run_on<D: Topology>(x: Pinned<i32, Topology::D>) -> i32 {
    spawn on(Topology::D) {
        let y = x;
    }
    return 0;
}
```

**`where Reachable<A, B>` constraints.** When a polymorphic function moves data between two
topology variables it must state that a transfer between them is possible. The
`where Reachable<S, D>` clause is discharged at each call site against the transfer cost graph
— instantiating `S`/`D` with a pair that has no path is a compile error. Multiple constraints
are comma-separated.

```rust
fn move_between<S: Topology, D: Topology>(
    src: Pinned<i32, Topology::S>,
    dst: Pinned<i32, Topology::D>,
) -> i32 where Reachable<S, D> {
    let staged = transfer(src, Memory::CPU_DRAM);
    return 0;
}

fn pipeline<A: Topology, B: Topology, C: Topology>( /* ... */ )
    -> i32 where Reachable<A, B>, Reachable<B, C> { /* ... */ }
```

**`Reachable<A, B>` as a comptime predicate.** The same relation is also a compile-time
boolean: `Reachable<A, B>` evaluates to whether a transfer path exists, where each argument is
a concrete topology (`Topology::CPU`) or a topology variable (`D`). Inside `if comptime`, a
statically-false predicate prunes its branch entirely, so an otherwise-invalid body never has
to type-check.

```rust
fn dispatch<D: Topology>(x: Pinned<i32, Topology::D>) -> i32 {
    if comptime Reachable<Topology::CPU, D> {
        let reachable = 1;   // only compiled when CPU -> D is reachable
    }
    return 0;
}
```

## 4. Logical and Relational Operators

- Compound assignment: `+=` (currently the only compound-assignment operator; `*=`, `-=`, `/=` are not yet parsed)
- Arithmetic: `+`, `-`, `*`, `/`, and `@` (matrix multiply)
- Relational Operators: `==`, `!=`, `<`, `>`, `<=`, `>=` (Returns a Boolean evaluation)
- Logical Operators: `&&`, `||`, `!` (Requires Boolean operands)
- Range: `..` (e.g. `0..4`, and inside a topology index such as `NPU[0..4]`)

## 5. Semantics of Data Movement: `transfer`

Data cannot be implicitly moved across address spaces. Moving data requires the `transfer` primitive, which explicitly tracks ownership and liveness across boundaries.

```rust
fn heterogeneous_pipeline(host_input: Tensor<f32, [1024], Memory::CPU_DRAM>) {
    spawn on(Topology::NPU[0]) {
        // Explicitly transfer data from Host DRAM to NPU HBM
        let local_data = transfer(host_input, Memory::NPU_HBM);

        // Execute computation on the local data
        let result = process(local_data);

        // Transfer result back to Host DRAM
        let host_result = transfer(result, Memory::CPU_DRAM);
    }
}
```

### 5.1 Multi-Hop Transfers via NIC

When transferring data across physically distant memory spaces (e.g., from CPU DRAM to a remote HBM across the network), Vx can utilize the NIC (Network Interface Controller). This creates a multi-hop transfer that the compiler's `TransferCostGraph` routes efficiently.

```rust
fn network_transfer_example(host_input: Tensor<f32, [1024], Memory::CPU_DRAM>) {
    // 1. Transfer from local CPU DRAM to the NIC RAM
    let nic_buffer = transfer(host_input, Memory::NIC_RAM);

    // 2. Transfer from NIC RAM over the network to Remote HBM
    let remote_data = transfer(nic_buffer, Memory::Remote_HBM);

    spawn on(Topology::NPU[0]) {
        // ... process remote_data ...
    }
}
```

### 5.2 Three questions that used to share the name `Transfer`

Data movement asks three different questions. Until Vx#353, `Transfer` answered two of them —
the reachability predicate and the implicit-movement trait — while the third went by the
lowercase keyword `transfer`, which read like the same thing and was not. They are now spelled
apart, and `Transfer` means only the third:

| spelling | question | keyed on |
| --- | --- | --- |
| `where Reachable<S, D>` | *may* these two topologies exchange data at all? | a pair of topologies |
| `impl Relocatable for T` | may a value of this type move implicitly across a boundary? | a user type |
| `impl Transfer<Memory::A, Memory::B> for Topology::X` | *how* does this machine move bytes across this edge? | (from space, to space, machine) |

**`Relocatable`** is an opt-in, not a capability. Implementing it says a value of that type is
allowed to be moved for you when it is touched from another topology; the compiler then inserts
the movement and warns (W1024) that it did, because a silent cross-topology copy is exactly the
cost this language exists to make visible. Write `.relocate()` explicitly to silence it.

```rust
trait Relocatable {
    fn relocate(self: MyModel) -> MyModel;
}

impl Relocatable for MyModel {
    fn relocate(self: MyModel) -> MyModel { /* ... */ }
}
```

**`impl Transfer<A, B> for Topology::X`** supplies the *code* for one machine's edge. The `for`
clause is required and is not decoration: an Ampere part fills shared memory with `cp.async`
and a Hopper part drives the same `Memory::L2 -> Memory::SMEM` edge differently, so a lowering
that named only the edge could not express the fleet directory it was written for.

```rust
impl Transfer<Memory::GPU_HBM, Memory::SMEM> for Topology::Dev {
    fn move_tile(src: &Tensor<f32, [2, 2]>, dst: &mut Tensor<f32, [2, 2]>) -> i32 {
        for i in 0..raw::extent(src) {
            raw::store(dst, i, raw::load(src, i));
        }
        raw::barrier();   // the trailing synchronization the contract demands
        return 0;
    }
}
```

Exactly one method, taking exactly two statically-shaped tile parameters in `(src, &mut dst)`
order, whose declared shape must equal the shape at the transfer site. Those restrictions are
current limits rather than the settled design; making a lowering generic over shape is future
work.

The body is ordinary Vx over a small primitive set (`raw::load`, `raw::store`, `raw::barrier`,
`raw::extent`, `raw::lane`, `raw::lanes`, `raw::async_copy`, `raw::async_wait`) that ordinary
code may not use. Every primitive takes a typed tile and an element index, never an address —
that is what keeps "which space did this touch" and "was it in bounds" checkable. What such a
body must guarantee, and what the compiler reads off it rather than trusting, is
[`custom_transfer_contract.md`](../custom_transfer_contract.md).

## 6. Control Flow

Standard Rust-like control flow is supported: `if`, `else`, `match`, `for`, `loop`, `break`, `continue`.
Loops can be annotated for spatial unrolling.

> [!WARNING]
> **Experimental / Unimplemented Feature**
> The `unroll across` syntax is currently planned but not yet implemented in the parser or lowering passes.

```rust
// Unroll this loop across 4 NPUs
unroll across(Topology::NPU[0..4]) { |npu_id|
    spawn on(npu_id) {
        let chunk = transfer(data[npu_id], Memory::NPU_HBM);
        process(chunk);
    }
}
```

## 7. Foreign Function Interface (FFI) & Safety

Vx supports calling external C functions via the `extern` block. By default, all external functions are considered `unsafe` because the compiler cannot statically verify their memory safety across the language boundary. Calling an `unsafe` function requires an `unsafe { ... }` block.

However, many C functions (like simple math functions, standard library I/O, or thoroughly tested user kernels) are inherently safe or have been manually verified by the programmer. Vx allows you to claim responsibility for this safety by annotating the FFI declaration with the `safe` keyword:

```rust
extern {
    // Unsafe by default. Requires `unsafe { ... }` at call sites.
    fn vx_malloc_f32(num_elements: i32) -> *mut f32;

    // Explicitly marked as safe. Can be called freely in pure Vx code!
    safe fn vx_decode_token(tokenizer_ptr: *mut i8, prev_token: i32, token: i32) -> *mut i8;
}
```

**Motivation**: The `safe` keyword delegates the safety assertion to the interface boundary. This prevents the codebase from being littered with repetitive `unsafe` blocks for functions that are already trusted, keeping your application logic clean and robust while maintaining strict boundaries for actual unsafe operations (like pointer arithmetic or arbitrary memory mapping).

## 8. Automatic Differentiation (Autodiff)

Vx provides first-class support for automatic differentiation via the `grad`, `vjp` (Vector-Jacobian Product), and `jvp` (Jacobian-Vector Product) keywords.

```rust
fn loss_function(weights: Tensor<f32, [128, 128]>) -> f32 {
    // ...
}

fn optimize() {
    // Computes the gradient of the loss function with respect to its inputs
    let gradients = grad(loss_function)(current_weights);
}
```

## 9. Formal Verification Contracts

Vx supports formal verification through contract programming. Functions and loops can be annotated with preconditions, postconditions, and invariants to mathematically prove program correctness at compile time.

```rust
// Preconditions (`requires`) must be true before the function executes
// Postconditions (`ensures`) are proven to be true after the function returns
fn safe_divide(x: i32, y: i32) -> i32
    requires y != 0
    ensures return <= x
{
    return x / y;
}

fn compute_sum(n: i32) -> i32
    requires n >= 0
{
    let mut sum = 0;
    let mut i = 0;
    // Loop invariants must hold before, during, and after loop execution
    loop
        invariant i <= n
        invariant sum >= 0
    {
        if i >= n { break; }
        sum += i;
        i += 1;
    }
    return sum;
}
```

## 10. Macros

Vx provides a robust macro system via the `macro_rules` keyword, enabling metaprogramming and code generation at compile time.

```rust
macro_rules! create_tensor {
    ($val:expr) => {
        Tensor::new($val)
    };
}
```

## 11. Tensor & Slice Initializers

A tensor can be constructed from an **initializer list** — a nested array literal — with its shape
inferred from the nesting. This replaces an explicit fill loop:

```rust
// A 2x4 tensor with these values (shape inferred from the [[..],[..]] nesting):
let q = Tensor<f32>([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]);

// Storage of the same shape and nothing in it -- the shape is written in the type:
let z = Tensor<f32, [2, 4]>::uninit();
```

The initializer list is the one construction still written the older way, because it supplies
*contents* rather than a shape and the type-applying constructors take neither. See
[types.md §6](./types.md#6-topology-aware-allocation) for `::new()` and `::uninit()`.

A **slice** (a row of a tensor, see the slice operators in
[`slice_operators.md`](../discussions/implementation_plans/slice_operators.md)) can be initialized
in place from a flat array literal:

```rust
let mut o = Tensor<f32>([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
o[0] = [1.0, 2.0, 3.0, 4.0];   // writes row 0
```
