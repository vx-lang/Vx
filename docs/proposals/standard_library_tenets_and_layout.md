# [RFC] The Vx standard library: tenets, layout and stability

- **Status:** draft for the language committee
- **Author:** Minzy Fox
- **Date:** 15 November 2026
- **Replaces:** `docs/std_library_design.md` (May 2026)
- **Builds on:** `docs/implementation_plans/core_library.md` and issue #451

## Summary

Vx has a good core library. It has numbers, iterators, `Option`, `Result`, hashing, atomics and
time. Applications now need the next level up: slices, byte strings, large vectors, a hash map,
files that report errors, and input and output traits. VxVector, the Parquet reader Vinyl and a
planned dataframe library all ran into the same gaps.

No document says what the standard library is for, what it leaves out, or how an API becomes
stable. This RFC proposes the following:

1. The **tenets** of the library: its scope, its API style, its performance rules, and the line
   between the language and the library.
1. The **layout**: two namespaces (`core` and `std`) that stay as they are today, three layers
   of dependencies inside them, and the modules in each.
1. The **device parts** of the library. Vx runs code on CPUs, GPUs and other devices. So the
   library has traits and types that Rust and Zig do not need.
1. The **rule for where code runs**: how a function, a type or a trait impl says that it runs
   only on the host or also on a device.
1. The **life cycle of an API**: internal, incubating, stable and removed, with a deprecation
   notice.

This RFC is about structure. It does not design each module. Each module gets its own tracking
issue later. Section 3 gives one or two lines per module to show its direction.

The last section lists open questions for the committee.

## Terms

- **Host:** the CPU that runs the program's `main` and talks to the operating system.
- **Device:** any other place that runs code, such as a GPU or an NPU. A device has no operating
  system.
- **Topology:** the Vx name for a place that runs code, such as `Topology::GPU`. A function says
  where it runs with an `on` clause: `fn f() on Topology::GPU`.
- **Runs where the caller runs:** a function written `on Topology::Any`, or generic over
  `D : Topology` with `on Topology::D`. The host and a device can both call it. The compiler makes
  one copy for each place.
- **Memory space:** a kind of memory, such as `Memory::CPU_DRAM`, `Memory::GPU_HBM` or
  `Memory::SMEM` (GPU shared memory). A tensor type names its memory space.
- **`HostOnly`:** a marker trait in `core::marker`. A type that implements it, or that holds one,
  cannot be used in device code (error E6038).
- **Layer:** a group of modules with the same dependencies: the core layer, the allocation layer
  and the OS layer. A layer is a rule that a test checks. It is not a namespace. Section 2
  defines the layers.

## Motivation

1. **Applications need a floor that does not exist yet.** The core library plan (#451) follows
   the module order of Rust's `core`. The modules that applications need first are in its later
   phases: slices (#534), `str` (#532), a `Vec` with large lengths (#3271), `String` as a Vx
   struct (#846) and a hash map written in Vx (#2152).
1. **The old design is out of date.** `docs/std_library_design.md` proposed to wrap Rust's
   standard library for collections, strings and even cryptography. Issue #2144 reversed that:
   plain code moves from the Rust runtime into Vx. No document records the new direction.
1. **Domain code entered `std`.** `std::nn` (neural network kernels), `std::llama` (a model
   loader), `std::bulk_hash` (a GPU hash table) and `stdlib/graph` are in the standard library.
   Nothing says whether they belong there.
1. **Vx runs on more than one kind of hardware.** Rust and Zig assume one kind of processor. Vx
   needs rules for library code that runs on a GPU, and for types that must never reach one.
1. **There is no stability promise.** Users cannot tell an incubating API from a settled one.
   Vx has no way to mark an API as deprecated.

## 1. Tenets

The tenets are the tests for every later decision. When two tenets disagree, the one that comes
first in this section wins.

### 1.1 Scope: what the standard library is for

- **S1. A shared vocabulary.** A type belongs in the library when independent libraries must
  agree on it to work together. Examples are `Slice`, `str`, `String`, `Vec`, `Result`, the
  error type, `Read` and `Write`, `Duration` and `Hash`. If two Parquet readers used different
  slice types, nothing could use both.
- **S2. Portable access to the machine.** The library gives one API for files, sockets, clocks,
  threads and the common device operations. A program then runs on Linux, macOS and each device
  without changes.
- **S3. Things that need the compiler.** Atomics, panics, intrinsics and tensor views need
  knowledge that only the compiler has. The library wraps them in normal APIs.
- **S4. Not a framework, not a domain library.** Neural network kernels, model loaders, file
  formats, graph algorithms and cryptographic algorithms do not belong in the standard library.
  They belong in libraries that use it.

**The inclusion test.** A new module must meet S1, S2 or S3. The core team must also be able to
maintain it. "Many programs could use it" alone is not enough.

### 1.2 What the language does, and what the library does

| The language does | The library does |
|---|---|
| Types, ownership, moves, borrows and `Drop` | Data structures and algorithms |
| Placement: topologies, memory spaces and the checks on them | Operating system access: files, sockets, clocks, processes |
| `spawn`, `tasks`, `across` loops and `transfer` | Synchronization types: `Mutex`, channels, semaphores |
| The tensor type and its element-wise operators | Tensor algorithms that a function can express |
| `panic`, `abort`, `assert` and the operator syntax | Formatting, parsing, hashing and text |
| `mlir!` blocks, the escape hatch to MLIR | Device vocabulary that a function can express (Section 4) |
| The capacity checker and the contracts (`ensures`, `valid`) | Contracts on its own unsafe functions |

The rules for this line:

- **L1. The compiler does not know library names.** A library item is an ordinary Vx item. Today
  the compiler recognizes some names: the tensor reductions (`dot`, `sum`, `max`, `min`), the
  `lanes::` routines (`src/hir/check/lanes.rs`) and the print macros. Each one either becomes a
  library function over a general feature, or the language reference documents it as a built-in.
  Issue #3468 is the first step for `lanes::`. The race proof also knows library names. `slot_of`
  in `src/hir/check/across_proof.rs` matches `atomic_fetch_add_i32`, `_u32`, `_i64` and `_u64` by
  name for its counter rule, and #3450 extends that rule. Under L1, the counter rule becomes a
  contract clause on the function, not a name match.
- **L2. A missing feature goes into the language, not into one type.** When the library cannot
  express something, the fix is a general language feature that any library can use. Issue #536
  shows how: each language item says which library module needs it.
- **L3. `core` never calls the operating system or an allocator.** The purity test in
  `shipped_programs_compile.rs` already enforces this rule. It is what lets device code call
  `x.exp()`.
- **L4. The library does not repeat a check that the language makes.** For example, the type
  system checks placement. The library does not check it again at run time.

What the standard library will not contain:

- Domain libraries: neural networks, model formats, graph algorithms, dataframes.
- File formats other than plain text and bytes. Open question Q12 covers JSON.
- Cryptographic algorithms (Section 7).
- An `async` runtime. Vx uses tasks and channels.
- A package manager or a build system.

### 1.3 API style

- **A1. Use Rust's names when the idea is the same.** Use `new`, `with_capacity`, `len`,
  `is_empty`, `get`, `push`, `iter`, `as_*`, `to_*`, `into_*`, `from_*` and `try_*`. The Vx syntax
  is close to Rust. So the same names lower the cost for readers. A module header lists each
  place where it differs from Rust, and why.
- **A2. One idea, one name.** A concept has the same name in every module.
- **A3. An expected failure returns a `Result`.** A missing file, bad input or a full buffer is an
  expected failure. A public API never reports a failure with a special value such as 0, -1 or a
  null pointer.
- **A4. A programmer error panics.** An index out of range and a broken invariant are programmer
  errors. The doc comment has a "Panics" line that says when.
- **A5. A checked form first, an unchecked form second.** `get` checks the index. `get_unchecked`
  is `unsafe` and states its precondition as a contract where the language can express it.
- **A6. Host containers use 64-bit sizes and indices.** A new API in the allocation layer or the
  OS layer does not use `i32` for a length, a capacity or an index. Device code is different. On
  a GPU, 64-bit index arithmetic costs registers and instructions. The indices of `across` loops
  are `i32` today, and so are the indices of the atomic tensor functions in `core::sync::atomic`.
  Open question Q11 covers `usize` against `i64`, and the index type in device code.
- **A7. Borrow when the function only reads.** A function that reads takes a slice or a `&str`,
  not an owned `Vec` or `String`.
- **A8. No hidden allocation.** `core` never allocates. A `std` function that allocates says so
  in its doc comment. When a caller buffer can avoid an allocation, the API takes one, as
  `core::fmt` does with a writer.
- **A9. No hidden data movement.** No library function moves data to another memory space unless
  its name or its signature shows it.
- **A10. Generic code, not dynamic dispatch.** Library APIs are generic over traits. Open question
  Q4 covers dynamic dispatch.
- **A11. Every public item MUST have a doc comment, and every public function and method MUST
  have a working example.** Section 1.8 gives the rules and the few exceptions.
- **A12. Plain English.** Doc comments and error messages follow the project's writing rules
  (`agents/AGENTS.md`).

### 1.4 Performance

- **P1. A check costs one compare and one branch.** The failure path of a check is out of line.
  It must not stop inlining or vectorization (#3268, #3269).
- **P2. No foreign call on a hot path.** Foreign calls are for the operating system and for large
  vetted libraries outside the standard library.
- **P3. Each container and algorithm has a benchmark.** The benchmark compares it with the Rust or
  C++ equivalent. The PR that adds the item records the result. A slower result needs a reason.
- **P4. A decline is a bug.** A standard library function that the flat code generator declines
  is a bug. The `KNOWN_DECLINES` list only gets shorter.
- **P5. No hidden work.** The library starts no background thread and keeps no global state that
  the caller did not ask for.
- **P6. Memory is visible.** Containers give `reserve`, `clear` and `shrink_to_fit`, so that a
  program can reuse buffers. Where possible, the capacity checker can see an allocation's size.

### 1.5 Heterogeneous hardware

- **H1. Every public function says where it runs.** It is `on Topology::Any`, generic over a
  topology, bound to a named topology, or host only. The parallel compiler rules require this:
  "a signature is the whole contract" (`docs/language_rules_for_a_parallel_compiler.md`).
  Section 5 gives the rules.
- **H2. One name, one meaning, on every device.** `f32::exp` means the same thing on a CPU and on
  a GPU. The compiler picks the implementation for each target. The caller does not.
- **H3. A type that cannot exist on a device is `HostOnly`.** Files, sockets, a mutex that waits
  for the host, and anything that holds a host pointer are `HostOnly`. The compiler refuses them in
  device code.
- **H4. No vendor names in the public API.** The public API has no `cuda` or `metal` module.
  Vendor code lives in target implementations inside the library (Section 4.3).
- **H5. Device vocabulary may enter the library. Device algorithms may not.** Lane indices,
  barriers, scoped atomics and target queries are vocabulary that every device library needs.
  A matrix multiply or an attention kernel is a domain algorithm.

### 1.6 Vx code and foreign code

- **N1.** The library writes plain computation in Vx: collections, text, parsing, hashing and
  math.
- **N2.** Foreign code is for operating system calls only. A separate library, not `std`, wraps a
  large vetted library such as a compression or TLS library.
- **N3.** Each foreign call sits behind a Vx interface. A Vx implementation can then replace it
  without a change for callers.

This replaces the "ecosystem delegation" plan of `docs/std_library_design.md`.

### 1.7 Contracts and proofs

**What Vx does today.**

- The prover (Z3) proves `requires`, `ensures` and loop `invariant` clauses at compile time.
  When it cannot prove a clause, compilation fails (E8001, E8006). It never turns a clause into a
  run-time check.
- So a contract costs nothing at run time, at every `-O` level. No optimization level and no
  build mode adds or removes anything for a contract.
- The compiler needs Z3. Without it, each contract that must be proved is a compile error. Contracts
  have no opt-out.
- A struct `invariant` stops other modules from changing the fields (E8014). The prover does not
  check the invariant yet, and it does not use it as a fact (#2832).
- Contracts give the optimizer little today. Only `valid(p, n)` reaches LLVM, as an
  `llvm.intr.assume`. A number fact such as `requires i < self.len()` does not reach LLVM. So
  it does not remove a run-time check that the library wrote, such as a bounds check or the
  overflow trap of `usize`. Issue #1142 starts this work for `usize`.
- The run-time checks in a program come from code that someone wrote: an `assert`, a bounds
  check, an overflow trap. `-O3` removes one only when LLVM proves it redundant by itself. Vinyl
  measured a loop where LLVM does this, and a loop where it does not (#3269).
- Few library functions have contracts. `std::vec` has 10 clauses, `core::convert` has 3 and
  `std::bulk_hash` has 1. The other modules have none.

**The rules.**

- **C1. A public function states its contract.** A result that callers need is an `ensures`, for
  example a length, a range, `valid` or `!frees`. A precondition is a run-time check by default,
  with a "Panics" line in the doc comment. Only the proven form of C3 has a `requires`. Vx fails
  the compile when it cannot prove a `requires`, and it has no run-time fallback. So a `requires`
  on an ordinary function, such as `split_at`, puts a proof on every call.
- **C2. A type with a rule about its fields declares it as an `invariant`.** `Vec` does this
  today: `self.len <= self.capacity`.
- **C3. An access has up to three forms.**
  - A checked form, such as `get`. It checks at run time and panics on a bad index. This is the
    default form. Its call asks the prover nothing.
  - A proven form. It has a `requires`, such as `i < self.len()`, and no run-time check. It needs
    no `unsafe`, because the caller's proof replaces the check. Rust, Zig and Mojo cannot offer
    this form without `unsafe`. It has its own name, so that a call shows that it needs a proof.
    Q20 asks for that name.
  - An `unsafe` form, such as `get_unchecked`, for a caller whose proof the prover cannot follow.
- **C4. A contract is part of the API.** A stronger `requires` or a weaker `ensures` can break a
  caller. Such a change goes through deprecation (Section 6).
- **C5. Library proofs stay inside the prover's work limit.** The prover counts steps, not
  seconds, so a proof that passes on one machine passes on all of them. CI proves the library on
  each change. A proof that needs more steps gets intermediate facts, not a larger limit.
- **C6. A proven fact removes the check that it makes redundant.** This is the goal, not the
  state today. Issue #1142 does it for `usize`. Bounds checks come next. Until then, rule P1
  keeps each check cheap.

### 1.8 Documentation

- **D1. Every public item MUST have a doc comment.** This covers modules, types, traits,
  functions, methods and constants. "The name says it all" is not an exception.
- **D2. Every public function and method MUST have at least one example.** By default, the
  example MUST compile and run. It MUST check its own result with `assert`, so that a wrong result
  fails the test. An example that only compiles is not enough by default. These exceptions need a
  stated reason that a reviewer accepts:
  - One example covers a small group of related items, such as the getters of one type.
  - An item has no example of its own, for example a trait method that the trait's example uses.
  - An example compiles but does not run, for example because it needs hardware that CI lacks.
- **D3. CI runs every example on both code generators.** A function that runs on a device also
  has an example inside a `spawn on` region. CI runs it on each device target that it has.
- **D4. A doc comment has these sections when they apply:**
  - "Panics": when the function panics.
  - "Errors": each error that it returns.
  - "Safety": what the caller of an `unsafe` function must ensure.
  - "Contract": its `requires` and `ensures`, in words.
  - "Runs on": its device rule (Section 5).
  - "Allocates": whether it allocates, and how much.
  - "Complexity": the cost of each container operation.
  - "Stage": incubating, or deprecated with the replacement.
- **D5. A module header states the module's purpose, layer, device rule and stage, and where it
  differs from Rust.**
- **D6. Doc comments are in plain English**, as `agents/AGENTS.md` requires. The STE lint
  checks them.

**What the rules need.** Today no tool enforces them, and no doc comment in the standard
library has an example. `standard_library_docs_lifecycle_and_testing.md` in this directory
proposes the tools and the tests that enforce these rules.

Until Vx has `pub` (#489), "public" means every item whose name does not start with `_`. The doc
rules apply to new and changed items at once. The existing modules meet them before they become
stable.

## 2. Layout

The library keeps its two namespaces, `core::` and `std::`. Inside them, each module belongs to
one of three layers. A layer says what a module may depend on.

```
OS layer          std::   needs an operating system. Host only.
                          io  fs  path  net  env  process  time  task  sync  rand  ffi
  ^
allocation layer  std::   needs an allocator, never the operating system.
                          alloc  box  vec  string  hash_map  hash_set  sync::arc
  ^
core layer        core::  needs nothing. Each item states where it runs (R1).
                          marker cmp ops clone default convert option result num iter
                          slice str char ascii hash fmt mem ptr cell sync::atomic time
                          tuple panic hint device target
```

- **Public paths stay.** `std::vec` stays `std::vec`. A program that imports a module today
  imports it from the same path after this RFC. Rust puts its allocation layer in a separate
  `alloc::` namespace. Vx does not copy that, for two reasons:
  - Vx has no re-export feature. Rust can move `Vec` to `alloc::vec` and keep `std::vec` as an
    alias. Vx cannot. So a move would break every `import std::vec;`.
  - Rust needs `alloc::` for programs that cannot link `std` at all, such as firmware. In Vx, the
    program that links `std` is the host program. A device never links it. The device question is
    which functions may run on the device. Section 5 answers that question for each function, and
    a path cannot answer it.
- **One direction.** Each layer imports only its own layer and the layers below it. A test
  enforces this, as the purity test does for `core` today. So an allocation-layer module never
  imports `std::fs`, even though both are under `std::`.
- **A layer does not decide where code runs.** The signature decides it (R1). The module path
  does not change it. The layer test checks that each public function in the core layer is
  `on Topology::Any` or generic over a topology. A core function with no `on` clause runs on the
  CPU only, and its "Runs on" line (D4) says why. Each public item in the OS layer is host only.
  Where the allocation layer runs depends on open question Q1.
- **Domain code moves out.** `std::nn`, `std::llama`, `std::bulk_hash` and `stdlib/graph` leave
  the standard library. Open question Q6 asks where they go.

The core library plan (`docs/implementation_plans/core_library.md`, phase 4) planned to split
`std` into `alloc` and `std`. This RFC proposes the dependency rule in place of that split. If a
public `alloc::` path is ever needed, a re-export feature must come first (Q7). Then the old
paths can stay as aliases.

## 3. Components

Each row gives the direction for one module. The tracking issues will hold the details.
"Exists" means the module is there today. It does not mean the module is complete.

### 3.1 `core`

| Module | Direction |
|---|---|
| `marker` | Exists. `Copy`, `Send`, `Sync`, `Sized`, `HostOnly`, `PhantomData`. Add the trait for values that can cross to a device (Section 4.1). |
| `cmp`, `ops`, `clone`, `default`, `convert` | Exist. Operators call the `ops` traits (#528). Add `TryFrom` (#570), and `AsRef` after slices. |
| `option`, `result` | Exist. Add the methods that return references, and `inspect`. |
| `num` | Exists. Add parsing from text, byte-order conversion (`to_le_bytes`, `from_le_bytes`) and generic number traits for numeric code. |
| `iter` | Exists. Close the gaps in the core library plan's Appendix A. |
| `slice` | New. `Slice<T>` and `SliceMut<T>`: `len`, `get`, `split_at`, `chunks`, `windows`, `copy_from_slice`, `sort_unstable`, `binary_search`. The `&[T]` syntax comes later (#534). |
| `str`, `char` | New. `str` is UTF-8 bytes plus a length: search, split, trim and parse. `char` starts with ASCII and Latin-1 (#532). |
| `ascii` | Exists. Add the byte iterators. |
| `hash` | Exists. Add `BuildHasher`, hashing of byte slices, and generated `Hash` impls (Q4 covers derive). |
| `fmt` | Exists. Add one `Writable` trait for text output (Q2) and a format macro. |
| `mem`, `ptr`, `cell`, `hint` | Exist. Add `align_of`, `MaybeUninit`, `copy_nonoverlapping` (#2501), `NonNull` (#2502) and `prefetch` (#3373). |
| `sync::atomic` | Exists. Atomics on scalars and on tensor elements, with scoped fences. |
| `time` | Exists. `Duration`. Add its operators once #528 lands. |
| `panic` | Exists. Add `assert_eq` and `assert_ne`. Define what a panic does on a device (Q15). |
| `device` | New. The device vocabulary of Section 4.2. |
| `target` | New. Compile-time facts about the target: pointer width, byte order, SIMD width and subgroup width. |

### 3.2 The allocation layer (`std::`)

| Module | Direction |
|---|---|
| `std::alloc` | Exists. Add the `Allocator` trait. It may be per memory space (Q1). Allocation failure is Q14. |
| `std::box` | Exists. `Box<T>`. |
| `std::vec` | Exists. 64-bit lengths, `reserve`, `set_len`, `resize`, `extend_from_slice` and `clear` (#3271). `as_slice` returns a `Slice`. |
| `std::string` | Exists as a wrapper over Rust. It becomes a Vx struct over bytes (#846): `push_str`, `from_utf8` and `as_str`. |
| `std::hash_map`, `std::hash_set` | Exist as wrappers over Rust for three type pairs. They become generic Vx code with SwissTable control bytes (#2152). |
| New containers | `VecDeque`, `BinaryHeap` and `BitSet`, then an ordered map. Each new container gets a path in the same style as the existing ones (Q7). |
| `std::sync::arc` | Exists. It needs atomics and an allocator, and never the operating system. |

Today some of these modules call the Rust runtime. A module joins the allocation layer when it
imports nothing from the OS layer. The path does not change when it joins.

### 3.3 The OS layer (`std::`)

| Module | Direction |
|---|---|
| `io` | The `Read`, `Write`, `Seek` and `BufRead` traits, plus `BufReader`, `BufWriter`, stdin, stdout and stderr. One error type with a kind. |
| `fs` | `File::open` returns a `Result` (#1341). Add metadata, directories, and an owning memory map that unmaps on drop (today `std::mmap`). |
| `path` | `Path` and `PathBuf` over bytes. |
| `net` | Exists. TCP listener, TCP stream, UDP socket and addresses. Blocking first. |
| `env`, `process` | Program arguments, environment variables, exit, and starting a child process and waiting for it. |
| `time` | Exists. `Instant` for a monotonic clock and `SystemTime` for wall time. `now()` returns an `Instant`, not seconds as `f32`. Add a cycle counter (#3371). |
| `task` | The language has `tasks` and `spawn`. The library adds the number of CPUs, pinning a task to CPUs (#3370) and task-local data. |
| `sync` | Exists. `Mutex`, channels and waits. Add `Semaphore`, `Barrier` and `Once`. |
| `device` | New. A run-time query of the devices that are present (Section 4.2). |
| `rand` | Exists. Add random bytes from the operating system. The existing seeded generator stays. |
| `ffi` | `CStr` and `CString`. `std::libc` becomes internal. |
| `testing` | Today `std::googletest`. Assertions for tests. Q17 asks if a test framework belongs in `std`. |

## 4. The device parts of the library

Rust and Zig target one kind of processor. Their libraries need no device vocabulary. Mojo does
need one, and its choice is a useful model:

- Mojo's standard library makes ordinary code work on every device. It uses compile-time target
  queries and a private layer of target implementations (`std/_plugin/{cuda,hip,metal}`).
- Mojo's GPU programming API (`gpu`, `layout`) and its kernels (`linalg`, `nn`) live outside the
  standard library, in MAX.
- Mojo has one device trait, `DevicePassable`. It names the type that a value becomes on the
  device.

Vx has an advantage over Mojo. The type system checks placement. So the library can state device
rules that the compiler enforces, not only conventions.

### 4.1 Device traits

| Trait | Purpose | Status |
|---|---|---|
| `HostOnly` | The type cannot be used on a device. The compiler refuses it and every type that holds it (E6038). | Exists |
| A device-crossing trait (name open, Q9) | The value can be copied to another memory space, or passed to a kernel. It is plain data with no host pointers. The compiler derives it from the fields, as it does for `Send`. Mojo's `DevicePassable` is the closest equivalent. | New |
| `Send`, `Sync` | A value can move to another task, or be shared between tasks. Q9 asks how they relate to the device-crossing trait. | Exist |
| `Allocator` per memory space | Allocation in a named memory space, if Q1 says yes. | New |
| A combine trait for lane routines | An associative combine, so that `lanes::sum` and its siblings work for any element type, not only `f32`. | New |

### 4.2 Device types and modules

- **`core::device`:** lane, block and subgroup indices, `barrier`, and the lane routines (`sum`,
  `max`, `argmax`, `sort`, `copy_async`). The lane routines move here from the compiler (L1).
- **`core::target`:** compile-time facts for each topology, such as subgroup width, SIMD width
  and whether `f64` is available.
- **`core::sync::atomic`:** already holds the scoped fences and the atomics on tensor elements.
- **Slices in any memory:** a `Slice` that carries its memory space, so that one byte reader or
  one column kernel works on host and device memory (Q1).
- **`std::device`:** which devices the host has at run time, and their memory sizes. VxVector
  needed this to choose between the GPU and the CPU before it trained.

These items stay out of the library: kernels (matrix multiply, attention, quantized products),
model loaders and the GPU hash table. They are domain libraries (H5).

### 4.3 Target implementations

Some functions need a different body on each target. Examples are `print`, the clock, math
intrinsics, atomic scopes and stack allocation. The library needs one defined place for "the CUDA
form of `f32::exp`". The public function stays the same (H2). A target module inside the library
supplies the body, and the library selects it by the target's `arch:`.

Fleet files do not hold target bodies. One `arch` serves many fleet files: `h100`, `h200` and
others are all `nvptx64`. So a body in a fleet file would be copied into each one. Fleet files
hold hardware facts, with citations to the hardware specification.

Each target body has the same contract as the public function, and a test runs the same cases on
each target that CI has.

## 5. Where library code runs

The rules below apply to every public item in the library.

- **R1. A function states where it runs, and a method too.** Every public function and method
  has an `on` clause: `on Topology::Any`, a generic `D : Topology`, or a named topology. A
  function or a method without one runs on the CPU only. The compiler checks every call against
  that clause.

  The compiler does not do this for methods today. Probes on the debug build of 14 November
  showed the following:

  - A function with no `on` clause gets `Topology::CPU` (`src/parser/decl.rs`).
  - A plain call from a GPU region to such a function is error E6001.
  - The compiler does not check a method call, `x.m()` or `Type::m()`. The check (`same_device` in
    `src/hir/check/calls.rs`) is on the plain-call paths only. A comment there says that
    `Type::method(..)` "checks no placement".
  - So device code can call `x.exp()` and the other `core::num` methods, which have no `on`
    clause. It can also call a method whose body calls a CPU-only function. In one probe, the
    Apple GPU code generator compiled that CPU-only function into the kernel. This is a soundness
    bug, and #3638 tracks its fix apart from this RFC.
  - Other checks still protect device code. Inside an `across` loop, the compiler checks
    the `writes_only()` contract through every call, and an `extern` function cannot claim it
    (E8011). The
    compiler refuses `HostOnly` types in device code (E6038). Other code reaches the GPU code
    generator, which can fail late with a code-generation error, not a type error.

  This is why `bulk_hash_fmix32`, a free function, says `on Topology::Any`, but the methods of
  `core::num` say nothing. Under R1, a function and a method with the same signature follow the
  same rule. That also follows the parallel-compiler rule that a signature is the whole contract.

  **How to reach R1.** First, give each method in `core` that kernels call an `on Topology::Any`
  clause. Then turn on the placement check for every method call (#3638). The other order breaks
  every kernel that calls a `core::num` method today.

  This RFC does not change the default. A default of "runs where the caller runs" changes the
  meaning of every program. Every host function that calls the operating system would then fail
  to compile until it gets a host-only mark.

- **R2. A type states if it is `HostOnly`.** A library type that holds an operating system
  handle, a host pointer or a wait for the host implements `HostOnly`:

  ```vx
  impl HostOnly for File {}
  ```

  The compiler then refuses the type, and every type that holds it, in device code.

- **R3. An impl is never wider than its type.** An impl of a trait for a `HostOnly` type is host
  only. An impl for a type that can be used on a device runs where its trait's methods run.

- **R4. A trait can require its methods to run anywhere.** `Hash`, `Hasher`, `PartialEq` and `Ord`
  must work in device code for device types. The trait states this, and an impl cannot be
  narrower. Q8 covers the syntax.

- **R5. Generic code follows its type arguments.** A function generic over `T : Hash` that runs
  `on Topology::D` works in device code only when `T` is not `HostOnly`.

- **R6. Tests prove the rule.** Each function that runs anywhere has a test that calls it from a
  `spawn on` region. CI runs that test on each device target that it has.

## 6. The life cycle of an API

Vx is at version 0.0.4. Before version 1.0, an API can change. But users still need to know which
APIs will stay and how a change reaches them.

| Stage | Meaning | How a user sees it |
|---|---|---|
| Internal | Not for users. It can change at any time. | The name starts with `_` until item visibility exists (#489). |
| Incubating | Open for trial. It can change or disappear in any release. It has a tracking issue. | The compiler refuses a use unless the program turns on the feature by name. The error says how to turn it on. |
| Stable | The API shape stays. A change goes through deprecation. | No marker. |
| Removed | Gone. | The release notes list it with its replacement. |

**Deprecated is a notice, not a stage.** An incubating or a stable item can be deprecated. It
keeps its stage, its feature name and its history. It still works. A warning at each use names
the replacement, or explains why there is none, and the release that removes the item.

The spelling of the opt-in is open (Q10). A command-line flag such as
`--enable-incubating=slice` is one possible form. Whatever the form, a feature that a library
turns on for itself must not turn on for the programs that use that library.

**To become stable**, an API must have all of the following:

- A doc comment on every public item, and a working example for every public function and
  method, or an accepted exception (Section 1.8).
- Its contracts, stated and proved in CI (Section 1.7).
- Tests on both code generators, and no decline by the flat code generator.
- A benchmark, if the API is on a hot path (P3).
- A stated device rule (Section 5).
- At least one real program that uses it (the "first user" rule).
- A review by the module's owner. A new module, or a new trait in `core`, also needs a committee
  review.

**Deprecation:**

- Before 1.0, a deprecated item stays for at least one release. After 1.0, it stays for at least
  two releases.
- The deprecation note names the replacement. Where possible, a tool rewrites the call.
- A removal is in the release notes, with a short migration note.
- `vx-api diff OLD.vxlib NEW.vxlib` already compares the interfaces of two library versions and
  prints the smallest safe version bump. CI can run it on each library change, so that a
  breaking change cannot go in unannounced.
- A move or a rename of a public path is a breaking change, and it goes through deprecation too.
  The old path must keep working during the deprecation. Vx needs a re-export feature for this
  (Q7), so no stable path moves before that feature exists.

**What the language must add.** Vx has no item attributes and no visibility today. The stages
above need both: something like `#[deprecated(since = "0.0.5", note = "use Slice::get")]`,
`#[incubating(feature = "slice", issue = 534)]`, and `pub`. Until they exist, a doc comment
tag such as `/// @incubating slice #534`, checked by a lint, is a possible interim step (Q10).
`standard_library_docs_lifecycle_and_testing.md` proposes the fields that this metadata holds and
how the compiler enforces it.

**Prior art.** Rust marks each library item stable or unstable, and unstable items work only on
nightly behind a feature flag. Zig makes no promise before 1.0 and lists breaking changes in its
release notes. Mojo recently added stability markers for its library APIs. Go promises that a
program written for Go 1 keeps working.

## 7. Cryptography

**Proposal: cryptographic algorithms stay out of the standard library.** Three small items stay
in, because other modules need them.

Languages differ on this point. Go, Zig, Java and Python put cryptography in the standard library.
Rust, C++ and Mojo do not. Mojo's `hash` package has only AHash and FNV-1a, which are not
cryptographic hashes.

Reasons to keep it out of Vx:

1. **Constant-time code needs language support.** Code that handles secrets must not branch on
   them or use them as memory indices. A young optimizer, and LLVM after it, can add such a
   branch. Vx has no way to mark a value as secret, and no optimization barrier with a contract.
1. **Release speed.** A security fix must not wait for a compiler release.
1. **Review cost.** The library must stay small enough for the core team to review.

What stays in:

- **Random bytes from the operating system** (`getrandom` or `/dev/urandom`). `HashMap` needs a
  random seed to resist collision attacks. The existing `std::rand` generator is not
  cryptographic, and its docs already say so.
- **Non-cryptographic hashes.** They exist today. Their docs must say that they are not
  cryptographic.
- **A constant-time comparison of two byte slices**, perhaps. It is small and generally useful.

When cryptography comes, it is a separate library. It first wraps a vetted library through
foreign calls. Algorithms written in Vx come only after the language can promise constant-time
code. A GPU batch hash, such as SHA-256 over many blocks, could be a distinctive Vx library later.

## 8. Where today's modules go

| Today | Proposed place | Note |
|---|---|---|
| `core::*` | `core` | No change. |
| `std::vec`, `std::string`, `std::box`, `std::hash_map`, `std::hash_set`, `std::alloc` | Same path, allocation layer | Each joins the layer when it stops calling the OS layer and the Rust runtime. |
| `std::sync::arc` | Same path, allocation layer | Needs atomics and an allocator. |
| `std::io`, `std::fs`, `std::net`, `std::time`, `std::rand` | Same path, OS layer | Each gets a `Result`-based API. |
| `std::mmap` | `std::fs` | Becomes an owning map type. |
| `std::libc` | internal to `std` | Not a public API. |
| `std::sync::{mutex, channel, wait}` | `std::sync` | No change. |
| `std::googletest` | `std::testing` or tools | Q17. |
| `std::tensor`, `std::simd`, `std::collective` | undecided | Q5. |
| `std::nn`, `std::llama`, `std::bulk_hash`, `stdlib/graph` | outside the standard library | Q6. Committee decision. |

## 9. How the library changes

- A change to a public API needs a PR and a short design note in its issue.
- CI refuses a PR that adds a public item with no doc comment, or a public function with no
  working example and no accepted exception (Section 1.8), once the tools exist.
- A new module, a new trait in `core`, or a change to a tenet needs an RFC in `docs/proposals/`
  and a committee review.
- Each module has an owner.
- Each new module has a first user: a real program that needs it. A module with no first user
  waits.
- Each module header states its layer, its device rule, its stage, and where it differs from Rust.
- Tracking issues for each layer come after the committee accepts this RFC.

## 10. Open questions

**Layout and scope**

- **Q1. Memory spaces in collections.** This RFC proposes that `Slice` carries a memory space, as
  in `Slice<T, Memory::M>`. The language already decides most of this. Pointers carry a memory
  space (`*mut f32 in Memory::GPU_HBM`), and so do tensors. A `Slice` is a pointer and a length.
  Without a memory space, it is either a hole in the placement checks or a host-only type. With
  one, one byte reader or column kernel works in any memory. An owning memory map in `std::fs`
  can then give a view in device memory where the machine declares a `: shared` edge (#3498).
  `Slice<T, Memory::M>` and a rank-1 view `Tensor<T, [?], Memory::M>` (#400) must not be two
  names for one idea (A2). This RFC proposes the following line. `Tensor` is for numbers, with
  shapes and operators. `Slice` is rank 1, for any `T`. Still open: do `Vec` and `Allocator`
  carry a memory space too? This choice shapes most of the allocation layer.
- **Q5. Tensors, SIMD and collectives.** Do `std::tensor`, `std::simd` and `std::collective`
  belong in `core`, in `std`, or in a device library? Do the `lanes::` routines become library
  functions (L1)?
- **Q6. Domain modules.** Do `std::nn`, `std::llama`, `std::bulk_hash` and `stdlib/graph` leave
  the standard library? If they leave, do they go to a directory in this repository or to separate
  repositories? Vx has no package manager yet (#1275), and two Vx objects cannot be linked into
  one library today (#3434).
- **Q7. Paths, re-exports and the prelude.** This RFC keeps every public path. Does Vx need a
  re-export feature anyway, so that a later move can keep the old path as an alias? Should
  `std::` re-export the `core::` modules, so that a program can write one path? Do new
  containers get flat paths (`std::vec_deque`), as today, or a `std::collections::` group? What
  goes in the implicit prelude (#529)?
- **Q12. Formats.** Do JSON, compression or regular expressions belong in `std`? This RFC proposes
  no.
- **Q13. Cryptography.** Does the committee agree that cryptographic algorithms stay out, and
  that OS random bytes and a constant-time comparison stay in?
- **Q17. Tests.** Is a test framework part of `std`, or part of the tools?

**Traits and API style**

- **Q2. Text output.** One `Writable` trait, as in Mojo, or `Display` plus `Debug`, as in Rust?
  One trait joins `fmt::Write` and `io::Write` into one byte-sink trait. Two traits separate the
  text for users from the text for programmers. The committee decides on the merits of each
  design. A compiler limit does not decide it. Today `Display` and `Debug` cannot both be on one
  type, because both name their method `fmt`. Under L2, that is a language gap to fix.
- **Q3. Errors.** An `Error` trait, as in Rust, or one concrete error type with a kind and a
  message, as in Mojo?
- **Q4. Dynamic dispatch and derive.** Does Vx get trait objects (`dyn`)? Does it get generated
  impls (`derive`) for `Clone`, `Hash`, `PartialEq` and debug output? Both are language decisions,
  but the library API depends on them.
- **Q11. Size type.** Are lengths and indices of host containers `usize` or `i64`? Vx's `usize`
  traps on overflow and wraps at 2^63. Which index type does device code use? Today the indices
  of `across` loops and of the atomic tensor functions are `i32`. Before a kernel starts, the
  host checks the size, so that these indices do not overflow (A6).
- **Q14. Allocation failure.** Does a failed allocation abort, as in Rust, or return an error, as
  in Zig? A device with a fixed amount of memory may need the error.

**Devices**

- **Q8. A trait whose methods run anywhere.** What is the syntax for a trait that requires its
  methods to run anywhere (R4)? Section 5, under R1, gives the order in which the method check
  turns on.

- **Q9. The device-crossing trait.** What is its name? Is it separate from `Send`? Does the
  compiler always derive it, or can a type opt out?

- **Q15. Panics on a device.** What does a panic do in a kernel? Today an `assert` in an Apple GPU
  kernel stops the program.

- **Q16. Blocking and non-blocking input and output.** Is blocking I/O with tasks enough, or does
  `std` need a non-blocking form?

**Contracts**

- **Q18. Facts for the optimizer.** Should a proven number fact, such as `i < self.len()`, reach
  LLVM as an `llvm.intr.assume`? Then `-O3` could remove the matching run-time check. An assume
  is only as sound as the prover. Too many assumes can also slow LLVM's optimizations. Or should
  the compiler remove the checks itself, as #1142 plans for `usize`?
- **Q19. The cost of proofs.** Under C1, only a call to a proven form asks the prover for a
  proof. Contracts make Z3 a hard requirement for a program that makes such a call. Each compile
  also checks the library's bodies again, until the library ships precompiled (#220). I did not
  measure this cost. Is it acceptable, and what is the limit?
- **Q20. The proven form of an access.** The proven form in C3 has its own name. What is that
  name, for example `at`? Does every container have one?

**Process**

- **Q10. Stability markers.** Do item attributes and `pub` visibility (#489) come first, or does a
  doc comment tag with a lint come first as an interim step? What is the spelling of the opt-in
  for an incubating feature? What is the schema of the attributes?

## References

- `docs/implementation_plans/core_library.md` and issues #451, #720, #721 and #536: the core
  library plan.
- Issue #2144: move plain code from the Rust runtime into Vx.
- `docs/language_rules_for_a_parallel_compiler.md`: why a signature must state where a function
  runs.
- `docs/lane_reductions.md` and issue #3468: the lane routines.
- Rust: the `core`, `alloc` and `std` split, the stability attributes, and RFC 1105 (API
  evolution).
- Zig: explicit allocators, no hidden allocation, and `std.crypto`.
- Mojo standard library at commit 2896002: `std/prelude`, `std/_plugin` and
  `std/builtin/device_passable.mojo`. The `gpu` package is in `max/mojo/max/gpu`.
- Go: the Go 1 compatibility promise.
