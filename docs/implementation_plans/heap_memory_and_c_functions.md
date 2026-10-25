# Heap memory and C functions in the Vx standard library

**Status:** in progress, 2026-10-25. Tracking issue: Vx#2144.
**Sources:** the reviews of Vx#2180
(https://github.com/vx-lang/Vx/pull/2180#issuecomment-6390913083) and Vx#2181
(https://github.com/vx-lang/Vx/pull/2181#issuecomment-6391038437).
**Scope:** how Vx code gets heap memory, which C functions the compiler and the libraries name,
and where a program's errors go.

______________________________________________________________________

## 0. Summary

The decisions this document makes:

1. **The standard library names no C function for memory.** `std::alloc` has three functions,
   `alloc`, `grow` and `dealloc`. Each is an `mlir!` block with one `vx` op.
1. **One function in the compiler lowers those ops.** It picks the C function for the target and
   for the placement. So the name `malloc` appears in one place in the compiler.
1. **The compiler's own allocations use the same ops.** A task's environment does today. A host
   `memref.alloc` will.
1. **A program can declare C's functions itself.** The compiler reuses a declaration of C's type.
   A declaration of another type is an error at the declaration.
1. **C's own functions, with C's types, are in `std::libc`.** Code written in C's style imports
   them from there.
1. **An `extern` declaration will belong to its module.** The checker checks each call against the
   declaration that its own module sees.
1. **Errors go to standard error.** This covers `eprint!`, `eprintln!`, a `panic`, and a failed
   check such as a division by zero.

## 1. The problems

**A program that declared `malloc` itself could not use `Vec`.** The `extern` declarations of a
program share one namespace. When `std::vec` called `malloc` through `std::alloc`, the checker
checked that call against the program's own declaration:

```vx
import std::vec;

extern "C" {
  fn malloc(size : i32) -> *mut i8;
}

fn main() -> i32 {
  let p = unsafe { malloc(16) };
  let mut v = Vec<i32>::new();
  v.push(1);
  print(v.get(0));
  return 0;
}
```

```
Error[E3003] at 35:30: in `vec_alloc`, imported from `std::vec` and used by this program:
Type mismatch in argument 1 for function 'malloc'. Expected i32, got i64
```

With `fn malloc(size : i64) -> *mut f32;`, as `llama2_math.vx` declares it, the compile stopped
later, with no position: `Codegen Error: ParseType("Unsupported MLIR element type in: none")`.

**The compiler declared `malloc` a second time (Vx#2175).** The task lowering and
`finalize-memref-to-llvm` look for an `llvm.func` named `malloc` or `free`. A program's declaration
is a `func.func` until `convert-func-to-llvm`. So they made a second declaration, and MLIR reported
"redefinition of symbol named 'malloc'", or renamed the copy to `malloc_0`, which did not link.

**Errors went to standard output.** A `panic` printed with `print!`. A failed `cf.assert` printed
its message with `puts`. So an error went into a program's data when its output went to a file.

## 2. Three ops for heap memory

```mlir
%p = vx.heap_alloc %bytes, %align {topology = 0 : i32} : !llvm.ptr
%q = vx.heap_grow %p, %old_bytes, %new_bytes, %align {topology = 0 : i32} : !llvm.ptr
vx.heap_free %p, %align {topology = 0 : i32} : !llvm.ptr
```

- `vx.heap_alloc` gives `bytes` of uninitialised memory at `align`, a power of two, or null.
- `vx.heap_grow` resizes a block and keeps its contents and its alignment. It gives null when
  there is no memory, and then the old block is still valid. A null block makes a new one.
- `vx.heap_free` gives back a block from either of them. Null is a no-op.

Each choice has a reason:

- **The alignment is an operand.** An `mlir!` input is a run-time value. The choice between
  `malloc` and `aligned_alloc` can then be made at run time. It folds when the alignment is a
  constant, as it is in `Vec`.
- **The topology is an attribute.** A device allocation goes through that topology's plugin, and
  the lowering must know which one.
- **`vx.heap_free` takes the alignment and the topology.** Memory from `_aligned_malloc` must go
  back through `_aligned_free`, and memory from `cudaMalloc` through `cudaFree`. Rust's `dealloc`
  takes the layout for the same reason.
- **`vx.heap_grow` is an op.** Device memory has no `realloc`, and only the lowering knows how to
  copy.
- **The ops make a raw pointer, not a memref.** A memref in an `mlir!` block worked only by chance:
  with no alignment asked, the aligned pointer was the one that `malloc` returned, and the drop pass
  left the memref alone.

`std::alloc`:

```vx
fn alloc(bytes : i64, align : i64) -> *mut i8
fn grow(p : *mut i8, old_bytes : i64, new_bytes : i64, align : i64) -> *mut i8
fn dealloc(p : *mut i8, align : i64) -> void
```

`std::vec`, `std::box` and later `String` call these. `Vec` asks for 16 bytes of alignment, the
most any Vx scalar type needs. It will pass `alignof<T>()` when Vx has it.

## 3. The lowering

`lowerHeapOps` in `src/dialect/VxLowering.cpp` lowers the three ops in `convert-vx-to-standard`:

| Topology and alignment | `vx.heap_alloc` | `vx.heap_grow` | `vx.heap_free` |
| --- | --- | --- | --- |
| Host, up to `malloc`'s | `malloc` | `realloc` | `free` |
| Host, larger | `aligned_alloc`, with the size rounded up to the alignment | a new block and a copy | `free` |
| Host on Windows, larger | `_aligned_malloc` | `_aligned_realloc` | `_aligned_free` |
| Device memory | the topology's plugin: `cudaMalloc`, `cudaMallocAsync`, `hipMalloc` | a new block and a device copy | the plugin: `cudaFree` |
| Pinned host memory, when the machine file declares it | the plugin: `cudaHostAlloc` | a new block and a copy | `cudaFreeHost` |
| Bare metal | the allocator that the machine file names | | |

C promises `malloc`'s alignment as `alignof(max_align_t)`. That is 16 on the 64-bit targets Vx
builds for today, and 8 on 32-bit ARM. The lowering knows the target, so the number belongs there,
not in `std`.

**The compiler's own allocations.** These go through the same function:

- The task lowering makes a task's environment with `vx.heap_alloc`. Before, it called
  `LLVM::lookupOrCreateMallocFn`.
- A host `memref.alloc` goes through it too. There are two ways:
  1. Vx lowers a host `memref.alloc` itself, before `finalize-memref-to-llvm`.
  1. `finalize-memref-to-llvm` runs with `use-generic-functions=true`. Vx then defines the three
     functions it calls, `_mlir_memref_to_llvm_alloc`, `_mlir_memref_to_llvm_aligned_alloc` and
     `_mlir_memref_to_llvm_free`, each as one op.

Then the name `malloc` appears in one place in the compiler. The working-set check and the
comptime interpreter each learn three ops, not C's functions. Windows, a bare-metal allocator,
pinned host memory, and `cudaMallocAsync` on a stream are each a change to this one function.

## 4. A program's own C functions

A program may declare `malloc` itself, or define a function with that name.

- **The compiler reuses a declaration of C's type.** The pass `vx-libc-externs-to-llvm` runs first in both
  pipelines. It turns a program's declaration of `malloc`, `free`, `realloc` or `aligned_alloc`,
  and the calls to it, into the `llvm.func` that the lowerings look for. When the heap ops lower
  `memref.alloc` too, this pass becomes the first step of the lowering function.
- **A declaration of another type is an error, E3059.** The error is at the declaration, and it
  gives the declaration to write. A size is a 64-bit integer, a pointer is a raw pointer to host
  memory, and `free` returns nothing. An `i32` size is wrong anyway: `size_t` has 64 bits, and such
  a call works only when the upper half of the register happens to be zero.
- **The compiler renames a definition with one of these names.** `src/name_clashes.rs` treats `malloc`,
  `realloc`, `aligned_alloc` and `free` as names that an `extern` block declares, even when none
  does. So the lowering's `free` keeps its name, and the program's function gets a new one.

## 5. C's functions in `std::libc`

Code written in C's style gets the functions as the C standard writes them:

```vx
extern {
  fn malloc(size : usize) -> *mut void;
  fn calloc(count : usize, size : usize) -> *mut void;
  fn realloc(ptr : *mut void, size : usize) -> *mut void;
  fn free(ptr : *mut void) -> void;
  fn aligned_alloc(alignment : usize, size : usize) -> *mut void;
}
```

- **`size_t` is `usize`, and `void*` is `*mut void`.** `char*` stays `*const i8`, as `open` has it.
  Two language changes make these types true:
  - `usize` must have the target's pointer width. It has 64 bits on every target today.
  - The conversion that `*mut i8` has to and from any pointer (`is_assignable`,
    `src/hir/expr.rs`) moves to `*mut void`. Then `*mut i8` means `i8` data again. Until then, the
    declarations use `*mut i8` in place of `*mut void`.
- **C11 has no `aligned_free`.** A block from `aligned_alloc` goes back through `free`. C23 adds
  `free_sized` and `free_aligned_sized`. Not every C library has them yet, so they can wait.
- **Microsoft's C runtime has no `aligned_alloc`.** It has `_aligned_malloc` and `_aligned_free`.
- **`std::alloc` is for Vx code, and `std::libc` is for code that wants C's functions.**

## 6. An `extern` declaration belongs to its module

Section 4 removes the clash for `std`, because `std` names no C function for memory. The
question stays for any two modules that declare one C function. The rule:

- The checker checks each call against the declaration that its own module sees.
- LLVM gets one declaration of the symbol.
- Two declarations that lower to the same LLVM type are fine, such as `*mut f32` and `*mut i8`,
  which are both `ptr`.
- Two that do not are an error that names both, such as an `i32` size and an `i64` size.

Then a program can import `std::libc` and still declare its own `malloc`. The rule depends only on
declarations, not on a function body, so it keeps the rules in
`docs/language_rules_for_a_parallel_compiler.md`.

## 7. Errors on standard error

- **`eprint!` and `eprintln!`** work as `print!` and `println!` do, on standard error. Each code
  generator puts two calls of the runtime's `vx_print_to_stderr` around them. These switch the
  `print_*` functions to standard error on the current thread.
- **A `panic`** prints `panic: <message>` with `eprintln!`.
- **A failed check:** the pass `vx-lower-host-asserts` runs before `convert-cf-to-llvm`. It turns a
  host `cf.assert` into a write of the message and a newline to standard error, and then `abort`.
  An empty message prints nothing. A `cf.assert` in a `gpu.module` keeps its lowering to
  `__assertfail`. The pass calls the runtime's `vx_stderr_write`, not C's `write`, so a program's
  own `fn write` cannot get in the way.
- **`vxc`** writes a program's standard error to its own standard error, whether the program fails
  or not.

A test can then check standard error alone, with `2>&1 >/dev/null`.

One limit stays: a tensor printed with `eprint!` still goes to standard output, because it prints
through MLIR's `printMemref`.

## 8. A placement-generic `alloc`

`fn alloc<M : Memory>(bytes : i64, align : i64) -> *mut i8 in M` needs the topology of `M` in the
op's attribute. That is the shape of Vx#2155, which gives the `core::ptr` functions a placement
parameter. An `mlir!` input is a run-time value, and nothing puts a generic parameter's value into
an attribute. So a host-only `std::alloc` can be written with `mlir!` today. A placement-generic
one needs a compiler intrinsic, or a way for `mlir!` to put a generic parameter's value into an
attribute.

The same gap shows for tensors: Vx#2200 asks for a placement parameter there too.

## 9. Status

| Item | Status | Where |
| --- | --- | --- |
| Reuse a program's declaration of C's type | done | Vx#2180 |
| E3059 for a declaration of another type | done | Vx#2180 |
| The three ops, and their host lowering | done | Vx#2212, merged into Vx#2180 |
| `std::alloc`, `std::box` and `std::libc` | done | Vx#2212 |
| A task's environment through the ops | done | Vx#2212 |
| Rename a definition named like a C allocator | done | Vx#2212 |
| `eprint!`, a panic and a failed check on standard error | in review | Vx#2215 |
| `std::vec` over `std::alloc`, the bounds message on standard error | in review | Vx#2216 |
| A host `memref.alloc` through the ops | open | |
| Device topologies, pinned host memory, Windows, bare metal | open | |
| An `extern` declaration that belongs to its module | open | |
| `usize` at the target's pointer width, and `*mut void` | open | |
| `alignof<T>()` for `Vec` | open | |
| A placement-generic `alloc` | open | Vx#2155, Vx#2200 |
| `eprint!` of a tensor | open | |
