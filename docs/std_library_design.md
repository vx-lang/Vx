# Vx Standard Library (`std.vx`) Design Proposal

**Author:** aditya@
**Date:** May 11, 2026

## 1. Executive Summary & Strategic Vision

A systems programming language cannot achieve critical mass without a robust, highly optimized standard library. However, reimplementing collections, networking, and file I/O from scratch introduces years of bug-fixing and security vulnerabilities.

**The Vx Standard Library (`std.vx`) strategy is one of Ecosystem Delegation.**
For Host operations (`Topology::Host`), Vx natively lowers source code to MLIR and statically links against a pre-compiled Rust standard library (`libvx_std.a`).

This approach provides three massive advantages:

1. **Zero-Overhead FFI:** Vx emits raw MLIR `llvm.call` instructions pointing to stable C-ABI endpoints, operating with the same performance as native C or Rust.
1. **Unified Middle-End:** We retain MLIR as our universal intermediate representation without needing to transpile source code to `.rs`.
1. **Instant Maturity:** We gain the memory safety, heavily audited cryptography, and blisteringly fast SIMD collections of the Rust ecosystem on Day 1.

______________________________________________________________________

## 2. Memory Safety Synergy

Rust is the ideal backing engine for Vx because its borrow checker naturally aligns with Vx's topological type constraints.

In Vx, variables are intrinsically bound to memory spaces (e.g., `Ref<T, Memory::Host_DRAM>`). Because Rust enforces strict aliasing rules (you cannot have a mutable reference and an immutable reference simultaneously), integrating Rust collections naturally protects the Vx compiler from Host-side race conditions. When Vx hands ownership of a pointer to a Rust `HashMap` across the FFI boundary, Rust's internal linear typing guarantees that the data structures remain corruption-free.

______________________________________________________________________

## 3. Core Data Structures & Interfaces

The following types will be exposed in `std.vx` via interface modules. The user will experience full IntelliSense and type-checking, while the compiler resolves the bodies externally.

### 3.1 Primitives and Numerics

- **Integers:** `i8`, `i16`, `i32`, `i64`, `i128` (and `u*` variants).
- **Floats:** `f16`, `bf16`, `f32`, `f64`.
- **Hardware Pointers:** `Ref<T, MemorySpace>`, `Pinned<T, Topology>`.

### 3.2 Core Enums

- `Option<T>`: Represents an optional value (`Some(T)` or `None`).
- `Result<T, E>`: Represents success (`Ok(T)`) or failure (`Err(E)`).

### 3.3 Collections

Backed by Rust's `std::collections`:

- `Vec<T>`: A contiguous, growable array type.
- `HashMap<K, V>`: A hash map implemented with quadratic probing and SIMD lookup.
- `HashSet<T>`: A hash set implemented as a `HashMap` where the value is `()`.

### 3.4 Strings

- `String`: A UTF-8 encoded, growable string.
- `str`: A string slice.

______________________________________________________________________

## 4. System & I/O Modules

By leveraging Rust's OS abstractions, Vx instantly supports cross-platform (Windows, macOS, Linux) I/O.

### 4.1 File System (`std::fs`)

- `File`: Represents a reference to an open file on the filesystem.
- Methods: `open()`, `create()`, `read_to_end()`, `write_all()`.

### 4.2 Networking (`std::net`)

- `TcpStream`: A TCP stream between a local and a remote socket.
- `TcpListener`: A TCP socket server, listening for connections.
- `UdpSocket`: A UDP socket.

______________________________________________________________________

## 5. The FFI Monomorphization Strategy

Exposing a generic type like `HashMap<K, V>` over a C-ABI requires architectural care since C does not support generics, and Rust mangles generic symbols.

### The Mechanics:

1. **The Interface Module (`std.vx`)**
   The Vx compiler reads the structural definition without an implementation:

   ```rust
   pub struct HashMap<K, V> {
       _ptr: *mut void, // Opaque pointer to Rust's HashMap
   }

   impl<K, V> HashMap<K, V> {
       pub extern "C" fn new() -> HashMap<K, V>;
       pub extern "C" fn insert(&mut self, key: K, value: V);
   }
   ```

1. **The Rust Backend (`libvx_std.rs`)**
   We utilize Rust macros to pre-compile monomorphized FFI shims for commonly used types. For advanced types, Vx's Semantic Analyzer instructs the build system to emit a custom C-ABI shim block prior to final linkage.

   ```rust
   use std::collections::HashMap;
   use std::ffi::c_void;

   #[no_mangle]
   pub extern "C" fn __vx_std_hashmap_i32_i32_new() -> *mut c_void {
       let map: HashMap<i32, i32> = HashMap::new();
       Box::into_raw(Box::new(map)) as *mut c_void
   }

   #[no_mangle]
   pub extern "C" fn __vx_std_hashmap_i32_i32_insert(ptr: *mut c_void, key: i32, value: i32) {
       let map = unsafe { &mut *(ptr as *mut HashMap<i32, i32>) };
       map.insert(key, value);
   }
   ```

1. **MLIR Lowering**
   When `map.insert(1, 2)` is called in Vx, the compiler resolves the generic arguments (`i32, i32`) and generates an MLIR call directly to the explicit symbol `@__vx_std_hashmap_i32_i32_insert`.

This technique ensures that Vx remains a pure frontend router, MLIR remains the pure middle-end, and we inherit the entire safety and performance profile of the Rust ecosystem.
