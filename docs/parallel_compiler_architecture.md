# High-Performance Parallel Compiler Architecture Proposal

Target Architecture: Rust-based Frontend with High-Throughput Concurrent Layout Pipeline

------------------------------
## 1. Executive Summary

This proposal outlines the technical architecture for a high-performance, module-based programming language compiler written in Rust. The primary architectural objective is to eliminate the single-threaded bottlenecks common in early compiler frontends (such as rustc).

By enforcing a nominal type system, requiring explicit boxing for recursive data structures, and utilizing a 256-bit bit-packed Global Identifier (GID) strategy, the compiler decouples module dependencies. This layout enables an embarrassingly parallel compilation pipeline capable of safely utilizing 100% of available CPU cores without complex query engines or heavy cross-thread lock contention.

A core pillar of this architecture is the **Mathematical Verification Engine**—a zero-cost abstraction enabled by `#[cfg(debug_assertions)]`. Rather than hoping threads behave correctly, the compiler injects structural invariants (Hooks) between phases to mathematically prove zero aliasing, cache bounds, and memory safety during parallel execution, verifying the Data-Oriented Execution Model at compile-time.

------------------------------
## 2. Global Identifier (GID) Design
To eliminate pointer-chasing, string lookups, and memory synchronization overhead across threads, every symbol, nominal type, and monomorphized variant in the program is represented by a unique, flat 256-bit integer array (`[u64; 4]`).

```
+-------------------+-------------------+-------------------+-------------------+
|  Word 0 (64-bit)  |  Word 1 (64-bit)  |  Word 2 (64-bit)  |  Word 3 (64-bit)  |
+-------------------+-------------------+-------------------+-------------------+
|    Module Hash    | Local Symbol Hash | Generic Context   | Reserved / Custom |
+-------------------+-------------------+-------------------+-------------------+
```

### Word Layout Specifications:

* **Word 0 (Module Hash):** A deterministic content hash of the module's fully qualified path. To prevent Birthday Paradox collisions, the compiler treats `Word 0` and `Word 1` as a composite 128-bit cryptographic hash.
* **Word 1 (Local Symbol Hash):** A stable, deterministic content hash of the local symbol's name. For anonymous types, a topological "DefPath" is used.
* **Word 2 (Generic Context Hash / Lifetime Bitfield):** This word operates as a fast-path 16-bit window bitfield for variance/region tracking. For generic instantiations (e.g., `List<i32>`), it acts as a 63-bit index pointing into a global arena, ensuring no hash collisions during structural equality checks.
* **Word 3 (Reserved Space / Cross-Crate Flags):** Allocated for metadata and compiler flag bitfields, keeping them local in the CPU cache.

#### The Fast-Path Structure (Bit 63 = 0)
When a function contains 4 or fewer parameters, it executes entirely on the fast path, avoiding heap allocations.

```
+---+-------------------+-------------------+-------------------+-------------------+
| 0 | Parameter 1 (15b) | Parameter 2 (16b) | Parameter 3 (16b) | Parameter 4 (16b) |
+---+-------------------+-------------------+-------------------+-------------------+
 63  62               48 47               32 31               16 15                0
```

#### The Slow-Path Structure (Bit 63 = 1)
For statistical outliers, the escape-hatch bit is flipped. The remaining 63 bits are a sequential index into an unbounded memory arena.

```
+---+-------------------------------------------------------------------------------+
| 1 |             63-bit Sequential Index to Global Unbounded Arena                 |
+---+-------------------------------------------------------------------------------+
 63  62                                                                            0
```

#### Rust Implementation
```rust
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]pub struct TypeId {
    pub words: [u64; 4],
}

const ESCAPE_HATCH_MASK: u64 = 1 << 63;
const INDEX_MASK: u64        = !ESCAPE_HATCH_MASK;

#[derive(Debug, Clone)]pub struct UnboundedFunctionMetadata {
    pub type_arguments: Vec<TypeId>,
    pub lifetime_regions: Vec<u32>,
    pub trait_vtables: Vec<u64>,
}

#[derive(Debug)]pub enum LifetimeSignature<'a> {
    FastPath(u64),
    SlowPath(&'a UnboundedFunctionMetadata),
}

impl TypeId {
    #[inline(always)]
    pub fn lifetime_context<'sess>(&self, arena: &'sess [UnboundedFunctionMetadata]) -> LifetimeSignature<'sess> {
        let word_2 = self.words[2];
        if (word_2 & ESCAPE_HATCH_MASK) != 0 {
            LifetimeSignature::SlowPath(&arena[(word_2 & INDEX_MASK) as usize])
        } else {
            LifetimeSignature::FastPath(word_2)
        }
    }
}
```

------------------------------
## 3. The Data-Oriented Execution Model

The core principle of Vx's high-performance design is that the compiler abandons the AST as early as possible. During parsing, the AST is lowered into **two distinct flat arrays**:

1. **`LOCAL_TYPE_STREAM`: `Vec<[u64; 4]>`** (Stores 256-bit GIDs)
2. **`LOCAL_HIR_STREAM`: `Vec<HirInstruction>`** (Dense instruction array with `u32` indices into the Type Stream)

### The Epoch / Worker Split (Zero-Lock Session)
To achieve parallel scaling without `RwLock` contention, the monolithic session is split:

1. **`GlobalSession` (The Frozen Epoch):** Absolute, immutable truth of everything compiled *before* the current phase.
2. **`LocalWorkerState` (The Phase Context):** Private, fully mutable context per thread.

```rust
pub struct GlobalSession {
    pub epoch: u64,
    pub registry: Arc<ImmutableGlobalRegistry>,
    pub slow_path_arena: Arc<Vec<UnboundedFunctionMetadata>>,
    pub generics_arena: Arc<Vec<Vec<TypeId>>>,
}

pub struct LocalWorkerState {
    pub global: Arc<GlobalSession>,
    pub local_slow_path_arena: Vec<UnboundedFunctionMetadata>,
    pub local_generics_arena: Vec<Vec<TypeId>>,
    pub local_type_stream: Vec<TypeId>,
    pub local_hir_stream: Vec<HirInstruction>,
}
```

------------------------------
## 4. The 8-Phase Parallel Pipeline & Verification Invariants

The compiler architecture rejects complex inter-stage pipelining in favor of a clean, phase-separated model bound by explicit synchronization barriers using Rayon's work-stealing thread pool.

### Phase 1: Parallel Parsing
* **Pre-conditions:** Source files exist on disk.
* **Logic:** Threads independently read a module file and parse it into an AST. Interface outlines are extracted.
* **Post-conditions:** Thread-local ASTs and symbol hashes are complete.

### Phase 2: Global Registry Build
* **Pre-conditions:** All AST files are parsed.
* **Logic:** Resolves cross-module aliases. A monolithic immutable `GlobalSession` is instantiated.
* **Post-conditions:** The global nominal layout table is entirely frozen and read-only. Cycle checks guarantee no infinitely recursive types.

### Phase 3: Parallel Body Type-Checking
* **Pre-conditions:** `GlobalSession` is available.
* **Logic:** Threads execute type-checking. Generics are pushed to `LOCAL_GENERICS_ARENA` with `LOCAL_DEFERRED_BIT` set.
* **Post-conditions (Verification Hook: Isolation):** Mathematically proves zero shared mutability (aliasing) and asserts that local deferred indices fit precisely within the thread's local arena length.

### Phase 4: Parallel Local Deduplication
* **Pre-conditions:** All worker threads have returned their `LocalWorkerState`.
* **Logic:** The compiler hashes the local arenas, buckets them, and deduplicates them in parallel into the global arenas, assigning absolute 63-bit indices.
* **Post-conditions:** Duplicate types within the same compilation epoch are collapsed.

### Phase 5: Cross-Thread Merging (Epoch Advance)
* **Pre-conditions:** Deduplication is finished.
* **Logic:** The deduplicated global arenas are wrapped in a new `Arc<GlobalSession>` with an incremented epoch. The old session is dropped.
* **Post-conditions (Verification Hook: LSP Memory Proof):** Uses a `Weak<GlobalSession>` pointer to prove the Rust borrow checker successfully destroyed the previous epoch, mathematically verifying zero memory leaks.

### Phase 6: SIMD Patch Pass
* **Pre-conditions:** `LOCAL_TYPE_STREAM` contains `LOCAL_DEFERRED_BIT` set on types.
* **Logic:** A vectorized `chunks_exact_mut(8)` sweeps over the streams, patching local deferred indices into global absolute indices in microseconds.
* **Post-conditions (Verification Hook: Absolute Identity Proof):** Mathematically proves that the SIMD unit cleared every single deferred bit, and that the new global indices correctly fit within the bounds of the newly advanced `GlobalSession` arenas.

### Phase 7: Parallel Module Deduplication & Codegen
* **Pre-conditions:** All types are globally resolved and patched.
* **Logic:** Monomorphization requests are routed via `Word 0` to their origin module buckets. Threads take exclusive ownership of buckets and run `sort_unstable()` and `dedup()`. Codegen executes over `LOCAL_HIR_STREAM`.
* **Post-conditions (Verification Hook: Monomorphization Router):** Proves that synthetic monomorphizations (external upstream generics) are successfully intercepted and mapped to the local caller's bucket, preventing writes into frozen upstream modules.

### Phase 8: Zero-Copy Metadata Serialization
* **Pre-conditions:** All streams are fully lowered.
* **Logic:** Dense dictionaries of GIDs are constructed and serialized directly to `.vxm` via `bytemuck`.
* **Post-conditions:** The artifact matches the exact bitwise layout of `TypeId` for downstream compilation.

------------------------------
## 5. Metadata Serialization Format

To support rapid cross-crate compilation without requiring downstream projects to re-parse upstream source files, the 256-bit ID scheme is tightly integrated into a zero-copy metadata format (`.vxm`).

### The "String/ID Dictionary" Pattern
Inside the compiled metadata binary file, an isolated block acts as the Type Dictionary. Full 32-byte IDs in signatures are replaced by lightweight 2-byte or 4-byte indices pointing into the local dictionary.

```rust
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]pub struct TypeId {
    pub words: [u64; 4],
}

// Zero-allocation serialization of the master dictionary
pub fn serialize_metadata_symbols(unique_types: &[TypeId], output: &mut Vec<u8>) {
    let len = unique_types.len() as u64;
    output.extend_from_slice(&len.to_le_bytes());
    let bytes: &[u8] = bytemuck::cast_slice(unique_types);
    output.extend_from_slice(bytes);
}

// Zero-allocation deserialization via `cast_slice`
pub fn deserialize_metadata_symbols(bytes: &[u8]) -> (&[TypeId], &[u8]) {
    let (len_bytes, remaining) = bytes.split_at(8);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let byte_len = len * std::mem::size_of::<TypeId>();
    let (dict_bytes, rest) = remaining.split_at(byte_len);
    (bytemuck::cast_slice(dict_bytes), rest)
}
```

### Cross-Crate Monomorphization
When loading metadata, pointer addresses or local IDs usually have to be "swizzled" (remapped). Because our Word 0 and Word 1 are cryptographic hashes, they require zero translation.

------------------------------
## 6. Performance Engineering Guarantees

By strictly adhering to this Data-Oriented parallel model, the architecture guarantees:

* **Linear Speedups:** Frontend scaling correlates perfectly with core count during parsing, type-checking, and bucket deduplication.
* **Cache Optimization:** Flat `[u64; 4]` structures ensure optimal L1/L2 cache-line utilization during heavy passes.
* **Zero Deadlocks:** The total removal of fine-grained runtime locking (`Mutex`/`RwLock`) across worker threads eliminates concurrent data races and thread stalling bugs mathematically.