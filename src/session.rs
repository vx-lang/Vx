//===- session.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the core compiler session and local worker states.
// It orchestrates the compilation pipeline across multiple threads, managing shared
// diagnostics, module registries, and global state required during the lowering
// of Vx source code to MLIR.
//
//===----------------------------------------------------------------------===//
use crate::gid::{TypeId, UnboundedFunctionMetadata};
use crate::hir::HirInstruction;
use std::sync::Arc;

// The frozen nominal-type registry (layouts + module indices + cycle-checked). Built once per
// compilation at the Phase 2 freeze point and shared read-only across worker threads. Re-exported
// here so `GlobalSession` and the verifier name it via `session`, but the type lives in `registry`.
pub use crate::registry::ImmutableGlobalRegistry;

/// Represents the frozen past of the compilation process.
/// It contains everything compiled before the current phase.
///
/// # Data-Oriented Design (DOD) Constraints
/// This session deliberately separates dynamic compiler metadata into strictly
/// typed, homogeneous arenas rather than using a single polymorphic `Enum` array
/// (e.g., `enum SlowPathData { Function(...), Generics(...) }`).
///
/// 1. **Cache-Line Density**: `UnboundedFunctionMetadata` is a dense structure of `u64`
///    bitfields. Iterating over it is highly predictable. `Vec<TypeId>`, however,
///    is a heap-allocated fat pointer with variable lengths. Mixing them would introduce
///    unpredictable struct padding and pointer-chasing overhead.
/// 2. **Hardware Prefetching**: By keeping `generics_arena` separate, the CPU's spatial
///    prefetcher can stream through the dense `slow_path_arena` at maximum bandwidth
///    during the critical lifetime subtyping pass, without stalling on scattered `Vec` pointers.
pub struct GlobalSession {
    pub epoch: u64,
    pub registry: Arc<ImmutableGlobalRegistry>,

    /// The dense, homogeneous arena for complex parameter and lifetime evaluation.
    /// Accessed heavily during Phase 1 type-checking and borrow checking.
    pub slow_path_arena: Arc<Vec<UnboundedFunctionMetadata>>,

    /// The heterogeneous arena for structural generic instantiations.
    /// Separated to prevent cache fragmentation in the `slow_path_arena`.
    pub generics_arena: Arc<Vec<TypeId>>,
    pub generics_offsets: Arc<Vec<(usize, usize)>>,
}

impl GlobalSession {
    /// A session with an *empty* frozen registry — for callers that don't build one (the
    /// sequential driver, unit tests). The parallel pipeline uses [`Self::with_registry`].
    pub fn new(epoch: u64) -> Self {
        Self::with_registry(
            epoch,
            ImmutableGlobalRegistry::build_and_validate(Vec::new())
                .expect("an empty registry is always valid"),
        )
    }

    /// A session holding the frozen registry built at the Phase 2 freeze point.
    pub fn with_registry(epoch: u64, registry: ImmutableGlobalRegistry) -> Self {
        Self {
            epoch,
            registry: Arc::new(registry),
            slow_path_arena: Arc::new(Vec::new()),
            generics_arena: Arc::new(Vec::new()),
            generics_offsets: Arc::new(Vec::new()),
        }
    }
}

/// Represents the mutable present of a single worker thread during Phase 1.
pub struct LocalWorkerState {
    pub global: Arc<GlobalSession>,

    // Completely lock-free, thread-local mutation
    pub local_slow_path_arena: Vec<UnboundedFunctionMetadata>,
    pub local_generics_arena: Vec<TypeId>,
    pub local_generics_offsets: Vec<(usize, usize)>,

    // The flat arrays replacing the AST
    pub local_type_stream: Vec<TypeId>, // array of 256-bit GIDs
    pub local_hir_stream: Vec<HirInstruction>,
}

use crate::gid::LifetimeSignature;

impl LocalWorkerState {
    pub fn new(global: Arc<GlobalSession>) -> Self {
        Self {
            global,
            local_slow_path_arena: Vec::new(),
            local_generics_arena: Vec::new(),
            local_generics_offsets: Vec::new(),
            local_type_stream: Vec::new(),
            local_hir_stream: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn resolve_lifetime<'a>(&'a self, type_id: &TypeId) -> LifetimeSignature<'a> {
        use crate::gid::{Word2, Word2Arena, Word2Scope};
        match type_id.classify_word2() {
            Word2::FastLifetime(bits) => LifetimeSignature::FastPath(bits),
            // A *pure* generic instantiation carries no lifetime metadata (its arena holds only type
            // arguments), so it is lifetime-unconstrained ('static). A type that is both generic and
            // borrowed lives in the SlowMeta arena, which carries the lifetimes.
            Word2::Index {
                arena: Word2Arena::Generics,
                ..
            } => LifetimeSignature::FastPath(0),
            Word2::Index {
                index,
                arena: Word2Arena::SlowMeta,
                scope,
            } => {
                let i = index as usize;
                match scope {
                    Word2Scope::Local => {
                        LifetimeSignature::SlowPath(&self.local_slow_path_arena[i])
                    }
                    Word2Scope::Global => {
                        LifetimeSignature::SlowPath(&self.global.slow_path_arena[i])
                    }
                }
            }
        }
    }
}
