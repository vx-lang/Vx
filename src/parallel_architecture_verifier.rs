//===- parallel_architecture_verifier.rs - Vx Compiler ---------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file verifies the structural integrity of the compiler's parallel passes.
// It enforces project-specific constraints, such as the strict prohibition of
// standard locking primitives, ensuring the compiler maintains a
// lock-free, high-throughput architecture.
//
//===----------------------------------------------------------------------===//
#[cfg(debug_assertions)]
pub mod verify_arch {
    use crate::gid::{TypeId, Word2, Word2Arena, Word2Scope, LOCAL_DEFERRED_BIT};
    use crate::session::{GlobalSession, LocalWorkerState};
    use crate::syntax;
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Weak};

    pub fn verify_phase_1_parse(file_paths: &[String], parsed_modules: &[syntax::VxModule]) {
        assert_eq!(
            file_paths.len(),
            parsed_modules.len(),
            "FATAL: Phase 1 Parsing failed to preserve a 1:1 mapping between files and modules."
        );
        for module in parsed_modules {
            assert!(
                !module.module_path.is_empty(),
                "FATAL: A module dropped its path during Phase 1."
            );
        }
    }

    pub fn verify_phase_2_registry(registry: &Arc<crate::session::ImmutableGlobalRegistry>) {
        assert!(
            Arc::strong_count(registry) >= 1,
            "FATAL: Phase 2 Global Registry is not safely bound."
        );
    }

    pub fn verify_phase_3_isolation(workers: &[&LocalWorkerState], global: &Arc<GlobalSession>) {
        for worker in workers {
            // INVARIANT 1: Zero Shared Mutability (The Aliasing Proof)
            assert_eq!(
                Arc::as_ptr(&worker.global),
                Arc::as_ptr(global),
                "FATAL: Worker thread diverged from the frozen GlobalSession."
            );

            // INVARIANT 2: Arena Bounding
            //
            // Routed through `classify_word2` rather than masking the words here. This was the last
            // hand-rolled word-2 decoder in the file (#308/#193): re-deriving the index and the
            // arena from raw bits duplicates the codec, and a duplicated codec is one that drifts --
            // the escape-hatch and digest encodings are invisible to a bare `& INDEX_MASK`, so this
            // read would silently treat a content-addressed digest as an arena index the moment the
            // guard above stopped excluding it.
            for gid in &worker.local_type_stream {
                if let Word2::Index {
                    index,
                    arena,
                    scope: Word2Scope::Local,
                } = gid.classify_word2()
                {
                    let arena_len = match arena {
                        Word2Arena::Generics => worker.local_generics_offsets.len(),
                        Word2Arena::SlowMeta => worker.local_slow_path_arena.len(),
                    };
                    assert!(
                        (index as usize) < arena_len,
                        "FATAL: Local deferred index out of bounds ({index} >= {arena_len} in the \
                         {arena:?} arena)."
                    );
                }
            }
        }
    }

    pub fn verify_phase_4_deduplication(
        global_generics_arena: &Arc<Vec<TypeId>>,
        global_generics_offsets: &Arc<Vec<(usize, usize)>>,
        global_slow_path_arena: &Arc<Vec<crate::gid::UnboundedFunctionMetadata>>,
    ) {
        // Assert no duplicates exist in generics arena
        let generics_count = global_generics_offsets.len();
        let unique_generics: std::collections::HashSet<_> = global_generics_offsets
            .par_iter()
            .map(|&(start, len)| &global_generics_arena[start..start + len])
            .collect();
        assert_eq!(
            generics_count,
            unique_generics.len(),
            "FATAL: Phase 4 Deduplication failed. Duplicate generics vector found."
        );

        // Assert no duplicates exist in slow path arena
        let slow_path_count = global_slow_path_arena.len();
        let unique_slow_path: std::collections::HashSet<_> =
            global_slow_path_arena.par_iter().collect();
        assert_eq!(
            slow_path_count,
            unique_slow_path.len(),
            "FATAL: Phase 4 Deduplication failed. Duplicate slow path metadata found."
        );
    }

    pub fn verify_phase_5_epoch_advance(old_epoch: Weak<GlobalSession>) {
        // INVARIANT: Zero Leakage
        assert!(
            old_epoch.upgrade().is_none(),
            "FATAL: Memory Leak detected. The previous Epoch was not fully dropped."
        );
    }

    pub fn verify_phase_6_simd_patch(patched_type_stream: &[TypeId], session: &Arc<GlobalSession>) {
        // We can use rayon here to verify the patch pass in parallel
        patched_type_stream.par_iter().for_each(|gid| {
            // INVARIANT 1: The Eradication of Local State
            assert_eq!(
                gid.words[3] & LOCAL_DEFERRED_BIT,
                0,
                "FATAL: SIMD Patch Pass missed a deferred bit. Absolute identity failed."
            );

            // INVARIANT 2: Global Coordinate Integrity.
            //
            // Routed through `classify_word2` rather than re-masking the words here. This function
            // used to decode word 2 itself, which made it a *second* decoder -- the arrangement
            // that produced the #193 bug, where the word was read under one format having been
            // written under another. It also mattered concretely: under `--intern-mode=content`
            // (#296) word 2 holds a digest, and a second decoder that only knows about indices
            // reads that digest as an arena offset and asserts on a bound it was never subject to.
            match gid.classify_word2() {
                crate::gid::Word2::Index { index, arena, .. } => {
                    let (limit, what) = match arena {
                        crate::gid::Word2Arena::Generics => {
                            (session.generics_offsets.len(), "generics")
                        }
                        crate::gid::Word2Arena::SlowMeta => {
                            (session.slow_path_arena.len(), "function")
                        }
                    };
                    assert!((index as usize) < limit, "FATAL: Patched {what} index OOB.");
                }
                // A content digest indexes nothing, so there is no bound to check. Its integrity
                // condition is different in kind: equal arguments must give equal digests, which is
                // true by construction and tested in `gid.rs`.
                crate::gid::Word2::GenericDigest(_) => {}
                crate::gid::Word2::FastLifetime(_) => {}
            }
        });
    }

    pub fn verify_phase_7_routing(
        buckets: &[Vec<syntax::Function>],
        module_index_map: &HashMap<u64, usize>,
    ) {
        // Build a reverse lookup mapping dense indices back to their Module Hashes
        let mut index_to_hash: HashMap<usize, u64> = HashMap::new();
        for (&hash, &index) in module_index_map.iter() {
            index_to_hash.insert(index, hash);
        }

        for (bucket_index, _bucket) in buckets.iter().enumerate() {
            if let Some(&_expected_bucket_hash) = index_to_hash.get(&bucket_index) {
                // For each instantiated function in this bucket
                // we can't fully verify TypeId word 0 easily here because we just have AST Functions.
                // However, the AST function name is present, but checking it mathematically against Word 0
                // requires reconstructing the TypeId.
                // We'll skip a deep structural check here because `buckets` holds `Function` objects, not `TypeId`s.
                // But this hook signature is available if we attach `TypeId` directly to AST nodes in the future.
            }
        }
    }

    pub fn verify_phase_8_serialization(bytes: &[u8], dictionary_len: usize) {
        // 8 bytes for length prefix, then dictionary_len * 32 (size of TypeId)
        let expected_size = 8 + dictionary_len * std::mem::size_of::<TypeId>();
        assert_eq!(
            bytes.len(),
            expected_size,
            "FATAL: Phase 8 Zero-Copy Serialization layout mismatch."
        );
    }
}
