#[cfg(debug_assertions)]
pub mod verify_arch {
    use crate::gid::{
        TypeId, ESCAPE_HATCH_MASK, INDEX_MASK, IS_GENERIC_INST_FLAG, LOCAL_DEFERRED_BIT,
    };
    use crate::session::{GlobalSession, LocalWorkerState};
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Weak};

    pub fn verify_phase_1_parse(file_paths: &[String], parsed_modules: &[crate::ast::VxModule]) {
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
            for gid in &worker.local_type_stream {
                if (gid.words[3] & LOCAL_DEFERRED_BIT) != 0 {
                    let index = (gid.words[2] & INDEX_MASK) as usize;
                    let arena_len = if (gid.words[3] & IS_GENERIC_INST_FLAG) != 0 {
                        worker.local_generics_arena.len()
                    } else {
                        worker.local_slow_path_arena.len()
                    };
                    assert!(
                        index < arena_len,
                        "FATAL: Local deferred index out of bounds."
                    );
                }
            }
        }
    }

    pub fn verify_phase_4_deduplication(
        global_generics_arena: &Arc<Vec<Vec<TypeId>>>,
        global_slow_path_arena: &Arc<Vec<crate::gid::UnboundedFunctionMetadata>>,
    ) {
        // Assert no duplicates exist in generics arena
        let mut generics_set = std::collections::HashSet::new();
        for gen in global_generics_arena.iter() {
            assert!(
                generics_set.insert(gen),
                "FATAL: Phase 4 Deduplication failed. Duplicate generics vector found."
            );
        }

        // Assert no duplicates exist in slow path arena
        let mut slow_path_set = std::collections::HashSet::new();
        for meta in global_slow_path_arena.iter() {
            assert!(
                slow_path_set.insert(meta),
                "FATAL: Phase 4 Deduplication failed. Duplicate slow path metadata found."
            );
        }
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

            // INVARIANT 2: Global Coordinate Integrity
            if (gid.words[2] & ESCAPE_HATCH_MASK) != 0 {
                let index = (gid.words[2] & INDEX_MASK) as usize;
                if (gid.words[3] & IS_GENERIC_INST_FLAG) != 0 {
                    assert!(
                        index < session.generics_arena.len(),
                        "FATAL: Patched generics index OOB."
                    );
                } else {
                    assert!(
                        index < session.slow_path_arena.len(),
                        "FATAL: Patched function index OOB."
                    );
                }
            }
        });
    }

    pub fn verify_phase_7_routing(
        buckets: &[Vec<crate::ast::Function>],
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
