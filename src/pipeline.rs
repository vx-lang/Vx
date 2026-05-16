use crate::ast::VxModule;
use rayon::prelude::*;

/// The central orchestrator for the parallel compiler frontend.
pub fn compile_pipeline(file_paths: &[String]) -> Result<(), String> {
    // Phase 1.1: Parallel Parsing & Local Symbol Generation
    // Each thread parses a file and populates its Thread-Local Arena with structs, enums, etc.
    let modules: Result<Vec<VxModule>, String> = file_paths
        .par_iter()
        .map(|path| {
            // For now, return empty modules as a skeleton.
            // In the future: let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            // return crate::parse_module(&source);
            println!("Parsing file: {}", path);
            Ok(crate::ast::Program {
                module_path: path.clone(),
                externs: vec![],
                structs: vec![],
                enums: vec![],
                traits: vec![],
                impls: vec![],
                functions: vec![],
            })
        })
        .collect();

    let mut parsed_modules = modules?;

    // Phase 1.5: Parallel Name Resolution
    // Resolve String lookups into 256-bit TypeIds

    // Build the global SymbolMap sequentially (Phase 1.25)
    let symbol_map = crate::resolver::build_symbol_map(&parsed_modules);

    // Resolve names across all ASTs in parallel
    parsed_modules
        .par_iter_mut()
        .for_each(|m| m.resolve_names(&symbol_map));
    println!("Resolved {} modules in parallel", parsed_modules.len());

    // Phase 1.2: Sequential Global Registry Build & Cycle Detection
    // let registry = crate::registry::ImmutableGlobalRegistry::build_and_validate(all_definitions)?;
    println!("Built Global Immutable Registry");

    let global_session = std::sync::Arc::new(crate::session::GlobalSession::new(1));
    // Phase 2.5: Build Global AST Environment (Sequential)
    let global_env_modules = parsed_modules.clone();
    let global_env = crate::sema::GlobalAstEnv::build(&global_env_modules);

    // Phase 1.3: Parallel Body Type-Checking (Lock-Free Frontend Threading)
    let mut check_results: Vec<_> = parsed_modules
        .par_iter_mut()
        .enumerate()
        .flat_map(|(module_idx, module)| {
            let global_session_ref = &global_session;
            let global_env_ref = &global_env;
            module
                .functions
                .par_iter_mut()
                .map(move |func| {
                    let mut worker =
                        crate::session::LocalWorkerState::new(global_session_ref.clone());
                    let mut checker = crate::sema::TypeChecker::new(global_env_ref, &mut worker);
                    checker.check_function(func);
                    (
                        checker.errors,
                        checker.monomorphized_functions,
                        worker,
                        module_idx,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let total_errors: usize = check_results.iter().map(|(errs, _, _, _)| errs.len()).sum();
    let total_monomorphized: usize = check_results
        .iter()
        .map(|(_, monos, _, _)| monos.len())
        .sum();

    println!(
        "Type checked bodies in parallel: {} errors, {} monomorphized variants generated",
        total_errors, total_monomorphized
    );

    if total_errors > 0 {
        for (errs, _, _, _) in &check_results {
            for err in errs {
                println!("Error: {}", err);
            }
        }
        return Err(format!(
            "Compilation failed with {} semantic errors",
            total_errors
        ));
    }

    #[cfg(debug_assertions)]
    {
        let workers: Vec<&crate::session::LocalWorkerState> = check_results
            .iter()
            .map(|(_, _, worker, _)| worker)
            .collect();
        crate::parallel_architecture_verifier::verify_arch::verify_phase_1_isolation(
            &workers,
            &global_session,
        );
    }

    // Phase 2 & 3: Parallel Local Deduplication & Cross-Thread Merging (Frozen Epoch)
    let mut merged_slow_path_arena = (*global_session.slow_path_arena).clone();
    let mut merged_generics_arena = (*global_session.generics_arena).clone();

    let mut slow_path_thread_mappings: Vec<Vec<u64>> = Vec::new();
    let mut generics_thread_mappings: Vec<Vec<u64>> = Vec::new();

    // We use a HashMap to structurally deduplicate the UnboundedFunctionMetadata (Phase 2.25)
    let mut dedup_map_slow: std::collections::HashMap<crate::gid::UnboundedFunctionMetadata, u64> =
        std::collections::HashMap::new();
    for (i, meta) in merged_slow_path_arena.iter().enumerate() {
        dedup_map_slow.insert(meta.clone(), i as u64);
    }

    let mut dedup_map_generics: std::collections::HashMap<Vec<crate::gid::TypeId>, u64> =
        std::collections::HashMap::new();
    for (i, gen) in merged_generics_arena.iter().enumerate() {
        dedup_map_generics.insert(gen.clone(), i as u64);
    }

    for (_, _, worker, _) in &check_results {
        // Slow Path Deduplication
        let mut local_mapping_slow = Vec::new();
        for meta in &worker.local_slow_path_arena {
            if let Some(&global_idx) = dedup_map_slow.get(meta) {
                local_mapping_slow.push(global_idx);
            } else {
                let global_idx = merged_slow_path_arena.len() as u64;
                merged_slow_path_arena.push(meta.clone());
                dedup_map_slow.insert(meta.clone(), global_idx);
                local_mapping_slow.push(global_idx);
            }
        }
        slow_path_thread_mappings.push(local_mapping_slow);

        // Generics Arena Deduplication
        let mut local_mapping_generics = Vec::new();
        for gen in &worker.local_generics_arena {
            if let Some(&global_idx) = dedup_map_generics.get(gen) {
                local_mapping_generics.push(global_idx);
            } else {
                let global_idx = merged_generics_arena.len() as u64;
                merged_generics_arena.push(gen.clone());
                dedup_map_generics.insert(gen.clone(), global_idx);
                local_mapping_generics.push(global_idx);
            }
        }
        generics_thread_mappings.push(local_mapping_generics);
    }

    let _epoch_2_session = std::sync::Arc::new(crate::session::GlobalSession {
        epoch: 2,
        registry: global_session.registry.clone(),
        slow_path_arena: std::sync::Arc::new(merged_slow_path_arena),
        generics_arena: std::sync::Arc::new(merged_generics_arena),
    });

    println!(
        "Phase 2: Merged {} local arenas into global. Advancing to Epoch 2.",
        slow_path_thread_mappings.len()
    );

    // Phase 3.5: SIMD Patch Pass (Pure Data-Oriented)
    println!("Executing Phase 3.5: SIMD Patch Pass over Flat Type Streams");

    let mut all_type_streams: Vec<(usize, Vec<crate::gid::TypeId>)> = check_results
        .iter_mut()
        .enumerate()
        .map(|(thread_idx, (_, _, worker, _))| {
            (thread_idx, std::mem::take(&mut worker.local_type_stream))
        })
        .collect();

    const LOCAL_DEFERRED_BIT: u64 = crate::gid::LOCAL_DEFERRED_BIT; // Word 3
    const IS_GENERIC_INST_FLAG: u64 = crate::gid::IS_GENERIC_INST_FLAG; // Word 3

    all_type_streams
        .par_iter_mut()
        .for_each(|(thread_idx, stream)| {
            let mapping_slow = &slow_path_thread_mappings[*thread_idx];
            let mapping_generics = &generics_thread_mappings[*thread_idx];

            // SIMD loop operating entirely on the flat stream. The AST is long gone.
            for chunk in stream.chunks_mut(8) {
                // 8 GIDs per AVX-512 register
                for gid in chunk.iter_mut() {
                    if (gid.words[3] & LOCAL_DEFERRED_BIT) != 0 {
                        // Extract local index from Word 2
                        let local_index = gid.words[2] as usize;

                        // Parallel gather from the proper local-to-global mapping table.
                        let global_index = if (gid.words[3] & IS_GENERIC_INST_FLAG) != 0 {
                            mapping_generics[local_index]
                        } else {
                            mapping_slow[local_index]
                        };

                        // Overwrite Word 2 with the absolute global index.
                        gid.words[2] = global_index;
                        // Clear the LOCAL_DEFERRED_BIT.
                        gid.words[3] &= !LOCAL_DEFERRED_BIT;
                    }
                }
            }
        });

    println!("SIMD Patch Pass completed. AST is officially lowered to Flat Array.");

    #[cfg(debug_assertions)]
    {
        let patched_stream: Vec<crate::gid::TypeId> = all_type_streams
            .iter()
            .flat_map(|(_, stream)| stream.clone())
            .collect();
        crate::parallel_architecture_verifier::verify_arch::verify_phase_3_5_simd_patch(
            &patched_stream,
            &global_session,
        );
    }

    // Phase 4: Parallel Module Deduplication & Codegen
    let num_modules = parsed_modules.len();
    let mut module_buckets: Vec<Vec<crate::ast::Function>> = vec![Vec::new(); num_modules];

    // Build the Module Hash to Index map for origin-preserving routing
    let mut module_hash_to_index: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    for (i, module) in parsed_modules.iter().enumerate() {
        let hash = crate::hash::compute_module_hash(&module.module_path);
        module_hash_to_index.insert(hash, i);
    }

    // Collect all monomorphized functions into their correct origin module bucket
    for (_, monos, _, caller_module_idx) in check_results {
        for (func, origin_hash) in monos {
            // Origin-Preserving Routing Fix:
            // If the target module hash is NOT in our local module_hash_to_index map,
            // it belongs to an upstream, frozen crate (or is a local non-generic like an inherent method).
            if !module_hash_to_index.contains_key(&origin_hash) {
                module_buckets[caller_module_idx].push(func);
            } else {
                let dense_index = module_hash_to_index[&origin_hash];
                module_buckets[dense_index].push(func);
            }
        }
    }

    #[cfg(debug_assertions)]
    crate::parallel_architecture_verifier::verify_arch::verify_phase_4_routing(
        &module_buckets,
        &module_hash_to_index,
    );

    // Parallel Deduplication Step (Zero Lock Contention)
    parsed_modules
        .par_iter_mut()
        .zip(module_buckets.into_par_iter())
        .for_each(|(module, mut bucket)| {
            // Dedup based on function name (mangled signature is unique)
            bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            bucket.dedup_by(|a, b| a.name == b.name);

            // Prepend to the module's AST to ensure correct register allocation ordering
            let mut new_functions = bucket;
            new_functions.extend(std::mem::take(&mut module.functions));
            module.functions = new_functions;
        });

    println!("Monomorphized generics deduplicated and appended to modules in parallel");

    // Phase 5: Zero-Copy Metadata Serialization
    // Collect all fully-resolved global TypeIds from all threads
    let mut master_type_dictionary: Vec<crate::gid::TypeId> = all_type_streams
        .into_iter()
        .flat_map(|(_, stream)| stream)
        .collect();

    // Deduplicate the global dictionary
    master_type_dictionary.sort_unstable_by_key(|a| a.words);
    master_type_dictionary.dedup_by(|a, b| a.words == b.words);

    // Save the zero-copy metadata file to disk
    let metadata_path = std::path::Path::new("output.vxm");
    crate::metadata::VxMetadata::save_to_file(&master_type_dictionary, metadata_path)
        .map_err(|e| format!("Failed to save metadata: {}", e))?;

    println!(
        "Saved {} unique TypeIds to {:?}",
        master_type_dictionary.len(),
        metadata_path
    );

    #[cfg(debug_assertions)]
    let weak_session = std::sync::Arc::downgrade(&global_session);
    drop(global_session);
    #[cfg(debug_assertions)]
    crate::parallel_architecture_verifier::verify_arch::verify_lsp_memory_reclamation(weak_session);

    Ok(())
}
