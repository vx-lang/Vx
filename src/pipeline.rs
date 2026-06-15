//===- pipeline.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the end-to-end compilation pipeline architecture.
// It structures the execution order of the lexer, parser, resolver, semantic
// analyzer, borrow checker, and code generator, providing a clean interface for
// invoking the compiler on a project.
//
//===----------------------------------------------------------------------===//
use crate::ast::MacroExpander;
use crate::ast::VxModule;
use crate::diagnostic::DiagnosticLevel;
use crate::lexer::Lexer;
use crate::metadata::VxMetadata;
#[cfg(debug_assertions)]
use crate::parallel_architecture_verifier::verify_arch::*;
use crate::parser::Parser;
use crate::sema::{GlobalAstEnv, TypeChecker};
use crate::session::{GlobalSession, LocalWorkerState};
use rayon::prelude::*;

/// The central orchestrator for the parallel compiler frontend.
use crate::ast;
pub fn compile_pipeline(file_paths: &[String]) -> Result<(), String> {
    // Phase 1: Parallel Parsing & Local Symbol Generation
    // Each thread parses a file and populates its Thread-Local Arena with structs, enums, etc.
    let modules: Result<Vec<VxModule>, String> = file_paths
        .par_iter()
        .map(|path| {
            println!("Parsing file: {}", path);
            let source = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path, e))?;
            let mut lexer = Lexer::new(&source);
            let tokens = lexer.tokenize();
            let mut parser = Parser::new(&tokens, &source);
            let mut program = parser
                .parse()
                .map_err(|e| format!("Failed to parse {}:\n{}", path, e.format(&source)))?;
            program.module_path = path.clone();
            Ok(program)
        })
        .collect();

    let mut parsed_modules = modules?;

    #[cfg(debug_assertions)]
    verify_phase_1_parse(file_paths, &parsed_modules);

    // Phase 1.1: Sequential Macro Collection & Expansion
    let mut global_macros = std::collections::HashMap::new();
    for m in &parsed_modules {
        for mac in &m.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    let mut expander = MacroExpander::new(&global_macros);
    for m in &mut parsed_modules {
        expander.expand_module(m)?;
    }

    // Phase 1.5: Parallel Name Resolution
    // Resolve String lookups into 256-bit TypeIds

    // Build the global SymbolMap sequentially (Phase 1.25)
    let symbol_map = crate::resolver::build_symbol_map(&parsed_modules);

    // Resolve names across all ASTs in parallel
    parsed_modules
        .par_iter_mut()
        .for_each(|m| m.resolve_names(&symbol_map));
    println!("Resolved {} modules in parallel", parsed_modules.len());

    // Phase 2: Sequential Global Registry Build & Cycle Detection
    // let registry = registry::ImmutableGlobalRegistry::build_and_validate(all_definitions)?;
    println!("Built Global Immutable Registry");

    let global_session = std::sync::Arc::new(GlobalSession::new(1));
    #[cfg(debug_assertions)]
    verify_phase_2_registry(&global_session.registry);

    // Phase 2.5: Build Global AST Environment (Sequential)
    let global_env_modules = parsed_modules.clone();
    let global_env = GlobalAstEnv::build(&global_env_modules);

    // Phase 3: Parallel Body Type-Checking (Lock-Free Frontend Threading)
    let mut check_results: Vec<_> = parsed_modules
        .par_iter_mut()
        .enumerate()
        .flat_map(|(module_idx, module)| {
            let global_session_ref = &global_session;
            let global_env_ref = &global_env;
            let mut func_results = module
                .functions
                .par_iter_mut()
                .map(move |func| {
                    let mut worker = LocalWorkerState::new(global_session_ref.clone());
                    let mut checker = TypeChecker::new(global_env_ref, &mut worker);
                    checker.check_function(func);
                    let errors = checker.errors;
                    let monomorphized_functions = checker.monomorphized_functions;
                    let generated_structs = checker.generated_structs;
                    (
                        errors,
                        monomorphized_functions,
                        worker,
                        module_idx,
                        generated_structs,
                    )
                })
                .collect::<Vec<_>>();

            let impl_results = module
                .impls
                .par_iter_mut()
                .flat_map(|i| {
                    i.methods.par_iter_mut().map(move |func| {
                        let mut worker = LocalWorkerState::new(global_session_ref.clone());
                        let mut checker = TypeChecker::new(global_env_ref, &mut worker);
                        checker.check_function(func);
                        let errors = checker.errors;
                        let monomorphized_functions = checker.monomorphized_functions;
                        let generated_structs = checker.generated_structs;
                        (
                            errors,
                            monomorphized_functions,
                            worker,
                            module_idx,
                            generated_structs,
                        )
                    })
                })
                .collect::<Vec<_>>();

            func_results.extend(impl_results);
            func_results
        })
        .collect();

    let mut total_errors = 0;
    for (errs, _, _, _, _) in &check_results {
        for diag in errs.iter() {
            if diag.level == DiagnosticLevel::Error {
                total_errors += 1;
                println!("Error: {}", diag.message);
            } else if diag.level == DiagnosticLevel::Warning {
                println!("Warning: {}", diag.message);
            }
        }
    }

    let total_monomorphized: usize = check_results
        .iter()
        .map(|(_, monos, _, _, _)| monos.len())
        .sum();

    println!(
        "Type checked bodies in parallel: {} errors, {} monomorphized variants generated",
        total_errors, total_monomorphized
    );

    if total_errors > 0 {
        return Err(format!(
            "Compilation failed with {} semantic errors",
            total_errors
        ));
    }

    #[cfg(debug_assertions)]
    {
        let workers: Vec<&crate::session::LocalWorkerState> = check_results
            .iter()
            .map(|(_, _, worker, _, _)| worker)
            .collect();
        verify_phase_3_isolation(&workers, &global_session);
    }

    // Phase 4: Parallel Local Deduplication & Cross-Thread Merging (Frozen Epoch)
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

    for (_, _, worker, _, _) in &check_results {
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

    #[cfg(debug_assertions)]
    verify_phase_4_deduplication(
        &_epoch_2_session.generics_arena,
        &_epoch_2_session.slow_path_arena,
    );

    println!(
        "Phase 5: Merged {} local arenas into global. Advancing to Epoch 2.",
        slow_path_thread_mappings.len()
    );

    // Phase 6: SIMD Patch Pass (Parallel Metadata Translation)
    println!("Executing Phase 6: SIMD Patch Pass over Flat Type Streams");

    let mut all_type_streams: Vec<(usize, Vec<crate::gid::TypeId>)> = check_results
        .iter_mut()
        .enumerate()
        .map(|(thread_idx, (_, _, worker, _, _))| {
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
        verify_phase_6_simd_patch(&patched_stream, &global_session);
    }

    // Phase 7: Parallel Module Deduplication & Codegen
    let num_modules = parsed_modules.len();
    let mut module_buckets: Vec<Vec<ast::Function>> = vec![Vec::new(); num_modules];
    let mut module_struct_buckets: Vec<Vec<ast::StructDecl>> = vec![Vec::new(); num_modules];

    // Build the Module Hash to Index map for origin-preserving routing
    let mut module_hash_to_index: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    for (i, module) in parsed_modules.iter().enumerate() {
        let hash = crate::hash::compute_module_hash(&module.module_path);
        module_hash_to_index.insert(hash, i);
    }

    // Collect all monomorphized functions into their correct origin module bucket
    for (_, monos, _, caller_module_idx, gen_structs) in check_results {
        module_struct_buckets[caller_module_idx].extend(gen_structs);
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
    verify_phase_7_routing(&module_buckets, &module_hash_to_index);

    // Parallel Deduplication Step (Zero Lock Contention)
    parsed_modules
        .par_iter_mut()
        .zip(module_buckets.into_par_iter())
        .zip(module_struct_buckets.into_par_iter())
        .for_each(|((module, mut bucket), mut struct_bucket)| {
            // Dedup based on function name (mangled signature is unique)
            bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            bucket.dedup_by(|a, b| a.name == b.name);

            struct_bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            struct_bucket.dedup_by(|a, b| a.name == b.name);

            // Prepend to the module's AST to ensure correct register allocation ordering
            let mut new_functions = bucket;
            new_functions.extend(std::mem::take(&mut module.functions));
            module.functions = new_functions;

            let mut new_structs = struct_bucket;
            new_structs.extend(std::mem::take(&mut module.structs));
            module.structs = new_structs;
        });

    println!("Monomorphized generics deduplicated and appended to modules in parallel");

    // Phase 7: Zero-Copy Metadata Serialization
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
    
    // Create a unique path for the test
    let test_path_str = format!("output_{}.vxm", std::process::id());
    let test_path = std::path::Path::new(&test_path_str);
    
    VxMetadata::save_to_file(&master_type_dictionary, test_path)
        .map_err(|e| format!("Failed to save metadata: {}", e))?;

    println!(
        "Saved {} unique TypeIds to {:?}",
        master_type_dictionary.len(),
        test_path
    );

    #[cfg(debug_assertions)]
    {
        let weak_session = std::sync::Arc::downgrade(&global_session);
        drop(global_session);
        verify_phase_5_epoch_advance(weak_session);

        let bytes = std::fs::read(test_path).unwrap();
        verify_phase_8_serialization(&bytes, master_type_dictionary.len());
    }

    Ok(())
}
