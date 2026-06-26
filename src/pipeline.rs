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
use crate::syntax::MacroExpander;
use crate::syntax::VxModule;
use crate::diagnostic::DiagnosticLevel;
use crate::lexer::Lexer;
use crate::metadata::VxMetadata;
#[cfg(debug_assertions)]
use crate::parallel_architecture_verifier::verify_arch::*;
use crate::parser::Parser;
use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::session::{GlobalSession, LocalWorkerState};
use rayon::prelude::*;

/// The central orchestrator for the parallel compiler frontend.
use crate::syntax;

#[derive(Debug)]
pub enum PipelineError {
    IO(String),
    Parse(String),
    Semantic(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::IO(msg) => write!(f, "IO Error: {}", msg),
            PipelineError::Parse(msg) => write!(f, "Parse Error: {}", msg),
            PipelineError::Semantic(msg) => write!(f, "Semantic Error: {}", msg),
        }
    }
}

pub fn compile_pipeline(file_paths: &[String]) -> Result<(), PipelineError> {
    let mut parsed_modules = parse_phase(file_paths)?;
    macro_expansion_phase(&mut parsed_modules)?;
    name_resolution_phase(&mut parsed_modules);

    // Phase 2: Sequential Global Registry Build & Cycle Detection
    println!("Built Global Immutable Registry");
    let global_session = std::sync::Arc::new(GlobalSession::new(1));
    #[cfg(debug_assertions)]
    verify_phase_2_registry(&global_session.registry);

    let global_env_modules: Vec<VxModule> =
        parsed_modules.iter().map(|m| m.clone_signature()).collect();
    let global_env = GlobalAstEnv::build(&global_env_modules);

    let mut check_results = type_check_phase(&mut parsed_modules, &global_session, &global_env)?;

    let (merged_slow, merged_gen, merged_off, slow_mappings, gen_mappings) =
        deduplication_phase(&check_results, &global_session);

    let _epoch_2_session = std::sync::Arc::new(crate::session::GlobalSession {
        epoch: 2,
        registry: global_session.registry.clone(),
        slow_path_arena: std::sync::Arc::new(merged_slow),
        generics_arena: std::sync::Arc::new(merged_gen),
        generics_offsets: std::sync::Arc::new(merged_off),
    });

    #[cfg(debug_assertions)]
    verify_phase_4_deduplication(
        &_epoch_2_session.generics_arena,
        &_epoch_2_session.generics_offsets,
        &_epoch_2_session.slow_path_arena,
    );

    let mut all_type_streams = extract_type_streams(&mut check_results);

    simd_patch_phase(&mut all_type_streams, &slow_mappings, &gen_mappings);

    #[cfg(debug_assertions)]
    {
        let patched_stream: Vec<crate::gid::TypeId> = all_type_streams
            .iter()
            .flat_map(|(_, stream)| stream.clone())
            .collect();
        verify_phase_6_simd_patch(&patched_stream, &global_session);
    }

    codegen_and_metadata_phase(&mut parsed_modules, check_results, all_type_streams)?;

    #[cfg(debug_assertions)]
    {
        let weak_session = std::sync::Arc::downgrade(&global_session);
        drop(global_session);
        verify_phase_5_epoch_advance(weak_session);
    }

    Ok(())
}

fn parse_phase(file_paths: &[String]) -> Result<Vec<VxModule>, PipelineError> {
    let modules: Result<Vec<VxModule>, PipelineError> = file_paths
        .par_iter()
        .map(|path| {
            println!("Parsing file: {}", path);
            let source = std::fs::read_to_string(path)
                .map_err(|e| PipelineError::IO(format!("Failed to read {}: {}", path, e)))?;
            let mut lexer = Lexer::new(&source);
            let tokens = lexer.tokenize();
            let mut parser = Parser::new(&tokens, &source);
            let mut program = parser.parse().map_err(|e| {
                PipelineError::Parse(format!("Failed to parse {}:\n{}", path, e.format(&source)))
            })?;
            program.module_path = path.clone().into();
            Ok(program)
        })
        .collect();

    let parsed_modules = modules?;

    #[cfg(debug_assertions)]
    verify_phase_1_parse(file_paths, &parsed_modules);

    Ok(parsed_modules)
}

fn macro_expansion_phase(parsed_modules: &mut [VxModule]) -> Result<(), PipelineError> {
    let mut global_macros = std::collections::HashMap::new();
    for m in parsed_modules.iter() {
        for mac in &m.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    let mut expander = MacroExpander::new(&global_macros);
    for m in parsed_modules.iter_mut() {
        expander.expand_module(m).map_err(PipelineError::Parse)?;
    }
    Ok(())
}

fn name_resolution_phase(parsed_modules: &mut Vec<VxModule>) {
    let symbol_map = crate::resolver::build_symbol_map(parsed_modules);
    parsed_modules
        .par_iter_mut()
        .for_each(|m| m.resolve_names(&symbol_map));
    println!("Resolved {} modules in parallel", parsed_modules.len());
}

type TypeCheckResult = (
    crate::diagnostic::DiagnosticsVec,
    Vec<(syntax::Function, u64)>,
    LocalWorkerState,
    usize,
    Vec<syntax::StructDecl>,
);

fn type_check_phase(
    parsed_modules: &mut Vec<VxModule>,
    global_session: &std::sync::Arc<GlobalSession>,
    global_env: &GlobalAstEnv,
) -> Result<Vec<TypeCheckResult>, PipelineError> {
    let check_results: Vec<_> = parsed_modules
        .par_iter_mut()
        .enumerate()
        .flat_map(|(module_idx, module)| {
            let global_session_ref = global_session;
            let global_env_ref = global_env;
            let mut func_results = module
                .functions
                .par_iter_mut()
                .map(move |func| {
                    let mut worker = LocalWorkerState::new(global_session_ref.clone());
                    let mut checker = TypeChecker::new(global_env_ref, &mut worker);
                    checker.check_function(func);

                    let errors = checker.errors;
                    let monos = checker.monomorphized_functions;
                    let gen_structs = checker.generated_structs;

                    (errors, monos, worker, module_idx, gen_structs)
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
                        let monos = checker.monomorphized_functions;
                        let gen_structs = checker.generated_structs;

                        (errors, monos, worker, module_idx, gen_structs)
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
        return Err(PipelineError::Semantic(format!(
            "Compilation failed with {} semantic errors",
            total_errors
        )));
    }

    #[cfg(debug_assertions)]
    {
        let workers: Vec<&crate::session::LocalWorkerState> = check_results
            .iter()
            .map(|(_, _, worker, _, _)| worker)
            .collect();
        verify_phase_3_isolation(&workers, global_session);
    }

    Ok(check_results)
}

type DeduplicationResult = (
    Vec<crate::gid::UnboundedFunctionMetadata>,
    Vec<crate::gid::TypeId>,
    Vec<(usize, usize)>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
);

fn deduplication_phase(
    check_results: &[TypeCheckResult],
    global_session: &GlobalSession,
) -> DeduplicationResult {
    let mut merged_slow_path_arena = (*global_session.slow_path_arena).clone();
    let mut merged_generics_arena = (*global_session.generics_arena).clone();
    let mut merged_generics_offsets = (*global_session.generics_offsets).clone();

    let mut slow_path_thread_mappings: Vec<Vec<u64>> = Vec::new();
    let mut generics_thread_mappings: Vec<Vec<u64>> = Vec::new();

    let mut dedup_map_slow: std::collections::HashMap<crate::gid::UnboundedFunctionMetadata, u64> =
        std::collections::HashMap::new();
    for (i, meta) in merged_slow_path_arena.iter().enumerate() {
        dedup_map_slow.insert(meta.clone(), i as u64);
    }

    let mut dedup_map_generics: std::collections::HashMap<Vec<crate::gid::TypeId>, u64> =
        std::collections::HashMap::new();
    for (i, &(start, len)) in merged_generics_offsets.iter().enumerate() {
        let gen = merged_generics_arena[start..start + len].to_vec();
        dedup_map_generics.insert(gen, i as u64);
    }

    for (_, _, worker, _, _) in check_results {
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

        let mut local_mapping_generics = Vec::new();
        for &(start, len) in &worker.local_generics_offsets {
            let gen = worker.local_generics_arena[start..start + len].to_vec();
            if let Some(&global_idx) = dedup_map_generics.get(&gen) {
                local_mapping_generics.push(global_idx);
            } else {
                let global_idx = merged_generics_offsets.len() as u64;
                let new_start = merged_generics_arena.len();
                merged_generics_arena.extend_from_slice(&gen);
                merged_generics_offsets.push((new_start, len));
                dedup_map_generics.insert(gen, global_idx);
                local_mapping_generics.push(global_idx);
            }
        }
        generics_thread_mappings.push(local_mapping_generics);
    }

    println!(
        "Phase 5: Merged {} local arenas into global. Advancing to Epoch 2.",
        slow_path_thread_mappings.len()
    );

    (
        merged_slow_path_arena,
        merged_generics_arena,
        merged_generics_offsets,
        slow_path_thread_mappings,
        generics_thread_mappings,
    )
}

fn extract_type_streams(
    check_results: &mut [TypeCheckResult],
) -> Vec<(usize, Vec<crate::gid::TypeId>)> {
    check_results
        .iter_mut()
        .enumerate()
        .map(|(thread_idx, (_, _, worker, _, _))| {
            (thread_idx, std::mem::take(&mut worker.local_type_stream))
        })
        .collect()
}

fn simd_patch_phase(
    all_type_streams: &mut [(usize, Vec<crate::gid::TypeId>)],
    slow_path_thread_mappings: &[Vec<u64>],
    generics_thread_mappings: &[Vec<u64>],
) {
    println!("Executing Phase 6: SIMD Patch Pass over Flat Type Streams");
    const LOCAL_DEFERRED_BIT: u64 = crate::gid::LOCAL_DEFERRED_BIT; // Word 3
    const IS_GENERIC_INST_FLAG: u64 = crate::gid::IS_GENERIC_INST_FLAG; // Word 3

    all_type_streams
        .par_iter_mut()
        .for_each(|(thread_idx, stream)| {
            let mapping_slow = &slow_path_thread_mappings[*thread_idx];
            let mapping_generics = &generics_thread_mappings[*thread_idx];

            for chunk in stream.chunks_mut(8) {
                for gid in chunk.iter_mut() {
                    let w2 = gid.words[2];
                    let w3 = gid.words[3];
                    let is_deferred = (w3 & LOCAL_DEFERRED_BIT) != 0;

                    if is_deferred {
                        let local_index = w2 as usize;
                        let is_generic = (w3 & IS_GENERIC_INST_FLAG) != 0;

                        let slow_val = mapping_slow[local_index];
                        let gen_val = mapping_generics[local_index];

                        // Predicated selection avoiding branch
                        let is_gen_mask = -(is_generic as i64) as u64; // all 1s if true, 0s if false
                        let global_index = (gen_val & is_gen_mask) | (slow_val & !is_gen_mask);

                        gid.words[2] = global_index;
                        gid.words[3] &= !LOCAL_DEFERRED_BIT;
                    }
                }
            }
        });

    println!("SIMD Patch Pass completed. AST is officially lowered to Flat Array.");
}

fn codegen_and_metadata_phase(
    parsed_modules: &mut [VxModule],
    check_results: Vec<TypeCheckResult>,
    all_type_streams: Vec<(usize, Vec<crate::gid::TypeId>)>,
) -> Result<(), PipelineError> {
    let num_modules = parsed_modules.len();
    let mut module_buckets: Vec<Vec<syntax::Function>> = vec![Vec::new(); num_modules];
    let mut module_struct_buckets: Vec<Vec<syntax::StructDecl>> = vec![Vec::new(); num_modules];

    let mut module_hash_to_index: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    for (i, module) in parsed_modules.iter().enumerate() {
        let hash = crate::hash::compute_module_hash(&module.module_path);
        module_hash_to_index.insert(hash, i);
    }

    for (_, monos, _, caller_module_idx, gen_structs) in check_results {
        module_struct_buckets[caller_module_idx].extend(gen_structs);
        for (func, origin_hash) in monos {
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

    parsed_modules
        .par_iter_mut()
        .zip(module_buckets.into_par_iter())
        .zip(module_struct_buckets.into_par_iter())
        .for_each(|((module, mut bucket), mut struct_bucket)| {
            bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            bucket.dedup_by(|a, b| a.name == b.name);

            struct_bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            struct_bucket.dedup_by(|a, b| a.name == b.name);

            let mut new_functions = bucket;
            new_functions.extend(std::mem::take(&mut module.functions));
            module.functions = new_functions;

            let mut new_structs = struct_bucket;
            new_structs.extend(std::mem::take(&mut module.structs));
            module.structs = new_structs;
        });

    println!("Monomorphized generics deduplicated and appended to modules in parallel");

    let mut master_type_dictionary: Vec<crate::gid::TypeId> = all_type_streams
        .into_iter()
        .flat_map(|(_, stream)| stream)
        .collect();

    master_type_dictionary.sort_unstable_by_key(|a| a.words);
    master_type_dictionary.dedup_by(|a, b| a.words == b.words);

    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join(format!("output_{}.vxm", std::process::id()));

    VxMetadata::save_to_file(&master_type_dictionary, &test_path)
        .map_err(|e| PipelineError::IO(format!("Failed to save metadata: {}", e)))?;

    println!(
        "Saved {} unique TypeIds to {:?}",
        master_type_dictionary.len(),
        test_path
    );

    #[cfg(debug_assertions)]
    {
        let bytes = std::fs::read(test_path).unwrap();
        verify_phase_8_serialization(&bytes, master_type_dictionary.len());
    }

    Ok(())
}
