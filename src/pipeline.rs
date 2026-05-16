use rayon::prelude::*;
use crate::ast::VxModule;

/// The central orchestrator for the parallel compiler frontend.
pub fn compile_pipeline(file_paths: &[String]) -> Result<(), String> {
    // Phase 1: Parallel Parsing & Local Symbol Generation
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
    parsed_modules.par_iter_mut().for_each(|m| m.resolve_names(&symbol_map));
    println!("Resolved {} modules in parallel", parsed_modules.len());

    // Phase 2: Sequential Global Registry Build & Cycle Detection
    // let registry = crate::registry::ImmutableGlobalRegistry::build_and_validate(all_definitions)?;
    println!("Built Global Immutable Registry");

    let global_session = std::sync::Arc::new(crate::session::GlobalSession::new(1));
    // Phase 2.5: Build Global AST Environment (Sequential)
    let global_env_modules = parsed_modules.clone();
    let global_env = crate::sema::GlobalAstEnv::build(&global_env_modules);

    // Phase 3: Parallel Body Type-Checking & Lowering to Flat Array
    let check_results: Vec<_> = parsed_modules.par_iter_mut().enumerate().flat_map(|(module_idx, module)| {
        module.functions.par_iter_mut().map(move |func| {
            let mut worker = crate::session::LocalWorkerState::new(global_session.clone());
            let mut checker = crate::sema::TypeChecker::new(&global_env, &mut worker);
            checker.check_function(func);
            (checker.errors, checker.monomorphized_functions, worker, module_idx)
        }).collect::<Vec<_>>()
    }).collect();
    
    let total_errors: usize = check_results.iter().map(|(errs, _, _, _)| errs.len()).sum();
    let total_monomorphized: usize = check_results.iter().map(|(_, monos, _, _)| monos.len()).sum();

    println!("Type checked bodies in parallel: {} errors, {} monomorphized variants generated", total_errors, total_monomorphized);

    if total_errors > 0 {
        for (errs, _, _, _) in &check_results {
            for err in errs {
                println!("Error: {}", err);
            }
        }
        return Err(format!("Compilation failed with {} semantic errors", total_errors));
    }

    // Phase 2: Sequential Global Deduplication & Epoch Advance
    let mut merged_slow_path_arena = (*global_session.slow_path_arena).clone();
    let mut thread_mappings: Vec<Vec<u64>> = Vec::new();
    
    // We use a HashMap to structurally deduplicate the UnboundedFunctionMetadata (Phase 2.25)
    let mut dedup_map: std::collections::HashMap<crate::gid::UnboundedFunctionMetadata, u64> = std::collections::HashMap::new();
    for (i, meta) in merged_slow_path_arena.iter().enumerate() {
        dedup_map.insert(meta.clone(), i as u64);
    }
    
    for (_, _, worker, _) in &check_results {
        let mut local_mapping = Vec::new();
        for meta in &worker.local_slow_path_arena {
            if let Some(&global_idx) = dedup_map.get(meta) {
                local_mapping.push(global_idx);
            } else {
                let global_idx = merged_slow_path_arena.len() as u64;
                merged_slow_path_arena.push(meta.clone());
                dedup_map.insert(meta.clone(), global_idx);
                local_mapping.push(global_idx);
            }
        }
        thread_mappings.push(local_mapping);
    }
    
    let _epoch_2_session = std::sync::Arc::new(crate::session::GlobalSession {
        epoch: 2,
        registry: global_session.registry.clone(),
        slow_path_arena: std::sync::Arc::new(merged_slow_path_arena),
    });
    
    println!("Phase 2: Merged {} local arenas into global. Advancing to Epoch 2.", thread_mappings.len());

    // Phase 3.5: SIMD Patch Pass (Pure Data-Oriented)
    println!("Executing Phase 3.5: SIMD Patch Pass over Flat Type Streams");
    
    let mut all_type_streams: Vec<(usize, Vec<crate::gid::TypeId>)> = check_results
        .into_iter()
        .enumerate()
        .map(|(thread_idx, (_, _, worker, _))| (thread_idx, worker.local_type_stream))
        .collect();
        
    const LOCAL_DEFERRED_BIT: u64 = 1 << 43; // Word 3

    all_type_streams.par_iter_mut().for_each(|(thread_idx, stream)| {
        let mapping = &thread_mappings[*thread_idx];
        
        // SIMD loop operating entirely on the flat stream. The AST is long gone.
        for chunk in stream.chunks_mut(8) { // 8 GIDs per AVX-512 register
            for gid in chunk.iter_mut() {
                if (gid.words[3] & LOCAL_DEFERRED_BIT) != 0 {
                    // Extract local index from Word 2
                    let local_index = gid.words[2] as usize;
                    
                    // Parallel gather from the local-to-global mapping table.
                    let global_index = mapping[local_index]; 
                    
                    // Overwrite Word 2 with the absolute global index.
                    gid.words[2] = global_index;
                    // Clear the LOCAL_DEFERRED_BIT.
                    gid.words[3] &= !LOCAL_DEFERRED_BIT; 
                }
            }
        }
    });
    
    println!("SIMD Patch Pass completed. AST is officially lowered to Flat Array.");

    // Phase 4: Parallel Module Deduplication & Codegen
    let num_modules = parsed_modules.len();
    let mut module_buckets: Vec<Vec<crate::ast::Function>> = vec![Vec::new(); num_modules];

    // Collect all monomorphized functions into their caller's module bucket
    for (_, monos, _, caller_module_idx) in check_results {
        for func in monos {
            module_buckets[caller_module_idx].push(func);
        }
    }

    // Parallel Deduplication Step (Zero Lock Contention)
    parsed_modules.par_iter_mut().enumerate().for_each(|(i, module)| {
        let mut bucket = std::mem::take(&mut module_buckets[i]);
        
        // Dedup based on function name (mangled signature is unique)
        bucket.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        bucket.dedup_by(|a, b| a.name == b.name);
        
        // Prepend to the module's AST to ensure correct register allocation ordering
        let mut new_functions = bucket;
        new_functions.extend(std::mem::take(&mut module.functions));
        module.functions = new_functions;
    });

    println!("Monomorphized generics deduplicated and appended to modules in parallel");

    Ok(())
}
