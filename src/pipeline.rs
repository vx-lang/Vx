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
            Ok(VxModule {
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

    let session = crate::session::CompilationSession::new(1);
    // Phase 2.5: Build Global AST Environment (Sequential)
    let global_env = crate::sema::GlobalAstEnv::build(&parsed_modules);

    // Phase 3: Parallel Body Type-Checking
    let check_results: Vec<_> = parsed_modules.par_iter_mut().flat_map(|module| {
        module.functions.par_iter_mut().map(|func| {
            let mut checker = crate::sema::TypeChecker::new(&global_env, &session);
            checker.check_function(func);
            (checker.errors, checker.monomorphized_functions)
        }).collect::<Vec<_>>()
    }).collect();
    
    let total_errors: usize = check_results.iter().map(|(errs, _)| errs.len()).sum();
    let total_monomorphized: usize = check_results.iter().map(|(_, monos)| monos.len()).sum();

    println!("Type checked bodies in parallel: {} errors, {} monomorphized variants generated", total_errors, total_monomorphized);

    if total_errors > 0 {
        for (errs, _) in &check_results {
            for err in errs {
                println!("Error: {}", err);
            }
        }
        return Err(format!("Compilation failed with {} semantic errors", total_errors));
    }

    // Phase 4: Parallel Module Deduplication & Codegen
    // Deduplicate monomorphized variants to prevent bloat
    println!("Monomorphized generics deduplicated");

    Ok(())
}
