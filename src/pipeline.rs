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
use crate::diagnostic::DiagnosticLevel;
use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::lexer::Lexer;
use crate::metadata::VxMetadata;
#[cfg(debug_assertions)]
use crate::parallel_architecture_verifier::verify_arch::*;
use crate::parser::Parser;
use crate::session::{GlobalSession, LocalWorkerState};
use crate::syntax::MacroExpander;
use crate::syntax::VxModule;
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

/// The pipeline's progress chatter, suppressible with `VX_PIPELINE_QUIET=1`.
///
/// EVAL-ONLY (#295/#296). Some of these `println!`s sit *inside* rayon parallel-fors -- `parse_phase`
/// emits one per module -- and `println!` takes the global stdout mutex, so each is a serialisation
/// point in the region whose scaling the paper measures. Diagnostics (`Error:` / `Warning:`) are
/// deliberately left unguarded: a corpus that stops compiling must stay visible even under a quiet
/// measurement run.
macro_rules! chatter {
    ($($t:tt)*) => {
        if !crate::intern_mode::quiet() {
            println!($($t)*);
        }
    };
}

pub fn compile_pipeline(file_paths: &[String]) -> Result<(), PipelineError> {
    let mut parsed_modules = parse_phase(file_paths)?;
    macro_expansion_phase(&mut parsed_modules)?;
    let symbol_map = name_resolution_phase(&mut parsed_modules);

    // Phase 2: Sequential Global Registry Build & Cycle Detection (the freeze point). Builds the
    // frozen nominal-type registry from the resolved modules; an infinite-sized recursive struct
    // (a by-value cycle) fails here.
    let registry = build_frozen_registry_with(&parsed_modules, &symbol_map)?;
    chatter!(
        "Built Global Immutable Registry ({} types)",
        registry.layouts.len()
    );
    let global_session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));
    #[cfg(debug_assertions)]
    verify_phase_2_registry(&global_session.registry);

    let global_env_modules: Vec<VxModule> =
        parsed_modules.iter().map(|m| m.clone_signature()).collect();
    let mut global_env = GlobalAstEnv::build(&global_env_modules);
    // `clone_signature` above strips non-generic function bodies, so `build` could not summarize
    // their return provenance (#243). Refill from the full modules (bodies intact) before the
    // parallel check reads it. The map is frozen after this point — the per-function checkers only
    // read it, preserving the lock-free `type_check_phase`.
    global_env.annotate_return_provenances(&parsed_modules);

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

/// Run the frontend through the Phase 6 SIMD patch and return the flattened, patched flat type
/// stream — the 256-bit GIDs each worker lowered from its functions' type references (via
/// `emit_function_type_gids`), interned and remapped local->global. Exposed for determinism
/// testing: because GIDs are content hashes (module + symbol), not scheduling-dependent counters,
/// the same files must produce the same GID set regardless of `rayon`'s scheduling. Composes the
/// same phase functions as `compile_pipeline`, minus codegen.
pub fn compile_pipeline_type_stream(
    file_paths: &[String],
) -> Result<Vec<crate::gid::TypeId>, PipelineError> {
    // EVAL-ONLY (#295): phase timing, so the sweep can attribute wall clock to serial vs
    // parallel work rather than inferring it from a plateau.
    use crate::intern_mode::timed;
    let mut parsed_modules = timed("parse", || parse_phase(file_paths))?;
    timed("macro_expand", || {
        macro_expansion_phase(&mut parsed_modules)
    })?;
    // `name_resolution_phase` now hands back the symbol map so the freeze point can reuse it
    // instead of rebuilding it (main, c4c35e31); the timing wrapper carries the value through.
    let symbol_map = timed("name_resolution", || {
        name_resolution_phase(&mut parsed_modules)
    });

    let registry = timed("registry_freeze", || {
        build_frozen_registry_with(&parsed_modules, &symbol_map)
    })?;
    let global_session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));
    let global_env_modules: Vec<VxModule> =
        parsed_modules.iter().map(|m| m.clone_signature()).collect();
    let mut global_env = timed("env_build", || GlobalAstEnv::build(&global_env_modules));
    // `clone_signature` above strips non-generic function bodies, so `build` could not summarize
    // their return provenance (#243). Refill from the full modules (bodies intact) before the
    // parallel check reads it. The map is frozen after this point — the per-function checkers only
    // read it, preserving the lock-free `type_check_phase`.
    global_env.annotate_return_provenances(&parsed_modules);

    let mut check_results = timed("type_check", || {
        type_check_phase(&mut parsed_modules, &global_session, &global_env)
    })?;
    let (_slow, _gen, _off, slow_mappings, gen_mappings) = timed("dedup_barrier", || {
        deduplication_phase(&check_results, &global_session)
    });

    let mut all_type_streams = extract_type_streams(&mut check_results);
    timed("simd_patch", || {
        simd_patch_phase(&mut all_type_streams, &slow_mappings, &gen_mappings)
    });

    Ok(all_type_streams
        .into_iter()
        .flat_map(|(_, stream)| stream)
        .collect())
}

fn parse_phase(file_paths: &[String]) -> Result<Vec<VxModule>, PipelineError> {
    let modules: Result<Vec<VxModule>, PipelineError> = file_paths
        .par_iter()
        .map(|path| {
            chatter!("Parsing file: {}", path);
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
    // Collecting the macro table stays serial: it is one pass over macro *declarations*, which are
    // few, and it must complete before any expansion since a macro defined in one module is visible
    // to all.
    let mut global_macros = std::collections::HashMap::new();
    for m in parsed_modules.iter() {
        for mac in &m.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    // Expansion is per-module and shares nothing. `MacroExpander` holds a single `&HashMap` and no
    // mutable state, so its methods took `&mut self` without ever being able to use it; now that
    // they take `&self`, one expander is shared across the parallel-for and the borrow checker
    // proves the isolation rather than a comment asserting it.
    let expander = MacroExpander::new(&global_macros);
    parsed_modules
        .par_iter_mut()
        .try_for_each(|m| expander.expand_module(m).map_err(PipelineError::Parse))
}

/// Resolve names, and hand back the symbol map so the freeze point does not rebuild it.
///
/// `build_frozen_registry` needs the same map, and used to compute its own — two full passes over
/// every module's declarations per compile, both on the serial spine. They are identical by
/// construction: `build_symbol_map` reads only module paths and the *names* of top-level structs,
/// enums and traits, and `resolve_names` attaches GIDs to type *references*, adding no declarations
/// and renaming none. Returning it is what makes reusing it obviously safe rather than a claim a
/// reader has to check.
fn name_resolution_phase(parsed_modules: &mut Vec<VxModule>) -> crate::resolver::SymbolMap {
    let symbol_map = crate::resolver::build_symbol_map(parsed_modules);
    parsed_modules
        .par_iter_mut()
        .for_each(|m| m.resolve_names(&symbol_map));
    chatter!("Resolved {} modules in parallel", parsed_modules.len());
    symbol_map
}

type TypeCheckResult = (
    crate::diagnostic::DiagnosticsVec,
    Vec<(syntax::Function, u64)>,
    LocalWorkerState,
    usize,
    Vec<syntax::StructDecl>,
);

// ---- Phase 3: lowering AST types to the flat GID stream ------------------------------------
// As a worker finishes type-checking a function, it lowers the function's *type references* from
// AST `Type`s into 256-bit GIDs pushed onto its `local_type_stream` (docs/parallel_compiler_
// architecture.md §2.5). Nominal types contribute the settled GID `resolve_names` already attached
// to the AST; a generic instantiation contributes a *deferred* GID (Phase 5 interns it, Phase 6
// patches it). This is what makes `LocalWorkerState::local_type_stream` -- and thus the dedup and
// SIMD-patch phases and their verification hooks -- operate on real data rather than an empty Vec.

/// Harvest the GIDs referenced by a function's signature into the worker's flat type stream.
fn emit_function_type_gids(func: &syntax::Function, worker: &mut LocalWorkerState) {
    for (_, ty) in &func.params {
        emit_type_gid(ty, worker);
    }
    emit_type_gid(&func.return_type, worker);
}

/// Push the GID(s) a single type reference lowers to. Recurses through reference/pointer wrappers
/// to the underlying nominal type; a `GenericInstance` becomes a deferred GID.
fn emit_type_gid(ty: &syntax::Type, worker: &mut LocalWorkerState) {
    use syntax::Type;
    match ty {
        Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => {
            worker.local_type_stream.push(*id); // settled: no deferred bit
        }
        Type::Tensor(..) => {
            if let Some(id) = crate::hir::flatten::tensor_gid_of(ty) {
                worker.local_type_stream.push(id);
            }
        }
        Type::GenericInstance(base, args) => {
            // `collect::<Option<Vec<_>>>`, not `filter_map`. Dropping an argument the key depends
            // on is what made `Foo<Bar<i32>>` and `Foo<Baz<i32>>` share a GID (#305) and
            // `Grid<i32,2,3>` and `Grid<i32,4,5>` share one (#309): every unresolvable argument
            // collapsed to the same empty argument list. Declining to emit is the conservative
            // failure -- the instantiation contributes no GID rather than a wrong one shared with
            // an unrelated type.
            if let (Some(base_id), Some(arg_ids)) = (
                nominal_gid(base),
                args.iter()
                    .map(nominal_gid)
                    .collect::<Option<Vec<crate::gid::TypeId>>>(),
            ) {
                let deferred = mint_deferred_generic(worker, base_id, arg_ids);
                worker.local_type_stream.push(deferred);
            }
        }
        Type::Ref(inner, _)
        | Type::Borrow { inner, .. }
        | Type::Pointer(inner, _, _)
        | Type::Verified(inner)
        | Type::Pinned(inner, _) => emit_type_gid(inner, worker),
        Type::Function(args, ret) | Type::Closure(args, ret) => {
            for a in args {
                emit_type_gid(a, worker);
            }
            emit_type_gid(ret, worker);
        }
        _ => {}
    }
}

/// The GID a type resolves to when it appears as a generic argument or a wrapped nominal: a
/// resolved nominal's attached GID, or a stable synthetic GID for a primitive scalar (module 0 =
/// builtin) so that e.g. `List<i32>` and `List<f32>` are distinguishable instantiations.
fn nominal_gid(ty: &syntax::Type) -> Option<crate::gid::TypeId> {
    use syntax::Type;
    match ty {
        Type::Struct(_, id) | Type::Enum(_, id) => *id,
        // Single source of truth for the primitive GID scheme, shared with HIR lowering so a scalar
        // has the same identity in a signature and in a lowered body.
        Type::Scalar(elem) => Some(crate::hir::flatten::scalar_gid(elem)),
        Type::Tensor(..) => crate::hir::flatten::tensor_gid_of(ty),
        Type::Ref(inner, _)
        | Type::Borrow { inner, .. }
        | Type::Pointer(inner, _, _)
        | Type::Verified(inner)
        | Type::Pinned(inner, _) => nominal_gid(inner),
        // A *nested* instantiation is identified by its base plus a content digest of its own
        // arguments, computed recursively (#305). It cannot be an arena index: the inner
        // instantiation holds only a worker-local index until the barrier, so two workers would
        // key the outer instantiation differently and dedup would never unify them. A digest is
        // the same in every worker the moment it is computed.
        Type::GenericInstance(base, args) => {
            let base_id = nominal_gid(base)?;
            let arg_ids: Vec<crate::gid::TypeId> =
                args.iter().map(nominal_gid).collect::<Option<_>>()?;
            let mut id = base_id;
            id.words[2] = crate::gid::ESCAPE_HATCH_MASK | crate::gid::generic_digest(&arg_ids);
            id.words[3] |= crate::gid::IS_GENERIC_INST_FLAG;
            Some(id)
        }
        // Const generic arguments (#309). Identity is the constant's *structure*, via a
        // span-free rendering -- `Type::Const`'s `Mangle` arm uses `format!("{:?}", expr)`, whose
        // Debug output embeds spans, so the same constant written at two source locations would
        // hash differently.
        Type::Const(expr) => Some(crate::gid::TypeId::new(
            0,
            crate::hash::DefPath::Named(&format!("$const::{}", const_identity_key(expr)))
                .compute_symbol_hash(),
            0,
            0,
        )),
        // A type *parameter* is an identity too: `Foo<T>` and `Foo<U>` are different type
        // expressions, and both differ from `Foo<i32>`. Dropping it is what made every
        // unresolvable argument collapse to the same empty key.
        Type::Generic(name, _) => Some(crate::gid::TypeId::new(
            0,
            crate::hash::DefPath::Named(&format!("$typaram::{name}")).compute_symbol_hash(),
            0,
            0,
        )),
        Type::Simd(elem, lanes) => Some(crate::gid::TypeId::new(
            0,
            crate::hash::DefPath::Named(&format!("$simd::{elem:?}::{lanes}")).compute_symbol_hash(),
            0,
            0,
        )),
        Type::Function(args, ret) | Type::Closure(args, ret) => {
            let mut ids: Vec<crate::gid::TypeId> =
                args.iter().map(nominal_gid).collect::<Option<_>>()?;
            ids.push(nominal_gid(ret)?);
            let mut id = crate::gid::TypeId::new(
                0,
                crate::hash::DefPath::Named(if matches!(ty, Type::Function(..)) {
                    "$fnty"
                } else {
                    "$closurety"
                })
                .compute_symbol_hash(),
                0,
                0,
            );
            id.words[2] = crate::gid::ESCAPE_HATCH_MASK | crate::gid::generic_digest(&ids);
            Some(id)
        }
        Type::Matrix => Some(crate::gid::TypeId::new(
            0,
            crate::hash::DefPath::Named("$matrix").compute_symbol_hash(),
            0,
            0,
        )),
        // `Module` and `Unknown` genuinely have no type identity. They should not reach a generic
        // argument position; `emit_type_gid` now declines the whole instantiation rather than
        // silently interning it with the argument missing.
        Type::Module(..) | Type::Unknown => None,
    }
}

/// A span-free structural rendering of a const-generic argument, for identity only.
///
/// Recurses through the shapes a const argument actually takes -- literals, references to const
/// parameters, and arithmetic over them (`Grid<i32, R, C>`, `Grid<i32, 2, 3>`, `Buf<N*2>`). Anything
/// else falls back to a variant tag, which keeps distinct shapes from colliding without claiming to
/// distinguish them precisely.
fn const_identity_key(expr: &syntax::Expr) -> String {
    use syntax::Expr;
    match expr {
        Expr::Number(n) => n.value.to_string(),
        Expr::Identifier(i) => i.name.to_string(),
        Expr::BinaryOp(b) => format!(
            "({} {:?} {})",
            const_identity_key(&b.lhs),
            b.op,
            const_identity_key(&b.rhs)
        ),
        Expr::UnaryOp(u) => format!("({:?} {})", u.op, const_identity_key(&u.expr)),
        other => format!("{:?}", std::mem::discriminant(other)),
    }
}

/// Mint a deferred generic GID: stash the argument GIDs in the worker's local generics arena and
/// return a GID whose word 2 is that local offset index, with `LOCAL_DEFERRED_BIT` +
/// `IS_GENERIC_INST_FLAG` set in word 3. Phase 5 (`deduplication_phase`) interns the arena and
/// Phase 6 (`simd_patch_phase`) remaps word 2 to the global offset index and clears the deferred
/// bit -- the local->global handoff the escape-hatch design exists for.
fn mint_deferred_generic(
    worker: &mut LocalWorkerState,
    base: crate::gid::TypeId,
    args: Vec<crate::gid::TypeId>,
) -> crate::gid::TypeId {
    mint_generic_in_mode(crate::intern_mode::mode(), worker, base, args)
}

/// The minting rule, with the strategy passed in rather than read from process-global state.
///
/// Tests pick a mode by calling this directly. Reading the global inside would make every
/// mode-sensitive test racy against every other: `cargo test` runs tests in parallel threads within
/// one process, so a test that flipped the mode could change what a concurrently running test
/// minted. That is not hypothetical -- it produced an intermittent failure before this split.
fn mint_generic_in_mode(
    mode: crate::intern_mode::InternMode,
    worker: &mut LocalWorkerState,
    base: crate::gid::TypeId,
    args: Vec<crate::gid::TypeId>,
) -> crate::gid::TypeId {
    if mode == crate::intern_mode::InternMode::Content {
        let mut id = base;
        id.set_generic_digest(crate::gid::generic_digest(&args));
        return id;
    }

    let start = worker.local_generics_arena.len();
    let len = args.len();
    worker.local_generics_arena.extend(args);
    let offset_index = worker.local_generics_offsets.len() as u64;
    worker.local_generics_offsets.push((start, len));

    let mut id = base; // reuse the base type's module (word 0) + symbol (word 1)
    id.set_arena_index(
        offset_index,
        crate::gid::Word2Arena::Generics,
        crate::gid::Word2Scope::Local,
    );
    id
}

// ---- Phase 2: the freeze point (build + validate the frozen registry) ----------------------
// Collects every module's top-level structs/enums into `TypeDefinition`s -- their GIDs (from the
// symbol map, matching what `resolve_names` attached) plus by-value dependency edges -- and runs
// cycle detection. An infinite-sized recursive struct (a by-value cycle) is a compile error. The
// resulting `ImmutableGlobalRegistry` is frozen into the `GlobalSession` and shared read-only.

/// Build and validate the frozen registry from post-`resolve_names` modules.
pub fn build_frozen_registry(
    modules: &[VxModule],
) -> Result<crate::registry::ImmutableGlobalRegistry, PipelineError> {
    let symbol_map = crate::resolver::build_symbol_map(modules);
    build_frozen_registry_with(modules, &symbol_map)
}

/// The freeze point, reusing a symbol map the caller already built.
///
/// The pipeline resolves names immediately before freezing the registry, and both steps need the
/// same map; computing it twice put a second full pass over every module's declarations on the
/// serial spine for no benefit. Callers outside the pipeline keep the one-argument form above.
pub fn build_frozen_registry_with(
    modules: &[VxModule],
    symbol_map: &crate::resolver::SymbolMap,
) -> Result<crate::registry::ImmutableGlobalRegistry, PipelineError> {
    use crate::registry::TypeDefinition;

    // Index every nominal decl by its GID so nested by-value fields resolve
    // cross-module during layout computation (#199). Name resolution ran in the
    // prior phase, so field `Type::Struct(_, Some(gid))` GIDs are populated.
    let mut gid_structs = rustc_hash::FxHashMap::default();
    let mut gid_enums = rustc_hash::FxHashMap::default();
    for module in modules {
        let Some(mod_syms) = symbol_map.get(&module.module_path) else {
            continue;
        };
        for s in &module.structs {
            if let Some(&id) = mod_syms.get(&s.name) {
                gid_structs.insert(id, s);
            }
        }
        for e in &module.enums {
            if let Some(&id) = mod_syms.get(&e.name) {
                gid_enums.insert(id, e);
            }
        }
    }
    let mut layout = crate::layout::LayoutComputer::new(gid_structs, gid_enums);

    let mut defs: Vec<TypeDefinition> = Vec::new();
    for module in modules {
        let Some(mod_syms) = symbol_map.get(&module.module_path) else {
            continue;
        };
        for s in &module.structs {
            let Some(&id) = mod_syms.get(&s.name) else {
                continue;
            };
            let deps = s
                .fields
                .iter()
                .filter_map(|(_, ty)| by_value_nominal_gid(ty))
                .collect();
            // An incomputable layout (generic, tensor field, by-value cycle) keeps
            // the earlier 0/0 stub; the cycle case is reported by build_and_validate.
            let (size_bytes, align_bytes, fields) = layout
                .layout_of(id)
                .map(|l| (l.size, l.align, l.fields))
                .unwrap_or((0, 0, Vec::new()));
            defs.push(TypeDefinition {
                id,
                name: s.name.to_string(),
                size_bytes,
                align_bytes,
                fields,
                by_value_dependencies: deps,
            });
        }
        for e in &module.enums {
            let Some(&id) = mod_syms.get(&e.name) else {
                continue;
            };
            let deps = e
                .variants
                .iter()
                .flat_map(|(_, payload)| payload.iter().flatten())
                .filter_map(by_value_nominal_gid)
                .collect();
            let (size_bytes, align_bytes) = layout
                .layout_of(id)
                .map(|l| (l.size, l.align))
                .unwrap_or((0, 0));
            defs.push(TypeDefinition {
                id,
                name: e.name.to_string(),
                size_bytes,
                align_bytes,
                fields: Vec::new(),
                by_value_dependencies: deps,
            });
        }
    }

    let mut registry = crate::registry::ImmutableGlobalRegistry::build_and_validate(defs)
        .map_err(PipelineError::Semantic)?;

    // Variant ordinals for payload-free (C-like) enums, so the flat lowerer resolves `Color::Green`
    // to its discriminant (`1`) and lowers a `match` over it (#227). A data-carrying (tagged-union)
    // enum is skipped -- constructing/matching it stays on the AST path.
    for module in modules {
        for e in &module.enums {
            let payload_free = e
                .variants
                .iter()
                .all(|(_, payload)| payload.as_ref().is_none_or(|p| p.is_empty()));
            if payload_free {
                let variants = e.variants.iter().map(|(name, _)| name.clone()).collect();
                registry
                    .enum_variants
                    .entry(e.name.clone())
                    .or_insert(variants);
            }
        }
    }

    // Base struct field types, keyed by the struct's GID (#291), so the flat path can substitute a
    // monomorphized instance's type arguments into a generic field type and recover a pointer
    // field's pointee (`Vec<i32>`'s `data : *mut T` -> `*mut i32`) — the field AST types the frozen
    // `layouts` erase to `Opaque`. The GID key keeps two modules' same-named structs distinct;
    // consumers resolve a bare name through `struct_fields_of` (#242).
    for module in modules {
        let mod_syms = symbol_map.get(&module.module_path);
        for s in &module.structs {
            let Some(&gid) = mod_syms.and_then(|m| m.get(&s.name)) else {
                continue;
            };
            registry.structs.insert(
                gid,
                crate::registry::StructFields {
                    generics: s.generics.iter().map(|g| g.name().into()).collect(),
                    fields: s.fields.clone(),
                },
            );
        }
        // Data-carrying enum decls, so the flat path can synthesize a monomorphized instance's
        // `{ tag, payload }` layout by substituting its type args into the variant payload types
        // (`Option<i32>` -> `{ i32, i32 }`). (#242)
        for e in &module.enums {
            registry.enum_data.insert(
                e.name.clone(),
                crate::registry::EnumData {
                    generics: e.generics.iter().map(|g| g.name().into()).collect(),
                    variants: e
                        .variants
                        .iter()
                        .map(|(n, p)| (n.clone(), p.clone().unwrap_or_default()))
                        .collect(),
                },
            );
        }
    }

    // Function signatures for call resolution in the flat HIR (#198): name -> (GID, return
    // type), GID minted with the resolver's formula. A name defined in more than one module with
    // distinct GIDs is ambiguous for the name-keyed map, so it is dropped rather than resolved wrong.
    let mut ambiguous_fns = std::collections::HashSet::new();
    for module in modules {
        let module_hash = crate::hash::compute_module_hash(&module.module_path);
        for f in &module.functions {
            let gid = crate::gid::TypeId::new(
                module_hash,
                crate::hash::DefPath::Named(f.name.as_ref()).compute_symbol_hash(),
                0,
                0,
            );
            match registry.fn_sigs.get(&f.name) {
                Some(existing) if existing.gid != gid => {
                    ambiguous_fns.insert(f.name.clone());
                }
                _ => {
                    registry.fn_sigs.insert(
                        f.name.clone(),
                        crate::registry::FnSig {
                            gid,
                            params: f.params.iter().map(|(_, t)| t.clone()).collect(),
                            ret_ty: f.return_type.clone(),
                            // Precompute the return-provenance code from the AST body now (it is
                            // present here), so a downstream compile that only has this interface can
                            // still refine the reborrow decision for a call to `f` (#265 step 7).
                            ret_prov: crate::hir::provenance::encode_return_provenance(
                                &crate::hir::provenance::compute_return_provenance(f),
                            ),
                        },
                    );
                }
            }
        }
        // `extern` declarations are callees too (e.g. libm `sqrtf`): register their signatures so the
        // flat HIR resolves a call to one (`flatten::lower_call`) and the emitter can declare + call it.
        // They have no Vx body -- the emitter emits a `func.func private` decl and the JIT links the
        // symbol (libm via `-lm`, `libvx_std_core`, ...).
        //
        // Unlike a Vx function (whose body is module-scoped), an `extern` names a *global* symbol that
        // links by name, so its identity is the name alone — module 0 (the builtin/global namespace),
        // not the declaring module's hash. That way the *same* `extern fn` declared in both a program
        // and an imported stdlib module (e.g. `vx_stdout_write` in a user file and in `std::io`) shares
        // one GID and is not a spurious "ambiguous" pair. A genuine conflict — the same name with a
        // different signature — is still flagged ambiguous (its GID collides but the `FnSig` differs).
        for ext in &module.externs {
            let gid = crate::gid::TypeId::new(
                0,
                crate::hash::DefPath::Named(ext.name.as_ref()).compute_symbol_hash(),
                0,
                0,
            );
            match registry.fn_sigs.get(&ext.name) {
                Some(existing) if existing.gid != gid || existing.ret_ty != ext.return_type => {
                    ambiguous_fns.insert(ext.name.clone());
                }
                _ => {
                    registry.fn_sigs.insert(
                        ext.name.clone(),
                        crate::registry::FnSig {
                            gid,
                            params: ext.params.iter().map(|(_, t)| t.clone()).collect(),
                            ret_ty: ext.return_type.clone(),
                            // An `extern` is an opaque foreign symbol with no analysable body, so its
                            // return provenance is unknown: the conservative top (any parameter).
                            ret_prov: crate::hir::provenance::encode_return_provenance(
                                &crate::hir::provenance::ReturnProvenance::AnyParam,
                            ),
                        },
                    );
                }
            }
        }
    }
    for name in ambiguous_fns {
        registry.fn_sigs.remove(&name);
    }

    // Method signatures keyed by (receiver GID, method name), minted from `impl` blocks (#218). This
    // is the GID-keyed method table that lets the frontend resolve `x.exp()` without walking borrowed
    // AST `ImplBlock`s in `GlobalAstEnv`. The method's GID is minted from its mangled name
    // (`<type>$<method>`, e.g. `f32$exp`); the key is `(receiver GID, unmangled method name)`.
    use crate::syntax::types::Mangle;
    let mut ambiguous_methods = std::collections::HashSet::new();
    for module in modules {
        let module_hash = crate::hash::compute_module_hash(&module.module_path);
        for imp in &module.impls {
            let Some(recv) = method_receiver_gid(&imp.target_type) else {
                continue; // generic/tensor/unresolved receiver -- deferred
            };
            for m in &imp.methods {
                let mangled = format!("{}${}", imp.target_type.mangle(), m.name);
                let gid = crate::gid::TypeId::new(
                    module_hash,
                    crate::hash::DefPath::Named(mangled.as_str()).compute_symbol_hash(),
                    0,
                    0,
                );
                let key = (recv, m.name.clone());
                match registry.methods.get(&key) {
                    // Same method minted in >1 module with distinct GIDs is ambiguous -- drop it,
                    // mirroring the `fn_sigs` policy.
                    Some(existing) if existing.gid != gid => {
                        ambiguous_methods.insert(key);
                    }
                    _ => {
                        registry.methods.insert(
                            key,
                            crate::registry::FnSig {
                                gid,
                                params: m.params.iter().map(|(_, t)| t.clone()).collect(),
                                ret_ty: m.return_type.clone(),
                                ret_prov: crate::hir::provenance::encode_return_provenance(
                                    &crate::hir::provenance::compute_return_provenance(m),
                                ),
                            },
                        );
                    }
                }
            }
        }
    }
    for key in ambiguous_methods {
        registry.methods.remove(&key);
    }

    Ok(registry)
}

/// The receiver GID for an `impl` target type, for the registry method table: a scalar's content-hash
/// GID or a resolved nominal's GID. Generic/tensor/unresolved receivers are `None` (deferred).
fn method_receiver_gid(ty: &crate::syntax::Type) -> Option<crate::gid::TypeId> {
    use crate::syntax::{ElementType, Type};
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(crate::hir::flatten::scalar_gid(e)),
        Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => Some(*id),
        _ => None,
    }
}

/// Build a serialized `.vxlib` module interface for `modules`: the frozen registry (types, layouts,
/// signatures) plus the flat-HIR bodies of the non-generic free functions that lower completely and
/// portably. This is the artifact producer -- a downstream compile deserializes it and resolves + links
/// the module with no AST (#220, `docs/discussions/implementation_plans/vxlib_bodies_and_loader.md`).
pub fn emit_module_interface(modules: &[VxModule]) -> Result<Vec<u8>, PipelineError> {
    emit_module_interface_reporting(modules).map(|(bytes, _)| bytes)
}

/// [`emit_module_interface`] plus the codec's per-table encoded/skipped accounting, so
/// `--emit-interface` can report an incomplete artifact at produce time (#292).
pub fn emit_module_interface_reporting(
    modules: &[VxModule],
) -> Result<(Vec<u8>, crate::metadata::InterfaceEmitReport), PipelineError> {
    let mut registry = build_frozen_registry(modules)?;
    harvest_bodies(&mut registry, modules)?;
    Ok(crate::metadata::serialize_registry_interface_reporting(
        &registry,
    ))
}

/// Lower each non-generic free function to flat HIR and stash a portable `FnBody` in `registry.bodies`,
/// keyed by the function's GID (from `fn_sigs`). A function that declines to lower, or whose type stream
/// still holds a per-compilation deferred GID, is skipped -- fail-closed, so the artifact carries only
/// linkable bodies. (Methods await flat method-call lowering, #217.)
fn harvest_bodies(
    registry: &mut crate::registry::ImmutableGlobalRegistry,
    modules: &[VxModule],
) -> Result<(), PipelineError> {
    // A separate, deterministic registry build backs the lowering session (identical GIDs); the frozen
    // registry we attach bodies to is not `Clone`.
    let session_reg = build_frozen_registry(modules)?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, session_reg));
    for module in modules {
        for func in &module.functions {
            if !func.generics.is_empty() {
                continue;
            }
            let Some(gid) = registry.fn_sigs.get(&func.name).map(|s| s.gid) else {
                continue; // ambiguous across modules -> dropped from fn_sigs
            };
            let mut worker = LocalWorkerState::new(session.clone());
            if !crate::hir::flatten::lower_function_to_hir(func, &mut worker) {
                continue;
            }
            if worker
                .local_type_stream
                .iter()
                .any(|t| t.is_local_deferred())
            {
                continue;
            }
            registry.bodies.insert(
                gid,
                crate::registry::FnBody {
                    name: func.name.clone(),
                    params: func.params.iter().map(|(_, t)| t.clone()).collect(),
                    ret_ty: func.return_type.clone(),
                    hir: worker.local_hir_stream,
                    types: worker.local_type_stream,
                },
            );
        }
    }
    Ok(())
}

/// The GID of a type held *by value* (a nominal struct/enum, seen through location wrappers that
/// add no indirection). A `Ref`/`Pointer`/`Borrow` breaks containment (and any cycle), so it is
/// not a by-value dependency and returns `None`. (Generic instantiations are not yet followed for
/// by-value cycle detection.)
fn by_value_nominal_gid(ty: &syntax::Type) -> Option<crate::gid::TypeId> {
    use syntax::Type;
    match ty {
        Type::Struct(_, id) | Type::Enum(_, id) => *id,
        Type::Pinned(inner, _) | Type::Verified(inner) => by_value_nominal_gid(inner),
        _ => None,
    }
}

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

                    // Lower this function's type references to the flat GID stream (Phase 3).
                    emit_function_type_gids(func, &mut worker);
                    // Lower the body to flat HIR bytecode; atomic — a no-op for functions
                    // outside the supported subset.
                    crate::hir::flatten::lower_function_to_hir(func, &mut worker);
                    #[cfg(debug_assertions)]
                    crate::hir::flatten::verify_hir_stream(&worker);

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

                        emit_function_type_gids(func, &mut worker);
                        crate::hir::flatten::lower_function_to_hir(func, &mut worker);
                        #[cfg(debug_assertions)]
                        crate::hir::flatten::verify_hir_stream(&worker);

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

    chatter!(
        "Type checked bodies in parallel: {} errors, {} monomorphized variants generated",
        total_errors,
        total_monomorphized
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

    chatter!(
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
    // The pass runs in every mode, and the *classification* decides what work exists. An earlier
    // version of this file early-returned for `locked`, which also skipped patching the slow-path
    // (lifetime) arena -- those GIDs are still minted worker-local in every mode, so skipping the
    // scan would have left unresolved local indices in the stream once a corpus exercised them.
    //
    // Charging the modes correctly falls out of the codec rather than from a flag: `locked` mints
    // final global generic indices and `content` mints digests, so neither has a Local-scope
    // generic GID for the loop to rewrite, while `deferred` does. The scan itself is shared work
    // that every mode needs for the slow path.
    chatter!("Executing Phase 6: SIMD Patch Pass over Flat Type Streams");
    use crate::gid::{Word2, Word2Scope};

    all_type_streams
        .par_iter_mut()
        .for_each(|(thread_idx, stream)| {
            let mapping_slow = &slow_path_thread_mappings[*thread_idx];
            let mapping_generics = &generics_thread_mappings[*thread_idx];

            for chunk in stream.chunks_mut(8) {
                for gid in chunk.iter_mut() {
                    // Only worker-local arena indices need patching to their global offset. The
                    // codec (`classify_word2`) is the single decoder: it masks the index and routes
                    // by arena, so the slow-path and generics index spaces (independent) can't be
                    // confused, and a fast-path lifetime bitfield is never touched.
                    if let Word2::Index {
                        index,
                        arena,
                        scope: Word2Scope::Local,
                    } = gid.classify_word2()
                    {
                        let global_index = match arena {
                            crate::gid::Word2Arena::Generics => mapping_generics[index as usize],
                            crate::gid::Word2Arena::SlowMeta => mapping_slow[index as usize],
                        };
                        gid.set_arena_index(global_index, arena, Word2Scope::Global);
                    }
                }
            }
        });

    chatter!("SIMD Patch Pass completed. AST is officially lowered to Flat Array.");
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

    chatter!("Monomorphized generics deduplicated and appended to modules in parallel");

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

    chatter!(
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

#[cfg(test)]
mod gid_stream_tests {
    use super::*;
    use crate::gid::{TypeId, LOCAL_DEFERRED_BIT};
    use std::sync::Arc;

    /// Deferred and content-addressed identity must describe the *same program*.
    ///
    /// Equality is checked after canonical renumbering rather than on raw bytes: `deferred` carries
    /// arena indices assigned at the barrier, `content` carries digests of the argument list, so the
    /// two never agree word-for-word. What must agree is which positions in the stream denote the
    /// same instantiation, which is what equality of programs actually means here.
    #[test]
    fn deferred_and_content_agree_under_canonical_renumbering() {
        use crate::intern_mode::{canonicalise_stream, InternMode};

        // Two distinct instantiations, the second repeated: exercises a fresh mint and a repeat,
        // and gives the canonical ranks something to disagree about if they can.
        let arg_a = vec![TypeId::new(0, 0x1111, 0, 0)];
        let arg_b = vec![TypeId::new(0, 0x2222, 0, 0)];
        let base = TypeId::new(0xAAAA, 0xBBBB, 0, 0);

        let run = |mode: InternMode| -> (Vec<TypeId>, Vec<TypeId>, Vec<(usize, usize)>) {
            let session = Arc::new(GlobalSession::new(1));
            let mut worker = LocalWorkerState::new(session.clone());
            for args in [&arg_a, &arg_b, &arg_a] {
                let id = mint_generic_in_mode(mode, &mut worker, base, args.clone());
                worker.local_type_stream.push(id);
            }
            let mut results: Vec<TypeCheckResult> = vec![(
                crate::diagnostic::DiagnosticsVec::new(),
                Vec::new(),
                worker,
                0,
                Vec::new(),
            )];
            let (_s, arena, offsets, slow, gen) = deduplication_phase(&results, &session);
            let mut streams = extract_type_streams(&mut results);
            simd_patch_phase(&mut streams, &slow, &gen);
            (streams.remove(0).1, arena, offsets)
        };

        let (deferred_stream, d_arena, d_offsets) = run(InternMode::Deferred);
        let (content_stream, c_arena, c_offsets) = run(InternMode::Content);

        // Content addressing stages nothing for the barrier -- the claim it exists to make.
        assert!(
            c_arena.is_empty() && c_offsets.is_empty(),
            "content mode must leave the global generics arena empty"
        );

        // Neither leaves a deferred GID: deferred patches them, content never creates them.
        for id in deferred_stream.iter().chain(content_stream.iter()) {
            assert_eq!(
                id.words[3] & LOCAL_DEFERRED_BIT,
                0,
                "no deferred bit should survive either mode"
            );
        }

        let resolve = |arena: &[TypeId], offsets: &[(usize, usize)]| {
            let arena = arena.to_vec();
            let offsets = offsets.to_vec();
            move |i: u64| {
                offsets
                    .get(i as usize)
                    .map(|&(start, len)| arena[start..start + len].to_vec())
            }
        };
        let d_canon = canonicalise_stream(&deferred_stream, resolve(&d_arena, &d_offsets));
        let c_canon = canonicalise_stream(&content_stream, |_| None);
        assert_eq!(
            d_canon, c_canon,
            "content addressing must describe the same program as deferred interning"
        );

        // The canonicalisation must not be vacuous: it has to tell the two distinct instantiations
        // apart, and the repeat must share the first one's rank. Mapping everything to one rank
        // would make the assertion above pass trivially.
        assert_eq!(d_canon.len(), 3);
        assert_eq!(d_canon[0], d_canon[2], "the repeat shares a rank");
        assert_ne!(
            d_canon[0], d_canon[1],
            "distinct instantiations keep distinct ranks"
        );
    }

    /// Two independent workers must agree on a content-addressed instantiation with no barrier
    /// between them -- the property the design turns on. Checked directly rather than inferred from
    /// streams matching, which a shared worker would also satisfy.
    #[test]
    fn content_addressed_workers_agree_without_coordinating() {
        use crate::intern_mode::InternMode;
        let base = TypeId::new(0xAAAA, 0xBBBB, 0, 0);
        let args = vec![TypeId::new(0, 0x1111, 0, 0)];
        let mut w1 = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        let mut w2 = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        let a = mint_generic_in_mode(InternMode::Content, &mut w1, base, args.clone());
        let b = mint_generic_in_mode(InternMode::Content, &mut w2, base, args);
        assert_eq!(a, b, "two workers must agree without coordinating");
        assert!(
            w1.local_generics_arena.is_empty(),
            "content mode must not stage anything for the barrier"
        );
    }

    fn parse_and_resolve(path: &str, src: &str) -> VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = path.into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map);
        mods.pop().unwrap()
    }

    /// The Phase 2 freeze runs cycle detection: an infinite-sized recursive struct (a by-value
    /// self-cycle) is a compile error.
    #[test]
    fn frozen_registry_detects_infinite_recursion() {
        let m = parse_and_resolve("crate::m", "struct List { next: List }");
        match build_frozen_registry(std::slice::from_ref(&m)) {
            Err(PipelineError::Semantic(msg)) => {
                assert!(msg.contains("Infinite-sized recursive layout"), "{msg}")
            }
            other => panic!("expected a Semantic cycle error, got {other:?}"),
        }
    }

    /// Indirection (a reference) breaks the by-value cycle; the registry then builds with the
    /// nominal types registered.
    #[test]
    fn frozen_registry_accepts_indirection_and_registers_types() {
        let m = parse_and_resolve(
            "crate::m",
            "struct Node { next: &Node, val: i32 }\nstruct Pair { a: i32, b: i32 }",
        );
        let reg = build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        assert_eq!(reg.layouts.len(), 2);
    }

    /// The GID-keyed method table (#218) resolves `(receiver GID, method) -> FnSig` for every `impl`
    /// method -- struct and scalar receivers -- the registry-backed replacement for walking borrowed
    /// AST `ImplBlock`s in `GlobalAstEnv`.
    #[test]
    fn registry_method_table_captures_impl_methods() {
        use crate::syntax::{ElementType, Type};
        let m = parse_and_resolve(
            "crate::m",
            "struct Point { x: i32, y: i32 }\n\
             impl Point { fn sum(self: Point) -> i32 { return self.x + self.y; } }\n\
             trait Sq { fn sq(self: Self) -> f32; }\n\
             impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } }\n",
        );
        let reg = build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let hash = crate::hash::compute_module_hash("crate::m");

        // Struct receiver: Point::sum -> i32.
        let point_gid = reg
            .resolve_in_module(hash, &crate::symbol::Symbol::from("Point"))
            .unwrap();
        let sig = reg
            .resolve_method(point_gid, &crate::symbol::Symbol::from("sum"))
            .expect("Point::sum in the method table");
        assert!(matches!(sig.ret_ty, Type::Scalar(ElementType::I32)));

        // Scalar receiver: f32::sq -> f32.
        let f32_gid = crate::hir::flatten::scalar_gid(&ElementType::F32);
        let sig = reg
            .resolve_method(f32_gid, &crate::symbol::Symbol::from("sq"))
            .expect("f32::sq in the method table");
        assert!(matches!(sig.ret_ty, Type::Scalar(ElementType::F32)));

        // No false positives: an absent method resolves to None.
        assert!(reg
            .resolve_method(f32_gid, &crate::symbol::Symbol::from("cube"))
            .is_none());

        // Dual-run parity: every impl method in the module is in the table with a matching return
        // type -- exactly the set an AST `impls` walk (what `GlobalAstEnv` stores) would find.
        for imp in &m.impls {
            let recv = method_receiver_gid(&imp.target_type).expect("a concrete impl receiver");
            for meth in &imp.methods {
                let sig = reg
                    .resolve_method(recv, &meth.name)
                    .expect("impl method present in the table");
                assert_eq!(
                    format!("{:?}", sig.ret_ty),
                    format!("{:?}", meth.return_type),
                    "return type parity for {}",
                    meth.name
                );
            }
        }
    }

    /// The `ModuleInterface` query surface (#219) resolves types, layouts, free functions and methods
    /// entirely from the frozen registry -- the AST-free import oracle the type checker will consult
    /// in place of `GlobalAstEnv`'s borrowed AST. Exercised through `&dyn ModuleInterface` so the trait
    /// dispatch itself is covered, not just the inherent accessors it delegates to.
    #[test]
    fn module_interface_serves_registry_backed_resolution() {
        use crate::registry::ModuleInterface;
        use crate::syntax::{ElementType, Type};
        let m = parse_and_resolve(
            "crate::m",
            "struct Point { x: i32, y: i32 }\n\
             fn origin() -> Point { return Point { x: 0i32, y: 0i32 }; }\n\
             impl Point { fn sum(self: Point) -> i32 { return self.x + self.y; } }\n\
             trait Sq { fn sq(self: Self) -> f32; }\n\
             impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } }\n",
        );
        let reg = build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let mi: &dyn ModuleInterface = &reg;
        let hash = crate::hash::compute_module_hash("crate::m");

        // resolve_type + layout_of: a nominal by (module, name) -> GID -> its structural layout.
        let point_gid = mi
            .resolve_type(hash, &crate::symbol::Symbol::from("Point"))
            .expect("Point resolves via the interface");
        let layout = mi.layout_of(point_gid).expect("Point has a layout");
        assert_eq!(layout.name, "Point");

        // resolve_fn: a free function's return type.
        let sig = mi
            .resolve_fn(&crate::symbol::Symbol::from("origin"))
            .expect("origin resolves via the interface");
        assert!(matches!(sig.ret_ty, Type::Struct(_, _)));

        // resolve_method: struct and scalar receivers, and a clean miss.
        let sum = mi
            .resolve_method(point_gid, &crate::symbol::Symbol::from("sum"))
            .expect("Point::sum via the interface");
        assert!(matches!(sum.ret_ty, Type::Scalar(ElementType::I32)));
        let f32_gid = crate::hir::flatten::scalar_gid(&ElementType::F32);
        assert!(mi
            .resolve_method(f32_gid, &crate::symbol::Symbol::from("sq"))
            .is_some());
        assert!(mi
            .resolve_method(point_gid, &crate::symbol::Symbol::from("nope"))
            .is_none());

        // resolve_unique_nominal: a name defined in exactly one module resolves to its GID.
        assert_eq!(
            mi.resolve_unique_nominal(&crate::symbol::Symbol::from("Point")),
            Some(point_gid)
        );
        assert!(mi
            .resolve_unique_nominal(&crate::symbol::Symbol::from("Absent"))
            .is_none());
    }

    /// `resolve_unique_nominal` (the registry-backed replacement for `GlobalAstEnv::struct_gids`,
    /// #219) declines a bare name two modules define with *distinct* GIDs -- the caller can't tell
    /// which is meant, so a `StructInit` gets no GID rather than the wrong one, exactly as the old
    /// side map did.
    #[test]
    fn resolve_unique_nominal_declines_cross_module_ambiguity() {
        use crate::registry::ModuleInterface;
        let a = parse_and_resolve("crate::a", "struct Point { x: i32 }");
        let b = parse_and_resolve(
            "crate::b",
            "struct Point { y: i32, z: i32 }\nstruct Only { w: i32 }",
        );
        let reg = build_frozen_registry(&[a, b]).expect("acyclic");
        let mi: &dyn ModuleInterface = &reg;

        // `Point` is defined in both modules with distinct GIDs -> ambiguous -> None.
        assert!(mi
            .resolve_unique_nominal(&crate::symbol::Symbol::from("Point"))
            .is_none());
        // `Only` is defined in exactly one module -> resolves.
        let only = mi
            .resolve_unique_nominal(&crate::symbol::Symbol::from("Only"))
            .expect("Only is unambiguous");
        assert_eq!(mi.layout_of(only).unwrap().name, "Only");
    }

    /// End-to-end for the #219 dual-run gate: freeze the registry over a module with a concrete scalar
    /// `impl` method (the shape of `impl Math for f32 { fn exp(..) }` in the stdlib), then type-check a
    /// caller *against that registry*. The in-situ parity gate in `check_methodcall_expr` fires at the
    /// `x.exp()` site -- the registry-backed `ModuleInterface` must resolve `(f32, "exp")` that the AST
    /// impl-walk resolves. Passing (no debug-assert panic) proves the interface is a sufficient method
    /// oracle at a real resolution site, not just in isolation.
    #[test]
    fn type_checker_method_resolution_agrees_with_registry() {
        let mut m = parse_and_resolve(
            "crate::m",
            "trait Math { fn exp(self: Self) -> Self; }\n\
             impl Math for f32 { fn exp(self: f32) -> f32 { return self; } }\n\
             fn use_it(x: f32) -> f32 { return x.exp(); }\n",
        );
        let reg = build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        assert!(
            !reg.methods.is_empty(),
            "the registry must capture the concrete impl method for the gate to run"
        );
        let session = Arc::new(GlobalSession::with_registry(1, reg));

        // The env borrows a *clone*, leaving `m` free to be mutated by `check_function`.
        let env_mods = vec![m.clone()];
        let env = GlobalAstEnv::build(&env_mods);
        let mut worker = LocalWorkerState::new(session);
        let mut checker = TypeChecker::new(&env, &mut worker);
        for f in &mut m.functions {
            checker.check_function(f);
        }
        // Reaching here means the dual-run gate held: the registry resolved `f32.exp` exactly where
        // the AST walk did. Guard against silent resolution failure too (warnings are fine).
        let hard_errors: Vec<_> = checker
            .errors
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(
            hard_errors.is_empty(),
            "type check reported errors: {:?}",
            hard_errors
        );
    }

    /// Cross-module return provenance (#265 step 7): the borrow checker refines a reborrow through an
    /// *imported* callee using the provenance code carried in the `.vxlib` interface, not the AST
    /// summary (empty for an import). `pick(a, b) -> &b.slot` derives from `b` only, so a caller that
    /// keeps the result live may mutate `a`'s storage but not `b`'s — and that precision must survive
    /// the compile boundary. Contrast with the conservative default (no interface), which assumes the
    /// result aliases *both* arguments.
    #[test]
    fn borrow_check_reads_return_provenance_from_a_vxlib_interface() {
        use crate::metadata::{deserialize_registry_interface, serialize_registry_interface};

        // The "library": Map + a per-parameter-provenance reference return + a mutator. Freeze its
        // registry and round-trip it through the interface codec, exactly as a `.vxlib` would.
        let lib = parse_and_resolve(
            "crate::lib",
            "struct Map { slot: i32, present: i32 }\n\
             fn insert(m: &mut Map, v: i32) -> void { m.slot = v; m.present = 1i32; }\n\
             fn pick(a: &Map, b: &Map) -> &i32 { return &b.slot; }\n",
        );
        let lib_reg = build_frozen_registry(std::slice::from_ref(&lib)).expect("acyclic");
        let imported = deserialize_registry_interface(&serialize_registry_interface(&lib_reg))
            .expect("round-trip");
        assert_eq!(
            imported
                .fn_sigs
                .get(&crate::symbol::Symbol::from("pick"))
                .unwrap()
                .ret_prov,
            2,
            "pick's return derives from parameter slot 1 (b), carried across the boundary"
        );

        // The consumer calls the imported `pick`/`insert`. It is checked with the *signatures*
        // visible (so the calls resolve) but no *bodies* — modeling what a `.vxlib` provides — so
        // `return_provenances` misses `pick` and the checker must read the registry's `ret_prov`.
        // (The redeclared signatures are a scaffold to feed the env; their bodies are stripped by
        // `clone_signature` and never consulted — the provenance comes only from the merged registry.)
        let consumer = parse_and_resolve(
            "crate::app",
            "struct Map { slot: i32, present: i32 }\n\
             fn insert(m: &mut Map, v: i32) -> void { m.slot = v; m.present = 1i32; }\n\
             fn pick(a: &Map, b: &Map) -> &i32 { return &b.slot; }\n\
             fn main() -> i32 {\n\
               let mut x = Map { slot: 1i32, present: 1i32 };\n\
               let mut y = Map { slot: 2i32, present: 1i32 };\n\
               let r = pick(&x, &y);\n\
               insert(&mut y, 99i32);\n\
               insert(&mut x, 99i32);\n\
               return *r;\n\
             }\n",
        );

        // Check `main` against a session whose registry is `reg`; count hard borrow errors. The env
        // is the signature-only clone (bodies stripped → `pick` absent from `return_provenances`),
        // and we deliberately do not annotate provenances — modeling a summary that lives only in the
        // interface.
        let run = |reg: crate::registry::ImmutableGlobalRegistry| -> usize {
            let session = Arc::new(GlobalSession::with_registry(1, reg));
            let env_mods = vec![consumer.clone_signature()];
            let env = GlobalAstEnv::build(&env_mods);
            let mut worker = LocalWorkerState::new(session);
            let mut checker = TypeChecker::new(&env, &mut worker);
            let mut main_fn = consumer
                .functions
                .iter()
                .find(|f| f.name.as_ref() == "main")
                .unwrap()
                .clone();
            checker.check_function(&mut main_fn);
            checker
                .errors
                .iter()
                .filter(|d| d.level == DiagnosticLevel::Error)
                .count()
        };

        // With the interface merged, `r` aliases only `y` (slot 1): mutating `y` conflicts, mutating
        // `x` is accepted — exactly one borrow error.
        assert_eq!(
            run(imported),
            1,
            "cross-module provenance: only the mutation of `y` (which `r` aliases) is rejected"
        );

        // Baseline: no interface (empty registry). The summary misses AND the registry misses, so the
        // checker falls to the conservative `AnyParam` — `r` is assumed to alias *both* args, so both
        // mutations are rejected. That delta is exactly what the `.vxlib` provenance buys.
        assert_eq!(
            run(build_frozen_registry(&[]).expect("empty registry")),
            2,
            "without the interface, the conservative default rejects both mutations"
        );
    }

    /// Phase 2 (#219 flip): a consumer resolves an imported call *entirely* from the merged registry —
    /// the callee is absent from the AST env — for both the type-check (the `E2002`-site `fn_sigs`
    /// fallback) and the borrow reborrow-tracking (`resolve_callee_ref_signature`'s registry fallback),
    /// then applies its `ret_prov`. `pick(a, b) -> b` (scalar refs, no struct GID) derives from slot 1,
    /// so a caller may mutably reborrow `x` but not `y` — resolved with no imported AST at all.
    #[test]
    fn borrow_check_resolves_imported_call_from_registry_only() {
        use crate::metadata::{deserialize_registry_interface, serialize_registry_interface};

        // Library: a scalar-reference per-parameter-provenance return. Freeze + round-trip its
        // interface. `pick` exists ONLY here and in the resulting registry — never in the consumer.
        let lib = parse_and_resolve(
            "crate::lib",
            "fn pick(a: &i32, b: &i32) -> &i32 { return b; }",
        );
        let lib_reg = build_frozen_registry(std::slice::from_ref(&lib)).expect("acyclic");
        let imported = deserialize_registry_interface(&serialize_registry_interface(&lib_reg))
            .expect("round-trip");
        assert_eq!(
            imported
                .fn_sigs
                .get(&crate::symbol::Symbol::from("pick"))
                .unwrap()
                .ret_prov,
            2
        );

        // Check a consumer that calls the registry-only `pick` and mutably reborrows one local. The
        // consumer declares only `bump`/`main` — `pick` is resolved purely from the registry.
        let check = |mutate: &str| -> usize {
            let src = format!(
                "fn bump(n: &mut i32) -> void {{ }}\n\
                 fn main() -> i32 {{\n\
                   let mut x = 10i32;\n\
                   let mut y = 20i32;\n\
                   let r = pick(&x, &y);\n\
                   bump(&mut {mutate});\n\
                   return *r;\n\
                 }}\n"
            );
            let consumer = parse_and_resolve("crate::app", &src);
            // Sanity: the consumer genuinely does NOT define `pick` — resolution must use the registry.
            assert!(consumer.functions.iter().all(|f| f.name.as_ref() != "pick"));
            let mut reg = build_frozen_registry(std::slice::from_ref(&consumer)).expect("app reg");
            reg.merge_from(
                deserialize_registry_interface(&serialize_registry_interface(&lib_reg)).unwrap(),
            );
            let session = Arc::new(GlobalSession::with_registry(1, reg));
            let env_mods = vec![consumer.clone_signature()];
            let env = GlobalAstEnv::build(&env_mods);
            let mut worker = LocalWorkerState::new(session);
            let mut checker = TypeChecker::new(&env, &mut worker);
            let mut main_fn = consumer
                .functions
                .iter()
                .find(|f| f.name.as_ref() == "main")
                .unwrap()
                .clone();
            checker.check_function(&mut main_fn);
            checker
                .errors
                .iter()
                .filter(|d| d.level == DiagnosticLevel::Error)
                .count()
        };

        // `r` derives from `b` (= `y`): reborrowing `x` is fine, reborrowing `y` conflicts — all with
        // `pick` resolved from the interface, no imported AST.
        assert_eq!(
            check("x"),
            0,
            "mutably reborrowing the non-aliased `x` is accepted"
        );
        assert_eq!(
            check("y"),
            1,
            "mutably reborrowing the aliased `y` is rejected"
        );
    }

    /// The freeze computes real layouts (#199), not the earlier 0/0 stub: field offsets honour
    /// natural alignment, nested nominals recurse by GID, and a C-like enum is an i32 discriminant.
    #[test]
    fn frozen_registry_computes_real_layouts() {
        let m = parse_and_resolve(
            "crate::m",
            "struct Pair { a: i8, b: i32 }\n\
             struct Wrap { flag: i8, inner: Pair }\n\
             enum Color { Red, Green, Blue }",
        );
        let reg = build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let hash = crate::hash::compute_module_hash("crate::m");
        let lookup = |name: &str| {
            let id = reg
                .resolve_in_module(hash, &crate::symbol::Symbol::from(name))
                .unwrap();
            reg.layouts[&id].clone()
        };

        // Pair { a: i8 @0, b: i32 @4 } -> size 8, align 4 (3 bytes of padding after `a`).
        let pair = lookup("Pair");
        assert_eq!((pair.size_bytes, pair.align_bytes), (8, 4));
        assert_eq!(pair.fields[0].offset, 0);
        assert_eq!(pair.fields[1].offset, 4);

        // Wrap { flag: i8 @0, inner: Pair @4 } -> size 12, align 4 (nested nominal recurses).
        let wrap = lookup("Wrap");
        assert_eq!((wrap.size_bytes, wrap.align_bytes), (12, 4));
        assert_eq!(wrap.fields[1].offset, 4);
        assert_eq!(wrap.fields[1].size, 8);

        // A payload-free enum is a bare i32 discriminant.
        let color = lookup("Color");
        assert_eq!((color.size_bytes, color.align_bytes), (4, 4));
    }

    fn parse_only(path: &str, src: &str) -> VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = path.into();
        prog
    }

    /// End-to-end cross-module identity through the flat pipeline (#194): module A defines `Foo`,
    /// module B holds it *by value* as `struct Bar { f: A::Foo }`. After name resolution attaches
    /// A's GID to the qualified reference, `build_frozen_registry` (Phase 2 freeze) must both index
    /// `Foo` under A's module hash and resolve Bar's by-value edge to A's `Foo` node -- proving the
    /// registry carries cross-module identity and the `module_indices` read path works end to end.
    #[test]
    fn frozen_registry_carries_cross_module_by_value_identity() {
        let mut modules = vec![
            parse_only("A", "struct Foo { x: i32 }"),
            parse_only("B", "struct Bar { f: A::Foo }"),
        ];
        // Resolve names across *both* modules (the parallel phase builds one symbol map for all).
        name_resolution_phase(&mut modules);

        // Builds only if Bar's by-value dependency on A::Foo resolved to a registered node.
        let reg = build_frozen_registry(&modules).expect("cross-module by-value dep resolves");

        let a_hash = crate::hash::compute_module_hash("A");
        let foo = reg
            .resolve_in_module(a_hash, &crate::symbol::Symbol::from("Foo"))
            .expect("A::Foo indexed under A's module hash");
        assert_eq!(foo.module_id(), a_hash, "word 0 is A's module hash");

        let bar = reg
            .resolve_in_module(
                crate::hash::compute_module_hash("B"),
                &crate::symbol::Symbol::from("Bar"),
            )
            .expect("Bar indexed under B's module hash");
        assert!(
            reg.layouts[&bar].by_value_dependencies.contains(&foo),
            "Bar carries A::Foo (cross-module) as a by-value dependency"
        );
    }

    /// Cross-module *by-value* recursion is still an infinite-sized layout: A::Node holds B::Other
    /// by value and B::Other holds A::Node by value. The Phase-2 freeze must detect the cycle across
    /// the module boundary — the cross-module GIDs from #194 make both edges resolvable, so
    /// `toposort` sees the loop.
    #[test]
    fn frozen_registry_detects_cross_module_by_value_recursion() {
        let mut modules = vec![
            parse_only("A", "struct Node { o: B::Other }"),
            parse_only("B", "struct Other { n: A::Node }"),
        ];
        name_resolution_phase(&mut modules);
        match build_frozen_registry(&modules) {
            Err(PipelineError::Semantic(msg)) => {
                assert!(msg.contains("Infinite-sized recursive layout"), "{msg}")
            }
            other => panic!("expected a cross-module cycle error, got {other:?}"),
        }
    }

    /// The real parallel `type_check_phase` lowers a scalar function body into its worker's
    /// `local_hir_stream` (not just the direct unit-test path). Proves the wiring end-to-end, with
    /// the debug `verify_hir_stream` hook active.
    #[test]
    fn type_check_phase_lowers_scalar_body_to_hir() {
        use crate::hir::bytecode::Opcode;
        let mut modules = vec![parse_only(
            "m",
            "fn add(a: i32, b: i32) -> i32 { return a + b; }",
        )];
        name_resolution_phase(&mut modules);
        let registry = build_frozen_registry(&modules).expect("registry");
        let session = Arc::new(GlobalSession::with_registry(1, registry));
        let env_mods: Vec<VxModule> = modules.iter().map(|m| m.clone_signature()).collect();
        let env = GlobalAstEnv::build(&env_mods);

        let results = type_check_phase(&mut modules, &session, &env).expect("type check ok");
        let worker = &results[0].2;
        let ops: Vec<Opcode> = worker.local_hir_stream.iter().map(|i| i.opcode).collect();
        assert_eq!(
            ops,
            vec![Opcode::Load, Opcode::Load, Opcode::Add, Opcode::Ret],
            "scalar body lowered through the parallel phase"
        );
    }

    /// A tensor-typed signature contributes the tensor's GID (element + shape) to the flat type
    /// stream, matching what the HIR body path (`flatten::tensor_gid_of`) emits — one identity for a
    /// tensor whether it appears in a signature or a lowered body (#199).
    #[test]
    fn tensor_signature_emits_the_tensor_gid() {
        let m = parse_only("m", "fn f(q: Tensor<f32, [2, 4]>) -> i32 { return 0; }");
        let mut worker = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        emit_function_type_gids(&m.functions[0], &mut worker);
        let expected = crate::hir::flatten::tensor_gid(
            &crate::syntax::ElementType::F32,
            &["2".to_string(), "4".to_string()],
        );
        assert!(
            worker.local_type_stream.contains(&expected),
            "signature emits the tensor GID matching the body path"
        );
    }

    /// A recursive *enum* held by value is infinite-sized too: `enum Tree { Leaf, Node(Tree) }`.
    /// With variant payloads now resolved, the registry sees the `Tree -> Tree` by-value edge and
    /// detects the cycle. (Regression guard for enum payload resolution.)
    #[test]
    fn frozen_registry_detects_recursive_enum_by_value() {
        let mut modules = vec![parse_only("m", "enum Tree { Leaf, Node(Tree) }")];
        name_resolution_phase(&mut modules);
        match build_frozen_registry(&modules) {
            Err(PipelineError::Semantic(msg)) => {
                assert!(msg.contains("Infinite-sized recursive layout"), "{msg}")
            }
            other => panic!("expected an enum cycle error, got {other:?}"),
        }
    }

    /// Indirection across the module boundary breaks the cycle: `n: &A::Node` is a reference, not a
    /// by-value dependency, so the registry builds with both types registered.
    #[test]
    fn frozen_registry_cross_module_cycle_broken_by_indirection() {
        let mut modules = vec![
            parse_only("A", "struct Node { o: B::Other }"),
            parse_only("B", "struct Other { n: &A::Node }"),
        ];
        name_resolution_phase(&mut modules);
        let reg =
            build_frozen_registry(&modules).expect("indirection breaks the cross-module cycle");
        assert_eq!(reg.layouts.len(), 2);
    }

    /// Order-sensitive determinism (#196): the flat GID stream must be byte-identical **in order**,
    /// and identical **across thread counts** — codegen indexes it by position (`type_idx`),
    /// so a scheduling-dependent race or an order-dependent phase is a correctness bug even when the
    /// *set* of GIDs matches. Running the whole pipeline under a 1-thread and an 8-thread rayon pool
    /// catches races the earlier same-pool set-comparison could not.
    #[test]
    fn flat_type_stream_order_is_deterministic_across_thread_counts() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("vx_det_types_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let srcs = [
            (
                "a.vx",
                "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
                 fn mul(a: i32, b: i32) -> i32 { return a * b; }",
            ),
            (
                "b.vx",
                "fn clamp(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
            ),
        ];
        let mut paths = Vec::new();
        for (name, src) in srcs {
            let p = dir.join(name);
            std::fs::File::create(&p)
                .unwrap()
                .write_all(src.as_bytes())
                .unwrap();
            paths.push(p.to_string_lossy().to_string());
        }

        let run = |threads: usize| -> Vec<[u64; 4]> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                compile_pipeline_type_stream(&paths)
                    .expect("pipeline")
                    .into_iter()
                    .map(|id| id.words)
                    .collect()
            })
        };

        let single = run(1);
        let many = run(8);
        assert!(!single.is_empty(), "expected a non-empty stream");
        assert_eq!(
            single, many,
            "flat type stream order differs across thread counts (non-deterministic)"
        );
        assert_eq!(many, run(8), "flat type stream order differs across reruns");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The HIR stream is new parallel-produced state whose order is contractual (codegen indexes
    /// it by register). Assert the concatenated stream is byte-identical across thread counts.
    #[test]
    fn hir_stream_is_deterministic_across_thread_counts() {
        use crate::hir::bytecode::Opcode;
        let build =
            || -> Vec<VxModule> {
                vec![
                parse_only("m", "fn f(a: i32, b: i32) -> i32 { let x = a * b; return x + a; }"),
                parse_only(
                    "n",
                    "fn g(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
                ),
            ]
            };
        let run = |threads: usize| -> Vec<(Opcode, u32, u32, u64)> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut modules = build();
                name_resolution_phase(&mut modules);
                let registry = build_frozen_registry(&modules).expect("registry");
                let session = Arc::new(GlobalSession::with_registry(1, registry));
                let env_mods: Vec<VxModule> = modules.iter().map(|m| m.clone_signature()).collect();
                let env = GlobalAstEnv::build(&env_mods);
                let results = type_check_phase(&mut modules, &session, &env).expect("type check");
                results
                    .iter()
                    .flat_map(|(_, _, w, _, _)| {
                        w.local_hir_stream
                            .iter()
                            .map(|i| (i.opcode, i.operand1.0, i.operand2.0, i.imm))
                    })
                    .collect()
            })
        };
        let single = run(1);
        let many = run(8);
        assert!(!single.is_empty(), "expected lowered HIR");
        assert_eq!(
            single, many,
            "HIR stream differs across thread counts (non-deterministic)"
        );
    }
}
