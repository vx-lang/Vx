//===- driver.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Compiler driver orchestrating file I/O, parsing, semantic analysis, and code generation.
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::DiagnosticLevel;
use crate::syntax::MacroExpander;
use crate::syntax_printer::AstPrinter;
use clap::{Parser, ValueEnum};
use codegen::MeliorGenerator;
use melior::ir::operation::OperationLike;
use std::path::PathBuf;

use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::module_loader::ModuleLoader;
use crate::session::{GlobalSession, LocalWorkerState};

use crate::codegen;
#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum Action {
    /// Only run the lexer and parser
    ParseOnly,
    /// Parse and typecheck the program, then print the AST
    PrintAst,
    /// Parse, typecheck, and emit the MLIR representation
    EmitMlir,
    /// Parse, typecheck, emit MLIR, and execute via the JIT Engine
    RunJit,
    /// Compile to an object file
    EmitObj,
    /// Parse, typecheck, emit MLIR, lower to LLVM dialect, and translate to LLVM IR
    EmitLlvm,
    /// Serialize this module's import interface (frozen registry + flat-HIR bodies) to a `.vxlib`
    /// artifact, so a downstream compile can import it without re-parsing the source (#220).
    EmitInterface,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct DriverOptions {
    /// Action to perform
    #[arg(short = 'a', long = "action", value_enum, default_value_t = Action::RunJit)]
    #[arg(overrides_with_all = ["compile", "parse_only", "print_ast", "emit_mlir", "emit_llvm", "run_jit", "emit_interface"])]
    pub action: Action,

    /// Output file
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Compile to object file (alias for --action emit-obj)
    #[arg(short = 'c', overrides_with = "action")]
    pub compile: bool,

    /// Parse only (alias for --action parse-only)
    #[arg(short = 'p', long = "parse-only", overrides_with = "action")]
    pub parse_only: bool,

    /// Print AST (alias for --action print-ast)
    #[arg(long = "print-ast", overrides_with = "action")]
    pub print_ast: bool,

    /// Emit MLIR (alias for --action emit-mlir)
    #[arg(long = "emit-mlir", overrides_with = "action")]
    pub emit_mlir: bool,

    /// Emit LLVM IR (alias for --action emit-llvm)
    #[arg(long = "emit-llvm", overrides_with = "action")]
    pub emit_llvm: bool,

    /// Run JIT (alias for --action run-jit)
    #[arg(long = "run", overrides_with = "action")]
    pub run_jit: bool,

    /// Emit a serialized module interface (alias for --action emit-interface): the frozen registry +
    /// flat-HIR bodies as a `.vxlib`, for a downstream compile to import without re-parsing (#220).
    #[arg(long = "emit-interface", overrides_with = "action")]
    pub emit_interface: bool,

    /// Link a precompiled `.vxlib` module interface: its function signatures resolve imported calls in
    /// the frontend (type + borrow check, incl. cross-module return provenance) and its portable
    /// flat-HIR bodies link in the flat codegen — all without parsing the library's source (#265 step
    /// 7 / #219). The consumer side of `--emit-interface`.
    #[arg(long = "link-interface", value_name = "FILE")]
    pub link_interface: Option<PathBuf>,

    /// Emit MLIR/LLVM backend diagnostics
    #[arg(long = "emit-backend-diagnostics")]
    pub emit_backend_diagnostics: bool,

    /// Use the legacy AST-walk `MeliorGenerator` codegen instead of the default flat-array path
    /// (`local_hir_stream` → `flat::emit_module_mlir`). The flat path is the default (#201) and falls
    /// back to this AST path per-program for anything outside the flat subset, so output never changes;
    /// `--legacy-codegen` forces the AST path for the whole compile.
    #[arg(long = "legacy-codegen")]
    pub legacy_codegen: bool,

    /// Discharge per-seam boundary obligations at cross-device transfers (assert
    /// pre-scan + z3 checks). Off by default; requires z3 on PATH (fails open if absent).
    #[arg(long = "verify-seams")]
    pub verify_seams: bool,

    /// Transport host-proven `assert` facts across a host->device `spawn` seam as
    /// `llvm.intr.assume` certificates at the kernel entry, so the device backend's
    /// `-O3` can fold on a relation that would otherwise be opaque past the launch
    /// boundary. Off by default. See `crate::codegen::lower::seam_cert`.
    #[arg(long = "emit-seam-certs")]
    pub emit_seam_certs: bool,

    /// Target backend for emitted LLVM IR: tags the module with the target triple and
    /// data layout (x86_64, aarch64, nvptx64, amdgcn). Affects `--emit-llvm` output.
    #[arg(long = "target")]
    pub target: Option<String>,

    /// Disable Vx optimizations
    #[arg(long = "disable-vx-optimizations")]
    pub disable_vx_optimizations: bool,

    /// Disable MLIR optimizations
    #[arg(long = "disable-mlir-optimizations")]
    pub disable_mlir_optimizations: bool,

    /// Disable LLVM optimizations
    #[arg(long = "disable-llvm-optimizations")]
    pub disable_llvm_optimizations: bool,

    /// Optimization level (-O0 to -O3)
    #[arg(short = 'O', num_args = 0..=1, default_missing_value = "3", default_value_t = 0)]
    pub opt_level: u8,

    /// Input source files
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Specify the language of the input file (e.g. mlir, vx)
    #[arg(short = 'x', long = "language")]
    pub language: Option<String>,

    /// Pass an argument to a specific backend tool (e.g., -X mlir=--pass-pipeline=...)
    #[arg(short = 'X')]
    pub tool_args: Vec<String>,

    /// Arguments to pass to the running program
    #[arg(last = true)]
    pub program_args: Vec<String>,
}

pub struct CompilerDriver {
    pub options: DriverOptions,
}

impl CompilerDriver {
    pub fn new(mut options: DriverOptions) -> Self {
        // Handle alias flags
        if options.parse_only {
            options.action = Action::ParseOnly;
        } else if options.print_ast {
            options.action = Action::PrintAst;
        } else if options.emit_mlir {
            options.action = Action::EmitMlir;
        } else if options.emit_llvm {
            options.action = Action::EmitLlvm;
        } else if options.run_jit {
            options.action = Action::RunJit;
        } else if options.emit_interface {
            options.action = Action::EmitInterface;
        } else if options.compile {
            options.action = Action::EmitObj;
        }

        Self { options }
    }

    pub fn execute(&self) -> Result<(), String> {
        if self.options.inputs.is_empty() {
            return Err("No input files provided".to_string());
        }

        let main_file = &self.options.inputs[0];
        let filename = main_file.to_string_lossy().to_string();

        let language = self
            .options
            .language
            .clone()
            .unwrap_or_else(|| "vx".to_string());

        let mut mlir_args: Vec<String> = Vec::new();
        for arg in &self.options.tool_args {
            if let Some(mlir_arg) = arg.strip_prefix("mlir=") {
                mlir_args.push(mlir_arg.to_string());
            }
        }

        if language == "mlir" {
            return self.execute_mlir_pipeline(main_file, &mlir_args);
        }

        self.execute_vx_pipeline(main_file, &filename, &mlir_args)
    }

    fn execute_mlir_pipeline(
        &self,
        main_file: &std::path::Path,
        mlir_args: &[String],
    ) -> Result<(), String> {
        let mlir_src = std::fs::read_to_string(main_file).map_err(|e| e.to_string())?;

        if self.options.action == Action::EmitMlir || self.options.action == Action::EmitLlvm {
            let optimized_mlir = apply_mlir_opt(&mlir_src, mlir_args, main_file)?;
            if self.options.action == Action::EmitLlvm {
                let llvm_ir = translate_to_llvm_ir(&optimized_mlir, main_file)?;
                println!("{}", llvm_ir);
            } else {
                println!("{}", optimized_mlir);
            }
            return Ok(());
        }

        if self.options.action == Action::RunJit {
            let mut args = vec![self.options.inputs[0].to_string_lossy().into_owned()];
            args.extend(self.options.program_args.clone());
            let out = crate::jit::execute_mlir(
                &mlir_src,
                args,
                self.options.opt_level,
                self.options.disable_llvm_optimizations,
            )
            .map_err(|e| e.to_string())?;
            println!("{}", out);
            return Ok(());
        }

        Err(format!(
            "Action {:?} is not supported for MLIR inputs",
            self.options.action
        ))
    }

    fn execute_vx_pipeline(
        &self,
        main_file: &std::path::Path,
        filename: &str,
        mlir_args: &[String],
    ) -> Result<(), String> {
        // Topology declarations are carried on the AST (`Program.topologies`) and indexed
        // per-compilation by `GlobalAstEnv`, so there is no process-global state to reset --
        // declarations cannot leak between compilations by construction.
        let mut program_arr = self.load_and_expand(filename)?;

        if self.options.action == Action::ParseOnly {
            return self.handle_parse_only(&program_arr, filename);
        }

        if self.options.action == Action::EmitInterface {
            return self.handle_emit_interface(&mut program_arr, filename);
        }

        let (mut main_ast, mut other_asts) =
            self.prepare_semantic_analysis(&mut program_arr, filename)?;
        self.run_semantic_analysis(&mut main_ast, &mut other_asts, filename)?;

        if self.options.action == Action::PrintAst {
            AstPrinter::print_program(&main_ast, &mut std::io::stdout()).unwrap();
            return Ok(());
        }

        self.run_codegen(main_ast, other_asts, filename, main_file, mlir_args)
    }

    fn load_and_expand(&self, filename: &str) -> Result<Vec<crate::syntax::Program>, String> {
        let mut loader = ModuleLoader::new();
        if let Err(e) = loader.load_main(filename) {
            return Err(format!("Frontend failed to parse '{}': {}", filename, e));
        }
        let mut program_arr = loader.into_programs();

        let mut global_macros = std::collections::HashMap::new();
        for m in &program_arr {
            for mac in &m.macros {
                global_macros.insert(mac.name.clone(), mac.rules.clone());
            }
        }
        let mut expander = MacroExpander::new(&global_macros);
        for m in &mut program_arr {
            if let Err(e) = expander.expand_module(m) {
                return Err(format!("Macro expansion failed: {}", e));
            }
        }
        Ok(program_arr)
    }

    fn handle_parse_only(
        &self,
        program_arr: &[crate::syntax::Program],
        filename: &str,
    ) -> Result<(), String> {
        let ast = program_arr
            .iter()
            .find(|p| {
                p.module_path == <std::string::String as Clone>::clone(&filename.to_string()).into()
            })
            .unwrap();
        println!("{:#?}", ast);
        Ok(())
    }

    /// `--emit-interface`: resolve the loaded modules and serialize their import interface (frozen
    /// registry + portable flat-HIR bodies) to a `.vxlib` artifact -- the producer side of the
    /// stdlib<->compiler decoupling (#220). No semantic analysis / codegen of the module is run.
    fn handle_emit_interface(
        &self,
        program_arr: &mut [crate::syntax::Program],
        filename: &str,
    ) -> Result<(), String> {
        let symbol_map = crate::resolver::build_symbol_map(program_arr);
        for m in program_arr.iter_mut() {
            m.resolve_names(&symbol_map);
        }
        let bytes = crate::pipeline::emit_module_interface(program_arr)
            .map_err(|e| format!("Failed to build module interface: {}", e))?;

        let out = self
            .options
            .output
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from(filename).with_extension("vxlib"));
        // The interface section carries everything; the type dictionary is left empty.
        crate::metadata::VxMetadata::save_with_interface(&[], &bytes, &out)
            .map_err(|e| format!("Failed to write {}: {}", out.display(), e))?;
        println!(
            "Wrote module interface ({} bytes) to {}",
            bytes.len(),
            out.display()
        );
        Ok(())
    }

    /// Read the `--link-interface` artifact and return its serialized module-interface bytes (the
    /// `interface_data` section of the `.vxlib`), or `None` when the flag is absent.
    /// [`crate::metadata::deserialize_registry_interface`] turns these back into a queryable registry
    /// that a downstream compile merges — resolving imported symbols with no parse of the library.
    fn linked_interface_bytes(&self) -> Result<Option<Vec<u8>>, String> {
        let Some(path) = &self.options.link_interface else {
            return Ok(None);
        };
        let buf = std::fs::read(path).map_err(|e| {
            format!(
                "Failed to read --link-interface '{}': {}",
                path.display(),
                e
            )
        })?;
        let meta = crate::metadata::VxMetadata::load_from_buffer(&buf);
        Ok(Some(meta.interface_data.to_vec()))
    }

    fn prepare_semantic_analysis(
        &self,
        program_arr: &mut Vec<crate::syntax::Program>,
        filename: &str,
    ) -> Result<
        (
            crate::syntax::Program,
            std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        ),
        String,
    > {
        let syntax_idx = program_arr
            .iter()
            .position(|p| {
                p.module_path == <std::string::String as Clone>::clone(&filename.to_string()).into()
            })
            .unwrap();
        let ast = program_arr.remove(syntax_idx);

        // Keep imported modules whole — including their *generic* free functions — so the resolution
        // env can index them (`GlobalAstEnv.generic_functions`) and instantiate cross-module generic
        // calls (e.g. `googletest::expect_eq`). Codegen skips the generic templates; only their
        // concrete instances (collected as monomorphizations) are emitted. See #204.
        let mut module_syntaxes = std::collections::HashMap::new();
        for p in program_arr.drain(..) {
            module_syntaxes.insert(p.module_path.clone(), p);
        }
        Ok((ast, module_syntaxes))
    }

    fn run_semantic_analysis(
        &self,
        ast: &mut crate::syntax::Program,
        other_asts: &mut std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        filename: &str,
    ) -> Result<(), String> {
        // A `--link-interface` compile resolves imported calls against the merged registry (their AST
        // is never parsed), so the session carries the deserialized interface; otherwise the driver
        // uses an empty registry and resolves everything against the AST env (#265 step 7 / #219).
        let global_session = match self.linked_interface_bytes()? {
            Some(bytes) => {
                let reg = crate::metadata::deserialize_registry_interface(&bytes)
                    .map_err(|e| format!("Failed to load --link-interface: {}", e))?;
                std::sync::Arc::new(GlobalSession::with_registry(1, reg))
            }
            None => std::sync::Arc::new(GlobalSession::new(1)),
        };

        // Build the resolution env from *owned* clones: full bodies for the imported modules (so
        // their methods/generics can be instantiated) plus the entry module's signature. Owning the
        // clones leaves `ast` and `other_asts` free to be type-checked *in place* below — which #203
        // needs, so codegen emits those bodies with method/generic calls rewritten to the
        // monomorphized instance names.
        let mut env_progs: Vec<crate::syntax::Program> = other_asts.values().cloned().collect();
        env_progs.push(ast.clone_signature());
        let mut env = GlobalAstEnv::build(&env_progs);
        // The entry module was pushed signature-stripped; refill its return-provenance summaries
        // from the full module so intra-module calls get per-parameter precision (#243). The
        // imported `other_asts` were pushed with bodies, so `build` already summarized them.
        env.annotate_return_provenances(std::slice::from_ref(ast));

        let mut worker = LocalWorkerState::new(global_session.clone());
        let mut checker = TypeChecker::new(&env, &mut worker);
        checker.verify_seams = self.options.verify_seams;

        // Reject/flag incoherent user-defined topology declarations before checking bodies.
        checker.check_topology_coherence(&ast.topologies);
        // Reject incoherent memory-space declarations (cycles, oversized sub-spaces, ...).
        checker.check_memory_coherence();

        for f in &mut ast.functions {
            checker.check_function(f);
        }
        for i in &mut ast.impls {
            for f in &mut i.methods {
                checker.check_function(f);
            }
        }

        // #203: codegen emits the imported modules' (non-generic) functions too, but above only the
        // entry module's bodies were checked — so any method or generic those bodies reach *only
        // transitively* (e.g. `dijkstra` calling `Graph::node_count` / `Vec<i32>::with_capacity`)
        // was never instantiated, and codegen would fail with "Function ... not found". Check them
        // *in place* (the env borrows the clones above, so `other_asts` is free to mutate): the
        // instantiations land in `monomorphized_functions`, and the method/generic calls in the
        // emitted bodies are rewritten to those instance names. Generic functions were already
        // dropped from `other_asts`; generic *methods* are instantiated on demand by these calls.
        let errors_before_imports = checker.errors.len();
        for p in other_asts.values_mut() {
            for f in &mut p.functions {
                if f.generics.is_empty() {
                    checker.check_function(f);
                }
            }
        }
        // Diagnostics from imported *library internals* are not the consumer's concern — this pass
        // exists to collect monomorphizations, not to re-validate dependencies (which are checked
        // when compiled on their own). Drop anything it added; keep the instantiations.
        checker.errors.inner.truncate(errors_before_imports);

        let has_errors = checker
            .errors
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error);

        // Print warnings first (so they appear before errors)
        for diag in checker.errors.iter() {
            if diag.level == DiagnosticLevel::Warning {
                eprintln!("{}", diag);
            }
        }

        // Eval metric M1: per-seam proof cost discharged during this compile. The
        // one-time solver startup is reported separately from the marginal per-seam
        // solving time (the persistent solver is spawned once and reused).
        if checker.seam_checks > 0 {
            eprintln!(
                "[seam] {} obligation(s): solver init {:.3} ms (once) + {:.3} ms solving \
                 total = {:.4} ms/seam marginal",
                checker.seam_checks,
                checker.solver_init_time.as_secs_f64() * 1e3,
                checker.seam_check_time.as_secs_f64() * 1e3,
                checker.seam_check_time.as_secs_f64() * 1e3 / checker.seam_checks as f64,
            );
        }

        if has_errors {
            let error_count = checker.errors.error_count();
            for diag in checker.errors.iter() {
                if diag.level == DiagnosticLevel::Error {
                    eprintln!("{}", diag);
                }
            }
            return Err(format!(
                "Semantic check failed on '{}': {} error(s) emitted",
                filename, error_count
            ));
        }

        let mut orig_functions = ast.functions.clone();
        orig_functions.retain(|f| f.generics.is_empty());

        let mut new_functions: Vec<_> = checker
            .monomorphized_functions
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        new_functions.extend(orig_functions);
        ast.functions = new_functions;
        ast.structs.extend(checker.generated_structs);

        Ok(())
    }

    fn run_codegen(
        &self,
        monomorphized_ast: crate::syntax::Program,
        module_syntaxes: std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        filename: &str,
        main_file: &std::path::Path,
        mlir_args: &[String],
    ) -> Result<(), String> {
        codegen::register_vx_passes();

        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        melior::utility::register_all_passes();
        let context = melior::Context::new();
        let emit_diagnostics = self.options.emit_backend_diagnostics;
        context.attach_diagnostic_handler(move |diagnostic| {
            if emit_diagnostics {
                eprintln!("{}", diagnostic);
            }
            true
        });
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        codegen::register_vx_dialect(&context);

        // The flat-array codegen path is the default: produce the module from `local_hir_stream` via
        // `flat::emit_module_mlir` instead of the AST walk. It declines (falls back) for anything
        // outside the flat subset, so it never regresses against the AST oracle; `--legacy-codegen`
        // forces the AST path (#201).
        let linked_interface = self.linked_interface_bytes()?;
        let flat_module = if self.options.legacy_codegen {
            None
        } else {
            Self::build_flat_module(
                &context,
                &monomorphized_ast,
                &module_syntaxes,
                linked_interface.as_deref(),
            )
        };

        let mut module = match flat_module {
            Some(m) => {
                eprintln!("[flat-codegen] emitted module via the flat path");
                m
            }
            None if self.options.link_interface.is_some() => {
                // The AST codegen path cannot link a `--link-interface` import: only the flat path
                // reads `body_of` (the imported function has no AST in this compile). So a flat
                // decline here is a hard, clean error — never an AST-fallback ICE ("Function … not
                // found"). The frontend type/borrow check (incl. cross-module provenance) already
                // succeeded; only codegen is blocked, by the flat subset's current coverage.
                return Err(format!(
                    "Cannot codegen '{}' with --link-interface: it uses constructs outside the \
                     flat-codegen subset, and an imported body links only on the flat path (the AST \
                     codegen has no AST for it). The frontend check passed; this is a flat-coverage \
                     limit — see docs/discussions/implementation_plans/cross_module_return_provenance.md.",
                    filename
                ));
            }
            None => {
                if !self.options.legacy_codegen {
                    eprintln!("[flat-codegen] program outside the flat subset; using the AST path");
                }
                let mut codegen =
                    MeliorGenerator::new(&context, monomorphized_ast.module_path.to_string());
                codegen.emit_seam_certs = self.options.emit_seam_certs;
                codegen
                    .generate(&monomorphized_ast, &module_syntaxes)
                    .map_err(|e| format!("Codegen Error: {:?}", e))?;
                codegen.into_module()
            }
        };

        if !module.as_operation().verify() {
            return Err(format!("MLIR verification failed for {}", filename));
        }

        let llvm_lower = matches!(
            self.options.action,
            Action::EmitLlvm | Action::RunJit | Action::EmitObj
        );
        let pipeline_str = get_optimization_pipeline(
            self.options.opt_level,
            llvm_lower,
            self.options.disable_mlir_optimizations,
        );

        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.enable_verifier(true);
        if let Err(e) = melior::utility::parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            &pipeline_str,
        ) {
            return Err(format!("Failed to parse MLIR pass pipeline: {}", e));
        }

        // Apply custom CLI MLIR args if provided
        for arg in mlir_args {
            if let Some(custom_pipeline) = arg.strip_prefix("--pass-pipeline=") {
                if let Err(e) = melior::utility::parse_pass_pipeline(
                    pass_manager.as_operation_pass_manager(),
                    custom_pipeline,
                ) {
                    return Err(format!("Failed to parse custom pass pipeline: {}", e));
                }
            }
        }

        if let Err(e) = pass_manager.run(&mut module) {
            return Err(format!("MLIR passes failed for {}: {}", filename, e));
        }

        match self.options.action {
            Action::EmitMlir | Action::EmitLlvm => {
                // `--emit-llvm` alone prints the portable, target-independent LLVM-dialect
                // MLIR. Asking for a concrete backend (`--target`) means you want real IR for
                // it: set that target's triple / data layout as real module attributes (which
                // `mlir-translate` propagates to the `.ll`) and translate to actual `.ll`.
                if self.options.action == Action::EmitLlvm {
                    if let Some(target) = &self.options.target {
                        if let Some((triple, datalayout)) = target_triple_and_datalayout(target) {
                            use melior::ir::operation::OperationMutLike;
                            module.as_operation_mut().set_attribute(
                                "llvm.target_triple",
                                melior::ir::attribute::StringAttribute::new(&context, triple)
                                    .into(),
                            );
                            module.as_operation_mut().set_attribute(
                                "llvm.data_layout",
                                melior::ir::attribute::StringAttribute::new(&context, datalayout)
                                    .into(),
                            );
                        }
                        let mlir_str = format!("{}", module.as_operation());
                        let llvm_ir = translate_to_llvm_ir(&mlir_str, main_file)?;
                        println!("{}", llvm_ir);
                        return Ok(());
                    }
                }
                println!("{}", module.as_operation());
            }
            Action::RunJit => {
                let mlir_str = format!("{}", module.as_operation());
                let mut args = vec![self.options.inputs[0].to_string_lossy().into_owned()];
                args.extend(self.options.program_args.clone());
                let out = crate::jit::execute_mlir(
                    &mlir_str,
                    args,
                    self.options.opt_level,
                    self.options.disable_llvm_optimizations,
                )
                .map_err(|e| e.to_string())?;
                println!("{}", out);
            }
            Action::EmitObj => {
                let current_dir = std::env::current_dir().unwrap();
                let vx_std_core = format!(
                    "{}/target/debug/{}vx_std_core{}",
                    current_dir.display(),
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                );
                let libnpu = format!(
                    "{}/target/jit/{}npu_shared{}",
                    current_dir.display(),
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                );

                let mlir_c_runner =
                    format!("libmlir_c_runner_utils{}", std::env::consts::DLL_SUFFIX);
                let mlir_runner = format!("libmlir_runner_utils{}", std::env::consts::DLL_SUFFIX);
                let mut shared_libs = vec![
                    mlir_c_runner.clone(),
                    mlir_runner.clone(),
                    vx_std_core.clone(),
                ];
                if cfg!(target_os = "macos") {
                    shared_libs.push(libnpu.clone());
                }

                let shared_libs_refs: Vec<&str> = shared_libs.iter().map(|s| s.as_ref()).collect();

                let engine = melior::ExecutionEngine::new(
                    &module,
                    self.options.opt_level as usize,
                    &shared_libs_refs,
                    true,
                    true,
                );

                let output_path = self.options.output.clone().unwrap_or_else(|| {
                    let mut p = main_file.to_path_buf();
                    p.set_extension("o");
                    p
                });

                engine.dump_to_object_file(output_path.to_str().unwrap());
            }
            _ => {}
        }

        Ok(())
    }

    /// Build the MLIR module via the flat-array codegen path, or `None` if any function is outside the
    /// flat subset (the caller then falls back to the AST path). Resolves names, freezes the registry,
    /// lowers every non-generic function across all modules to flat HIR, and emits one module via
    /// `flat::emit_module_mlir` — mirroring the differential harness (#201). Programs whose structs need
    /// a registry-backed `StructInit` GID annotation (the driver type-checks against an empty registry)
    /// simply decline here and fall back to the AST path — never a wrong result.
    fn build_flat_module<'c>(
        context: &'c melior::Context,
        main_ast: &crate::syntax::Program,
        module_syntaxes: &std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        linked_interface: Option<&[u8]>,
    ) -> Option<melior::ir::Module<'c>> {
        // The main module first (its monomorphs win any name collision), then the imports; resolve
        // names so the registry freeze sees settled struct/enum GIDs.
        let mut mods: Vec<crate::syntax::Program> = vec![main_ast.clone()];
        mods.extend(module_syntaxes.values().cloned());
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        for m in &mut mods {
            m.resolve_names(&symbol_map);
        }
        let mut registry = crate::pipeline::build_frozen_registry(&mods).ok()?;
        // `--link-interface`: fold the precompiled interface's signatures + portable flat-HIR bodies
        // into this compile's registry, so an imported call resolves (`fn_sigs`) and its body links
        // (`body_of`) with no parse of the library (#265 step 7 / #220). Own entries win on collision.
        if let Some(bytes) = linked_interface {
            let imported = crate::metadata::deserialize_registry_interface(bytes).ok()?;
            registry.merge_from(imported);
        }
        let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

        // Re-run the type checker against the *frozen registry* purely to annotate each `StructInit`
        // with its struct GID (the driver's own semantic analysis ran against an empty registry, so
        // those GIDs are `None` and the flat lowerer would decline every struct). The bodies were
        // already checked + monomorphized, so this pass only settles the annotation; its diagnostics
        // and any re-collected monomorphs are discarded. (#215)
        // Sub-space descriptors for the flat emitter (P0-1): the frozen registry carries no memory
        // decls, so build them here from the per-compilation env — keyed by each space's dispatch id,
        // the same identity an `Opcode::Transfer` carries in its `imm`. Lets the flat path re-attach the
        // `space`/`within`/`granule`/`capacity`/`scope` + bump-allocated `offset`/`slots` attrs the AST
        // path emits (otherwise dropped on the default path — B1).
        let subspaces: Vec<crate::codegen::flat::SubspaceInfo>;
        {
            let env_mods = mods.clone();
            let env = GlobalAstEnv::build(&env_mods);
            for m in &mut mods {
                let mut scratch = LocalWorkerState::new(session.clone());
                let mut checker = TypeChecker::new(&env, &mut scratch);
                for f in &mut m.functions {
                    checker.check_function(f);
                }
            }
            subspaces = env
                .memories
                .values()
                .map(|decl| {
                    let space = crate::syntax::MemorySpace::from_name(decl.name.as_ref());
                    crate::codegen::flat::SubspaceInfo {
                        dispatch_id: crate::arch::memory_space_dispatch_id(&space) as u64,
                        name: space.name(),
                        within: decl.parent.as_ref().map(|p| p.name()),
                        granule: decl.granule.as_ref().map(|g| g.0),
                        capacity: decl.capacity.as_ref().map(|c| c.0),
                        scope: decl.scope.as_ref().map(|s| {
                            match s {
                                crate::syntax::Scope::Device => "device",
                                crate::syntax::Scope::Sm => "sm",
                                crate::syntax::Scope::Cta => "cta",
                                crate::syntax::Scope::Thread => "thread",
                            }
                            .to_string()
                        }),
                    }
                })
                .collect();
        }

        // Index every non-generic function by name (the main-module version wins any collision).
        let mut fn_map: std::collections::HashMap<crate::symbol::Symbol, &crate::syntax::Function> =
            std::collections::HashMap::new();
        for m in &mods {
            for f in &m.functions {
                if f.generics.is_empty() {
                    fn_map.entry(f.name.clone()).or_insert(f);
                }
            }
        }

        // Lower everything reachable from the main module (its own functions + the monomorphs
        // type-checking appended are the roots; a BFS over each body's called names pulls in
        // transitively-called *imported* functions on demand — not the whole imported module). An
        // extern is not a function (not in `fn_map`), so it is skipped here and declared
        // `func.func private` at emit; the JIT links it. Decline (→ AST fallback) if any reachable
        // function is outside the flat subset — never a wrong result.
        let mut worklist: Vec<crate::symbol::Symbol> = mods[0]
            .functions
            .iter()
            .filter(|f| f.generics.is_empty())
            .map(|f| f.name.clone())
            .collect();
        let mut lowered_names = std::collections::HashSet::new();
        let mut entries: Vec<(crate::syntax::Function, LocalWorkerState)> = Vec::new();
        while let Some(name) = worklist.pop() {
            if !lowered_names.insert(name.clone()) {
                continue;
            }
            let Some(f) = fn_map.get(&name).copied() else {
                continue; // an extern or a non-function name use -> not lowered here
            };
            let mut worker = LocalWorkerState::new(session.clone());
            if !crate::hir::flatten::lower_function_to_hir(f, &mut worker) {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg] HIR lowering declined: {}", f.name);
                }
                return None;
            }
            let mut uses = std::collections::HashSet::new();
            for s in &f.body {
                TypeChecker::extract_uses_stmt(s, &mut uses);
            }
            for u in uses {
                worklist.push(crate::symbol::Symbol::from(u.as_str()));
            }
            entries.push((f.clone(), worker));
        }
        // Append any *imported* function bodies the lowering referenced but did not lower locally
        // (absent from `fn_map` — their AST was never parsed). Their portable flat-HIR body comes from
        // the merged `.vxlib` interface via `body_of`; give each a signature-only `Function` to emit
        // against (the emitter reads only its `params`/`ret_ty` header). An import with no portable
        // body (an extern, or a generic) is skipped here — an extern is declared `private` at emit and
        // linked by the JIT (#265 step 7 / #220).
        let mut imported_entries: Vec<(
            crate::syntax::Function,
            Vec<crate::hir::bytecode::HirInstruction>,
            Vec<crate::gid::TypeId>,
        )> = Vec::new();
        for name in &lowered_names {
            if fn_map.contains_key(name) {
                continue;
            }
            let Some(sig) = session.registry.fn_sigs.get(name) else {
                continue;
            };
            let Some(body) = session.registry.body_of(sig.gid) else {
                continue;
            };
            let synth = crate::syntax::Function {
                name: body.name.clone(),
                generics: vec![],
                params: body
                    .params
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        (
                            crate::symbol::Symbol::from(format!("a{i}").as_str()),
                            t.clone(),
                        )
                    })
                    .collect(),
                topology: crate::syntax::Topology::CPU,
                return_type: body.ret_ty.clone(),
                requires: vec![],
                ensures: vec![],
                where_transfers: vec![],
                body: vec![],
                doc_comment: None,
            };
            imported_entries.push((synth, body.hir.clone(), body.types.clone()));
        }
        let mut funcs: Vec<(&crate::syntax::Function, &[_], &[_])> = entries
            .iter()
            .map(|(f, w)| {
                (
                    f,
                    w.local_hir_stream.as_slice(),
                    w.local_type_stream.as_slice(),
                )
            })
            .collect();
        funcs.extend(
            imported_entries
                .iter()
                .map(|(f, hir, types)| (f, hir.as_slice(), types.as_slice())),
        );
        let tensor_types: Vec<_> = entries
            .iter()
            .flat_map(|(_, w)| w.local_tensor_types.iter().cloned())
            .collect();
        // Per-function string tables, index-aligned with `funcs` (each `PrintStr`'s `imm` indexes into
        // its own function's table; the module emitter numbers them from a running base).
        let string_tables: Vec<&[String]> = entries
            .iter()
            .map(|(_, w)| w.local_string_table.as_slice())
            .collect();
        let agg_layouts: Vec<_> = entries
            .iter()
            .flat_map(|(_, w)| w.local_agg_layouts.iter().cloned())
            .collect();
        let text = match crate::codegen::flat::emit_module_mlir(
            &funcs,
            &session.registry,
            &tensor_types,
            &string_tables,
            &agg_layouts,
            &subspaces,
        ) {
            Some(t) => t,
            None => {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg] emit declined (a function outside the emitter subset)");
                }
                return None;
            }
        };

        let parsed = melior::ir::Module::parse(context, &format!("module {{\n{text}}}\n"));
        if parsed.is_none() && std::env::var("VX_FLAT_DBG").is_ok() {
            eprintln!("[flat-dbg] emitted flat MLIR failed to parse:\n{text}");
        }
        parsed
    }
}

fn get_optimization_pipeline(
    opt_level: u8,
    llvm_lower: bool,
    disable_mlir_optimizations: bool,
) -> String {
    let mut passes = vec![];

    if opt_level > 0 || llvm_lower {
        passes.push("convert-vx-to-standard".to_string());
    }

    if opt_level > 0 && !disable_mlir_optimizations {
        passes.push("canonicalize".to_string());
        passes.push("cse".to_string());
        passes.push(
            "func.func(affine-loop-fusion,affine-loop-tile,affine-loop-unroll,affine-scalrep)"
                .to_string(),
        );
        passes.push("lower-affine".to_string());
        passes.push("canonicalize".to_string());
        passes.push("cse".to_string());
    }

    if llvm_lower {
        passes.push("vx-to-llvm".to_string());
        passes.push("func.func(convert-linalg-to-loops,lower-affine)".to_string());
        passes.push("convert-scf-to-cf".to_string());
        passes.push("expand-strided-metadata".to_string());
        passes.push("finalize-memref-to-llvm".to_string());
        passes.push("convert-vector-to-llvm".to_string());
        passes.push("convert-func-to-llvm".to_string());
        passes.push("convert-index-to-llvm".to_string());
        passes.push("convert-math-to-llvm".to_string());
        passes.push("convert-math-to-libm".to_string());
        passes.push("convert-cf-to-llvm".to_string());
        passes.push("convert-arith-to-llvm".to_string());
        passes.push("reconcile-unrealized-casts".to_string());
    }

    format!("builtin.module({})", passes.join(","))
}

extern "C" {
    fn run_vx_opt(
        argc: std::os::raw::c_int,
        argv: *const *const std::os::raw::c_char,
    ) -> std::os::raw::c_int;
}

pub fn apply_mlir_opt(
    mlir_src: &str,
    mlir_args: &[String],
    _main_file: &std::path::Path,
) -> Result<String, String> {
    if mlir_args.is_empty() {
        return Ok(mlir_src.to_string());
    }

    let mut temp_in = tempfile::Builder::new()
        .prefix("vx_opt_in_")
        .suffix(".mlir")
        .tempfile()
        .map_err(|e| e.to_string())?;

    let temp_out = tempfile::Builder::new()
        .prefix("vx_opt_out_")
        .suffix(".mlir")
        .tempfile()
        .map_err(|e| e.to_string())?;

    std::io::Write::write_all(temp_in.as_file_mut(), mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut args = vec![
        "vx-opt".to_string(),
        temp_in.path().to_string_lossy().into_owned(),
        "-o".to_string(),
        temp_out.path().to_string_lossy().into_owned(),
    ];
    args.extend_from_slice(mlir_args);

    let c_args: Vec<std::ffi::CString> = args
        .into_iter()
        .map(|a| std::ffi::CString::new(a).unwrap())
        .collect();
    let c_ptrs: Vec<*const std::os::raw::c_char> = c_args.iter().map(|a| a.as_ptr()).collect();

    let status = unsafe { run_vx_opt(c_ptrs.len() as std::os::raw::c_int, c_ptrs.as_ptr()) };

    if status != 0 {
        return Err("vx-opt failed. Check stderr for details.".to_string());
    }

    let out_str = std::fs::read_to_string(temp_out.path()).unwrap_or_default();
    Ok(out_str)
}

/// The LLVM target triple and data layout for a named backend.
fn target_triple_and_datalayout(target: &str) -> Option<(&'static str, &'static str)> {
    match target {
        "x86_64" | "x86-64" | "x86" => Some((
            "x86_64-unknown-linux-gnu",
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
        )),
        "aarch64" | "arm64" => Some((
            "aarch64-unknown-linux-gnu",
            "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
        )),
        "nvptx64" | "nvptx" => Some((
            "nvptx64-nvidia-cuda",
            "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
        )),
        "amdgcn" | "amdgpu" => Some((
            "amdgcn-amd-amdhsa",
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",
        )),
        _ => None,
    }
}

pub fn translate_to_llvm_ir(
    mlir_src: &str,
    _main_file: &std::path::Path,
) -> Result<String, String> {
    use std::io::Write;
    let mut child = std::process::Command::new("mlir-translate")
        .arg("--mlir-to-llvmir")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn mlir-translate: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(mlir_src.as_bytes())
            .map_err(|e| e.to_string())?;
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(format!(
            "mlir-translate failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod flat_codegen_tests {
    use super::*;

    fn ctx() -> melior::Context {
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        let context = melior::Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        melior::utility::register_all_llvm_translations(&context);
        codegen::register_vx_dialect(&context);
        context
    }

    fn parse(src: &str) -> crate::syntax::Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut p = parser.parse().expect("parse");
        p.module_path = "crate::t".into();
        p
    }

    /// `--flat-codegen` produces a module for an in-subset program (scalar arithmetic, a scalar helper
    /// call, an extern call) and declines for one outside it, so the driver falls back to the AST path
    /// — never a wrong result (#201).
    #[test]
    fn flat_module_built_for_in_subset_declined_otherwise() {
        let context = ctx();
        let empty = std::collections::HashMap::new();

        // In subset: scalar arithmetic + a scalar helper call (both functions emit; the call resolves).
        let prog = parse(
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
             fn main() -> i32 { return add(3, 4) * 5; }",
        );
        assert!(CompilerDriver::build_flat_module(&context, &prog, &empty, None).is_some());

        // In subset: a libm extern call (declared `func.func private`, linked by the JIT).
        let ext = parse(
            "extern { safe fn sqrtf(x: f32) -> f32; }\n\
             fn main() -> i32 { print(sqrtf(16.0)); return 0; }",
        );
        assert!(CompilerDriver::build_flat_module(&context, &ext, &empty, None).is_some());

        // In subset: structs (including a struct return) now build through the flat path -- the flat
        // build runs a registry-backed type-check to annotate `StructInit` GIDs (#215).
        let strukt = parse(
            "struct P { x: i32, y: i32 }\n\
             fn mk() -> P { return P { x: 1, y: 2 }; }\n\
             fn main() -> i32 { let p = mk(); return p.x + p.y; }",
        );
        assert!(CompilerDriver::build_flat_module(&context, &strukt, &empty, None).is_some());

        // Outside the subset: a bare call to an *undefined* Vx function has no body to lower -> the
        // whole program declines -> AST fallback (never a wrong result).
        let unknown = parse("fn main() -> i32 { return mystery(1); }");
        assert!(CompilerDriver::build_flat_module(&context, &unknown, &empty, None).is_none());
    }
}
