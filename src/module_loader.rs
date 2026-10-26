//===- module_loader.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file handles the filesystem interaction and parsing of Vx source files.
// It is responsible for recursively discovering dependencies, reading source code
// from disk, invoking the parser, and feeding the resulting ASTs into the global
// module registry.
//
//===----------------------------------------------------------------------===//
use crate::config::Schedule;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::symbol::Symbol;
use crate::syntax::Program;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ModuleError {
    IO(String, std::io::Error),
    /// The error names its file and position itself.
    Parse(crate::error::Error),
    Resolution(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::IO(file, err) => write!(f, "Failed to open file '{}': {}", file, err),
            ModuleError::Parse(err) => write!(f, "{}", err),
            ModuleError::Resolution(err) => write!(f, "Resolution error: {}", err),
        }
    }
}

/// What an import resolved to, for [`ModuleLoader::load_all`].
enum Resolved {
    /// A `.vx` file, to be parsed in the next wave.
    Source(PathBuf),
    /// The serialized interface of a precompiled `.vxlib`, never parsed.
    Interface(Vec<u8>),
}

pub struct ModuleLoader {
    /// Where the standard library is: `--std-path`, or found from the compiler itself.
    std_roots: Vec<PathBuf>,
    /// The directory of each file compiled as a root, so a program can import a file beside it.
    entry_dirs: Vec<PathBuf>,
    /// `--package name=dir`: an import that starts with `name` is read from `dir` and nowhere else.
    packages: Vec<(String, PathBuf)>,
    pub loaded_modules: HashMap<crate::symbol::Symbol, Program>,
    /// Imports resolved to a precompiled `.vxlib` **interface** instead of `.vx` source (#219): the
    /// module's source is never parsed, and its serialized module-interface bytes are collected here
    /// (keyed by module name) for the driver to deserialize + merge into the frozen registry. This is
    /// the automatic form of `--link-interface`.
    pub loaded_interfaces: HashMap<crate::symbol::Symbol, Vec<u8>>,
    /// Set by [`ModuleLoader::load_all`], which returns the modules it parsed rather than storing
    /// them here. Read by [`ModuleLoader::into_programs`], so that asking this loader for modules
    /// it never kept is a crash instead of an empty compile.
    returned_modules_directly: bool,
    /// Where each parsed module was read from, to parse it again with its clashing names renamed.
    paths: HashMap<Symbol, PathBuf>,
    /// The modules loaded as roots rather than imported, in load order.
    roots: Vec<Symbol>,
    /// The name each file got when it was first loaded, by its canonical path. A file loaded
    /// both as a root and by an import path keeps that one name.
    file_names: HashMap<PathBuf, Symbol>,
}

impl ModuleLoader {
    /// A loader that finds the standard library from the compiler itself.
    pub fn new() -> Self {
        Self::with_std_paths(Self::default_std_roots())
    }

    /// A loader that reads the standard library from `std_paths` only (`--std-path`).
    pub fn with_std_paths(std_paths: Vec<PathBuf>) -> Self {
        Self {
            std_roots: std_paths,
            entry_dirs: Vec::new(),
            packages: Vec::new(),
            loaded_modules: HashMap::new(),
            loaded_interfaces: HashMap::new(),
            returned_modules_directly: false,
            paths: HashMap::new(),
            roots: Vec::new(),
            file_names: HashMap::new(),
        }
    }

    /// The standard library beside the compiler, as an installed toolchain lays it out (`bin/vxc`
    /// and `stdlib/` under one prefix), or else in the source tree the compiler was built from.
    /// Never the current directory: one command must read the same files wherever it runs.
    fn default_std_roots() -> Vec<PathBuf> {
        let beside_the_compiler = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|bin| bin.join("..").join("stdlib")));
        let source_tree = Path::new(env!("CARGO_MANIFEST_DIR")).join("stdlib");
        for stdlib in beside_the_compiler.into_iter().chain([source_tree]) {
            if stdlib.join("std").is_dir() {
                // `stdlib` itself is a root so libraries beyond `std` resolve too:
                // `import graph::traversal` is `stdlib/graph/traversal.vx`.
                return vec![stdlib.join("std"), stdlib];
            }
        }
        Vec::new()
    }

    /// Read imports that start with `name` from `dir`: `import name::a::b` is `dir/a/b.vx`. The
    /// search paths are not tried for them, so the package is the only place `name` comes from.
    pub fn add_package(&mut self, name: &str, dir: &Path) {
        self.packages.push((name.to_string(), dir.to_path_buf()));
    }

    /// Let the program import files that sit beside `root`, one of the files it compiles.
    fn add_entry_dir(&mut self, root: &Path) {
        let dir = match root.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
            _ => PathBuf::from("."),
        };
        if !self.entry_dirs.contains(&dir) {
            self.entry_dirs.push(dir);
        }
    }

    pub fn load_main(&mut self, filename: &str) -> Result<(), ModuleError> {
        self.add_entry_dir(Path::new(filename));
        let main_program = Self::parse_file(Path::new(filename), filename.into())?;
        self.paths.insert(filename.into(), PathBuf::from(filename));
        self.roots.push(filename.into());

        let imports = main_program.imports.clone();
        self.loaded_modules
            .insert(filename.to_string().into(), main_program);

        for import in imports {
            self.load_import(&import.path)?;
        }

        Ok(())
    }

    /// Load `roots` and every module they import, transitively, and return the programs in the
    /// order they were discovered: the roots first, then each wave of imports.
    ///
    /// The files of one wave do not depend on each other, so a wave is parsed in parallel under a
    /// parallel schedule; this is `vxc -j`'s loader. Resolving the next wave's import names is
    /// parallel too: each is a handful of path probes, which is nothing until there are a
    /// thousand of them, when done one after another they cost more than parsing the files did.
    /// The order comes from the source, never from the threads, so every schedule loads the same
    /// modules in the same order.
    ///
    /// A root is known by its file name and an import by its import path, as in `load_main`. An
    /// import that resolves to a `.vxlib` interface lands in `loaded_interfaces` and is not parsed.
    pub fn load_all(
        &mut self,
        roots: &[String],
        sched: Schedule,
    ) -> Result<Vec<Program>, ModuleError> {
        self.returned_modules_directly = true;
        let mut programs: Vec<Program> = Vec::new();
        let mut seen: HashSet<Symbol> = HashSet::new();
        let mut wave: Vec<(Symbol, PathBuf)> = Vec::new();
        for root in roots {
            self.add_entry_dir(Path::new(root));
            if seen.insert(root.as_str().into()) {
                wave.push((root.as_str().into(), PathBuf::from(root)));
                self.roots.push(root.as_str().into());
            }
        }
        while !wave.is_empty() {
            let parse = |(module_path, path): &(Symbol, PathBuf)| {
                Self::parse_file(path, module_path.clone())
            };
            let parsed: Vec<Result<Program, ModuleError>> = if sched.is_seq() {
                wave.iter().map(parse).collect()
            } else {
                wave.par_iter().map(parse).collect()
            };
            // The next wave's imports, in source order, each once.
            let mut pending: Vec<(Symbol, Vec<Symbol>)> = Vec::new();
            for (result, (_, path)) in parsed.into_iter().zip(&wave) {
                let mut program = result?;
                self.name_file_once(&mut program, path)?;
                for import in &program.imports {
                    let module_name = Self::module_name(&import.path);
                    if seen.insert(module_name.clone()) {
                        pending.push((module_name, import.path.clone()));
                    }
                }
                programs.push(program);
            }
            let resolve = |(_, path): &(Symbol, Vec<Symbol>)| self.resolve_import(path);
            let resolved: Vec<Result<Resolved, ModuleError>> = if sched.is_seq() {
                pending.iter().map(resolve).collect()
            } else {
                pending.par_iter().map(resolve).collect()
            };
            for (module_name, path) in &wave {
                self.paths.insert(module_name.clone(), path.clone());
            }
            wave = Vec::new();
            for ((module_name, _), found) in pending.into_iter().zip(resolved) {
                match found? {
                    Resolved::Source(path) => wave.push((module_name, path)),
                    Resolved::Interface(bytes) => {
                        self.loaded_interfaces.insert(module_name, bytes);
                    }
                }
            }
        }
        self.rename_clashes(&mut programs)?;
        Ok(programs)
    }

    /// Parse again each module that declares or sees a name another module also declares, with
    /// that name renamed where it is not the declaration the whole compile keeps.
    fn rename_clashes(&self, programs: &mut [Program]) -> Result<(), ModuleError> {
        let plan =
            crate::name_clashes::plan(programs, &self.roots).map_err(ModuleError::Resolution)?;
        for program in programs.iter_mut() {
            let Some(renames) = plan.get(&program.module_path) else {
                continue;
            };
            let path = &self.paths[&program.module_path];
            let mut renamed =
                Self::parse_file_renaming(path, program.module_path.clone(), renames)?;
            // The imports were already followed as first parsed.
            assert_eq!(
                renamed.imports, program.imports,
                "renaming a clashing name changed an import path"
            );
            renamed.file = program.file.clone();
            *program = renamed;
        }
        Ok(())
    }

    /// Where an import leads: a source file to parse, or a precompiled interface, which is
    /// preferred when both sit where the source would and is read here so nothing about
    /// resolution is left for the caller to do serially.
    fn resolve_import(&self, path: &[Symbol]) -> Result<Resolved, ModuleError> {
        if let Some(artifact) = self.resolve_artifact_path(path) {
            let buf = fs::read(&artifact)
                .map_err(|e| ModuleError::IO(artifact.to_string_lossy().into_owned(), e))?;
            let meta = crate::metadata::VxMetadata::load_from_buffer(&buf);
            return Ok(Resolved::Interface(meta.interface_data.to_vec()));
        }
        match self.resolve_module_path(path) {
            Some(p) => Ok(Resolved::Source(p)),
            None => Err(ModuleError::Resolution(format!(
                "Could not resolve import '{}'",
                Self::module_name(path)
            ))),
        }
    }

    /// The modules this loader kept, for the `load_main` entry point that stores them.
    ///
    /// [`ModuleLoader::load_all`] hands its modules straight back to its caller instead, so it
    /// leaves nothing here. Calling this after it used to return an empty `Vec`, and a caller
    /// following the older shape -- `load_all(..)?; let programs = loader.into_programs();` --
    /// compiled nothing at all, with no error anywhere to say so.
    pub fn into_programs(self) -> Result<Vec<Program>, ModuleError> {
        assert!(
            !self.returned_modules_directly,
            "load_all already returned the modules it loaded; this loader kept none, so asking \
             it for them would compile an empty program"
        );
        let mut programs: Vec<Program> = self.loaded_modules.values().cloned().collect();
        self.rename_clashes(&mut programs)?;
        Ok(programs)
    }

    /// The name a module is known by: its import path joined with `::`.
    fn module_name(path: &[Symbol]) -> Symbol {
        path.iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join("::")
            .as_str()
            .into()
    }

    /// Give `program` the file name that the first load of the same file got. Otherwise a file
    /// loaded as a root and again by an import path has two names, and its impl blocks no
    /// longer compare equal, so E3049 reports one impl as two.
    fn name_file_once(&mut self, program: &mut Program, path: &Path) -> Result<(), ModuleError> {
        let key = fs::canonicalize(path)
            .map_err(|e| ModuleError::IO(path.to_string_lossy().into_owned(), e))?;
        let first = self
            .file_names
            .entry(key)
            .or_insert_with(|| program.file.clone().expect("a parsed file has a name"));
        program.file = Some(first.clone());
        Ok(())
    }

    /// Read and parse one file as the module `module_path`.
    fn parse_file(path: &Path, module_path: Symbol) -> Result<Program, ModuleError> {
        Self::parse_file_renaming(path, module_path, &HashMap::new())
    }

    fn parse_file_renaming(
        path: &Path,
        module_path: Symbol,
        renames: &HashMap<Symbol, crate::name_clashes::Rename>,
    ) -> Result<Program, ModuleError> {
        let name = path.to_string_lossy().into_owned();
        let source = fs::read_to_string(path).map_err(|e| ModuleError::IO(name.clone(), e))?;

        let mut lexer = Lexer::new(&source);
        let mut tokens = lexer.tokenize();
        if !renames.is_empty() {
            crate::name_clashes::rename_tokens(&mut tokens, renames)
                .map_err(|e| ModuleError::Resolution(format!("{name}: {e}")))?;
        }
        let mut parser = Parser::new(&tokens, &source);

        let mut program = parser
            .parse()
            .map_err(|e| ModuleError::Parse(crate::error::Error(e.format(&source, Some(&name)))))?;
        program.module_path = module_path;
        program.file = Some(name.as_str().into());
        Ok(program)
    }

    /// Resolve an import path against the standard library roots, then the directories of the
    /// root files, trying the given file extension. Shared by `.vx` source resolution and `.vxlib`
    /// artifact resolution (#219).
    fn resolve_with_ext(&self, path: &[crate::symbol::Symbol], ext: &str) -> Option<PathBuf> {
        if path.len() > 1 {
            if let Some((_, dir)) = self.packages.iter().find(|(name, _)| *path[0] == **name) {
                let mut current_path = dir.clone();
                for component in &path[1..] {
                    current_path.push(component.as_ref());
                }
                current_path.set_extension(ext);
                return current_path.exists().then_some(current_path);
            }
        }
        for search_path in self.std_roots.iter().chain(&self.entry_dirs) {
            let mut current_path = search_path.clone();
            if !path.is_empty() && *path[0] == *"std" {
                for component in &path[1..] {
                    current_path.push(component.as_ref());
                }
            } else {
                for component in path {
                    current_path.push(component.as_ref());
                }
            }
            current_path.set_extension(ext);

            if current_path.exists() {
                return Some(current_path);
            }
        }
        None
    }

    fn resolve_module_path(&self, path: &[crate::symbol::Symbol]) -> Option<PathBuf> {
        self.resolve_with_ext(path, "vx")
    }

    /// Resolve an import to a precompiled `.vxlib` interface artifact, if one sits where the `.vx`
    /// source would (#219). Preferred over the source: loading it skips parsing the module entirely.
    fn resolve_artifact_path(&self, path: &[crate::symbol::Symbol]) -> Option<PathBuf> {
        self.resolve_with_ext(path, "vxlib")
    }

    fn load_import(&mut self, path: &[Symbol]) -> Result<(), ModuleError> {
        let module_name = Self::module_name(path);
        if self.loaded_modules.contains_key(&module_name)
            || self.loaded_interfaces.contains_key(&module_name)
        {
            return Ok(());
        }

        // A precompiled interface is preferred over the source, and taking it means the module's
        // source is never parsed -- the automatic form of `--link-interface`. The artifact is
        // self-contained (its own dependencies were baked into its registry when it was emitted),
        // so there is no further import recursion to do for it.
        //
        // Both loaders ask `resolve_import` rather than each probing for the artifact themselves,
        // so the two cannot come to different conclusions about which file an import leads to.
        let resolved_path = match self.resolve_import(path)? {
            Resolved::Interface(bytes) => {
                self.loaded_interfaces.insert(module_name, bytes);
                return Ok(());
            }
            Resolved::Source(p) => p,
        };

        let mut program = Self::parse_file(&resolved_path, module_name.clone())?;
        self.name_file_once(&mut program, &resolved_path)?;
        self.paths.insert(module_name.clone(), resolved_path);

        let imports = program.imports.clone();
        self.loaded_modules.insert(module_name, program);

        for import in imports {
            self.load_import(&import.path)?;
        }

        Ok(())
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loader that reads the repository's `tests` directory as the package `tests`, which the
    /// fixtures below import from.
    fn loader() -> ModuleLoader {
        let mut loader = ModuleLoader::new();
        loader.add_package(
            "tests",
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"),
        );
        loader
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("vx_module_loader_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A program in another directory imports a file that sits beside it.
    #[test]
    fn an_import_resolves_beside_the_entry_file() {
        let dir = scratch_dir("beside_the_entry_file");
        fs::write(dir.join("helper.vx"), "fn helper() -> i32 { return 1; }\n").unwrap();
        let main = dir.join("main.vx");
        fs::write(
            &main,
            "import helper;\nfn main() -> i32 { return helper(); }\n",
        )
        .unwrap();
        let programs = ModuleLoader::new()
            .load_all(&[main.to_string_lossy().into_owned()], Schedule::Sequential)
            .expect("the file beside the entry file should be found");
        assert_eq!(programs.len(), 2, "expected main.vx and helper.vx");
    }

    /// `tests/modules/jobs_wave_b.vx` is under the current directory of `cargo test`, but no flag
    /// names that directory, so the import must fail wherever the compiler runs.
    #[test]
    fn an_import_is_not_read_from_the_current_directory() {
        assert!(
            Path::new("tests/modules/jobs_wave_b.vx").exists(),
            "this test needs the file under the current directory, or it checks nothing"
        );
        let dir = scratch_dir("not_the_current_directory");
        let main = dir.join("main.vx");
        fs::write(
            &main,
            "import tests::modules::jobs_wave_b;\nfn main() -> i32 { return 0; }\n",
        )
        .unwrap();
        match ModuleLoader::new()
            .load_all(&[main.to_string_lossy().into_owned()], Schedule::Sequential)
        {
            Ok(_) => panic!("the import was read from the current directory"),
            Err(err) => assert!(
                err.to_string()
                    .contains("Could not resolve import 'tests::modules::jobs_wave_b'"),
                "unexpected error: {err}"
            ),
        }
    }

    /// A relative root would be read from the current directory.
    #[test]
    fn the_standard_library_roots_are_absolute_paths() {
        let roots = ModuleLoader::default_std_roots();
        assert!(
            !roots.is_empty(),
            "no standard library beside the compiler or in its source tree"
        );
        for root in &roots {
            assert!(root.is_absolute(), "{} is a relative path", root.display());
        }
    }

    /// `load_all` hands its modules back rather than keeping them, so a caller who follows the
    /// `load_main` shape gets nothing from this loader. That used to be an empty `Vec` and an
    /// empty compile; it is a crash now, because a compiler that emits nothing and reports
    /// success is the worse of the two.
    #[test]
    #[should_panic(expected = "load_all already returned the modules")]
    fn asking_load_all_s_loader_for_its_modules_is_refused() {
        let mut loader = loader();
        let _ = loader.load_all(
            &["tests/modules/jobs_wave_b.vx".to_string()],
            Schedule::Sequential,
        );
        let _ = loader.into_programs();
    }

    /// The two loaders follow imports by different machinery -- one recurses, one walks waves in
    /// parallel -- but they must agree on where an import leads. They ask one function now; before
    /// that each probed for a `.vxlib` beside the source itself, so the rule about which wins was
    /// written down twice and could be changed in one place only.
    #[test]
    fn both_loaders_find_the_same_modules() {
        let entry = "tests/frontend/pass/jobs_loads_imports_in_waves.vx";
        let names = |mut programs: Vec<Program>| -> Vec<String> {
            let mut out: Vec<String> = programs
                .drain(..)
                .map(|p| p.module_path.as_ref().to_string())
                .collect();
            out.sort();
            out
        };

        let mut recursive = loader();
        recursive.load_main(entry).expect("the fixture should load");
        let from_load_main = names(recursive.into_programs().unwrap());

        let mut waves = loader();
        let from_load_all = names(
            waves
                .load_all(&[entry.to_string()], Schedule::Sequential)
                .expect("the fixture should load"),
        );

        assert_eq!(
            from_load_main, from_load_all,
            "the two loaders disagree about which modules this program is made of"
        );
        assert!(
            from_load_all.len() > 1,
            "the fixture stopped importing anything, so this compares nothing: {from_load_all:?}"
        );
    }

    /// The `load_main` entry point does keep them, and this is the shape that reads them back.
    #[test]
    fn load_main_keeps_the_modules_it_parsed() {
        let mut loader = loader();
        loader
            .load_main("tests/modules/jobs_wave_b.vx")
            .expect("the fixture module should parse");
        assert!(
            !loader.into_programs().unwrap().is_empty(),
            "load_main kept nothing, so a compile through this loader would be empty"
        );
    }

    /// A file that is a root and is also imported, by itself here, is loaded twice. Both
    /// loads name the file the way the root was written, in both loaders.
    #[test]
    fn a_file_loaded_twice_has_one_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.vx"), "import pkg::a;\n").unwrap();
        // Written with `./`, so the root's name differs from the path the import finds.
        let root = format!("{}/./a.vx", dir.path().display());
        let files = |programs: Vec<Program>| -> Vec<Option<Symbol>> {
            programs.into_iter().map(|p| p.file).collect()
        };

        let mut recursive = ModuleLoader::new();
        recursive.add_package("pkg", dir.path());
        recursive.load_main(&root).unwrap();
        let mut waves = ModuleLoader::new();
        waves.add_package("pkg", dir.path());
        let from_load_all = waves
            .load_all(std::slice::from_ref(&root), Schedule::Sequential)
            .unwrap();

        let expected = vec![Some(Symbol::from(root.as_str())); 2];
        assert_eq!(files(recursive.into_programs().unwrap()), expected);
        assert_eq!(files(from_load_all), expected);
    }
}
