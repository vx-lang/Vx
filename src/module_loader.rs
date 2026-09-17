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
    Parse(String, crate::error::Error),
    Resolution(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::IO(file, err) => write!(f, "Failed to open file '{}': {}", file, err),
            ModuleError::Parse(file, err) => write!(f, "Parse failed in '{}': {}", file, err),
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
    search_paths: Vec<PathBuf>,
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
}

impl ModuleLoader {
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        if let Ok(env_path) = std::env::var("VX_STD_PATH") {
            // Read as a PATH-style list rather than a single directory. Set to one directory it
            // replaced BOTH roots below, so `import graph::traversal` stopped resolving -- which
            // is exactly what an installed toolchain would have set it to.
            search_paths.extend(std::env::split_paths(&env_path));
        } else {
            // Beside the compiler, as an installed toolchain lays it out: `bin/vxc` and `stdlib/`
            // under one prefix. Checked before the working directory so a downloaded toolchain
            // resolves imports from any directory, with no environment variable set.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(bin_dir) = exe.parent() {
                    let prefix = bin_dir.join("..");
                    if prefix.join("stdlib").join("std").is_dir() {
                        search_paths.push(prefix.join("stdlib").join("std"));
                        search_paths.push(prefix.join("stdlib"));
                    }
                }
            }
            search_paths.push(PathBuf::from("stdlib/std"));
            // `stdlib` itself is a root so top-level libraries beyond `std` resolve, e.g.
            // `import graph::traversal` -> `stdlib/graph/traversal.vx`.
            search_paths.push(PathBuf::from("stdlib"));
        }

        search_paths.push(PathBuf::from("."));

        Self {
            search_paths,
            loaded_modules: HashMap::new(),
            loaded_interfaces: HashMap::new(),
            returned_modules_directly: false,
        }
    }

    pub fn load_main(&mut self, filename: &str) -> Result<(), ModuleError> {
        let main_program = Self::parse_file(Path::new(filename), filename.into())?;

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
            if seen.insert(root.as_str().into()) {
                wave.push((root.as_str().into(), PathBuf::from(root)));
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
            for result in parsed {
                let program = result?;
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
        Ok(programs)
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
    pub fn into_programs(self) -> Vec<Program> {
        assert!(
            !self.returned_modules_directly,
            "load_all already returned the modules it loaded; this loader kept none, so asking \
             it for them would compile an empty program"
        );
        self.loaded_modules.into_values().collect()
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

    /// Read and parse one file as the module `module_path`.
    fn parse_file(path: &Path, module_path: Symbol) -> Result<Program, ModuleError> {
        let name = path.to_string_lossy().into_owned();
        let source = fs::read_to_string(path).map_err(|e| ModuleError::IO(name.clone(), e))?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, &source);

        let mut program = parser
            .parse()
            .map_err(|e| ModuleError::Parse(name, crate::error::Error(e.format(&source))))?;
        program.module_path = module_path;
        Ok(program)
    }

    /// Resolve an import path against the search paths, trying the given file extension. Shared by
    /// `.vx` source resolution and `.vxlib` artifact resolution (#219).
    fn resolve_with_ext(&self, path: &[crate::symbol::Symbol], ext: &str) -> Option<PathBuf> {
        for search_path in &self.search_paths {
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

        let program = Self::parse_file(&resolved_path, module_name.clone())?;

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

    /// `load_all` hands its modules back rather than keeping them, so a caller who follows the
    /// `load_main` shape gets nothing from this loader. That used to be an empty `Vec` and an
    /// empty compile; it is a crash now, because a compiler that emits nothing and reports
    /// success is the worse of the two.
    #[test]
    #[should_panic(expected = "load_all already returned the modules")]
    fn asking_load_all_s_loader_for_its_modules_is_refused() {
        let mut loader = ModuleLoader::new();
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

        let mut recursive = ModuleLoader::new();
        recursive.load_main(entry).expect("the fixture should load");
        let from_load_main = names(recursive.into_programs());

        let mut waves = ModuleLoader::new();
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
        let mut loader = ModuleLoader::new();
        loader
            .load_main("tests/modules/jobs_wave_b.vx")
            .expect("the fixture module should parse");
        assert!(
            !loader.into_programs().is_empty(),
            "load_main kept nothing, so a compile through this loader would be empty"
        );
    }
}
