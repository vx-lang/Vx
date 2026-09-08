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
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::syntax::Program;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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

pub struct ModuleLoader {
    search_paths: Vec<PathBuf>,
    pub loaded_modules: HashMap<crate::symbol::Symbol, Program>,
    /// Imports resolved to a precompiled `.vxlib` **interface** instead of `.vx` source (#219): the
    /// module's source is never parsed, and its serialized module-interface bytes are collected here
    /// (keyed by module name) for the driver to deserialize + merge into the frozen registry. This is
    /// the automatic form of `--link-interface`.
    pub loaded_interfaces: HashMap<crate::symbol::Symbol, Vec<u8>>,
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
        }
    }

    pub fn load_main(&mut self, filename: &str) -> Result<(), ModuleError> {
        let source =
            fs::read_to_string(filename).map_err(|e| ModuleError::IO(filename.to_string(), e))?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, &source);

        let mut main_program = parser.parse().map_err(|e| {
            ModuleError::Parse(filename.to_string(), crate::error::Error(e.format(&source)))
        })?;
        main_program.module_path = filename.to_string().into();

        let imports = main_program.imports.clone();
        self.loaded_modules
            .insert(filename.to_string().into(), main_program);

        for import in imports {
            self.load_import(&import.path)?;
        }

        Ok(())
    }

    pub fn into_programs(self) -> Vec<Program> {
        self.loaded_modules.into_values().collect()
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

    fn load_import(&mut self, path: &[crate::symbol::Symbol]) -> Result<(), ModuleError> {
        let module_name = path
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join("::");
        if self.loaded_modules.contains_key(&*module_name)
            || self.loaded_interfaces.contains_key(&*module_name)
        {
            return Ok(());
        }

        // Prefer a precompiled `.vxlib` interface if one sits where the source would (#219): collect
        // its serialized module-interface bytes and skip parsing the module's source entirely — the
        // automatic form of `--link-interface`. The artifact is self-contained (its own dependencies
        // were baked into its registry at emit time), so no further import recursion is needed.
        if let Some(artifact) = self.resolve_artifact_path(path) {
            let buf = fs::read(&artifact)
                .map_err(|e| ModuleError::IO(artifact.to_string_lossy().into_owned(), e))?;
            let meta = crate::metadata::VxMetadata::load_from_buffer(&buf);
            self.loaded_interfaces
                .insert(module_name.as_str().into(), meta.interface_data.to_vec());
            return Ok(());
        }

        let resolved_path = match self.resolve_module_path(path) {
            Some(p) => p,
            None => {
                return Err(ModuleError::Resolution(format!(
                    "Could not resolve import '{}'",
                    module_name
                )))
            }
        };

        let source = fs::read_to_string(&resolved_path)
            .map_err(|e| ModuleError::IO(resolved_path.to_string_lossy().into_owned(), e))?;

        // Need to keep the source string alive, we might leak it or use a string interner.
        // For now, since AST holds string slices to source, `ModuleLoader` should probably return Strings too?
        // Wait! `Parser` takes `&'a str`. `syntax::Program` borrows from source? No, `syntax::Program` clones strings. Let's check `parser.rs`.
        // `Parser::new(&tokens, source)` takes `&'a str`.
        // Does `Program` contain lifetimes? No. `Program` uses `String`.

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, &source);

        let mut program = parser.parse().map_err(|e| {
            ModuleError::Parse(
                resolved_path.to_string_lossy().into_owned(),
                crate::error::Error(e.format(&source)),
            )
        })?;
        program.module_path = module_name.clone().into();

        let imports = program.imports.clone();
        self.loaded_modules
            .insert(module_name.clone().into(), program);

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
