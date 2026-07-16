//===- module_loader.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
}

impl ModuleLoader {
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        if let Ok(env_path) = std::env::var("VX_STD_PATH") {
            search_paths.push(PathBuf::from(env_path));
        } else {
            search_paths.push(PathBuf::from("stdlib/std"));
            // `stdlib` itself is a root so top-level libraries beyond `std` resolve, e.g.
            // `import graph::traversal` -> `stdlib/graph/traversal.vx`.
            search_paths.push(PathBuf::from("stdlib"));
        }

        search_paths.push(PathBuf::from("."));

        Self {
            search_paths,
            loaded_modules: HashMap::new(),
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

    fn resolve_module_path(&self, path: &[crate::symbol::Symbol]) -> Option<PathBuf> {
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
            current_path.set_extension("vx");

            if current_path.exists() {
                return Some(current_path);
            }
        }
        None
    }

    fn load_import(&mut self, path: &[crate::symbol::Symbol]) -> Result<(), ModuleError> {
        let module_name = path
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join("::");
        if self.loaded_modules.contains_key(&*module_name) {
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
