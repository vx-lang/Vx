use crate::ast::Program;
use crate::lexer::Lexer;
use crate::parser::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub struct ModuleLoader {
    search_paths: Vec<PathBuf>,
    loaded_modules: HashMap<String, Program>,
}

impl ModuleLoader {
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        // Add default stdlib path
        if let Ok(env_path) = std::env::var("VX_STD_PATH") {
            search_paths.push(PathBuf::from(env_path));
        } else {
            search_paths.push(PathBuf::from("stdlib/std"));
        }

        Self {
            search_paths,
            loaded_modules: HashMap::new(),
        }
    }

    pub fn load_main(&mut self, filename: &str) -> Result<Vec<Program>, String> {
        let source = fs::read_to_string(filename)
            .map_err(|e| format!("Failed to open file: {} - {}", filename, e))?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, &source);

        let mut main_program = parser.parse()?;
        main_program.module_path = filename.to_string();

        let imports = main_program.imports.clone();
        self.loaded_modules
            .insert(filename.to_string(), main_program);

        for import in imports {
            self.load_import(&import.path)?;
        }

        Ok(self.loaded_modules.clone().into_values().collect())
    }

    fn load_import(&mut self, path: &[String]) -> Result<(), String> {
        let module_name = path.join("::");
        if self.loaded_modules.contains_key(&module_name) {
            return Ok(());
        }

        // Try to resolve the path
        // For example `std::vec` -> `stdlib/std/vec.vx`
        let mut resolved_path = None;

        for search_path in &self.search_paths {
            // Check if it's a stdlib import
            let mut current_path = search_path.clone();
            if !path.is_empty() && path[0] == "std" {
                // `std::collections::vec` -> `stdlib/std/collections/vec.vx`
                for component in &path[1..] {
                    current_path.push(component);
                }
                current_path.set_extension("vx");
            } else {
                for component in path {
                    current_path.push(component);
                }
                current_path.set_extension("vx");
            }

            if current_path.exists() {
                resolved_path = Some(current_path);
                break;
            }
        }

        let resolved_path = match resolved_path {
            Some(p) => p,
            None => return Err(format!("Could not resolve import '{}'", module_name)),
        };

        let source = fs::read_to_string(&resolved_path)
            .map_err(|e| format!("Failed to open imported file: {:?} - {}", resolved_path, e))?;

        // Need to keep the source string alive, we might leak it or use a string interner.
        // For now, since AST holds string slices to source, `ModuleLoader` should probably return Strings too?
        // Wait! `Parser` takes `&'a str`. `ast::Program` borrows from source? No, `ast::Program` clones strings. Let's check `parser.rs`.
        // `Parser::new(tokens, source)` takes `&'a str`.
        // Does `Program` contain lifetimes? No. `Program` uses `String`.

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, &source);

        let mut program = parser.parse()?;
        program.module_path = module_name.clone();

        let imports = program.imports.clone();
        self.loaded_modules.insert(module_name.clone(), program);

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
