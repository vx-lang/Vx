use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::module_loader::ModuleLoader;
use crate::session::{GlobalSession, LocalWorkerState};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct IdeDiagnostic {
    pub line_start: usize,
    pub col_start: usize,
    pub line_end: usize,
    pub col_end: usize,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct HoverInfo {
    pub value: String,
    pub line_start: usize,
    pub col_start: usize,
    pub line_end: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone)]
pub struct DefinitionLocation {
    pub uri: String,
    pub line_start: usize,
    pub col_start: usize,
    pub line_end: usize,
    pub col_end: usize,
}

#[derive(Default)]
pub struct AnalysisHost {
    pub files: HashMap<String, String>,
}

impl AnalysisHost {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    pub fn apply_change(&mut self, uri: String, text: String) {
        self.files.insert(uri, text);
    }

    pub fn snapshot(&self) -> Analysis {
        Analysis {
            files: self.files.clone(),
        }
    }
}

pub struct Analysis {
    pub files: HashMap<String, String>,
}

impl Analysis {
    pub fn diagnostics(&self, uri: &str) -> Vec<IdeDiagnostic> {
        let text = match self.files.get(uri) {
            Some(t) => t,
            None => return vec![],
        };

        let mut diagnostics = Vec::new();

        // Write to temp file for the module loader (simplified for now)
        let temp_path = "/Users/adityak/go/Vx/vx-analyzer/temp.vx";
        std::fs::write(temp_path, text).unwrap_or_default();

        let mut loader = ModuleLoader::new();
        if let Err(e) = loader.load_main(temp_path) {
            let err_str = format!("{}", e);
            let mut line = 1;
            let mut col = 1;

            if let Some(idx) = err_str.find("Error at ") {
                let rest = &err_str[idx + 9..];
                if let Some(colon_idx) = rest.find(':') {
                    if let Ok(l) = rest[..colon_idx].parse::<usize>() {
                        line = l;
                        let rest_after_col = &rest[colon_idx + 1..];
                        if let Some(second_colon) = rest_after_col.find(':') {
                            if let Ok(c) = rest_after_col[..second_colon].parse::<usize>() {
                                col = c;
                            }
                        }
                    }
                }
            }

            diagnostics.push(IdeDiagnostic {
                line_start: line.saturating_sub(1),
                col_start: col.saturating_sub(1),
                line_end: line.saturating_sub(1),
                col_end: col,
                message: err_str,
            });

            return diagnostics;
        }

        use crate::parser::MacroExpander;

        let mut modules: Vec<_> = loader.loaded_modules.values().cloned().collect();

        // 1. Macro Expansion
        let mut global_macros = std::collections::HashMap::new();
        for m in modules.iter() {
            for mac in &m.macros {
                global_macros.insert(mac.name.clone(), mac.rules.clone());
            }
        }
        let expander = MacroExpander::new(&global_macros);
        for m in modules.iter_mut() {
            let _ = expander.expand_module(m); // Ignore expansion errors for IDE
        }

        // 2. Name Resolution
        let symbol_map = crate::resolver::build_symbol_map(&modules);
        for m in modules.iter_mut() {
            m.resolve_names(&symbol_map);
        }

        let global_env_modules: Vec<_> = modules.iter().map(|m| m.clone_signature()).collect();
        let mut global_env = GlobalAstEnv::build(&global_env_modules);
        global_env.annotate_return_provenances(&modules);
        let session = std::sync::Arc::new(GlobalSession::new(0));

        for p in &mut modules {
            for func in &mut p.functions {
                let mut worker_state = LocalWorkerState::new(session.clone());
                let mut checker = TypeChecker::new(&global_env, &mut worker_state);
                checker.check_function(func);

                for d in checker.errors.into_iter() {
                    let (ls, cs) = match &d.source_span {
                        Some(s) => (s.line.saturating_sub(1), s.column.saturating_sub(1)),
                        None => (0, 0),
                    };
                    let (le, ce) = match &d.source_span {
                        Some(s) => (
                            s.line.saturating_sub(1),
                            s.column.saturating_sub(1) + s.length.max(1),
                        ),
                        None => (0, 0),
                    };
                    diagnostics.push(IdeDiagnostic {
                        line_start: ls,
                        col_start: cs,
                        line_end: le,
                        col_end: ce,
                        message: d.message.to_string(),
                    });
                }
            }
        }

        diagnostics
    }

    pub fn hover(&self, uri: &str, line: usize, col: usize) -> Option<HoverInfo> {
        let text = self.files.get(uri)?;

        // 1. Find the word under the cursor
        let target_line = text.lines().nth(line)?;

        if col >= target_line.len() {
            return None;
        }

        let mut start_col = col;
        let mut end_col = col;
        let chars: Vec<char> = target_line.chars().collect();

        while start_col > 0
            && (chars[start_col - 1].is_alphanumeric() || chars[start_col - 1] == '_')
        {
            start_col -= 1;
        }

        while end_col < chars.len() && (chars[end_col].is_alphanumeric() || chars[end_col] == '_') {
            end_col += 1;
        }

        if start_col == end_col {
            return None; // Not a word
        }

        let word: String = chars[start_col..end_col].iter().collect();
        let word_sym: crate::symbol::Symbol = word.clone().into();

        // 2. Load the environment to resolve the word
        let temp_path = "/Users/adityak/go/Vx/vx-analyzer/temp_hover.vx";
        std::fs::write(temp_path, text).unwrap_or_default();
        let mut loader = ModuleLoader::new();
        if let Err(e) = loader.load_main(temp_path) {
            std::fs::write("/tmp/vx_hover_err.log", format!("{:?}", e)).ok();
        } else {
            std::fs::write("/tmp/vx_hover_err.log", "load_main succeeded").ok();
        }

        let modules: Vec<_> = loader.loaded_modules.values().cloned().collect();
        let global_env_modules: Vec<_> = modules.iter().map(|m| m.clone_signature()).collect();
        let mut global_env = GlobalAstEnv::build(&global_env_modules);
        global_env.annotate_return_provenances(&modules);

        // 3. Look up the word in the environment
        let mut hover_text = String::new();

        let mut doc = None;

        if let Some((ty, is_unsafe, params, top, _, _)) = global_env.functions.get(&word_sym) {
            hover_text.push_str(&format!("fn {}(", word));
            let params_str = params
                .iter()
                .map(|t| format!("{}", t))
                .collect::<Vec<_>>()
                .join(", ");
            hover_text.push_str(&params_str);
            hover_text.push_str(&format!(") -> {}", ty));
            if *is_unsafe {
                hover_text = format!("unsafe {}", hover_text);
            }
            hover_text.push_str(&format!(" [Topology: {:?}]", top.kind()));

            if let Some(f) = global_env.syntax_functions.get(&word_sym) {
                doc = f.doc_comment.clone();
            }
        } else if let Some(struct_decl) = global_env.structs.get(&word_sym) {
            hover_text.push_str(&format!("struct {} {{\n", word));
            for (name, ty) in &struct_decl.fields {
                hover_text.push_str(&format!("    {}: {},\n", name, ty));
            }
            hover_text.push('}');
            doc = struct_decl.doc_comment.clone();
        } else if let Some(enum_decl) = global_env.enums.get(&word_sym) {
            hover_text.push_str(&format!("enum {} {{\n", word));
            for var in &enum_decl.variants {
                hover_text.push_str(&format!("    {},\n", var.0));
            }
            hover_text.push('}');
            doc = enum_decl.doc_comment.clone();
        }

        if let Some(doc_str) = doc {
            let clean_doc = doc_str
                .lines()
                .map(|line| {
                    let s = line.trim_start();
                    if let Some(stripped) = s.strip_prefix("/// ") {
                        stripped
                    } else if let Some(stripped) = s.strip_prefix("///") {
                        stripped
                    } else {
                        s
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            hover_text = format!("{}\n\n```vx\n{}\n```", clean_doc, hover_text);
        } else {
            hover_text = format!("```vx\n{}\n```", hover_text);
        }

        if hover_text.is_empty() {
            return None;
        }

        Some(HoverInfo {
            value: hover_text,
            line_start: line,
            col_start: start_col,
            line_end: line,
            col_end: end_col,
        })
    }

    pub fn goto_definition(
        &self,
        uri: &str,
        line: usize,
        col: usize,
    ) -> Option<DefinitionLocation> {
        let text = self.files.get(uri)?;

        let target_line = text.lines().nth(line)?;
        if col >= target_line.len() {
            return None;
        }

        let mut start_col = col;
        let mut end_col = col;
        let chars: Vec<char> = target_line.chars().collect();

        while start_col > 0
            && (chars[start_col - 1].is_alphanumeric() || chars[start_col - 1] == '_')
        {
            start_col -= 1;
        }
        while end_col < chars.len() && (chars[end_col].is_alphanumeric() || chars[end_col] == '_') {
            end_col += 1;
        }

        if start_col == end_col {
            return None;
        }
        let word: String = chars[start_col..end_col].iter().collect();

        // Simple heuristic: search all files for definition
        for (search_uri, search_text) in &self.files {
            let mut lexer = crate::lexer::Lexer::new(search_text);
            let tokens = lexer.tokenize();

            for i in 0..tokens.len() {
                // If it's `fn`, `struct`, `enum`, `let`, `mut` followed by the word
                match tokens[i].kind {
                    crate::lexer::TokenTypeBase::Fn
                    | crate::lexer::TokenTypeBase::Struct
                    | crate::lexer::TokenTypeBase::Enum
                    | crate::lexer::TokenTypeBase::Let
                    | crate::lexer::TokenTypeBase::Mut
                        if i + 1 < tokens.len() =>
                    {
                        if let crate::lexer::TokenTypeBase::Identifier(id) = tokens[i + 1].kind {
                            if id == word {
                                return Some(DefinitionLocation {
                                    uri: search_uri.clone(),
                                    line_start: tokens[i + 1].line.saturating_sub(1),
                                    col_start: tokens[i + 1].column.saturating_sub(1),
                                    line_end: tokens[i + 1].line.saturating_sub(1),
                                    col_end: tokens[i + 1].column.saturating_sub(1)
                                        + tokens[i + 1].length,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }
}
