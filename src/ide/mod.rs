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
    files: HashMap<String, String>,
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

        use crate::syntax::macro_expand::MacroExpander;

        let mut modules: Vec<_> = loader.loaded_modules.values().cloned().collect();

        // 1. Macro Expansion
        let mut global_macros = std::collections::HashMap::new();
        for m in modules.iter() {
            for mac in &m.macros {
                global_macros.insert(mac.name.clone(), mac.rules.clone());
            }
        }
        let mut expander = MacroExpander::new(&global_macros);
        for m in modules.iter_mut() {
            let _ = expander.expand_module(m); // Ignore expansion errors for IDE
        }

        // 2. Name Resolution
        let symbol_map = crate::resolver::build_symbol_map(&modules);
        for m in modules.iter_mut() {
            m.resolve_names(&symbol_map);
        }

        let global_env_modules: Vec<_> = modules.iter().map(|m| m.clone_signature()).collect();
        let global_env = GlobalAstEnv::build(&global_env_modules);
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

    pub fn hover(&self, _uri: &str, _line: usize, _col: usize) -> Option<HoverInfo> {
        // To be implemented: extract the node at the specified line/col
        // and return hover text representing the type or docstring.
        None
    }
}
