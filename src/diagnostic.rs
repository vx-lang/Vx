//===- diagnostic.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Diagnostic reporting module for formatting and displaying compiler errors and warnings.
//
//===----------------------------------------------------------------------===//

use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Source-level location for diagnostics with line/column information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

impl SourceSpan {
    pub fn new(line: usize, column: usize, length: usize) -> Self {
        Self {
            line,
            column,
            length,
        }
    }

    /// Convert from the AST-level Span (which has the same fields).
    pub fn from_ast_span(span: &crate::syntax::Span) -> Self {
        Self {
            line: span.line,
            column: span.column,
            length: span.length,
        }
    }
}

/// Stable warning/error codes for diagnostics.
/// Warning codes use the W prefix, error codes use the E prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticCode {
    // --- Warnings (W1xxx) ---
    /// Unused variable binding
    W1001,
    /// Unused function definition
    W1002,
    /// Unreachable code after return, break, or continue
    W1003,
    /// Unnecessary mutable binding (`let mut x` where x is never reassigned)
    W1004,
    /// Shadowed variable in same scope
    W1005,
    /// Redundant borrow (`&&x`)
    W1006,
    /// Implicit type widening in `as` cast
    W1007,
    /// Empty match arm body
    W1008,
    /// Unused function parameter
    W1009,
    /// Unnecessary unsafe block (no unsafe ops inside)
    W1010,
    /// Redundant `as` cast to same type
    W1013,
    /// Narrowing cast loses precision
    W1014,
    /// Immediately dereferenced borrow (`*&x`)
    W1020,
    /// Transfer to same memory space (no-op)
    W1022,
    /// Spawn on Topology::Current (no-op)
    W1023,

    // --- Parser Errors (E1xxx) ---
    /// Unexpected token
    E1001,
    /// Unexpected end of file
    E1002,
    /// Expected identifier
    E1003,
    /// Expected type
    E1004,
    /// Expected expression
    E1005,
    /// Unclosed delimiter (paren/brace/bracket)
    E1006,
    /// Missing semicolon
    E1007,
    /// Missing comma
    E1008,
    /// Invalid operator
    E1009,
    /// Unknown topology variant
    E1010,
    /// Unknown memory space
    E1011,
    /// Unknown element type
    E1012,
    /// Invalid macro invocation syntax
    E1013,

    // --- Name Resolution Errors (E2xxx) ---
    /// Undefined variable
    E2001,
    /// Undefined function
    E2002,
    /// Unknown enum
    E2003,
    /// Unknown enum variant
    E2004,
    /// Unknown struct field
    E2005,
    /// Module does not export function
    E2006,
    /// Method not found on type
    E2007,

    // --- Type Errors (E3xxx) ---
    /// Type mismatch in variable declaration
    E3001,
    /// Type mismatch in return
    E3002,
    /// Type mismatch in function argument
    E3003,
    /// Type mismatch in binary operation
    E3004,
    /// Type mismatch in relational operation
    E3005,
    /// Type mismatch in logical operation
    E3006,
    /// If branch type mismatch
    E3007,
    /// Enum payload type mismatch
    E3008,
    /// Enum payload arity mismatch
    E3009,
    /// Function argument count mismatch
    E3010,
    /// Unsupported cast
    E3011,
    /// Type mismatch in struct field initialization
    E3012,
    /// Missing struct field in initialization
    E3013,
    /// Range type mismatch
    E3014,
    /// Trait not implemented
    E3015,
    /// Generic type deduction failure
    E3016,
    /// Closure argument count or type mismatch
    E3017,

    // --- Borrow/Ownership Errors (E4xxx) ---
    /// Use of moved or consumed linear variable
    E4001,
    /// Cannot access mutably borrowed variable
    E4002,
    /// Cannot borrow as mutable (already immutably borrowed)
    E4003,
    /// Cannot borrow (already mutably borrowed)
    E4004,

    // --- Safety Errors (E5xxx) ---
    /// Unsafe function call outside unsafe block
    E5001,
    /// Unsafe memory operation outside unsafe block
    E5002,

    // --- Topology/Hardware Errors (E6xxx) ---
    /// Topology mismatch in function call
    E6001,
    /// Cannot transfer between memory spaces (no hardware path)
    E6002,
    /// Cannot transfer non-reference type
    E6003,

    // --- Tensor/Math Errors (E7xxx) ---
    /// Matmul dimension mismatch
    E7001,
    /// Matmul element type mismatch
    E7002,
    /// Reshape arithmetic mismatch
    E7003,
    /// Non-differentiable return type (autodiff)
    E7004,

    // --- Contract/Verification Errors (E8xxx) ---
    /// Cannot prove postcondition
    E8001,
    /// Comptime assert failed
    E8002,
}

impl std::fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel {
    Warning,
    Error,
}

/// A secondary note attached to a diagnostic, providing additional context.
#[derive(Debug, Clone)]
pub struct Note {
    pub message: Cow<'static, str>,
    pub span: Option<SourceSpan>,
}

/// A suggested text edit that can fix the diagnostic.
#[derive(Debug, Clone)]
pub struct FixIt {
    pub message: Cow<'static, str>,
    pub span: SourceSpan,
    pub replacement: Cow<'static, str>,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub code: Option<DiagnosticCode>,
    pub message: Cow<'static, str>,
    pub span: Option<Span>,
    pub source_span: Option<SourceSpan>,
    pub notes: Vec<Note>,
    pub fix_its: Vec<FixIt>,
}

impl Diagnostic {
    pub fn error(msg: impl Into<Cow<'static, str>>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            code: None,
            message: msg.into(),
            span: None,
            source_span: None,
            notes: Vec::new(),
            fix_its: Vec::new(),
        }
    }

    pub fn warning(msg: impl Into<Cow<'static, str>>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            code: None,
            message: msg.into(),
            span: None,
            source_span: None,
            notes: Vec::new(),
            fix_its: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_source_span(mut self, span: SourceSpan) -> Self {
        self.source_span = Some(span);
        self
    }

    pub fn with_code(mut self, code: DiagnosticCode) -> Self {
        self.code = Some(code);
        self
    }

    pub fn with_note(mut self, msg: impl Into<Cow<'static, str>>) -> Self {
        self.notes.push(Note {
            message: msg.into(),
            span: None,
        });
        self
    }

    pub fn with_note_at(mut self, msg: impl Into<Cow<'static, str>>, span: SourceSpan) -> Self {
        self.notes.push(Note {
            message: msg.into(),
            span: Some(span),
        });
        self
    }

    pub fn with_fix_it(
        mut self,
        msg: impl Into<Cow<'static, str>>,
        span: SourceSpan,
        replacement: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.fix_its.push(FixIt {
            message: msg.into(),
            span,
            replacement: replacement.into(),
        });
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.level {
            DiagnosticLevel::Warning => "Warning",
            DiagnosticLevel::Error => "Error",
        };

        // Include diagnostic code if present
        if let Some(code) = &self.code {
            write!(f, "{}[{}]", prefix, code)?;
        } else {
            write!(f, "{}", prefix)?;
        }

        // Include source location if present
        if let Some(s) = &self.source_span {
            write!(f, " at {}:{}", s.line, s.column)?;
        } else if let Some(s) = &self.span {
            write!(f, " at {}..{}", s.start, s.end)?;
        }

        write!(f, ": {}", self.message)?;

        // Print notes
        for note in &self.notes {
            write!(f, "\n  note: {}", note.message)?;
            if let Some(span) = &note.span {
                write!(f, " (at {}:{})", span.line, span.column)?;
            }
        }

        // Print fix-its
        for fix in &self.fix_its {
            if fix.replacement.is_empty() {
                write!(f, "\n  help: {} -- remove", fix.message)?;
            } else {
                write!(
                    f,
                    "\n  help: {} -- replace with `{}`",
                    fix.message, fix.replacement
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticsVec {
    pub inner: Vec<Diagnostic>,
}

impl DiagnosticsVec {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn report(&mut self, level: DiagnosticLevel, message: impl Into<Cow<'static, str>>) {
        if level == DiagnosticLevel::Error && self.error_count() >= 10 {
            if self.error_count() == 10 {
                self.inner.push(Diagnostic {
                    level: DiagnosticLevel::Error,
                    code: None,
                    message: "Too many errors. Please fix the errors and try again.".into(),
                    span: None,
                    source_span: None,
                    notes: Vec::new(),
                    fix_its: Vec::new(),
                });
            }
            return;
        }

        self.inner.push(Diagnostic {
            level,
            code: None,
            message: message.into(),
            span: None,
            source_span: None,
            notes: Vec::new(),
            fix_its: Vec::new(),
        });
    }

    /// Emit a warning with a diagnostic code and source span.
    /// Returns a mutable reference to the diagnostic for chaining notes/fix-its.
    pub fn warn(
        &mut self,
        code: DiagnosticCode,
        message: impl Into<Cow<'static, str>>,
        span: Option<SourceSpan>,
    ) -> &mut Diagnostic {
        self.inner.push(Diagnostic {
            level: DiagnosticLevel::Warning,
            code: Some(code),
            message: message.into(),
            span: None,
            source_span: span,
            notes: Vec::new(),
            fix_its: Vec::new(),
        });
        self.inner.last_mut().unwrap()
    }

    /// Emit an error with a diagnostic code and optional source span.
    /// Returns a mutable reference to the diagnostic for chaining notes/fix-its.
    pub fn error_with_code(
        &mut self,
        code: DiagnosticCode,
        message: impl Into<Cow<'static, str>>,
        span: Option<SourceSpan>,
    ) -> &mut Diagnostic {
        self.inner.push(Diagnostic {
            level: DiagnosticLevel::Error,
            code: Some(code),
            message: message.into(),
            span: None,
            source_span: span,
            notes: Vec::new(),
            fix_its: Vec::new(),
        });
        self.inner.last_mut().unwrap()
    }

    pub fn error_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .count()
    }

    pub fn push(&mut self, message: String) {
        self.report(DiagnosticLevel::Error, message);
    }

    pub fn push_warning(&mut self, message: String) {
        self.report(DiagnosticLevel::Warning, message);
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Diagnostic> {
        self.inner.iter()
    }

    /// Check if a specific warning code has been emitted.
    pub fn has_warning(&self, code: DiagnosticCode) -> bool {
        self.inner
            .iter()
            .any(|d| d.level == DiagnosticLevel::Warning && d.code == Some(code))
    }

    /// Check if a specific error code has been emitted.
    pub fn has_error(&self, code: DiagnosticCode) -> bool {
        self.inner
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error && d.code == Some(code))
    }
}

impl IntoIterator for DiagnosticsVec {
    type Item = Diagnostic;
    type IntoIter = std::vec::IntoIter<Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a> IntoIterator for &'a DiagnosticsVec {
    type Item = &'a Diagnostic;
    type IntoIter = std::slice::Iter<'a, Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl Default for DiagnosticsVec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_display_error_no_span() {
        let d = Diagnostic::error("type mismatch");
        assert_eq!(d.to_string(), "Error: type mismatch");
    }

    #[test]
    fn test_diagnostic_display_warning_with_span() {
        let d = Diagnostic::warning("unused variable").with_span(Span { start: 5, end: 10 });
        assert_eq!(d.to_string(), "Warning at 5..10: unused variable");
    }

    #[test]
    fn test_diagnostic_error_with_span() {
        let d = Diagnostic::error("undeclared").with_span(Span { start: 0, end: 3 });
        assert_eq!(d.level, DiagnosticLevel::Error);
        assert_eq!(d.span, Some(Span { start: 0, end: 3 }));
    }

    #[test]
    fn test_diagnostics_vec_error_cap_at_10() {
        let mut diags = DiagnosticsVec::new();
        for i in 0..15 {
            diags.push(format!("error {}", i));
        }
        // 10 real errors + 1 "too many errors" message = 11 total
        assert_eq!(diags.error_count(), 11);
        assert_eq!(diags.len(), 11);
        // The 11th message should be the cap message
        assert!(diags.inner[10].message.contains("Too many errors"));
    }

    #[test]
    fn test_diagnostics_vec_warnings_not_capped() {
        let mut diags = DiagnosticsVec::new();
        for i in 0..20 {
            diags.push_warning(format!("warning {}", i));
        }
        assert_eq!(diags.len(), 20);
        assert_eq!(diags.error_count(), 0);
    }

    #[test]
    fn test_diagnostics_vec_mixed_error_warning_counting() {
        let mut diags = DiagnosticsVec::new();
        diags.push("error 1".to_string());
        diags.push_warning("warning 1".to_string());
        diags.push("error 2".to_string());
        diags.push_warning("warning 2".to_string());
        assert_eq!(diags.error_count(), 2);
        assert_eq!(diags.len(), 4);
    }

    #[test]
    fn test_diagnostics_vec_empty() {
        let diags = DiagnosticsVec::new();
        assert!(diags.is_empty());
        assert_eq!(diags.len(), 0);
        assert_eq!(diags.error_count(), 0);
    }

    #[test]
    fn test_diagnostics_vec_into_iter() {
        let mut diags = DiagnosticsVec::new();
        diags.push("e1".to_string());
        diags.push_warning("w1".to_string());
        let collected: Vec<_> = diags.into_iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].level, DiagnosticLevel::Error);
        assert_eq!(collected[1].level, DiagnosticLevel::Warning);
    }

    // ---- New tests for extended diagnostic infrastructure ----

    #[test]
    fn test_diagnostic_with_code_display() {
        let d = Diagnostic::warning("unused variable 'x'").with_code(DiagnosticCode::W1001);
        assert_eq!(d.to_string(), "Warning[W1001]: unused variable 'x'");
    }

    #[test]
    fn test_diagnostic_with_source_span_display() {
        let d = Diagnostic::warning("unreachable code")
            .with_code(DiagnosticCode::W1003)
            .with_source_span(SourceSpan::new(10, 5, 3));
        assert_eq!(d.to_string(), "Warning[W1003] at 10:5: unreachable code");
    }

    #[test]
    fn test_diagnostic_with_notes_display() {
        let d = Diagnostic::warning("unused variable 'x'")
            .with_code(DiagnosticCode::W1001)
            .with_note("declared here");
        assert_eq!(
            d.to_string(),
            "Warning[W1001]: unused variable 'x'\n  note: declared here"
        );
    }

    #[test]
    fn test_diagnostic_with_fix_it_display() {
        let d = Diagnostic::warning("unused variable 'x'")
            .with_code(DiagnosticCode::W1001)
            .with_fix_it("prefix with underscore", SourceSpan::new(2, 9, 1), "_x");
        assert_eq!(
            d.to_string(),
            "Warning[W1001]: unused variable 'x'\n  help: prefix with underscore -- replace with `_x`"
        );
    }

    #[test]
    fn test_diagnostic_with_fix_it_remove_display() {
        let d = Diagnostic::warning("unnecessary `mut`")
            .with_code(DiagnosticCode::W1004)
            .with_fix_it("remove `mut`", SourceSpan::new(2, 9, 4), "");
        assert_eq!(
            d.to_string(),
            "Warning[W1004]: unnecessary `mut`\n  help: remove `mut` -- remove"
        );
    }

    #[test]
    fn test_warn_method_creates_coded_warning() {
        let mut diags = DiagnosticsVec::new();
        diags.warn(
            DiagnosticCode::W1001,
            "unused variable 'x'",
            Some(SourceSpan::new(5, 9, 1)),
        );
        assert_eq!(diags.warning_count(), 1);
        assert!(diags.has_warning(DiagnosticCode::W1001));
        assert!(!diags.has_warning(DiagnosticCode::W1003));
    }

    #[test]
    fn test_warn_method_chaining() {
        let mut diags = DiagnosticsVec::new();
        let d = diags.warn(
            DiagnosticCode::W1001,
            "unused variable 'x'",
            Some(SourceSpan::new(5, 9, 1)),
        );
        d.notes.push(Note {
            message: "declared here".into(),
            span: None,
        });
        d.fix_its.push(FixIt {
            message: "prefix with underscore".into(),
            span: SourceSpan::new(5, 9, 1),
            replacement: "_x".into(),
        });
        assert_eq!(diags.inner[0].notes.len(), 1);
        assert_eq!(diags.inner[0].fix_its.len(), 1);
    }

    #[test]
    fn test_diagnostic_code_display() {
        assert_eq!(format!("{}", DiagnosticCode::W1001), "W1001");
        assert_eq!(format!("{}", DiagnosticCode::W1003), "W1003");
        assert_eq!(format!("{}", DiagnosticCode::E2001), "E2001");
        assert_eq!(format!("{}", DiagnosticCode::E3003), "E3003");
    }

    #[test]
    fn test_warning_count() {
        let mut diags = DiagnosticsVec::new();
        diags.push("error".to_string());
        diags.push_warning("warning1".to_string());
        diags.warn(DiagnosticCode::W1001, "warning2", None);
        assert_eq!(diags.error_count(), 1);
        assert_eq!(diags.warning_count(), 2);
    }

    #[test]
    fn test_error_with_code_creates_coded_error() {
        let mut diags = DiagnosticsVec::new();
        diags.error_with_code(
            DiagnosticCode::E2001,
            "Undefined variable 'x'",
            Some(SourceSpan::new(5, 9, 1)),
        );
        assert_eq!(diags.error_count(), 1);
        assert!(diags.has_error(DiagnosticCode::E2001));
        assert!(!diags.has_error(DiagnosticCode::E2002));
        let d = &diags.inner[0];
        assert_eq!(d.level, DiagnosticLevel::Error);
        assert_eq!(d.code, Some(DiagnosticCode::E2001));
        assert!(d.message.contains("Undefined variable"));
        assert_eq!(d.source_span, Some(SourceSpan::new(5, 9, 1)));
    }

    #[test]
    fn test_error_with_code_display() {
        let d = Diagnostic::error("Undefined variable 'x'")
            .with_code(DiagnosticCode::E2001)
            .with_source_span(SourceSpan::new(5, 9, 1));
        assert_eq!(d.to_string(), "Error[E2001] at 5:9: Undefined variable 'x'");
    }

    #[test]
    fn test_error_with_code_and_note() {
        let mut diags = DiagnosticsVec::new();
        let d = diags.error_with_code(
            DiagnosticCode::E4001,
            "Use of moved variable 'x'",
            Some(SourceSpan::new(10, 5, 1)),
        );
        d.notes.push(Note {
            message: "moved here".into(),
            span: Some(SourceSpan::new(8, 5, 1)),
        });
        let output = format!("{}", diags.inner[0]);
        assert!(output.contains("Error[E4001]"));
        assert!(output.contains("moved here"));
        assert!(output.contains("at 8:5"));
    }

    #[test]
    fn test_error_with_code_and_fix_it() {
        let mut diags = DiagnosticsVec::new();
        let d = diags.error_with_code(
            DiagnosticCode::E5001,
            "Call to unsafe function 'malloc'",
            Some(SourceSpan::new(3, 12, 6)),
        );
        d.fix_its.push(FixIt {
            message: "wrap in unsafe block".into(),
            span: SourceSpan::new(3, 12, 6),
            replacement: "unsafe { malloc(...) }".into(),
        });
        let output = format!("{}", diags.inner[0]);
        assert!(output.contains("Error[E5001]"));
        assert!(output.contains("wrap in unsafe block"));
    }

    #[test]
    fn test_has_error_negative() {
        let mut diags = DiagnosticsVec::new();
        diags.push("unstructured error".to_string());
        // has_error only matches coded errors
        assert!(!diags.has_error(DiagnosticCode::E2001));
    }
}
