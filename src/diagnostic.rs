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

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: Cow<'static, str>,
    pub span: Option<Span>,
}

impl Diagnostic {
    pub fn error(msg: impl Into<Cow<'static, str>>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            message: msg.into(),
            span: None,
        }
    }

    pub fn warning(msg: impl Into<Cow<'static, str>>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            message: msg.into(),
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.level {
            DiagnosticLevel::Warning => "Warning",
            DiagnosticLevel::Error => "Error",
        };
        if let Some(s) = &self.span {
            write!(f, "{}: {} at {}..{}", prefix, self.message, s.start, s.end)
        } else {
            write!(f, "{}: {}", prefix, self.message)
        }
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
                    message: "Too many errors. Please fix the errors and try again.".into(),
                    span: None,
                });
            }
            return;
        }

        self.inner.push(Diagnostic {
            level,
            message: message.into(),
            span: None,
        });
    }

    pub fn error_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
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
        assert_eq!(d.to_string(), "Warning: unused variable at 5..10");
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
}
