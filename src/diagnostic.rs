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
