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

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.level {
            DiagnosticLevel::Warning => write!(f, "Warning: {}", self.message),
            DiagnosticLevel::Error => write!(f, "Error: {}", self.message),
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

    pub fn push(&mut self, message: String) {
        self.inner.push(Diagnostic {
            level: DiagnosticLevel::Error,
            message,
        });
    }

    pub fn push_warning(&mut self, message: String) {
        self.inner.push(Diagnostic {
            level: DiagnosticLevel::Warning,
            message,
        });
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
