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
    /// Implicit cross-topology transfer inserted via a `Relocatable` impl (a real data
    /// movement happens silently at the use site; write the transfer explicitly to
    /// silence). See docs/discussions/brainstorming/hardware_monad_topology.md.
    ///
    /// `Relocatable` answers "may this value move implicitly?" and is keyed on a user
    /// type. That is a different question from "what code moves bytes across this
    /// hardware edge?", which is `impl Transfer<Memory::A, Memory::B> for Topology::X`.
    /// Both were called `Transfer` before Vx#353.
    W1024,
    /// Use of a user-defined topology with no registered descriptor (not declared via
    /// `Topology <Name> { ... }` and not registered by a plugin). Often a typo of a
    /// built-in; defaults to host-like placement.
    W1025,
    /// A user-defined topology's memory is unreachable from the host (no transfer path),
    /// so data can never be moved to it. See the topology coherence check.
    W1026,
    /// A declared `relaxed` transfer edge does not preserve visibility (the seam engine
    /// shows a consumer may read stale data). See the topology coherence check.
    W1027,
    /// The working set of a memory space exceeds `capacity`, but the space is declared
    /// `overcommit`, so the cumulative-budget errors (E6010, and the cross-call E6027/E6028)
    /// are downgraded to this warning.
    W1028,
    /// A tensor placed in a memory space that declares a `capacity` has a *dynamic* (non-
    /// literal) shape, so the capacity check (E6009/E6010) could not run — the placement is
    /// unverified. Silence by making the shape static, or bounding it (see P1-1). Emitted only
    /// when the destination space actually declares a capacity. See P0-4 in
    /// docs/discussions/heterogeneous_target_gap_analysis.md.
    W1029,
    /// A topology's device index is not a compile-time constant (`GPU[i]` for a runtime `i`), so
    /// it cannot be resolved to a device instance and falls back to index 0. Every such spawn
    /// therefore targets the same device. Vx models one representative device per declared kind
    /// (#284), so a fleet program should index with constants or const generics.
    W1030,
    /// A proof obligation could not be discharged because no SMT solver was available, so the
    /// property is **unverified** rather than proved. Distinct from W1027, which means the solver
    /// ran and found a violation. Emitted only under `VX_ALLOW_UNVERIFIED`; without it a missing
    /// solver is an error, because silence used to be indistinguishable from success (Vx#374).
    W1031,

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
    /// An array literal whose elements are not scalars, or which is empty.
    ///
    /// An array literal lowers to `tensor.from_elements`, whose element type must be a scalar,
    /// so `[a, b]` for tensors -- placed or not -- has nothing to lower to, and an empty
    /// literal has no element type to give it. Both used to be accepted by the checker (the
    /// element type silently stayed at its `f32` default) and then crash codegen with an
    /// internal error rather than a diagnostic. See Vx#354.
    E3018,

    // --- Borrow/Ownership Errors (E4xxx) ---
    /// Use of moved or consumed linear variable
    E4001,
    /// Cannot access mutably borrowed variable
    E4002,
    /// Cannot borrow as mutable (already immutably borrowed)
    E4003,
    /// Cannot borrow (already mutably borrowed)
    E4004,
    /// A returned reference escapes the function borrowing a function-local (dangling return)
    E4005,

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
    /// A value is used from a topology that cannot see the memory space it lives in.
    /// The diagnostic names the value's space, the visible set of the topology reading it,
    /// and the cost of the transfer that would fix it -- so a misplaced handoff (an
    /// un-transferred KV cache in a disaggregated prefill/decode split, say) is a compile
    /// error that carries its own remedy. A `managed: cached` space the topology can reach
    /// across a declared seam is coherent in hardware and is not reported here.
    E6003,
    /// Transfer violates the boundary contract at a seam (per-seam local-completeness
    /// / soundness obligation is `sat`; a stale read can violate the contract).
    E6004,
    /// A user-defined topology declaration is incoherent: it cannot see its own default
    /// memory space (`default_space ∉ visibility`). See the topology coherence check.
    E6005,
    /// A `Memory` declaration's `within:` hierarchy forms a cycle (a space contains itself).
    E6006,
    /// A `Memory` sub-space's `capacity` exceeds its parent's capacity (a child cannot be
    /// larger than what contains it).
    E6007,
    /// A `Memory` declaration has a non-positive `capacity`, `bandwidth`, or `granule`.
    E6008,
    /// A statically-shaped tensor placed in a memory space exceeds that space's `capacity`.
    E6009,
    /// The working set placed in a memory space (the sum of its tiles) exceeds `capacity`.
    /// Downgraded to W1028 when the space is declared `overcommit`.
    E6010,
    /// A sub-space's `scope` is broader than its parent's (locality must narrow down `within:`).
    E6011,
    /// The same `Memory` or `Topology` name is declared by two compilation inputs (e.g. a
    /// `--machine` file and the program). Declarations are name-keyed, so one would silently
    /// shadow the other and the machine model in force would depend on load order (#281).
    E6012,
    /// A declared `transfer` edge carries an explicit cost *and* has one derivable from its
    /// endpoints' `bandwidth:` figures. An edge gets exactly one cost source, because two answers
    /// to "what does this hop cost" is not a model: the compiler routed by the declared number and
    /// reported the derived one, and nothing detected the disagreement.
    E6013,
    /// A program stages through host memory while a machine model is in force, and no host was
    /// declared. `--machine` describes an accelerator and says nothing about the machine it hangs
    /// off, so the host end of that seam was being reasoned about without anything describing it.
    /// `--host <file>` names one; `--host default` names the machine compiling the program.
    ///
    /// A host declares no capacity -- host memory is virtual, and a hard limit would reject
    /// programs that page rather than fail -- so this is about the host being *stated* rather than
    /// assumed, not about a budget.
    E6014,
    /// A structurally invalid transfer lowering (`impl transfer A -> B { ... }`): the same edge
    /// implemented twice in one compilation (which one is in force would be load order), or a
    /// lowering with no functions (an empty body cannot move anything, and accepting it would
    /// make `impl transfer` an inert annotation rather than code).
    E6015,
    /// A `Topology` or `Memory` declaration whose identity cannot be relied on. Two forms: the
    /// declared name shadows a built-in topology (every use of `Topology::<Name>` resolves to the
    /// built-in, so the declaration is silently ignored -- including its `arch:`); or two declared
    /// names collide on one dispatch id (custom ids are derived from the name by hashing), in
    /// which case which declaration is in force would be hash-iteration order -- observed as the
    /// same program getting a device image on some runs and not others for a topology, and as a
    /// `transfer` carrying the other space's capacity and granule for a memory space.
    E6016,
    /// A misuse of the `raw::` transfer-lowering primitives (Vx#353 A2): a `raw::` call
    /// outside an `impl transfer` body, an unknown primitive name, a tile argument that
    /// is not a bare parameter name (the primitives are indexed, not addressed), a store
    /// into a tile not held by `&mut`, or a wrongly typed index/value.
    E6017,
    /// A `raw::` bounds obligation (`0 <= index < extent`) that could not be proven.
    /// Prove it with a loop bound or invariant the SMT prover can see, or assert it in
    /// an `unsafe` block -- which records the obligation as asserted-not-proven, the
    /// same standing an unverified `spec:` figure has.
    E6018,
    /// `raw::barrier()` anywhere but a top-level statement of the lowering body. The
    /// barrier's contract requires every lane to reach it; under a conditional or a
    /// loop that cannot be guaranteed syntactically, so it is rejected outright
    /// (conservative by design -- restructure the body so the barrier is unconditional).
    E6019,
    /// `raw::async_copy` in a lowering for an edge no declared topology equips with a
    /// copy engine. The capability lives in the machine file (`transfer A -> B
    /// copy_engine`); using an absent primitive is a compile error, not a fallback.
    E6020,
    /// A violation of the async/synchronization discipline in a lowering body: a
    /// destination read while an `async_copy` into it is still outstanding, a body that
    /// ends with copies no `async_wait` covers, or a lowering for a synchronizing edge
    /// whose body does not end with `raw::barrier()` (the seam obligation of
    /// `hir/seam.rs`: a relaxed publication makes a stale read reachable).
    E6021,
    /// An `impl transfer` lowering whose edge endpoints are not visible to a topology
    /// that declares the edge -- the lowering would execute on a part that cannot
    /// address the spaces it moves bytes between (contract constraint C6).
    E6022,
    /// An `impl transfer` lowering whose declared tile shape is not the shape the
    /// transfer at hand actually moves. A lowering is selected by edge, so nothing else
    /// relates the two, and the `raw::` primitives take their extents from the
    /// declaration: a smaller declaration copies part of the tile and leaves the rest
    /// uninitialised, a larger one stores past the end (observed as a SIGSEGV).
    E6023,
    /// A proof obligation could not be discharged because no SMT solver was available. Fails the
    /// compilation by default: an undischarged obligation is not a proved one, and treating the
    /// two alike is what let a missing z3 certify every seam in silence (Vx#374). Set
    /// `VX_ALLOW_UNVERIFIED=1` to downgrade this to W1031 and compile anyway.
    E6024,
    /// A placement naming a location the machine does not have: a memory space no declared
    /// topology holds, written either as the space or as the device that would hold it. The
    /// derivation between the two spellings has a like-named fallback, so an undeclared name
    /// resolves to a space that exists only in the placement that mentions it.
    E6025,
    /// A tensor whose element type the target hardware cannot represent, placed on it anyway.
    ///
    /// The machine model states what a device has (`dtypes: [f32, f16, ...]`); this is the check
    /// that a placement stays inside it. An H100 has no fp4, so an fp4 tensor placed on one asks
    /// for silicon that is not there -- and the placement is in the type, so the question is
    /// answerable here rather than at a kernel launch on the machine that lacks the type.
    ///
    /// Only fires against a topology that declares `dtypes:`. An undeclared machine constrains
    /// nothing, which is what keeps every machine file written before the field kept working.
    E6026,
    /// A working set that overflows a space only across call boundaries: the peak along some
    /// call path -- what each caller still holds when it calls, plus the deepest callee's own
    /// peak -- exceeds the space's declared capacity, while every function on the path fits by
    /// itself (that case is E6010's). Computed by folding per-function capacity summaries over
    /// the call graph, after the per-function checks. Downgraded to W1028 when the space is
    /// declared `overcommit`.
    E6027,
    /// A recursive cycle that places tiles in a space with a declared capacity. The recursion
    /// depth is not known at compile time, so the true peak is unbounded and the placement is
    /// refused conservatively. Downgraded to W1028 when the space is declared `overcommit`.
    E6028,

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

/// The machine-readable payload of an admission verdict, carried alongside the human-readable
/// message so `--diagnostics-json` reports numbers rather than making a consumer parse prose
/// back out of a sentence (#282). Only the diagnostics an admission campaign keys on carry one.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticFacts {
    /// A capacity verdict: one tile against a space (E6009), or a function's whole working set
    /// against it (E6010 / W1028). `tiles` is `None` for the single-tile case.
    Capacity {
        space: String,
        required_bytes: u64,
        available_bytes: u64,
        tiles: Option<usize>,
    },
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
    /// Structured fields for `--diagnostics-json`; `None` for diagnostics with no campaign-
    /// relevant numbers. Never affects the rendered text.
    pub facts: Option<DiagnosticFacts>,
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
            facts: None,
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
            facts: None,
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
                    facts: None,
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
            facts: None,
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
            facts: None,
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
            facts: None,
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
