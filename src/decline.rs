//===- decline.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Why the flat path gave up on a function.
//
// The flat lowerer declines whatever it cannot lower yet, and the AST path stays the oracle for
// that function. Declining is expected. Declining without saying why is what makes "how much of
// the language does the flat path carry" a reading exercise instead of a query.
//
// A leaf module: both the flattener and the flat emitter produce these, and the corpus sweep
// counts them.
//
//===----------------------------------------------------------------------===//

use std::fmt;

/// Why one function did not lower through the flat path.
///
/// Every variant names something the flat path does not do *yet*. None of them is an error: a
/// declined function compiles through the AST path and produces the same MLIR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Decline {
    /// An expression the flat lowerer has no case for. Carries the `Expr` variant name.
    UnsupportedExpr(&'static str),

    /// A statement the flat lowerer has no case for. Carries the `Statement` variant name.
    UnsupportedStmt(&'static str),

    /// A callee with no signature in the registry -- an ambiguous cross-module name dropped from
    /// `fn_sigs`, or an unregistered symbol.
    UnresolvedCallee(String),

    /// A builtin called with operands the flat path does not lower: the wrong arity, or a tensor
    /// whose rank or element type the emitted kernel does not accept.
    BuiltinShape {
        builtin: &'static str,
        why: &'static str,
    },

    /// A value whose lowered type is not the one this construct needs -- indexing something that
    /// is not a pointer, taking a field of a nominal the flat path has no layout for.
    TypeNotModelled { what: &'static str },

    /// A construct the flat path lowers only in memory mode, reached in register mode.
    NeedsMemoryMode { construct: &'static str },

    /// Something the flat path does not lower yet that is none of the above. `what` is a short
    /// fixed phrase, never a formatted value, so the histogram groups.
    Unsupported { what: &'static str },

    /// A gap in the flat *emitter* that has not been given a phrase of its own yet, naming the
    /// source line that produced it.
    ///
    /// The emitter bails from roughly 190 places, most of which no corpus program has ever
    /// reached. Writing a description for each up front would be 190 guesses about which ones
    /// matter. Carrying the site instead means the histogram names the exact line to describe,
    /// and only the lines something actually reaches get the work.
    EmitterGap { site: &'static str },
}

impl Decline {
    /// A stable key for grouping. Deliberately drops the payload of the variants that carry one:
    /// a histogram of "unresolved callee `foo`, unresolved callee `bar`" answers nothing that
    /// "unresolved callee x2" does not.
    pub fn key(&self) -> &'static str {
        match self {
            Decline::UnsupportedExpr(_) => "unsupported-expr",
            Decline::UnsupportedStmt(_) => "unsupported-stmt",
            Decline::UnresolvedCallee(_) => "unresolved-callee",
            Decline::BuiltinShape { .. } => "builtin-shape",
            Decline::TypeNotModelled { .. } => "type-not-modelled",
            Decline::NeedsMemoryMode { .. } => "needs-memory-mode",
            Decline::Unsupported { .. } => "unsupported",
            Decline::EmitterGap { .. } => "emitter-gap",
        }
    }

    /// The key plus the one detail that says *which* construct, for a histogram that names the
    /// work rather than the category: `unsupported-expr(Closure)`.
    pub fn detail_key(&self) -> String {
        match self {
            Decline::UnsupportedExpr(k) => format!("unsupported-expr({k})"),
            Decline::UnsupportedStmt(k) => format!("unsupported-stmt({k})"),
            Decline::UnresolvedCallee(_) => "unresolved-callee".to_string(),
            Decline::BuiltinShape { builtin, .. } => format!("builtin-shape({builtin})"),
            Decline::TypeNotModelled { what } => format!("type-not-modelled({what})"),
            Decline::NeedsMemoryMode { construct } => format!("needs-memory-mode({construct})"),
            Decline::Unsupported { what } => format!("unsupported({what})"),
            Decline::EmitterGap { site } => format!("emitter-gap({site})"),
        }
    }
}

impl fmt::Display for Decline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decline::UnsupportedExpr(k) => write!(f, "no flat lowering for the {k} expression"),
            Decline::UnsupportedStmt(k) => write!(f, "no flat lowering for the {k} statement"),
            Decline::UnresolvedCallee(n) => write!(f, "no signature for the callee '{n}'"),
            Decline::BuiltinShape { builtin, why } => {
                write!(f, "'{builtin}' called with {why}")
            }
            Decline::TypeNotModelled { what } => write!(f, "{what}"),
            Decline::NeedsMemoryMode { construct } => {
                write!(f, "{construct} lowers only in memory mode")
            }
            Decline::Unsupported { what } => write!(f, "no flat lowering for {what}"),
            Decline::EmitterGap { site } => write!(f, "an emitter gap at {site}"),
        }
    }
}

/// What a lowering step returns: the value, or the reason the flat path gave up.
pub type Lowered<T> = Result<T, Decline>;

/// An emitter bail-out that has no phrase of its own yet, stamped with where it is.
///
/// A macro rather than a function so `line!()` expands at the bail site.
#[macro_export]
macro_rules! emitter_gap {
    () => {
        $crate::decline::Decline::EmitterGap {
            site: concat!("flat.rs:", line!()),
        }
    };
}
