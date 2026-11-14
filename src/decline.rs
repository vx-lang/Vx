//===- decline.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Which name an `EmitterGap` carries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GapFunction {
    /// The name as the program's author wrote it.
    Source(String),

    /// A name the compiler made: a generic instance's type arguments, a method's receiver.
    /// The author wrote something shorter, so the message says this is not a source name.
    Instance(String),
}

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
    /// source line that produced it and, once `in_function` or `in_instance` has filled it in,
    /// the function it was found in.
    ///
    /// The emitter bails from roughly 190 places, most of which no corpus program has ever
    /// reached. Writing a description for each up front would be 190 guesses about which ones
    /// matter. Carrying the site instead means the histogram names the exact line to describe,
    /// and only the lines something actually reaches get the work. The site alone is not enough
    /// to act on: it names a line of the emitter, so the function is carried too.
    EmitterGap {
        site: &'static str,
        /// The function being emitted. `None` for a gap outside the emit of a function.
        function: Option<GapFunction>,
    },
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

    /// Fill in the function a gap was found in, named the way the program's author wrote it. A
    /// decline that already names its construct is returned unchanged: only a gap has nothing to
    /// say until someone tells it.
    pub fn in_function(self, function: &str) -> Decline {
        match self {
            Decline::EmitterGap {
                site,
                function: None,
            } => Decline::EmitterGap {
                site,
                function: Some(GapFunction::Source(function.to_string())),
            },
            other => other,
        }
    }

    /// Fill in the name a gap was found under when the compiler made that name: a generic
    /// instance or a method. The function record has no field holding the source spelling, so
    /// the message says the name is an instance rather than pretending it is one the author
    /// wrote.
    pub fn in_instance(self, function: &str) -> Decline {
        match self {
            Decline::EmitterGap {
                site,
                function: None,
            } => Decline::EmitterGap {
                site,
                function: Some(GapFunction::Instance(function.to_string())),
            },
            other => other,
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
            Decline::EmitterGap { site, .. } => format!("emitter-gap({site})"),
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
            Decline::EmitterGap { site, function } => {
                write!(f, "an emitter gap at {site}")?;
                match function {
                    Some(GapFunction::Source(name)) => write!(f, " in function '{name}'")?,
                    Some(GapFunction::Instance(name)) => {
                        write!(f, " in function '{name}' (a name the compiler made)")?
                    }
                    None => {}
                }
                Ok(())
            }
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
            site: concat!(file!(), ":", line!()),
            function: None,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bare gap names the emitter line and nothing else, which is the message that sent a
    /// reader bisecting a model layer by layer.
    #[test]
    fn a_gap_says_where_it_bailed_before_it_knows_more() {
        let gap = Decline::EmitterGap {
            site: "src/example.rs:1",
            function: None,
        };
        assert_eq!(gap.to_string(), "an emitter gap at src/example.rs:1");
        assert_eq!(gap.detail_key(), "emitter-gap(src/example.rs:1)");
    }

    /// Once the emitter knows which function it was walking, the message names the function it
    /// was found in. It names no instruction: an opcode is a compiler instruction, and the
    /// author never wrote one.
    #[test]
    fn a_gap_names_the_function_it_was_found_in() {
        let named = Decline::EmitterGap {
            site: "src/example.rs:1",
            function: None,
        }
        .in_function("check_token");
        assert_eq!(
            named.to_string(),
            "an emitter gap at src/example.rs:1 in function 'check_token'"
        );
        assert_eq!(named.detail_key(), "emitter-gap(src/example.rs:1)");
    }

    /// A generic instance or a method is named by a name the compiler made, which the program's
    /// author never wrote. The function record has no field for the source spelling, so the
    /// message says the name is one the compiler made.
    #[test]
    fn a_gap_says_when_the_name_is_an_instance_the_compiler_named() {
        let named = Decline::EmitterGap {
            site: "src/example.rs:1",
            function: None,
        }
        .in_instance("sum$i64");
        assert_eq!(
            named.to_string(),
            "an emitter gap at src/example.rs:1 in function 'sum$i64' (a name the compiler made)"
        );
        assert_eq!(named.detail_key(), "emitter-gap(src/example.rs:1)");
    }

    /// Every other decline already carries the construct it names, so filling in a function
    /// must leave it alone.
    #[test]
    fn a_decline_that_names_its_own_construct_is_left_alone() {
        let typed = Decline::TypeNotModelled {
            what: "a call argument with no type",
        };
        assert_eq!(typed.clone().in_function("check_token"), typed);
    }
}
