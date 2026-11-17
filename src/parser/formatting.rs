//! Record where each piece of syntax starts and ends, for `vx-format`.
//!
//! The formatter parses a file with the compiler's own parser, so the two cannot disagree on the
//! grammar. When `Parser::recorder` is set, the parse functions also record the token range of
//! each piece of syntax that the formatter has a rule for. The compiler leaves it unset.

use super::{ParseResult, Parser};
use crate::lexer::TokenType;
use crate::syntax::Statement;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClauseKind {
    On,
    Requires,
    Ensures,
    Where,
}

/// The kinds of syntax that the parser records. Add a kind here when a formatting rule needs
/// one, and record it where the parser reads that syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecordedKind {
    /// A function, trait method or extern function, from its first keyword to the end of its
    /// body or its `;`. A `pub` before it is not included.
    FunctionDeclaration,
    /// The part of a function declaration before its body or `;`.
    FunctionHeader,
    /// The `<...>` of a function.
    GenericParameters,
    /// The parameters of a function, with their `(` and `)`.
    ParameterList,
    /// One parameter: `mut name: Type`.
    Parameter,
    /// The type of one parameter.
    ParameterType,
    /// The type after `->`.
    ReturnType,
    Clause(ClauseKind),
    /// A function body that the parser could not read. See `parse_function_body`.
    UnreadableBody,
    ImplBlock,
    TraitBlock,
    ExternBlock,
    /// `Name { field: value, ... }`, from the first name of the path to the `}`.
    StructLiteral,
    /// One `field: value` of a struct literal.
    StructField,
    /// `match value { arms }`, from `match` to the `}`.
    Match,
    /// One `pattern => body` of a match, without the comma after it.
    MatchArm,
    /// `if condition { value } else { value }`, with one value and no statement in each branch.
    IfValue,
    /// One `{ value }` branch of an `IfValue`.
    IfValueBranch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SyntaxRecord {
    pub kind: RecordedKind,
    /// Indices into the token slice that the parser reads.
    pub tokens: Range<usize>,
}

impl<'a> Parser<'a> {
    /// Parse a whole file and return the recorded syntax.
    ///
    /// Only the grammar is checked. A second definition of a name is a compile error, but the
    /// formatter must still format that file.
    pub(crate) fn formatting_file(&mut self) -> ParseResult<'a, Vec<SyntaxRecord>> {
        self.recorder = Some(Vec::new());
        self.parse()?;
        Ok(self.recorder.take().unwrap_or_default())
    }

    /// Parse statements outside any function, as in an editor selection or a short test input.
    pub(crate) fn formatting_statements(&mut self) -> ParseResult<'a, ()> {
        while !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut Vec::new())?;
        }
        Ok(())
    }

    pub(crate) fn recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Record that the syntax `kind` spans the tokens from `start` to the current position.
    pub(crate) fn record(&mut self, kind: RecordedKind, start: usize) {
        if let Some(records) = &mut self.recorder {
            records.push(SyntaxRecord {
                kind,
                tokens: start..self.pos,
            });
        }
    }

    /// Read the statements of a function body, up to its `}`. The `{` was read just before.
    ///
    /// For the formatter, a body that the parser cannot read is recorded and skipped. The
    /// formatter then keeps that one declaration as written and still formats the rest of the
    /// file. The compiler gets the error as usual.
    pub(crate) fn parse_function_body(&mut self, body: &mut Vec<Statement>) -> ParseResult<'a, ()> {
        let open = self.pos - 1;
        let generics = self.generic_params.len();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let Err(error) = self.parse_statement_into(body) else {
                continue;
            };
            let Some(close) = self.recording().then(|| self.closing_brace(open)).flatten() else {
                return Err(error);
            };
            self.pos = close;
            self.generic_params.truncate(generics);
            self.pending_stmts.clear();
            self.prefix_stmts.clear();
            self.match_is_statement = false;
            if let Some(records) = &mut self.recorder {
                records.push(SyntaxRecord {
                    kind: RecordedKind::UnreadableBody,
                    tokens: open..close + 1,
                });
            }
        }
        Ok(())
    }

    /// The `}` that closes the `{` at `open`.
    fn closing_brace(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(open) {
            match token.kind {
                TokenType::LeftBrace => depth += 1,
                TokenType::RightBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(index);
                    }
                }
                _ => {}
            }
        }
        None
    }
}
