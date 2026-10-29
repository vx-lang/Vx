//! Source-syntax foundations. The existing CLI renderer is not switched to this pipeline yet.

pub mod dispatch;
pub mod document;
mod formatters;
pub mod ownership;
mod recognize;

#[cfg(test)]
mod tests;

pub use formatters::{builtin_registry, format_declaration};

use crate::lexer::{Lexer, Token, TokenType};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenId(pub usize);

#[derive(Clone, Debug)]
pub struct SourceToken<'a> {
    pub token: Token<'a>,
    pub bytes: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapConstraint {
    MustTouch,
    MustSeparate,
    Flexible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrainedGap {
    pub left: TokenId,
    pub right: TokenId,
    pub constraint: GapConstraint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroKind {
    Invocation,
    Definition,
}

#[derive(Clone, Debug)]
pub struct MacroView {
    pub kind: MacroKind,
    pub tokens: Range<usize>,
    pub bytes: Range<usize>,
    pub token_trees: Vec<Range<usize>>,
    pub original_line: usize,
    pub original_column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderClauseKind {
    On,
    Requires,
    Ensures,
    Where,
}

#[derive(Clone, Debug)]
pub struct HeaderClauseView {
    pub kind: HeaderClauseKind,
    pub tokens: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct TypeView {
    pub tokens: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct ParameterView {
    pub tokens: Range<usize>,
    pub ty: TypeView,
}

#[derive(Clone, Debug)]
pub struct FunctionHeaderView {
    pub tokens: Range<usize>,
    pub name: TokenId,
    pub generics: Range<usize>,
    pub parameter_open: TokenId,
    pub parameter_close: TokenId,
    pub parameters: Vec<ParameterView>,
    pub return_type: TypeView,
    pub clauses: Vec<HeaderClauseView>,
    pub terminator: TokenId,
}

#[derive(Clone, Debug)]
pub struct FunctionDeclView {
    pub tokens: Range<usize>,
    pub bytes: Range<usize>,
    pub header: FunctionHeaderView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageWarning {
    pub bytes: Range<usize>,
    pub message: String,
}

impl CoverageWarning {
    pub fn render(&self, path: &str, source: &str) -> String {
        let line = source[..self.bytes.start]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1;
        format!("WARN: {path}:{line}: {}", self.message)
    }
}

/// Tokens, views, and constraints always refer to the unmodified source snapshot.
pub struct SourceFileView<'a> {
    pub text: &'a str,
    pub tokens: Vec<SourceToken<'a>>,
    pub macros: Vec<MacroView>,
    pub exclusions: Vec<Range<usize>>,
    pub functions: Vec<FunctionDeclView>,
    pub gaps: Vec<ConstrainedGap>,
    pub warnings: Vec<CoverageWarning>,
    matching: Vec<Option<usize>>,
}

impl<'a> SourceFileView<'a> {
    pub fn parse(text: &'a str) -> Self {
        let mut result = Self::parse_tokens(text);
        result.matching = vec![None; result.tokens.len()];
        result.recognize();
        result
    }

    pub fn token_text(&self, id: TokenId) -> &'a str {
        &self.text[self.tokens[id.0].bytes.clone()]
    }

    pub fn token_bytes(&self, tokens: Range<usize>) -> Range<usize> {
        if tokens.is_empty() {
            let point = self.tokens[tokens.start].bytes.start;
            point..point
        } else {
            self.tokens[tokens.start].bytes.start..self.tokens[tokens.end - 1].bytes.end
        }
    }

    pub fn matching_delimiter(&self, token: TokenId) -> Option<TokenId> {
        self.matching[token.0].map(TokenId)
    }

    pub fn is_opaque(&self, token: TokenId) -> bool {
        self.macros
            .iter()
            .any(|node| node.tokens.contains(&token.0))
    }

    pub fn is_excluded(&self, bytes: &Range<usize>) -> bool {
        self.exclusions.iter().any(|range| overlaps(range, bytes))
    }

    pub fn verify_preservation(&self, output: &str) -> Result<(), String> {
        self.verify_span_preservation(0..self.text.len(), output)
    }

    pub fn verify_span_preservation(
        &self,
        bytes: Range<usize>,
        output: &str,
    ) -> Result<(), String> {
        let rendered = SourceFileView::parse_tokens(output);
        let original: Vec<_> = self
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| {
                token.bytes.start >= bytes.start
                    && token.bytes.end <= bytes.end
                    && !matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof)
            })
            .collect();
        let actual: Vec<_> = rendered
            .tokens
            .iter()
            .filter(|token| !matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof))
            .collect();
        if original.len() != actual.len() {
            return Err("Formatting changed the number of source tokens".into());
        }
        let mut positions = vec![None; self.tokens.len()];
        for ((id, before), after) in original.iter().zip(&actual) {
            if before.token.kind != after.token.kind
                || self.text[before.bytes.clone()] != output[after.bytes.clone()]
            {
                return Err("Formatting changed a source token".into());
            }
            positions[*id] = Some(after.bytes.clone());
        }
        for gap in &self.gaps {
            if gap.constraint == GapConstraint::Flexible {
                continue;
            }
            let (Some(left), Some(right)) = (&positions[gap.left.0], &positions[gap.right.0])
            else {
                continue;
            };
            let touching = left.end == right.start;
            if touching != matches!(gap.constraint, GapConstraint::MustTouch) {
                return Err("Formatting changed grammatical adjacency".into());
            }
        }
        for preserved in self
            .exclusions
            .iter()
            .cloned()
            .chain(self.macros.iter().map(|view| view.bytes.clone()))
        {
            if !overlaps(&bytes, &preserved) {
                continue;
            }
            if preserved.start < bytes.start || preserved.end > bytes.end {
                return Err("A preserved structure crosses the formatting boundary".into());
            }
            let owned: Vec<_> = original
                .iter()
                .filter(|(_, token)| {
                    token.bytes.start >= preserved.start && token.bytes.end <= preserved.end
                })
                .collect();
            if let (Some((first, first_token)), Some((last, last_token))) =
                (owned.first().copied(), owned.last().copied())
            {
                let leading = first_token.bytes.start - preserved.start;
                let trailing = preserved.end - last_token.bytes.end;
                let start = positions[*first]
                    .as_ref()
                    .expect("preserved token has an output position")
                    .start
                    .checked_sub(leading)
                    .ok_or("Formatting removed preserved leading whitespace")?;
                let end = positions[*last]
                    .as_ref()
                    .expect("preserved token has an output position")
                    .end
                    + trailing;
                if output.get(start..end) != Some(&self.text[preserved]) {
                    return Err("Formatting changed opaque or excluded source bytes".into());
                }
            } else if output != &self.text[preserved] {
                return Err("Formatting changed an excluded source range".into());
            }
        }
        Ok(())
    }

    fn parse_tokens(text: &str) -> SourceFileView<'_> {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(offset, _)| offset + 1));
        let tokens = Lexer::new_with_comments(text)
            .tokenize()
            .into_iter()
            .map(|token| {
                let start = if matches!(token.kind, TokenType::Eof) {
                    text.len()
                } else {
                    let line = starts[token.line - 1];
                    line + text[line..]
                        .char_indices()
                        .nth(token.column - 1)
                        .map_or(text.len() - line, |(offset, _)| offset)
                };
                let end = start + token.length;
                assert!(
                    end <= text.len() && text.is_char_boundary(start) && text.is_char_boundary(end),
                    "Lexer positions must reference original source bytes"
                );
                SourceToken {
                    token,
                    bytes: start..end,
                }
            })
            .collect();
        SourceFileView {
            text,
            tokens,
            matching: Vec::new(),
            macros: Vec::new(),
            exclusions: Vec::new(),
            functions: Vec::new(),
            gaps: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

pub(super) fn overlaps(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}
