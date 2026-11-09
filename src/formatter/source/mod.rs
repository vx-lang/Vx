//! Source-preserving syntax views, registered layouts, and the shared CLI formatting pipeline.

pub mod dispatch;
pub mod document;
pub(crate) mod engine;
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
    pub context: DeclarationContext,
    pub tokens: Range<usize>,
    pub bytes: Range<usize>,
    pub header: FunctionHeaderView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclarationContext {
    Root,
    Impl,
    Trait,
    Extern,
    Other,
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

    /// Find tokens intersecting a byte interval without scanning earlier source.
    pub(crate) fn token_range(&self, bytes: &Range<usize>) -> Range<usize> {
        let start = self
            .tokens
            .partition_point(|token| token.bytes.end <= bytes.start);
        let end = self
            .tokens
            .partition_point(|token| token.bytes.start < bytes.end);
        start.min(end)..end
    }

    pub fn matching_delimiter(&self, token: TokenId) -> Option<TokenId> {
        self.matching[token.0].map(TokenId)
    }

    pub fn is_opaque(&self, token: TokenId) -> bool {
        let end = self
            .macros
            .partition_point(|node| node.tokens.start <= token.0);
        end > 0 && self.macros[end - 1].tokens.contains(&token.0)
    }

    pub fn is_excluded(&self, bytes: &Range<usize>) -> bool {
        let first = self
            .exclusions
            .partition_point(|range| range.end <= bytes.start);
        self.exclusions
            .get(first)
            .is_some_and(|range| overlaps(range, bytes))
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
        let range = self.token_range(&bytes);
        let original: Vec<_> = self.tokens[range.clone()]
            .iter()
            .enumerate()
            .map(|(index, token)| (index + range.start, token))
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
        let mut positions = vec![None; range.len()];
        for ((id, before), after) in original.iter().zip(&actual) {
            if before.token.kind != after.token.kind
                || self.text[before.bytes.clone()] != output[after.bytes.clone()]
            {
                return Err("Formatting changed a source token".into());
            }
            positions[*id - range.start] = Some(after.bytes.clone());
        }
        let gap_start = self.gaps.partition_point(|gap| gap.left.0 < range.start);
        for gap in self.gaps[gap_start..]
            .iter()
            .take_while(|gap| gap.right.0 < range.end)
        {
            if gap.constraint == GapConstraint::Flexible {
                continue;
            }
            let (Some(left), Some(right)) = (
                &positions[gap.left.0 - range.start],
                &positions[gap.right.0 - range.start],
            ) else {
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
            .skip(
                self.exclusions
                    .partition_point(|item| item.end <= bytes.start),
            )
            .take_while(|item| item.start < bytes.end)
            .cloned()
            .chain(
                self.macros
                    .iter()
                    .skip(
                        self.macros
                            .partition_point(|view| view.bytes.end <= bytes.start),
                    )
                    .take_while(|view| view.bytes.start < bytes.end)
                    .map(|view| view.bytes.clone()),
            )
        {
            if !overlaps(&bytes, &preserved) {
                continue;
            }
            if preserved.start < bytes.start || preserved.end > bytes.end {
                return Err("A preserved structure crosses the formatting boundary".into());
            }
            let start = original.partition_point(|(_, token)| token.bytes.start < preserved.start);
            let end = original.partition_point(|(_, token)| token.bytes.end <= preserved.end);
            let owned = &original[start..end];
            if let (Some((first, first_token)), Some((last, last_token))) =
                (owned.first(), owned.last())
            {
                let leading = first_token.bytes.start - preserved.start;
                let trailing = preserved.end - last_token.bytes.end;
                let start = positions[*first - range.start]
                    .as_ref()
                    .expect("preserved token has an output position")
                    .start
                    .checked_sub(leading)
                    .ok_or("Formatting removed preserved leading whitespace")?;
                let end = positions[*last - range.start]
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
        for macro_view in self
            .macros
            .iter()
            .skip(
                self.macros
                    .partition_point(|view| view.bytes.end <= bytes.start),
            )
            .take_while(|view| view.bytes.start < bytes.end)
        {
            if macro_view.tokens.start < range.start {
                continue;
            }
            let Some(position) = &positions[macro_view.tokens.start - range.start] else {
                continue;
            };
            let prefix = &output[..position.start];
            let original_line_start = self.text[..bytes.start]
                .rfind('\n')
                .map_or(0, |offset| offset + 1);
            let entry_column = self.text[original_line_start..bytes.start].chars().count();
            let column = prefix.rfind('\n').map_or_else(
                || entry_column + prefix.chars().count(),
                |newline| prefix[newline + 1..].chars().count(),
            );
            if column != macro_view.original_column - 1 {
                return Err("Formatting changed an opaque macro's column".into());
            }
        }
        Ok(())
    }

    fn parse_tokens(text: &str) -> SourceFileView<'_> {
        let mut offset = 0;
        let mut line = 1;
        let mut column = 1;
        let tokens = Lexer::new_with_comments(text)
            .tokenize()
            .into_iter()
            .map(|mut token| {
                let start = offset;
                offset += token.length;
                assert!(
                    offset <= text.len()
                        && text.is_char_boundary(start)
                        && text.is_char_boundary(offset),
                    "Lossless lexer tokens must partition the original source bytes"
                );
                token.line = line;
                token.column = column;
                for character in text[start..offset].chars() {
                    if character == '\n' {
                        line += 1;
                        column = 1;
                    } else {
                        column += 1;
                    }
                }
                SourceToken {
                    token,
                    bytes: start..offset,
                }
            })
            .collect();
        assert_eq!(
            offset,
            text.len(),
            "Lossless lexer must retain the whole source"
        );
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
