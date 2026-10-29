use crate::lexer::{Token, TokenType};
use std::ops::Range;
use vx_config::FormatConfig;

/// Source properties shared by every formatting rule.
pub(super) struct SyntaxContext {
    matching: Vec<Option<usize>>,
    protected: Vec<bool>,
}

impl SyntaxContext {
    pub fn new(tokens: &[Token<'_>]) -> Self {
        let mut matching = vec![None; tokens.len()];
        let mut stack = Vec::new();
        let mut protected = vec![false; tokens.len()];
        let mut disabled = tokens.first().is_some_and(|token| {
            matches!(&token.kind, TokenType::Comment(text) | TokenType::DocComment(text)
                if text.contains("//! vx-format: OFF"))
        });
        for (index, token) in tokens.iter().enumerate() {
            protected[index] = disabled;
            if let TokenType::Comment(text) | TokenType::DocComment(text) = &token.kind {
                if text.contains("vx-format-begin: OFF") {
                    disabled = true;
                    protected[index] = true;
                } else if text.contains("vx-format-begin: ON")
                    || text.contains("vx-format-end: OFF")
                {
                    disabled = false;
                    protected[index] = true;
                }
            }
            let (kind, opening) = match token.kind {
                TokenType::LeftParen => (0, true),
                TokenType::RightParen => (0, false),
                TokenType::LeftBracket => (1, true),
                TokenType::RightBracket => (1, false),
                TokenType::LeftBrace => (2, true),
                TokenType::RightBrace => (2, false),
                _ => continue,
            };
            if opening {
                stack.push((kind, index));
            } else if let Some((previous, open)) = stack.pop() {
                if previous == kind {
                    matching[open] = Some(index);
                    matching[index] = Some(open);
                }
            }
        }
        Self {
            matching,
            protected,
        }
    }

    pub fn matching_delimiter(&self, index: usize) -> Option<usize> {
        self.matching[index]
    }
}

/// A rule sees the source and the renderer's current position, without mutable output access.
pub(super) struct FormatContext<'a, 'source> {
    pub tokens: &'a [Token<'source>],
    pub syntax: &'a SyntaxContext,
    pub config: &'a FormatConfig,
    pub index: usize,
    pub column: usize,
    pub indent_level: usize,
}

impl FormatContext<'_, '_> {
    pub fn can_format(&self, range: Range<usize>) -> bool {
        range.start < range.end
            && range.end <= self.tokens.len()
            && !self.syntax.protected[range]
                .iter()
                .any(|protected| *protected)
    }
}
