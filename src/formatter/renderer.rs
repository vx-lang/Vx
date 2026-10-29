use super::context::FormatContext;
use super::layout::{LayoutEdit, LayoutPiece};
use crate::lexer::{Lexer, Token, TokenType};

/// Render rule output without permitting rules to write arbitrary source text.
pub(super) fn render(edit: &LayoutEdit, context: &FormatContext<'_, '_>, output: &mut String) {
    for piece in &edit.pieces {
        match piece {
            LayoutPiece::Token(index) => output.push_str(&context.tokens[*index].kind.to_string()),
            LayoutPiece::Space => output.push(' '),
            LayoutPiece::Line { indent } => {
                output.push('\n');
                output.extend(std::iter::repeat_n(' ', indent * context.config.indent));
            }
        }
    }
}

/// Check the whole output so rules cannot merge tokens across replacement boundaries.
pub(super) fn assert_preserves_tokens(tokens: &[Token<'_>], output: &str) {
    let rendered = Lexer::new_with_comments(output).tokenize();
    let expected = tokens
        .iter()
        .filter(|token| !matches!(token.kind, TokenType::Whitespace(_) | TokenType::Eof));
    let actual = rendered
        .iter()
        .filter(|token| !matches!(token.kind, TokenType::Whitespace(_) | TokenType::Eof));
    assert!(
        expected
            .map(|token| &token.kind)
            .eq(actual.map(|token| &token.kind)),
        "formatting changed the source tokens"
    );
}
