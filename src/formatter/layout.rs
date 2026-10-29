use super::context::FormatContext;
use crate::lexer::TokenType;
use std::ops::Range;

/// Rules can preserve tokens and change only the whitespace between them.
pub(super) enum LayoutPiece {
    Token(usize),
    Space,
    Line { indent: usize },
}

pub(super) struct LayoutEdit {
    pub range: Range<usize>,
    pub pieces: Vec<LayoutPiece>,
}

#[derive(Default)]
pub(super) struct LayoutPlan {
    edit: Option<LayoutEdit>,
    owner: Option<&'static str>,
}

impl LayoutPlan {
    pub fn into_edit(self) -> Option<LayoutEdit> {
        self.edit
    }

    pub fn propose(
        &mut self,
        rule: &'static str,
        context: &FormatContext<'_, '_>,
        edit: LayoutEdit,
    ) {
        assert_eq!(
            edit.range.start, context.index,
            "format rule {rule} must start at the current token"
        );
        assert!(
            edit.range.end <= context.tokens.len(),
            "format rule {rule} exceeds the token range"
        );
        if !context.can_format(edit.range.clone()) {
            return;
        }
        assert!(
            self.edit.is_none(),
            "format rules {:?} and {rule} proposed overlapping layouts",
            self.owner
        );

        let mut expected = edit
            .range
            .clone()
            .filter(|index| !matches!(context.tokens[*index].kind, TokenType::Whitespace(_)));
        for piece in &edit.pieces {
            if let LayoutPiece::Token(index) = piece {
                assert_eq!(
                    expected.next(),
                    Some(*index),
                    "format rule {rule} changed source token order"
                );
            }
        }
        assert!(
            expected.next().is_none(),
            "format rule {rule} removed source tokens"
        );

        // Replacements must leave the enclosing renderer's delimiter state unchanged.
        for index in edit.range.clone() {
            match context.tokens[index].kind {
                TokenType::LeftParen
                | TokenType::RightParen
                | TokenType::LeftBracket
                | TokenType::RightBracket
                | TokenType::LeftBrace
                | TokenType::RightBrace => {
                    assert!(
                        context
                            .syntax
                            .matching_delimiter(index)
                            .is_some_and(|other| edit.range.contains(&other)),
                        "format rule {rule} must include complete delimiter pairs"
                    );
                }
                TokenType::Eof => panic!("format rule {rule} cannot replace the end of the file"),
                _ => {}
            }
        }
        self.owner = Some(rule);
        self.edit = Some(edit);
    }
}

pub(super) fn flat_width(context: &FormatContext<'_, '_>, pieces: &[LayoutPiece]) -> usize {
    pieces
        .iter()
        .map(|piece| match piece {
            LayoutPiece::Token(index) => context.tokens[*index].kind.to_string().chars().count(),
            LayoutPiece::Space => 1,
            LayoutPiece::Line { .. } => panic!("a multiline layout has no single-line width"),
        })
        .sum()
}
