use super::super::context::FormatContext;
use super::super::layout::{flat_width, LayoutEdit, LayoutPiece, LayoutPlan};
use super::super::registry::FormatRule;
use crate::lexer::TokenType;
use std::ops::Range;

pub(in crate::formatter) struct FunctionSignatures;

impl FormatRule for FunctionSignatures {
    fn name(&self) -> &'static str {
        "function_signatures"
    }

    fn starts_at(&self, token: &TokenType<'_>) -> bool {
        matches!(token, TokenType::Fn)
    }

    fn apply(&self, context: &FormatContext<'_, '_>, layout: &mut LayoutPlan) {
        if let Some(edit) = signature_layout(context) {
            layout.propose(self.name(), context, edit);
        }
    }
}

#[derive(Default)]
struct Depth {
    parens: usize,
    brackets: usize,
    angles: usize,
    braces: usize,
}

impl Depth {
    fn is_zero(&self) -> bool {
        self.parens == 0 && self.brackets == 0 && self.angles == 0 && self.braces == 0
    }

    fn advance(&mut self, kind: &TokenType<'_>) {
        match kind {
            TokenType::LeftParen => self.parens += 1,
            TokenType::RightParen => self.parens = self.parens.saturating_sub(1),
            TokenType::LeftBracket => self.brackets += 1,
            TokenType::RightBracket => self.brackets = self.brackets.saturating_sub(1),
            TokenType::LeftAngle => self.angles += 1,
            TokenType::RightAngle => self.angles = self.angles.saturating_sub(1),
            TokenType::LeftBrace => self.braces += 1,
            TokenType::RightBrace => self.braces = self.braces.saturating_sub(1),
            _ => {}
        }
    }
}

fn compact(context: &FormatContext<'_, '_>, range: Range<usize>) -> Vec<LayoutPiece> {
    let tokens = &context.tokens[range.clone()];
    let mut result = Vec::new();
    let mut previous: Option<&TokenType<'_>> = None;
    let mut whitespace = false;
    for (index, token) in tokens.iter().enumerate() {
        if matches!(token.kind, TokenType::Whitespace(_)) {
            whitespace = true;
            continue;
        }
        if let Some(previous) = previous {
            let no_space = matches!(
                token.kind,
                TokenType::Comma
                    | TokenType::Colon
                    | TokenType::DoubleColon
                    | TokenType::RightParen
                    | TokenType::RightBracket
                    | TokenType::RightAngle
            ) || (matches!(
                token.kind,
                TokenType::LeftParen | TokenType::LeftBracket | TokenType::LeftAngle
            ) && !matches!(
                previous,
                TokenType::Colon | TokenType::Comma | TokenType::Arrow
            )) || matches!(
                previous,
                TokenType::LeftParen
                    | TokenType::LeftBracket
                    | TokenType::LeftAngle
                    | TokenType::DoubleColon
                    | TokenType::Ampersand
                    | TokenType::Star
            );
            let space = whitespace
                || matches!(
                    previous,
                    TokenType::Fn | TokenType::Comma | TokenType::Colon | TokenType::Arrow
                )
                || matches!(token.kind, TokenType::Arrow);
            let separate_references = matches!(previous, TokenType::Ampersand)
                && matches!(token.kind, TokenType::Ampersand);
            if separate_references || (space && !no_space) {
                result.push(LayoutPiece::Space);
            }
        }
        result.push(LayoutPiece::Token(range.start + index));
        previous = Some(&token.kind);
        whitespace = false;
    }
    result
}

/// Reflow named declarations only. Comments retain their existing layout.
fn signature_layout(context: &FormatContext<'_, '_>) -> Option<LayoutEdit> {
    let tokens = context.tokens;
    let start = context.index;
    let column = context.column;
    let indent_level = context.indent_level;
    let config = context.config;
    let name = tokens[start + 1..]
        .iter()
        .find(|token| !matches!(token.kind, TokenType::Whitespace(_)))?;
    if !matches!(name.kind, TokenType::Identifier(_)) {
        return None;
    }

    let mut depth = Depth::default();
    let mut open = None;
    let mut end = None;
    for (index, token) in tokens.iter().enumerate().skip(start + 1) {
        match token.kind {
            TokenType::Comment(_) | TokenType::DocComment(_) | TokenType::Eof => return None,
            TokenType::LeftParen if open.is_none() && depth.is_zero() => open = Some(index),
            TokenType::LeftBrace | TokenType::Semicolon if depth.is_zero() => {
                end = Some(index);
                break;
            }
            _ => {}
        }
        depth.advance(&token.kind);
    }
    let open = open?;
    let close = context.syntax.matching_delimiter(open)?;
    let end = end?;
    if close >= end || !context.can_format(start..end) {
        return None;
    }
    let mut parameters: Vec<Range<usize>> = Vec::new();
    let mut parameter_start = open + 1;
    let mut depth = Depth::default();
    for (index, token) in tokens.iter().enumerate().take(close).skip(open + 1) {
        if matches!(token.kind, TokenType::Comma) && depth.is_zero() {
            parameters.push(parameter_start..index + 1);
            parameter_start = index + 1;
        }
        depth.advance(&token.kind);
    }
    if tokens[parameter_start..close]
        .iter()
        .any(|token| !matches!(token.kind, TokenType::Whitespace(_)))
    {
        parameters.push(parameter_start..close);
    }

    let inline = compact(context, start..end);
    let body_space = usize::from(matches!(tokens[end].kind, TokenType::LeftBrace));
    let mut pieces = if parameters.is_empty()
        || column
            .saturating_add(flat_width(context, &inline))
            .saturating_add(body_space + 1)
            <= config.fn_signature_width
    {
        inline
    } else {
        let mut multiline = compact(context, start..open + 1);
        for parameter in parameters {
            multiline.push(LayoutPiece::Line {
                indent: indent_level + 1,
            });
            multiline.extend(compact(context, parameter));
        }
        multiline.push(LayoutPiece::Line {
            indent: indent_level,
        });
        multiline.extend(compact(context, close..end));
        multiline
    };
    if body_space != 0 {
        pieces.push(LayoutPiece::Space);
    }
    Some(LayoutEdit {
        range: start..end,
        pieces,
    })
}
