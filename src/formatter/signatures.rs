use crate::lexer::{Token, TokenType};
use std::ops::Range;
use vx_config::FormatConfig;

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

fn compact(tokens: &[Token<'_>]) -> String {
    let mut result = String::new();
    let mut previous: Option<&TokenType<'_>> = None;
    let mut whitespace = false;
    for token in tokens {
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
                result.push(' ');
            }
        }
        result.push_str(&token.kind.to_string());
        previous = Some(&token.kind);
        whitespace = false;
    }
    result
}

/// Reflow named declarations only. Leave comments and function calls to the normal formatter.
pub(super) fn format_signature(
    tokens: &[Token<'_>],
    start: usize,
    column: usize,
    indent_level: usize,
    config: &FormatConfig,
) -> Option<(usize, String)> {
    let name = tokens[start + 1..]
        .iter()
        .find(|token| !matches!(token.kind, TokenType::Whitespace(_)))?;
    if !matches!(name.kind, TokenType::Identifier(_)) {
        return None;
    }

    let mut depth = Depth::default();
    let mut open = None;
    let mut close = None;
    let mut end = None;
    for (index, token) in tokens.iter().enumerate().skip(start + 1) {
        match token.kind {
            TokenType::Comment(_) | TokenType::DocComment(_) | TokenType::Eof => return None,
            TokenType::LeftParen if open.is_none() && depth.is_zero() => open = Some(index),
            TokenType::RightParen if open.is_some() && close.is_none() && depth.parens == 1 => {
                close = Some(index);
            }
            TokenType::LeftBrace | TokenType::Semicolon if depth.is_zero() => {
                end = Some(index);
                break;
            }
            _ => {}
        }
        depth.advance(&token.kind);
    }
    let (open, close, end) = (open?, close?, end?);
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

    let inline = compact(&tokens[start..end]);
    let body_space = usize::from(matches!(tokens[end].kind, TokenType::LeftBrace));
    let mut header = if parameters.is_empty()
        || column
            .saturating_add(inline.chars().count())
            .saturating_add(body_space + 1)
            <= config.fn_signature_width
    {
        inline
    } else {
        let mut multiline = compact(&tokens[start..=open]);
        for parameter in parameters {
            multiline.push('\n');
            multiline.push_str(&" ".repeat((indent_level + 1) * config.indent));
            multiline.push_str(&compact(&tokens[parameter]));
        }
        multiline.push('\n');
        multiline.push_str(&" ".repeat(indent_level * config.indent));
        multiline.push_str(&compact(&tokens[close..end]));
        multiline
    };
    if body_space != 0 {
        header.push(' ');
    }
    Some((end, header))
}
