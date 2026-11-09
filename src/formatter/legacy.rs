//! Bounded compatibility formatting. This module has no syntax-specific rule dispatch.
use super::source::document::{DelimiterKind, OpenDelimiter, RenderState};
use super::source::ownership::OwnershipMap;
use super::source::SourceFileView;
use crate::lexer::{Token, TokenType};
use std::collections::HashMap;
use std::ops::Range;
use vx_config::FormatConfig;

pub(super) fn render(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    bytes: Range<usize>,
    config: &FormatConfig,
    state: &mut RenderState,
) -> Result<String, String> {
    ownership.validate_legacy_edit(bytes.clone())?;
    let mut input = Vec::new();
    for original in &source.tokens[source.token_range(&bytes)] {
        if !super::source::overlaps(&original.bytes, &bytes) {
            continue;
        }
        let start = original.bytes.start.max(bytes.start);
        let end = original.bytes.end.min(bytes.end);
        let mut token = original.token.clone();
        if start != original.bytes.start || end != original.bytes.end {
            if !matches!(token.kind, TokenType::Whitespace(_)) {
                return Err("A legacy interval splits a source token".into());
            }
            token.kind = TokenType::Whitespace(&source.text[start..end]);
            token.length = end - start;
        }
        input.push(token);
    }
    let tokens = adjust_spacing(normalize_and_expand_blocks(input));
    let mut tokens = tokens.iter().peekable();
    let mut output = String::new();
    while let Some(token) = tokens.next() {
        if let TokenType::Whitespace(text) = &token.kind {
            if text.contains('\n') {
                emit(
                    &"\n".repeat(text.bytes().filter(|byte| *byte == b'\n').count()),
                    state,
                    &mut output,
                );
            } else if state.column > 0
                && !(tokens.peek().is_none() && bytes.end == source.text.len())
            {
                emit(text, state, &mut output);
            }
            continue;
        }
        let closing = match token.kind {
            TokenType::RightParen => Some(DelimiterKind::Parenthesis),
            TokenType::RightBracket => Some(DelimiterKind::Bracket),
            TokenType::RightBrace => Some(DelimiterKind::Brace),
            _ => None,
        };
        if let Some(kind) = closing {
            let open = state
                .delimiters
                .pop()
                .ok_or("Missing enclosing delimiter in legacy handoff")?;
            if open.kind != kind {
                return Err("Mismatched legacy delimiter state".into());
            }
            state.indent_level = state
                .indent_level
                .checked_sub(usize::from(open.adds_indent))
                .ok_or("Invalid legacy indentation state")?;
        }
        if state.column == 0 {
            emit(
                &" ".repeat(state.indent_level * config.indent),
                state,
                &mut output,
            );
        }
        emit(&token.kind.to_string(), state, &mut output);
        let opening = match token.kind {
            TokenType::LeftParen => Some((DelimiterKind::Parenthesis, false)),
            TokenType::LeftBrace => Some((DelimiterKind::Brace, true)),
            TokenType::LeftBracket => Some((
                DelimiterKind::Bracket,
                tokens.peek().is_some_and(
                    |next| matches!(&next.kind, TokenType::Whitespace(text) if text.contains('\n')),
                ),
            )),
            _ => None,
        };
        if let Some((kind, adds_indent)) = opening {
            state.delimiters.push(OpenDelimiter { kind, adds_indent });
            state.indent_level += usize::from(adds_indent);
        }
    }
    Ok(output)
}

fn emit(text: &str, state: &mut RenderState, output: &mut String) {
    output.push_str(text);
    state.emitted(text);
}

fn normalize_and_expand_blocks(mut tokens: Vec<Token>) -> Vec<Token> {
    // Pass 1a: Ensure spaces before `{`
    let mut spaced_tokens: Vec<Token> = Vec::with_capacity(tokens.len() + 10);
    for token in &tokens {
        if let TokenType::LeftBrace = token.kind {
            if !spaced_tokens.is_empty()
                && !matches!(spaced_tokens.last().unwrap().kind, TokenType::Whitespace(_))
            {
                spaced_tokens.push(Token {
                    kind: TokenType::Whitespace(" "),
                    line: token.line,
                    column: token.column,
                    length: 1,
                    expansion: token.expansion,
                });
            }
        }
        spaced_tokens.push(token.clone());
    }
    tokens = spaced_tokens;

    // Pass 1b: Match braces in O(N) using a stack
    let mut stack = Vec::new();
    let mut left_to_right = HashMap::new();
    let mut right_to_left = HashMap::new();
    for (idx, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenType::LeftBrace => stack.push(idx),
            TokenType::RightBrace => {
                if let Some(lb_idx) = stack.pop() {
                    left_to_right.insert(lb_idx, idx);
                    right_to_left.insert(idx, lb_idx);
                }
            }
            _ => {}
        }
    }

    // Pass 1c: Process blocks (collapsing `unsafe` or expanding others)
    let mut new_tokens: Vec<Token> = Vec::with_capacity(tokens.len() + 20);
    let mut i = 0;
    while i < tokens.len() {
        if let TokenType::LeftBrace = tokens[i].kind {
            if let Some(&rb_idx) = left_to_right.get(&i) {
                let lb_idx = i;
                let has_newline = tokens[(lb_idx + 1)..rb_idx].iter().any(|t| {
                    if let TokenType::Whitespace(ws) = t.kind {
                        ws.contains('\n')
                    } else {
                        false
                    }
                });

                let mut is_unsafe_block = false;
                for j in (0..new_tokens.len()).rev() {
                    match new_tokens[j].kind {
                        TokenType::Whitespace(_)
                        | TokenType::Comment(_)
                        | TokenType::DocComment(_) => continue,
                        TokenType::Unsafe => {
                            is_unsafe_block = true;
                            break;
                        }
                        _ => break,
                    }
                }

                // A line comment inside runs to the end of its line, so a block holding one is
                // never put on one line: the comment would swallow the code after it.
                let holds_comment = tokens[lb_idx..=rb_idx]
                    .iter()
                    .any(|t| matches!(t.kind, TokenType::Comment(_) | TokenType::DocComment(_)));
                if is_unsafe_block && !holds_comment {
                    // Try to measure the total length if we collapsed it to a single line
                    let mut collapsed_len = 0;
                    for j in (0..new_tokens.len()).rev() {
                        if let TokenType::Whitespace(ws) = new_tokens[j].kind {
                            if let Some(pos) = ws.rfind('\n') {
                                collapsed_len += ws.len() - pos - 1;
                                break;
                            } else {
                                collapsed_len += new_tokens[j].length;
                            }
                        } else {
                            collapsed_len += new_tokens[j].length;
                        }
                    }
                    for t in &tokens[lb_idx..=rb_idx] {
                        if let TokenType::Whitespace(ws) = t.kind {
                            if ws.contains('\n') {
                                collapsed_len += 1;
                            } else {
                                collapsed_len += t.length;
                            }
                        } else {
                            collapsed_len += t.length;
                        }
                    }

                    if collapsed_len + 2 <= 80 {
                        // Collapse it! Replace all newlines inside with spaces.
                        new_tokens.push(tokens[lb_idx].clone());

                        if !matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
                            new_tokens.push(Token {
                                kind: TokenType::Whitespace(" "),
                                line: tokens[lb_idx].line,
                                column: tokens[lb_idx].column,
                                length: 1,
                                expansion: tokens[lb_idx].expansion,
                            });
                        }

                        for token in &tokens[(lb_idx + 1)..rb_idx] {
                            let mut t = token.clone();
                            if let TokenType::Whitespace(ref mut ws) = t.kind {
                                if ws.contains('\n') {
                                    *ws = " ";
                                }
                            }
                            new_tokens.push(t);
                        }

                        if !matches!(new_tokens.last().unwrap().kind, TokenType::Whitespace(_)) {
                            new_tokens.push(Token {
                                kind: TokenType::Whitespace(" "),
                                line: tokens[rb_idx].line,
                                column: tokens[rb_idx].column,
                                length: 1,
                                expansion: tokens[rb_idx].expansion,
                            });
                        }

                        new_tokens.push(tokens[rb_idx].clone());
                        i = rb_idx + 1;
                        continue;
                    }
                }

                let has_non_ws = tokens[(lb_idx + 1)..rb_idx].iter().any(|t| {
                    !matches!(
                        t.kind,
                        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
                    )
                });

                if !has_newline && has_non_ws {
                    new_tokens.push(tokens[lb_idx].clone());

                    if matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
                        let mut t = tokens[lb_idx + 1].clone();
                        t.kind = TokenType::Whitespace("\n");
                        new_tokens.push(t);
                        i = lb_idx + 2;
                    } else {
                        new_tokens.push(Token {
                            kind: TokenType::Whitespace("\n"),
                            line: 0,
                            column: 0,
                            length: 1,
                            expansion: 0,
                        });
                        i = lb_idx + 1;
                    }
                    continue;
                }
            }
        } else if let TokenType::RightBrace = tokens[i].kind {
            if let Some(&lb_idx) = right_to_left.get(&i) {
                let has_newline = tokens[(lb_idx + 1)..i].iter().any(|t| {
                    if let TokenType::Whitespace(ws) = t.kind {
                        ws.contains('\n')
                    } else {
                        false
                    }
                });

                let has_non_ws = tokens[(lb_idx + 1)..i].iter().any(|t| {
                    !matches!(
                        t.kind,
                        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
                    )
                });

                if !has_newline && has_non_ws {
                    let needs_newline = match new_tokens.last() {
                        Some(t) => {
                            if let TokenType::Whitespace(ws) = t.kind {
                                !ws.contains('\n')
                            } else {
                                true
                            }
                        }
                        None => true,
                    };

                    if needs_newline {
                        if let Some(t) = new_tokens.last_mut() {
                            if let TokenType::Whitespace(_) = t.kind {
                                t.kind = TokenType::Whitespace("\n");
                            } else {
                                new_tokens.push(Token {
                                    kind: TokenType::Whitespace("\n"),
                                    line: 0,
                                    column: 0,
                                    length: 1,
                                    expansion: 0,
                                });
                            }
                        }
                    }
                }
            }
        }

        new_tokens.push(tokens[i].clone());
        i += 1;
    }

    new_tokens
}

fn adjust_spacing(tokens: Vec<Token>) -> Vec<Token> {
    let mut new_tokens = Vec::with_capacity(tokens.len());
    let mut last_non_ws: Option<TokenType> = None;

    let mut next_non_ws_arr = vec![None; tokens.len()];
    let mut curr_next = None;
    for (i, token) in tokens.iter().enumerate().rev() {
        next_non_ws_arr[i] = curr_next.clone();
        if !matches!(
            token.kind,
            TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
        ) {
            curr_next = Some(token.kind.clone());
        }
    }

    for (i, mut token) in tokens.into_iter().enumerate() {
        if !matches!(
            token.kind,
            TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
        ) {
            last_non_ws = Some(token.kind.clone());
            new_tokens.push(token);
            continue;
        }

        if let TokenType::Whitespace(ref mut ws) = token.kind {
            let next_non_ws = &next_non_ws_arr[i];

            if ws.contains('\n') {
                if let Some(prev) = &last_non_ws {
                    if matches!(prev, TokenType::For | TokenType::If | TokenType::While) {
                        *ws = " ";
                    }
                }
                if let Some(next) = next_non_ws {
                    if matches!(
                        next,
                        TokenType::EqEq
                            | TokenType::NotEq
                            | TokenType::LessEq
                            | TokenType::GreaterEq
                            | TokenType::LeftAngle
                            | TokenType::RightAngle
                            | TokenType::AndAnd
                            | TokenType::OrOr
                    ) {
                        *ws = " ";
                    }
                }
            }

            if let (Some(TokenType::RightBrace), Some(TokenType::Else)) =
                (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
            if let (Some(TokenType::Else), Some(TokenType::LeftBrace)) = (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
        }

        new_tokens.push(token);
    }

    new_tokens
}
