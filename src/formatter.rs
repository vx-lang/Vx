//===- formatter.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file provides the core logic for the Vx code formatter.
// It iterates over token streams produced by the lexer and applies intelligent
// spacing, indentation, and line-wrapping rules to generate idiomatic and
// readable Vx source code.
//
//===----------------------------------------------------------------------===//
use crate::lexer::{Lexer, Token, TokenType};
use std::collections::HashMap;

pub fn format_file(content: &str, indent_spaces: usize) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let tokens = lexer.tokenize();

    if let Some(token) = tokens.first() {
        if let TokenType::Comment(c) = &token.kind {
            if c.contains("//! vx-format: OFF") {
                return content.to_string();
            }
        }
    }

    let tokens = normalize_and_expand_blocks(tokens);
    let tokens = adjust_spacing(tokens);
    emit_formatted_string(tokens, indent_spaces, content.len())
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
                        TokenType::Whitespace(_) | TokenType::Comment(_) => continue,
                        TokenType::Unsafe => {
                            is_unsafe_block = true;
                            break;
                        }
                        _ => break,
                    }
                }

                if is_unsafe_block {
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
                            });
                        }

                        new_tokens.push(tokens[rb_idx].clone());
                        i = rb_idx + 1;
                        continue;
                    }
                }

                let has_non_ws = tokens[(lb_idx + 1)..rb_idx]
                    .iter()
                    .any(|t| !matches!(t.kind, TokenType::Whitespace(_) | TokenType::Comment(_)));

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

                let has_non_ws = tokens[(lb_idx + 1)..i]
                    .iter()
                    .any(|t| !matches!(t.kind, TokenType::Whitespace(_) | TokenType::Comment(_)));

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
        if !matches!(token.kind, TokenType::Whitespace(_) | TokenType::Comment(_)) {
            curr_next = Some(token.kind.clone());
        }
    }

    for (i, mut token) in tokens.into_iter().enumerate() {
        if !matches!(token.kind, TokenType::Whitespace(_) | TokenType::Comment(_)) {
            last_non_ws = Some(token.kind.clone());
            new_tokens.push(token);
            continue;
        }

        if let TokenType::Whitespace(ref mut ws) = token.kind {
            let next_non_ws = &next_non_ws_arr[i];

            if ws.contains('\n') {
                if let Some(prev) = &last_non_ws {
                    if matches!(prev, TokenType::For | TokenType::If) {
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

fn emit_formatted_string(tokens: Vec<Token>, indent_spaces: usize, original_len: usize) -> String {
    let mut formatted = String::with_capacity(original_len + original_len / 10);
    let mut indent_level: isize = 0;

    fn write_indent(out: &mut String, level: isize, spaces: usize) {
        if level > 0 {
            for _ in 0..(level as usize * spaces) {
                out.push(' ');
            }
        }
    }

    let mut is_new_line = true;
    let mut format_enabled = true;

    for token in tokens {
        if token.kind == TokenType::Eof {
            break;
        }

        match token.kind {
            TokenType::Whitespace(ws) => {
                if !format_enabled {
                    formatted.push_str(ws);
                    if ws.contains('\n') {
                        is_new_line = true;
                    }
                } else {
                    if ws.contains('\n') {
                        let newlines = ws.chars().filter(|&c| c == '\n').count();
                        for _ in 0..newlines {
                            formatted.push('\n');
                        }
                        is_new_line = true;
                    } else {
                        if !is_new_line {
                            formatted.push_str(ws);
                        }
                    }
                }
            }
            TokenType::RightBrace => {
                indent_level -= 1;
                if indent_level < 0 {
                    indent_level = 0;
                }

                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('{');
                indent_level += 1;
            }
            TokenType::Comment(c) => {
                let mut turning_off = false;
                if c.contains("vx-format-begin: OFF") {
                    format_enabled = false;
                    turning_off = true;
                } else if c.contains("vx-format-begin: ON") || c.contains("vx-format-end: OFF") {
                    format_enabled = true;
                }

                if (format_enabled || turning_off) && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push_str(c);
            }
            other => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                use std::fmt::Write;
                write!(&mut formatted, "{}", other).unwrap();
            }
        }
    }

    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_off_top_level() {
        let input = "//! vx-format: OFF\nfn   foo()   {\nlet  x=1;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, input);
    }

    #[test]
    fn test_format_off_block() {
        let input = "fn foo() {\n// vx-format-begin: OFF\n      let   x = 1;\n// vx-format-end: OFF\nlet y = 2;\n}";
        let expected = "fn foo() {\n  // vx-format-begin: OFF\n      let   x = 1;\n  // vx-format-end: OFF\n  let y = 2;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_newline_stripping() {
        let input = "for \n i in 0..10 {\n  if sum_idx\n == 0 { a = 1; }\n}";
        let expected = "for i in 0..10 {\n  if sum_idx == 0 {\n    a = 1;\n  }\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_else_single_line_expansion() {
        let input = "if a > 0 { x = 1; } else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_dangling_else() {
        let input = "if a > 0 {\n  x = 1;\n}\n else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }
}
