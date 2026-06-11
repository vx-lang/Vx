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
use crate::lexer::{Lexer, TokenType};

pub fn format_file(content: &str, indent_spaces: usize) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let mut tokens = lexer.tokenize();

    // Pass 1: Token Stream Normalization - Expand single-line blocks
    let mut i = 0;
    while i < tokens.len() {
        if let TokenType::RightBrace = tokens[i].kind {
            let rb_idx = i;
            let mut lb_idx = None;
            let mut depth = 1;
            for j in (0..rb_idx).rev() {
                match tokens[j].kind {
                    TokenType::RightBrace => depth += 1,
                    TokenType::LeftBrace => {
                        depth -= 1;
                        if depth == 0 {
                            lb_idx = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(lb_idx) = lb_idx {
                let has_newline = tokens[(lb_idx + 1)..rb_idx].iter().any(|t| {
                    if let TokenType::Whitespace(ref ws) = t.kind {
                        ws.contains('\n')
                    } else {
                        false
                    }
                });
                let has_non_ws = tokens[(lb_idx + 1)..rb_idx]
                    .iter()
                    .any(|t| !matches!(t.kind, TokenType::Whitespace(_) | TokenType::Comment(_)));

                if !has_newline && has_non_ws {
                    // Expand the block by inserting newlines after { and before }
                    if matches!(tokens[rb_idx - 1].kind, TokenType::Whitespace(_)) {
                        tokens[rb_idx - 1].kind = TokenType::Whitespace("\n".to_string());
                    } else {
                        tokens.insert(
                            rb_idx,
                            crate::lexer::Token {
                                kind: TokenType::Whitespace("\n".to_string()),
                                line: 0,
                                column: 0,
                                length: 1,
                            },
                        );
                        i += 1;
                    }

                    if matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
                        tokens[lb_idx + 1].kind = TokenType::Whitespace("\n".to_string());
                    } else {
                        tokens.insert(
                            lb_idx + 1,
                            crate::lexer::Token {
                                kind: TokenType::Whitespace("\n".to_string()),
                                line: 0,
                                column: 0,
                                length: 1,
                            },
                        );
                        i += 1;
                    }
                }
            }
        }
        i += 1;
    }

    for i in 0..tokens.len() {
        if matches!(tokens[i].kind, TokenType::Whitespace(_)) {
            let prev_non_ws = (0..i).rev().find_map(|j| {
                if !matches!(
                    tokens[j].kind,
                    TokenType::Whitespace(_) | TokenType::Comment(_)
                ) {
                    Some(tokens[j].kind.clone())
                } else {
                    None
                }
            });
            let next_non_ws = ((i + 1)..tokens.len()).find_map(|j| {
                if !matches!(
                    tokens[j].kind,
                    TokenType::Whitespace(_) | TokenType::Comment(_)
                ) {
                    Some(tokens[j].kind.clone())
                } else {
                    None
                }
            });

            if let TokenType::Whitespace(ref mut ws) = tokens[i].kind {
                if ws.contains('\n') {
                    if let Some(prev) = &prev_non_ws {
                        if matches!(prev, TokenType::For | TokenType::If) {
                            *ws = " ".to_string();
                        }
                    }
                    if let Some(next) = &next_non_ws {
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
                            *ws = " ".to_string();
                        }
                    }
                }

                // Enforce } else { formatting unconditionally on whitespaces between them
                if let (Some(TokenType::RightBrace), Some(TokenType::Else)) =
                    (&prev_non_ws, &next_non_ws)
                {
                    *ws = " ".to_string();
                }
                if let (Some(TokenType::Else), Some(TokenType::LeftBrace)) =
                    (&prev_non_ws, &next_non_ws)
                {
                    *ws = " ".to_string();
                }
            }
        }
    }

    let mut formatted = String::with_capacity(content.len() + content.len() / 10);
    let mut indent_level: isize = 0;
    let indent_str = " ".repeat(indent_spaces);

    let mut is_new_line = true;
    let mut format_enabled = true;

    for token in tokens {
        if token.kind == TokenType::Eof {
            break;
        }

        match token.kind {
            TokenType::Whitespace(ws) => {
                if !format_enabled {
                    formatted.push_str(&ws);
                    if ws.contains('\n') {
                        is_new_line = true;
                    }
                } else {
                    if ws.contains('\n') {
                        // Output the newlines and reset the start-of-line flag
                        let newlines = ws.chars().filter(|&c| c == '\n').count();
                        for _ in 0..newlines {
                            formatted.push('\n');
                        }
                        is_new_line = true;
                    } else {
                        // Only output inline spaces if we aren't at the very start of a line
                        if !is_new_line {
                            formatted.push_str(&ws);
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
                    for _ in 0..indent_level {
                        formatted.push_str(&indent_str);
                    }
                }
                is_new_line = false;
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if format_enabled && is_new_line {
                    for _ in 0..indent_level {
                        formatted.push_str(&indent_str);
                    }
                }
                is_new_line = false;
                formatted.push('{');
                indent_level += 1;
            }
            TokenType::Comment(c) => {
                if c.contains("//! vx-format: OFF") {
                    return content.to_string();
                }

                let mut turning_off = false;
                if c.contains("vx-format-begin: OFF") {
                    format_enabled = false;
                    turning_off = true;
                } else if c.contains("vx-format-begin: ON") || c.contains("vx-format-end: OFF") {
                    format_enabled = true;
                }

                if (format_enabled || turning_off) && is_new_line {
                    for _ in 0..indent_level {
                        formatted.push_str(&indent_str);
                    }
                }
                is_new_line = false;
                formatted.push_str(&c);
            }
            other => {
                if format_enabled && is_new_line {
                    for _ in 0..indent_level {
                        formatted.push_str(&indent_str);
                    }
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
