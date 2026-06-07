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
    let mut formatted = String::new();
    let mut indent_level: isize = 0;
    let indent_str = " ".repeat(indent_spaces);

    let mut is_new_line = true;
    let mut format_enabled = true;

    loop {
        let token = lexer.next_token();
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
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                }
                is_new_line = false;
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if format_enabled && is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
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
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                }
                is_new_line = false;
                formatted.push_str(&c);
            }
            other => {
                if format_enabled && is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                }
                is_new_line = false;
                formatted.push_str(&other.to_string());
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
}
