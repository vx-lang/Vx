//===- formatter.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
use vx_config::FormatConfig;

mod signatures;

pub fn format_file(content: &str, indent_spaces: usize) -> String {
    format_file_with_config(
        content,
        &FormatConfig {
            indent: indent_spaces,
            ..FormatConfig::default()
        },
    )
}

pub fn format_file_with_config(content: &str, config: &FormatConfig) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let tokens = lexer.tokenize();

    if let Some(token) = tokens.first() {
        if let TokenType::Comment(c) | TokenType::DocComment(c) = &token.kind {
            if c.contains("//! vx-format: OFF") {
                return content.to_string();
            }
        }
    }

    // A file that does not lex, such as one that ends inside a string, is left as it is: an
    // unknown token is written back as one character, which would lose the rest of its text.
    if tokens
        .iter()
        .any(|t| matches!(t.kind, TokenType::Unknown(_)))
    {
        return content.to_string();
    }

    let tokens = normalize_and_expand_blocks(tokens);
    let tokens = adjust_spacing(tokens);
    emit_formatted_string(tokens, config, content.len())
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

fn emit_formatted_string(tokens: Vec<Token>, config: &FormatConfig, original_len: usize) -> String {
    let indent_spaces = config.indent;
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
    // For each open `[`, whether its line ends after it, so its rows are indented one level.
    let mut brackets: Vec<bool> = Vec::new();

    let all_tokens = tokens;
    let mut tokens = all_tokens.iter().cloned().enumerate().peekable();
    while let Some((index, token)) = tokens.next() {
        if token.kind == TokenType::Eof {
            break;
        }

        match token.kind {
            TokenType::Fn if format_enabled => {
                let column = if is_new_line {
                    indent_level.max(0) as usize * indent_spaces
                } else {
                    formatted.rsplit('\n').next().unwrap_or("").chars().count()
                };
                if let Some((end, header)) = signatures::format_signature(
                    &all_tokens,
                    index,
                    column,
                    indent_level.max(0) as usize,
                    config,
                ) {
                    if is_new_line {
                        write_indent(&mut formatted, indent_level, indent_spaces);
                    }
                    formatted.push_str(&header);
                    is_new_line = false;
                    while tokens.peek().is_some_and(|(next, _)| *next < end) {
                        tokens.next();
                    }
                } else {
                    if is_new_line {
                        write_indent(&mut formatted, indent_level, indent_spaces);
                    }
                    formatted.push_str("fn");
                    is_new_line = false;
                }
            }
            TokenType::LeftBracket => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('[');
                let rows_below = matches!(
                    tokens.peek().map(|(_, t)| &t.kind),
                    Some(TokenType::Whitespace(ws)) if ws.contains('\n')
                );
                if rows_below {
                    indent_level += 1;
                }
                brackets.push(rows_below);
            }
            TokenType::RightBracket => {
                if brackets.pop() == Some(true) {
                    indent_level = (indent_level - 1).max(0);
                }
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push(']');
            }
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
            TokenType::Comment(c) | TokenType::DocComment(c) => {
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

    // Strip trailing whitespace from all lines
    let mut final_out = String::with_capacity(formatted.len());
    let mut iter = formatted.split('\n').peekable();
    while let Some(line) = iter.next() {
        final_out.push_str(line.trim_end_matches([' ', '\t']));
        if iter.peek().is_some() {
            final_out.push('\n');
        }
    }

    final_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_fit_on_one_line_or_wrap_at_the_configured_width() {
        let cases = [
            (
                100,
                2,
                "fn add(\nfirst: i32,\nsecond: i32\n) -> i32 {\nfirst + second\n}",
                "fn add(first: i32, second: i32) -> i32 {\n  first + second\n}",
            ),
            (
                39,
                2,
                "fn add(first: i32, second: i32) -> i32 {\nfirst + second\n}",
                "fn add(\n  first: i32,\n  second: i32\n) -> i32 {\n  first + second\n}",
            ),
            (
                40,
                4,
                "extern {\nfn add(first: i32, second: i32) -> i32;\n}",
                "extern {\n    fn add(\n        first: i32,\n        second: i32\n    ) -> i32;\n}",
            ),
            (
                40,
                2,
                "fn use_pair<T>(value: Tensor<T, [2, 2]>, pair: (i32, i32)) -> T;",
                "fn use_pair<T>(\n  value: Tensor<T, [2, 2]>,\n  pair: (i32, i32)\n) -> T;",
            ),
            (
                100,
                2,
                "pub fn add(first: i32,second: i32)->i32 {\nfirst + second\n}",
                "pub fn add(first: i32, second: i32) -> i32 {\n  first + second\n}",
            ),
        ];
        for (width, indent, input, expected) in cases {
            let config = FormatConfig {
                indent,
                fn_signature_width: width,
            };
            let output = format_file_with_config(input, &config);
            assert_eq!(output, expected, "width {width}, indent {indent}");
            assert_eq!(
                format_file_with_config(&output, &config),
                output,
                "formatting must be stable"
            );
            let content_tokens = |text: &str| {
                Lexer::new_with_comments(text)
                    .tokenize()
                    .into_iter()
                    .filter(|token| {
                        !matches!(token.kind, TokenType::Whitespace(_) | TokenType::Eof)
                    })
                    .map(|token| token.kind.to_string())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                content_tokens(input),
                content_tokens(&output),
                "source tokens must be preserved"
            );
        }
    }

    #[test]
    fn signature_width_includes_return_type_brace_and_indentation() {
        let input = "fn add(first: i32, second: i32) -> i32;";
        let exact_width = input.chars().count();
        let config = FormatConfig {
            indent: 2,
            fn_signature_width: exact_width,
        };
        assert_eq!(format_file_with_config(input, &config), input);
        let config = FormatConfig {
            fn_signature_width: exact_width - 1,
            ..config
        };
        assert_eq!(
            format_file_with_config(input, &config),
            "fn add(\n  first: i32,\n  second: i32\n) -> i32;"
        );
    }

    #[test]
    fn signature_layout_leaves_calls_comments_and_disabled_text_to_existing_rules() {
        let cases = [
            "fn call() {\n  add(first, second);\n}",
            "fn add(\nfirst: i32, // first value\nsecond: i32\n) -> i32;",
            "//! vx-format: OFF\nfn   add(first: i32,second: i32)->i32;",
            "// vx-format-begin: OFF\nfn   add(first: i32,second: i32)->i32;\n// vx-format-end: OFF",
        ];
        for input in cases {
            let narrow = FormatConfig {
                indent: 2,
                fn_signature_width: 10,
            };
            let wide = FormatConfig {
                fn_signature_width: 100,
                ..narrow.clone()
            };
            assert_eq!(
                format_file_with_config(input, &narrow),
                format_file_with_config(input, &wide)
            );
        }
    }

    #[test]
    fn test_format_off_top_level() {
        let input = "//! vx-format: OFF\nfn   foo()   {\nlet  x=1;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, input);
    }

    #[test]
    fn the_rows_of_a_multi_line_array_are_indented() {
        let input = "const T : Tensor<i64, [3]> = [\n1,\n2,\n3\n];\n\nfn main() -> i32 {\n  let a : Tensor<i64, [3]> = [\n  1, 2,\n  3\n  ];\n  let b = [ 1, 2 ];\n  return a[0] + b[1];\n}";
        let expected = "const T : Tensor<i64, [3]> = [\n  1,\n  2,\n  3\n];\n\nfn main() -> i32 {\n  let a : Tensor<i64, [3]> = [\n    1, 2,\n    3\n  ];\n  let b = [ 1, 2 ];\n  return a[0] + b[1];\n}";
        assert_eq!(format_file(input, 2), expected);
    }

    #[test]
    fn escapes_in_strings_are_kept() {
        let input =
            "fn main() -> i32 {\n  print!(\"a\\tb\\n say \\\"hi\\\" \\\\ \");\n  return 0;\n}";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn a_file_that_ends_inside_a_string_is_left_as_it_is() {
        let input = "fn main() -> i32 {\n    let s = \"abc";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn a_short_block_with_a_comment_stays_on_several_lines() {
        let input = "fn main() -> i32 {\n  let mut x = 1;\n  unsafe {\n    // set x\n    x = 2;\n  }\n  return x;\n}";
        assert_eq!(format_file(input, 2), input);
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
    fn test_format_while_loop() {
        let input = "while \n keep_going { step(); }";
        let expected = "while keep_going {\n  step();\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
        assert_eq!(format_file(&formatted, 2), formatted);
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
