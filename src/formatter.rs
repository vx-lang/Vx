//===- formatter.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the core logic for the Vx code formatter.
// It composes registered syntax layouts with bounded legacy formatting.
// Final validation protects original tokens, grammatical gaps, and preserved source.
//
//===----------------------------------------------------------------------===//
use vx_config::FormatConfig;

mod legacy;
pub mod source;

pub use source::dispatch::{Coverage, FormatOutcome};

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
    format_file_with_report(content, config).text
}

pub fn format_file_with_report(content: &str, config: &FormatConfig) -> FormatOutcome {
    source::engine::format_source(content, config)
}

/// Run the shared pipeline with explicitly registered syntax formatters.
pub fn format_file_with_registry(
    content: &str,
    config: &FormatConfig,
    registry: &source::dispatch::NodeRegistry<'_>,
) -> FormatOutcome {
    source::engine::format_source_with_registry(content, config, registry)
}

#[cfg(test)]
mod activation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Lexer, TokenType};

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
                ..FormatConfig::default()
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
            ..FormatConfig::default()
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
    fn colon_spacing_applies_to_parameters_and_generic_bounds() {
        let input = "fn identity<T  :Copy>(value:T, other: package::Value<T>) -> T {\n  value\n}";
        for (space_before_colon, expected) in [
            (
                false,
                "fn identity<T: Copy>(value: T, other: package::Value<T>) -> T {\n  value\n}",
            ),
            (
                true,
                "fn identity<T : Copy>(value : T, other : package::Value<T>) -> T {\n  value\n}",
            ),
        ] {
            let config = FormatConfig {
                space_before_colon,
                ..FormatConfig::default()
            };
            let result = format_file_with_report(input, &config);
            assert!(result.warnings.is_empty(), "{:?}", result.warnings);
            assert_eq!(
                result.text, expected,
                "space_before_colon = {space_before_colon}"
            );
            assert_eq!(format_file_with_config(&result.text, &config), result.text);
        }
    }

    #[test]
    fn spaces_before_colons_count_toward_signature_width() {
        let input = "fn add(first: i32, second: i32) -> i32 {\n  first + second\n}";
        let config = FormatConfig {
            fn_signature_width: input.lines().next().unwrap().chars().count(),
            ..FormatConfig::default()
        };
        assert_eq!(format_file_with_config(input, &config), input);
        let config = FormatConfig {
            space_before_colon: true,
            ..config
        };
        assert_eq!(
            format_file_with_config(input, &config),
            "fn add(\n  first : i32,\n  second : i32\n) -> i32 {\n  first + second\n}"
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
                ..FormatConfig::default()
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
        let expected = "fn foo() {\n// vx-format-begin: OFF\n      let   x = 1;\n// vx-format-end: OFF\n  let y = 2;\n}";
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
