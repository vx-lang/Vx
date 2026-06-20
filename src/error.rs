//===- error.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the error handling and diagnostic reporting infrastructure.
// It provides formatting routines to display rich compiler errors with source code
// snippets, line numbers, and colorful annotations to help developers quickly
// identify and resolve syntax or semantic issues.
//
//===----------------------------------------------------------------------===//

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub fn format_compiler_error(
    source: &str,
    line: usize,
    col: usize,
    len: usize,
    msg: &str,
) -> String {
    if line > 0 {
        if let Some(src_line) = source.lines().nth(line - 1) {
            let mut out = String::with_capacity(msg.len() + src_line.len() * 2 + 64);
            use std::fmt::Write;
            let _ = writeln!(out, "Error at {}:{}: {}", line, col, msg);

            out.push_str(src_line);
            out.push('\n');

            let mut pointer = String::new();
            for (i, c) in src_line.chars().enumerate() {
                if i < col - 1 {
                    if c == '\t' {
                        pointer.push('\t');
                    } else {
                        pointer.push(' ');
                    }
                } else {
                    break;
                }
            }

            // Replace up to 4 trailing spaces with tildes for better visual anchoring
            let mut prefix_tildes = 0;
            while prefix_tildes < 4 && pointer.ends_with(' ') {
                pointer.pop();
                prefix_tildes += 1;
            }
            for _ in 0..prefix_tildes {
                pointer.push('~');
            }

            for _ in 0..len {
                pointer.push('^');
            }
            pointer.push_str("~~~~");
            out.push_str(&pointer);
            out.push('\n');
            return out;
        }
    }

    // Fallback if line is out of bounds
    format!("Error at {}:{}: {}\n", line, col, msg)
}

pub fn format_compiler_warning(
    source: &str,
    line: usize,
    col: usize,
    len: usize,
    code: Option<crate::diagnostic::DiagnosticCode>,
    msg: &str,
) -> String {
    let code_str = match code {
        Some(c) => format!("[{}]", c),
        None => String::new(),
    };

    if line > 0 {
        if let Some(src_line) = source.lines().nth(line - 1) {
            let mut out = String::with_capacity(msg.len() + src_line.len() * 2 + 64);
            use std::fmt::Write;
            let _ = writeln!(out, "Warning{} at {}:{}: {}", code_str, line, col, msg);

            out.push_str(src_line);
            out.push('\n');

            let mut pointer = String::new();
            for (i, c) in src_line.chars().enumerate() {
                if i < col - 1 {
                    if c == '\t' {
                        pointer.push('\t');
                    } else {
                        pointer.push(' ');
                    }
                } else {
                    break;
                }
            }

            // Replace up to 4 trailing spaces with tildes for better visual anchoring
            let mut prefix_tildes = 0;
            while prefix_tildes < 4 && pointer.ends_with(' ') {
                pointer.pop();
                prefix_tildes += 1;
            }
            for _ in 0..prefix_tildes {
                pointer.push('~');
            }

            for _ in 0..len {
                pointer.push('^');
            }
            pointer.push_str("~~~~");
            out.push_str(&pointer);
            out.push('\n');
            return out;
        }
    }

    // Fallback if line is out of bounds
    format!("Warning{} at {}:{}: {}\n", code_str, line, col, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_compiler_error_len_1_with_padding() {
        // "    + " (col = 5)
        let source = "    + ";
        let formatted = format_compiler_error(source, 1, 5, 1, "Expected operand");

        // Should produce:
        // Error at 1:5: Expected operand
        //     +
        // ~~~~^~~~~
        let expected = "Error at 1:5: Expected operand\n    + \n~~~~^~~~~\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_compiler_error_len_1_no_padding() {
        // "+ " (col = 1)
        let source = "+ ";
        let formatted = format_compiler_error(source, 1, 1, 1, "Unexpected");

        // Should produce:
        // Error at 1:1: Unexpected
        // +
        // ^~~~~
        let expected = "Error at 1:1: Unexpected\n+ \n^~~~~\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_compiler_error_len_4() {
        // "asdf" (col = 1)
        let source = "asdf";
        let formatted = format_compiler_error(source, 1, 1, 4, "Unknown identifier");

        // Should produce:
        // Error at 1:1: Unknown identifier
        // asdf
        // ^^^^~~~~
        let expected = "Error at 1:1: Unknown identifier\nasdf\n^^^^~~~~\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_compiler_error_len_4_with_padding() {
        // "    asdf" (col = 5) - full 4 spaces of padding
        let source = "    asdf";
        let formatted = format_compiler_error(source, 1, 5, 4, "Unknown identifier");

        // Should produce:
        // Error at 1:5: Unknown identifier
        //     asdf
        // ~~~~^^^^~~~~
        let expected = "Error at 1:5: Unknown identifier\n    asdf\n~~~~^^^^~~~~\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_compiler_error_len_4_with_partial_padding() {
        // "  asdf" (col = 3) - only 2 spaces of padding available to replace
        let source = "  asdf";
        let formatted = format_compiler_error(source, 1, 3, 4, "Unknown identifier");

        // Should produce:
        // Error at 1:3: Unknown identifier
        //   asdf
        // ~~^^^^~~~~
        let expected = "Error at 1:3: Unknown identifier\n  asdf\n~~^^^^~~~~\n";
        assert_eq!(formatted, expected);
    }
}
