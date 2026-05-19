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
pub fn format_compiler_error(
    source: &str,
    line: usize,
    col: usize,
    len: usize,
    msg: &str,
) -> String {
    let mut out = format!("Error at {}:{}: {}\n", line, col, msg);
    let lines: Vec<&str> = source.lines().collect();

    if line > 0 && line <= lines.len() {
        let src_line = lines[line - 1];
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
            } else if i == col - 1 {
                pointer.push('^');
            } else if i < col - 1 + len {
                pointer.push('~');
            } else {
                break;
            }
        }
        out.push_str(&pointer);
        out.push('\n');
    }
    out
}
