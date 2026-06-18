//===- vx-format.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the standalone Vx code formatter.
// It parses raw Vx source code using the compiler's lexer and outputs a standardized,
// cleanly indented version of the code, enforcing consistent styling conventions
// across all Vx projects.
//
//===----------------------------------------------------------------------===//
use std::fs;
use std::path::Path;

use vxc::formatter::format_file;

use clap::Parser;
use rayon::prelude::*;

#[derive(Parser)]
#[command(author, version, about = "Vx Code Formatter")]
struct Cli {
    /// Number of spaces for indentation
    #[arg(long, default_value_t = 2)]
    indent: usize,

    /// Files to format
    #[arg(required = true)]
    files: Vec<String>,
}

fn process_file(file_path: &str, indent_spaces: usize) -> anyhow::Result<()> {
    let path = Path::new(file_path);
    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    let content = fs::read_to_string(path)?;
    let formatted = format_file(&content, indent_spaces);
    if formatted != content {
        fs::write(path, formatted)?;
        println!("Formatted {}", file_path);
    } else {
        println!("Unchanged {}", file_path);
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    cli.files.par_iter().for_each(|file_path| {
        if let Err(e) = process_file(file_path, cli.indent) {
            eprintln!("Error processing {}: {:?}", file_path, e);
        }
    });
}
