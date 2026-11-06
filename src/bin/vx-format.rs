//===- vx-format.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use vx_config::{default_format_config_toml, resolve_format_config, FormatConfig};
use vxc::formatter::format_file;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

#[derive(Clone, ValueEnum)]
enum PrintConfig {
    Default,
}

#[derive(Parser)]
#[command(author, version, about = "Vx Code Formatter")]
struct Cli {
    /// Print built-in configuration as TOML without formatting files
    #[arg(long, value_enum, conflicts_with_all = ["indent", "config", "files", "check"])]
    print_config: Option<PrintConfig>,

    /// Show formatting differences without modifying files
    #[arg(long)]
    check: bool,

    /// Override indentation width (defaults to two spaces without configuration)
    #[arg(long)]
    indent: Option<usize>,

    /// Configuration file for all inputs (otherwise discover vx-format.toml per input)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Files to format
    #[arg(required_unless_present = "print_config")]
    files: Vec<String>,
}

fn process_file(
    file_path: &str,
    content: &str,
    config: &FormatConfig,
    check: bool,
) -> anyhow::Result<bool> {
    let path = Path::new(file_path);
    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    let formatted = format_file(content, config.indent);
    let changed = formatted != content;
    if changed {
        if check {
            let diff = similar::TextDiff::from_lines(content, &formatted);
            let report = diff
                .unified_diff()
                .context_radius(3)
                .header(file_path, &format!("{file_path} (formatted)"))
                .to_string();
            std::io::stdout().lock().write_all(report.as_bytes())?;
        } else {
            fs::write(path, formatted)
                .map_err(|error| anyhow::anyhow!("{}: {}", path.display(), error))?;
            println!("Formatted {}", file_path);
        }
    } else if !check {
        println!("Unchanged {}", file_path);
    }
    Ok(changed)
}

fn run(cli: Cli) -> anyhow::Result<bool> {
    if let Some(PrintConfig::Default) = cli.print_config {
        std::io::stdout()
            .lock()
            .write_all(default_format_config_toml().as_bytes())?;
        return Ok(false);
    }

    // Validate every configuration and read every input before modifying files.
    let inputs = cli
        .files
        .iter()
        .map(|file_path| {
            let path = Path::new(file_path);
            let config = resolve_format_config(path, cli.config.as_deref(), cli.indent)?;
            let content = fs::read_to_string(path)
                .map_err(|error| anyhow::anyhow!("{}: {}", path.display(), error))?;
            Ok((file_path, content, config.settings))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let changes = inputs
        .par_iter()
        .map(|(path, content, config)| process_file(path, content, config, cli.check))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(cli.check && changes.into_iter().any(|changed| changed))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::from(1),
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::from(2)
        }
    }
}
