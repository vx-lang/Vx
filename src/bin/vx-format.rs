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

use vx_config::{resolve_format_config, FormatConfig, SettingOverride};
use vxc::formatter::{check_rule_names, default_config_toml, format_file_with_report};

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

#[derive(Clone, ValueEnum)]
enum PrintConfig {
    Default,
}

#[derive(Parser)]
#[command(name = "vx-format", author, version = vxc::driver::VERSION, about = "Vx Code Formatter")]
struct Cli {
    /// Print built-in configuration as TOML without formatting files
    #[arg(long, value_enum, conflicts_with_all = ["indent", "set", "config", "files", "check"])]
    print_config: Option<PrintConfig>,

    /// Show formatting differences without modifying files
    #[arg(long)]
    check: bool,

    /// Override indentation width (defaults to two spaces without configuration)
    #[arg(long)]
    indent: Option<usize>,

    /// Override one setting of [format], as in `max_width=80` or `rules.function_headers=false`
    #[arg(long, value_name = "KEY=VALUE", value_parser = SettingOverride::parse)]
    set: Vec<SettingOverride>,

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
    let outcome = format_file_with_report(content, config);
    for warning in &outcome.warnings {
        eprintln!("{}", warning.render(file_path, content));
    }
    let formatted = outcome.text;
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
    } else if !check && outcome.warnings.is_empty() {
        println!("Unchanged {}", file_path);
    }
    Ok(changed)
}

fn run(cli: Cli) -> anyhow::Result<bool> {
    if let Some(PrintConfig::Default) = cli.print_config {
        std::io::stdout()
            .lock()
            .write_all(default_config_toml().as_bytes())?;
        return Ok(false);
    }

    let mut overrides: Vec<SettingOverride> = cli
        .indent
        .map(|indent| SettingOverride::parse(&format!("indent={indent}")))
        .transpose()
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .collect();
    overrides.extend(cli.set.iter().cloned());
    for setting in &overrides {
        if let Some(rule) = setting.key().strip_prefix("rules.") {
            check_rule_names(&FormatConfig::default().with_rule(rule, true))
                .map_err(|message| anyhow::anyhow!("--set {}: {message}", setting.text()))?;
        }
    }

    // Validate every configuration and read every input before modifying files.
    let inputs = cli
        .files
        .iter()
        .map(|file_path| {
            let path = Path::new(file_path);
            let config = resolve_format_config(path, cli.config.as_deref(), &overrides)?;
            // The rules from `--set` are already checked, so an unknown one is in the file.
            check_rule_names(&config.settings).map_err(|message| match &config.source {
                Some(source) => anyhow::anyhow!("{}: {message}", source.display()),
                None => anyhow::Error::msg(message),
            })?;
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

/// The stack of each thread that formats files. The formatter reads a file with the compiler's
/// parser, which needs the stack of a main thread for deeply nested expressions.
const FORMAT_THREAD_STACK: usize = 8 * 1024 * 1024;

fn main() -> ExitCode {
    rayon::ThreadPoolBuilder::new()
        .stack_size(FORMAT_THREAD_STACK)
        .build_global()
        .expect("the thread pool is built once, before any formatting");
    match run(Cli::parse()) {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::from(1),
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::from(2)
        }
    }
}
