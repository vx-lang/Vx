//===- update_mlir_test_checks.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility inspired by LLVM's update_llc_test_checks.py.
// It runs the compiler commands specified in `// RUN:` lines and updates the
// `// <PREFIX>:` lines in the file to match the actual MLIR output.
//
// Usage: cargo run --bin update_mlir_test_checks -- tests/optimizations/pass/*.vx
//
//===----------------------------------------------------------------------===//

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "update_mlir_test_checks",
    version,
    about = "Rewrite FileCheck lines in Vx MLIR tests from live vxc output"
)]
struct Cli {
    /// Test files whose CHECK lines should be refreshed
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

fn build_vxc() -> std::path::PathBuf {
    println!("Building vxc...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "vxc"])
        .status()
        .expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build vxc. Exiting.");
        std::process::exit(1);
    }

    env::current_dir()
        .unwrap()
        .join("target")
        .join("debug")
        .join("vxc")
}

fn parse_run_lines(content: &str, file_path: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut runs = Vec::new();
    let mut prefixes = Vec::new();

    for line in content.lines() {
        if line.trim().starts_with("// RUN:") {
            let cmd = line.split_once("RUN:").unwrap().1.trim();
            let vxc_cmd = cmd.split('|').next().unwrap().trim();
            let prefix = if cmd.contains("FileCheck") {
                if let Some(pos) = cmd.find("--check-prefix=") {
                    cmd[pos + 15..]
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .to_string()
                } else {
                    "CHECK".to_string()
                }
            } else {
                "CHECK".to_string()
            };

            let vxc_cmd = vxc_cmd.replace("%s", file_path);

            runs.push((vxc_cmd, prefix.clone()));
            if !prefixes.contains(&prefix) {
                prefixes.push(prefix);
            }
        }
    }
    (runs, prefixes)
}

fn filter_existing_checks(content: &str, prefixes: &[String]) -> Vec<String> {
    let mut new_lines = Vec::new();
    for line in content.lines() {
        let mut is_check = false;
        for prefix in prefixes {
            let check_str = format!("// {}:", prefix);
            if line.trim().starts_with(&check_str) {
                is_check = true;
                break;
            }
        }
        if !is_check {
            new_lines.push(line.to_string());
        }
    }

    while let Some(last) = new_lines.last() {
        if last.trim().is_empty() {
            new_lines.pop();
        } else {
            break;
        }
    }
    new_lines.push("".to_string());
    new_lines
}

fn parse_command_args(cmd: &str) -> Vec<String> {
    let mut args: Vec<String> = vec![];
    let mut current_arg = String::new();
    let mut in_quotes = false;
    for c in cmd.chars() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if c == ' ' && !in_quotes {
            if !current_arg.is_empty() {
                args.push(current_arg.clone());
                current_arg.clear();
            }
        } else {
            current_arg.push(c);
        }
    }
    if !current_arg.is_empty() {
        args.push(current_arg);
    }
    args
}

fn extract_mlir_checks(stdout: &str, prefix: &str) -> Vec<String> {
    let mut checks = Vec::new();
    let mut in_mlir = false;
    for line in stdout.lines() {
        if line.starts_with("module {")
            || line.starts_with("module ")
            || line.starts_with("\"builtin.module\"")
        {
            in_mlir = true;
        }
        if !in_mlir || line.trim().is_empty() {
            continue;
        }
        checks.push(format!("// {}: {}", prefix, line));
    }
    checks
}

fn extract_llvm_checks(stdout: &str, prefix: &str) -> Vec<String> {
    let mut checks = Vec::new();
    let mut in_llvm = false;
    for line in stdout.lines() {
        if line.starts_with("; ModuleID")
            || line.starts_with("declare ")
            || line.starts_with("define ")
        {
            in_llvm = true;
        }
        if !in_llvm || line.trim().is_empty() {
            continue;
        }
        checks.push(format!("// {}: {}", prefix, line));
    }
    checks
}

fn process_test_file(file_path: &Path, vxc_bin: &Path) -> Result<(), String> {
    let display = file_path.display().to_string();
    println!("Updating {}...", display);

    let original_content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read {}: {}", display, e))?;

    let (runs, prefixes) = parse_run_lines(&original_content, &display);
    if runs.is_empty() {
        return Err(format!("No RUN lines found in {}", display));
    }

    let mut new_lines = filter_existing_checks(&original_content, &prefixes);

    for (vxc_cmd, prefix) in runs {
        let mut args = parse_command_args(&vxc_cmd);

        let exec_name = args.remove(0);
        if exec_name != "vxc" && exec_name != "vx-opt" {
            return Err(format!(
                "Unknown executable in RUN line: {} (in {})",
                exec_name, display
            ));
        }

        let output = Command::new(vxc_bin)
            .args(&args)
            .output()
            .expect("Failed to execute compiler");

        let stdout = String::from_utf8_lossy(&output.stdout);

        let is_llvm = args.contains(&"--emit-llvm".to_string());

        let checks = if is_llvm {
            extract_llvm_checks(&stdout, &prefix)
        } else {
            extract_mlir_checks(&stdout, &prefix)
        };

        new_lines.extend(checks);
        new_lines.push("".to_string());
    }

    let new_content = new_lines.join("\n");
    if new_content != original_content {
        fs::write(file_path, new_content)
            .map_err(|e| format!("Error writing {}: {}", display, e))?;
        println!("  Updated.");
    } else {
        println!("  Unchanged.");
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let mut missing = Vec::new();
    for path in &cli.files {
        if !path.exists() {
            missing.push(path.display().to_string());
        }
    }
    if !missing.is_empty() {
        for path in &missing {
            eprintln!("error: file not found: {}", path);
        }
        std::process::exit(1);
    }

    let vxc_bin = build_vxc();

    let mut failed = false;
    for file_path in &cli.files {
        if let Err(e) = process_test_file(file_path, &vxc_bin) {
            eprintln!("error: {}", e);
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
}
