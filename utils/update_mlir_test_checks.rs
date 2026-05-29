//===- update_mlir_test_checks.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: update_mlir_test_checks <test_files...>");
        std::process::exit(1);
    }

    println!("Building vxc...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "vxc"])
        .status()
        .expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build vxc. Exiting.");
        std::process::exit(1);
    }

    let vxc_bin = env::current_dir()
        .unwrap()
        .join("target")
        .join("debug")
        .join("vxc");

    for file_path in args {
        println!("Updating {}...", file_path);

        let original_content = match fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Failed to read {}: {}", file_path, e);
                continue;
            }
        };

        // 1. Find all RUN lines and prefixes
        let mut runs = Vec::new();
        let mut prefixes = Vec::new();

        for line in original_content.lines() {
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

                let vxc_cmd = vxc_cmd.replace("%s", &file_path);

                runs.push((vxc_cmd, prefix.clone()));
                if !prefixes.contains(&prefix) {
                    prefixes.push(prefix);
                }
            }
        }

        if runs.is_empty() {
            eprintln!("Warning: No RUN lines found in {}", file_path);
            continue;
        }

        // 2. Filter out all existing checks for all known prefixes
        let mut new_lines = Vec::new();
        for line in original_content.lines() {
            let mut is_check = false;
            for prefix in &prefixes {
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

        // 3. For each RUN line, run vxc and append output
        for (vxc_cmd, prefix) in runs {
            let mut args: Vec<String> = vec![];
            let mut current_arg = String::new();
            let mut in_quotes = false;
            for c in vxc_cmd.chars() {
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

            let exec_name = args.remove(0);
            if exec_name != "vxc" && exec_name != "vx-opt" {
                eprintln!("Unknown executable in RUN line: {}", exec_name);
                continue;
            }

            let output = Command::new(&vxc_bin)
                .args(&args)
                .output()
                .expect("Failed to execute compiler");

            let stdout = String::from_utf8_lossy(&output.stdout);

            let is_llvm = args.contains(&"--emit-llvm".to_string());
            let mut in_mlir = false;
            for line in stdout.lines() {
                if !is_llvm
                    && (line.starts_with("module {")
                        || line.starts_with("module ")
                        || line.starts_with("\"builtin.module\""))
                {
                    in_mlir = true;
                } else if is_llvm
                    && (line.starts_with("; ModuleID")
                        || line.starts_with("declare ")
                        || line.starts_with("define "))
                {
                    in_mlir = true;
                }
                if !in_mlir {
                    continue;
                }
                if line.trim().is_empty() {
                    continue;
                }
                new_lines.push(format!("// {}: {}", prefix, line));
            }
            new_lines.push("".to_string());
        }

        let new_content = new_lines.join("\n");
        if new_content != original_content {
            if let Err(e) = fs::write(&file_path, new_content) {
                eprintln!("Error writing {}: {}", file_path, e);
            } else {
                println!("  Updated.");
            }
        } else {
            println!("  Unchanged.");
        }
    }
}
