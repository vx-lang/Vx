//===- update_mlir_test_checks.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// A utility inspired by LLVM's update_llc_test_checks.py.
// It runs `vxc --emit-mlir` on the given test files and updates the `// CHECK:`
// lines in the file to match the actual MLIR output.
//
// Usage: cargo run --bin update_mlir_test_checks -- tests/middle_end/pass/*.vx
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

    // Build the vxc compiler first
    println!("Building vxc...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "vxc"])
        .status()
        .expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build vxc. Exiting.");
        std::process::exit(1);
    }

    let vxc_bin = Path::new("target/debug/vxc");

    for file_path in args {
        println!("Updating {}...", file_path);

        // Read the original file
        let original_content = match fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Failed to read {}: {}", file_path, e);
                continue;
            }
        };

        // Run vxc --emit-mlir on the file
        let output = Command::new(vxc_bin)
            .args(["--emit-mlir", &file_path])
            .output()
            .expect("Failed to execute vxc");

        if !output.status.success() {
            eprintln!(
                "Warning: vxc failed on {}:\n{}",
                file_path,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        let mlir_output = String::from_utf8_lossy(&output.stdout);

        // Strip existing CHECK lines and trailing whitespace
        let mut new_lines = Vec::new();
        for line in original_content.lines() {
            if !line.trim().starts_with("// CHECK:") {
                new_lines.push(line.to_string());
            }
        }

        // Remove trailing empty lines before appending checks
        while let Some(last) = new_lines.last() {
            if last.trim().is_empty() {
                new_lines.pop();
            } else {
                break;
            }
        }

        // Add a blank line separator
        new_lines.push("".to_string());

        // Append new CHECK lines for MLIR only
        let mut in_mlir = false;
        for line in mlir_output.lines() {
            if line.starts_with("module {") || line.starts_with("\"builtin.module\"") {
                in_mlir = true;
            }
            if !in_mlir {
                continue;
            }
            if line.trim().is_empty() {
                continue; // Skip empty lines in MLIR output to keep tests clean
            }
            new_lines.push(format!("// CHECK: {}", line));
        }

        // Ensure trailing newline
        new_lines.push("".to_string());

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
