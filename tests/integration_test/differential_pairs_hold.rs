//===- differential_pairs_hold.rs - Vx Compiler -----------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each differential pair's Vx side still reaches the verdict its `pair.env` claims.
//
// The pairs are the published evidence that Vx refuses at compile time what CUDA
// discovers at run time, so a stale one is a wrong claim rather than a stale test.
// Nothing walked `utils/differential/`, and pair 04 went stale exactly that way: it
// places three tiles and read none of them, so once a tile's life ran to its last
// reader instead of to the end of the function, all three were released where they
// were made, the peak became one tile, and the pair compiled clean where it claims
// E6010. It was found by a reader, not by the suite.
//
// This checks the verdict (refused / admitted) and the diagnostic code, which is
// what the pair's own `E_CODE` promises. A pair whose code is `uncoded` refuses
// through a message that carries no code, so only the refusal is asserted.
//
//===----------------------------------------------------------------------===//

use std::path::PathBuf;
use std::process::Command;

fn field<'a>(env: &'a str, key: &str) -> Option<&'a str> {
    env.lines().find_map(|l| {
        l.strip_prefix(key)?
            .trim()
            .strip_prefix('"')?
            .strip_suffix('"')
    })
}

#[test]
fn every_differential_pair_reaches_its_claimed_verdict() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pairs_dir = root.join("utils/differential/pairs");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&pairs_dir)
        .expect("utils/differential/pairs is missing")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert!(
        dirs.len() >= 6,
        "expected the published pairs, found {}",
        dirs.len()
    );

    let mut wrong = Vec::new();
    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let env = std::fs::read_to_string(dir.join("pair.env")).expect("pair.env");
        let Some(src) = field(&env, "VX_SOURCE=") else {
            wrong.push(format!("  {name}: pair.env has no VX_SOURCE"));
            continue;
        };
        let code = field(&env, "E_CODE=").unwrap_or("uncoded");
        let args: Vec<String> = field(&env, "VX_ARGS=")
            .unwrap_or("--action emit-mlir -o /dev/null")
            .split_whitespace()
            .map(str::to_string)
            .collect();

        let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
            .current_dir(&root)
            .arg(src)
            .args(&args)
            .output()
            .expect("failed to run vxc");
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let refused = log.lines().any(|l| l.starts_with("Error"));

        // `none` is the admitting pair: it states a placement the machine accepts, and
        // is what keeps the refusing pairs from being satisfied by a compiler that
        // refuses everything.
        if code == "none" {
            if refused {
                wrong.push(format!(
                    "  {name}: claims a clean admit, but was refused:\n      {}",
                    log.lines().find(|l| l.starts_with("Error")).unwrap_or("")
                ));
            }
            continue;
        }
        if !refused {
            wrong.push(format!(
                "  {name}: claims {code}, but the program compiled clean"
            ));
            continue;
        }
        if code != "uncoded" && !log.contains(code) {
            wrong.push(format!(
                "  {name}: claims {code}, refused with something else:\n      {}",
                log.lines().find(|l| l.starts_with("Error")).unwrap_or("")
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "differential pairs no longer match what they claim:\n{}",
        wrong.join("\n")
    );
}
