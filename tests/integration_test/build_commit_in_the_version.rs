//===- build_commit_in_the_version.rs - Vx Compiler ------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A build with `VX_VERSION_WITH_COMMIT=1` makes `vxc --version` name the commit that vxc was built
// from, so two builds of one version can be told apart. The commit must be the one checked out
// now: Cargo builds vxc again before this test runs, so an older commit means the build did not
// notice a new one. A build without it prints the version alone. `vx-format`, `vx-api` and
// `vx-opt` print their own name and the same version. `vx-opt` also prints LLVM's lines after it.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

/// The short hash of the commit checked out in the Vx sources, or `None` when they are not a git
/// checkout of their own.
fn checked_out_commit() -> Option<String> {
    let vx_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(vx_root)
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let top = git(&["rev-parse", "--show-toplevel"])?;
    if std::fs::canonicalize(top).ok()? != std::fs::canonicalize(vx_root).ok()? {
        return None;
    }
    git(&["rev-parse", "--short=8", "HEAD"])
}

/// The first line `--version` prints for a tool named `name`.
fn expected_version_line(name: &str) -> String {
    // Cargo builds this test and the tools in one run, with the same environment.
    let with_commit = option_env!("VX_VERSION_WITH_COMMIT") == Some("1");
    match checked_out_commit().filter(|_| with_commit) {
        Some(commit) => format!("{name} {} ({commit})", env!("CARGO_PKG_VERSION")),
        None => format!("{name} {}", env!("CARGO_PKG_VERSION")),
    }
}

fn version_output(program: &str) -> String {
    let out = Command::new(program)
        .arg("--version")
        .output()
        .expect("run --version");
    assert!(out.status.success(), "{program} --version failed: {out:?}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn vxc_version_names_the_commit_it_was_built_from() {
    let expected = format!("{}\n", expected_version_line("vxc"));
    assert_eq!(version_output(env!("CARGO_BIN_EXE_vxc")), expected);
}

#[test]
fn vx_format_version_names_vx_format() {
    let expected = format!("{}\n", expected_version_line("vx-format"));
    assert_eq!(version_output(env!("CARGO_BIN_EXE_vx-format")), expected);
}

#[test]
fn vx_api_version_names_vx_api() {
    let expected = format!("{}\n", expected_version_line("vx-api"));
    assert_eq!(version_output(env!("CARGO_BIN_EXE_vx-api")), expected);
}

#[test]
fn vx_opt_version_starts_with_its_own_name() {
    let text = version_output(env!("CARGO_BIN_EXE_vx-opt"));
    let first_line = text.lines().next().unwrap_or_default();
    assert_eq!(first_line, expected_version_line("vx-opt"));
    // LLVM's own lines still follow.
    assert!(text.contains("LLVM version"), "{text}");
}
