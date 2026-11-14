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
// notice a new one. A build without it prints the version alone.
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

#[test]
fn vxc_version_names_the_commit_it_was_built_from() {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg("--version")
        .output()
        .expect("run vxc --version");
    assert!(out.status.success(), "vxc --version failed: {out:?}");
    // Cargo builds this test and vxc in one run, with the same environment.
    let with_commit = option_env!("VX_VERSION_WITH_COMMIT") == Some("1");
    let expected = match checked_out_commit().filter(|_| with_commit) {
        Some(commit) => format!("vxc {} ({commit})\n", env!("CARGO_PKG_VERSION")),
        None => format!("vxc {}\n", env!("CARGO_PKG_VERSION")),
    };
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}
