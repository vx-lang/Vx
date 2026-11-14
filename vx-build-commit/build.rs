//===- build.rs - Vx Build Commit ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Asks git for the commit that `vxc --version` names, and tells Cargo which files change when
//! that commit changes.
//!
//! This is a crate of its own so that a commit or a checkout reruns only this script. The
//! compiler's own build script also compiles the C++ runtime, and a rerun repeats all of it.
//!
//! It asks git only when `VX_VERSION_WITH_COMMIT=1`. The release build and CI set it. Without
//! it a commit changes nothing here, so it does not recompile vxc.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Set to 1 to make `vxc --version` name the commit.
const WITH_COMMIT: &str = "VX_VERSION_WITH_COMMIT";

/// Whether the build asked for the commit. Any value other than 1 or 0 stops the build.
fn with_commit() -> bool {
    match env::var(WITH_COMMIT) {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Err(env::VarError::NotPresent) => false,
        Ok(value) => panic!("{WITH_COMMIT} must be 1 or 0, not '{value}'"),
        Err(env::VarError::NotUnicode(value)) => {
            panic!("{WITH_COMMIT} must be 1 or 0, not {value:?}")
        }
    }
}

/// The output of `git <args>` in this crate's directory, or `None` when git is missing or fails.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The short hash of the checked-out commit, or `None` when the Vx sources are not a git
/// checkout. A source archive unpacked inside some other repository must not take its commit.
fn commit(vx_root: &Path) -> Option<String> {
    let top = git(&["rev-parse", "--show-toplevel"])?;
    if fs::canonicalize(top).ok()? != fs::canonicalize(vx_root).ok()? {
        return None;
    }
    git(&["rev-parse", "--short=8", "HEAD"])
}

/// The files that change when the checked-out commit changes. Each one exists: Cargo reruns a
/// build script on every build while a file it watches is missing.
fn files_to_watch() -> Vec<PathBuf> {
    let git_path = |name: &str| git(&["rev-parse", "--git-path", name]).map(PathBuf::from);
    // HEAD names the branch, or holds the commit itself when HEAD is detached. In a git
    // worktree it is not at .git/HEAD, so ask git where it is.
    let mut watch: Vec<PathBuf> = git_path("HEAD").into_iter().collect();
    if let Some(table) = git_path("reftable").filter(|dir| dir.is_dir()) {
        // A repository in the reftable format keeps every ref, HEAD too, in this directory.
        watch.push(table);
    } else if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        // A commit moves the branch, not HEAD. The branch is a file of its own, or a line in
        // packed-refs after `git gc`. The next commit then writes the file again, in the
        // nearest directory that exists now.
        if let Some(file) = git_path(&branch) {
            if file.is_file() {
                watch.push(file);
            } else {
                watch.extend(git_path("packed-refs"));
                watch.extend(
                    file.ancestors()
                        .skip(1)
                        .find(|dir| dir.is_dir())
                        .map(PathBuf::from),
                );
            }
        }
    }
    watch.retain(|path| path.exists());
    watch
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by Cargo"));
    let vx_root = manifest_dir
        .parent()
        .expect("this crate is a directory of the Vx sources");
    println!("cargo:rerun-if-env-changed={WITH_COMMIT}");
    let suffix = match with_commit().then(|| commit(vx_root)).flatten() {
        Some(commit) => {
            for path in files_to_watch() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
            format!(" ({commit})")
        }
        None => {
            println!("cargo:rerun-if-changed=build.rs");
            String::new()
        }
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("set by Cargo"));
    let source =
        format!("#[macro_export]\nmacro_rules! commit_suffix {{ () => {{ {suffix:?} }}; }}\n");
    fs::write(out_dir.join("commit_suffix.rs"), source).expect("OUT_DIR is writable");
}
