//===- objects_need_the_macos_of_the_host.rs - Vx Compiler -------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An object that `vxc --action emit-obj` writes on a Mac names the oldest macOS that can
// run it. LLVM took that version from the kernel's and gave macOS 28 on macOS 27, so
// each link warned that the object was built for a newer macOS. The object must name
// `MACOSX_DEPLOYMENT_TARGET`, as clang does, or else the major and minor version of the
// running macOS.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

/// The `minos` of the macOS `LC_BUILD_VERSION` in a 64-bit Mach-O object, as `major.minor.patch`.
fn minimum_macos(object: &Path) -> String {
    let bytes = std::fs::read(object).unwrap();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert_eq!(word(0), 0xfeed_facf, "not a 64-bit Mach-O object");
    let mut at = 32;
    for _ in 0..word(16) {
        // LC_BUILD_VERSION, for platform 1, macOS.
        if word(at) == 0x32 && word(at + 8) == 1 {
            let v = word(at + 12);
            return format!("{}.{}.{}", v >> 16, (v >> 8) & 0xff, v & 0xff);
        }
        at += word(at + 4) as usize;
    }
    panic!("the object has no LC_BUILD_VERSION for macOS");
}

fn emit(dir: &Path, deployment_target: Option<&str>) -> String {
    let source = dir.join("add.vx");
    let object = dir.join("add.o");
    std::fs::write(
        &source,
        "pub fn add(a : i32, b : i32) -> i32 {\n  return a + b;\n}\n",
    )
    .unwrap();
    let mut vxc = Command::new(env!("CARGO_BIN_EXE_vxc"));
    vxc.arg(&source)
        .args(["--action", "emit-obj", "-o"])
        .arg(&object);
    match deployment_target {
        Some(v) => vxc.env("MACOSX_DEPLOYMENT_TARGET", v),
        None => vxc.env_remove("MACOSX_DEPLOYMENT_TARGET"),
    };
    let out = vxc.output().expect("vxc did not start");
    assert!(
        out.status.success(),
        "vxc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    minimum_macos(&object)
}

#[test]
fn an_object_needs_the_macos_that_runs_the_compiler() {
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .expect("sw_vers did not start");
    let host = String::from_utf8_lossy(&out.stdout);
    let mut parts = host.trim().split('.');
    let major = parts.next().unwrap();
    let minor = parts.next().unwrap_or("0");
    assert_eq!(emit(dir.path(), None), format!("{major}.{minor}.0"));
}

#[test]
fn an_object_needs_the_macos_that_the_deployment_target_names() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(emit(dir.path(), Some("14.2")), "14.2.0");
}
