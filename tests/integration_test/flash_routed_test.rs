//===- flash_routed_test.rs - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Executes `flash_attention_into` end to end and checks its closed form.
//!
//! The call is deliberately flat-only (Vx#378), which is why this cannot ride
//! the backend harness: that runner drives the legacy AST codegen in-process,
//! and the one thing it would test is the panic saying the lowering does not
//! exist. Shelling out to `vxc --action run-jit` runs the default (flat) path
//! -- the same binary and route a user takes.
//!
//! On a machine with no accelerator this executes the serial fallback nest on
//! the host, which is the point: the region must answer identically whether
//! the runtime routes it to a vendor flash kernel, runs the nest's own device
//! image, or lands here. The value asserted is deterministic across libms --
//! uniform q and k make every softmax weight exp(0) = 1 exactly -- and was
//! matched bit-for-bit against the routed FlashAttention-2 path on an A100
//! (tests/flat/flash_attention_routed.vx documents all three runs).
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

#[test]
fn flash_attention_into_computes_the_closed_form() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("tests/flat/flash_attention_routed.vx");
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&src)
        .args(["--action", "run-jit"])
        .current_dir(root)
        .output()
        .expect("vxc should execute");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "vxc --action run-jit failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("0.0750122"),
        "the closed form (mean of V's f16 rows, 0.0750122) is missing:\n{stdout}"
    );
}
