//===- ane_attention_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Fused attention placed on the Apple Neural Engine.
//!
//! In `tests/flat/` and driven from here because `flash_attention_into` has no
//! AST lowering and `tests/backend/pass` is compiled with `--legacy-codegen`.
//!
//! macOS only. The program is valid anywhere, but on a machine with no Apple
//! dispatcher there is nothing to route to and the host fallback for a
//! 256x256x8192 attention is not something to put in a test suite.

use std::path::Path;
use std::process::Command;

#[test]
#[cfg(target_os = "macos")]
fn attention_on_the_neural_engine_computes_the_uniform_mean() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("tests/flat/ane_attention_f16.vx");
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
    // Uniform: q and k are zero, so the softmax is uniform over 256 keys and the
    // output row is the mean of v. One v row is 1.0, so the mean is 1/256, exact
    // in f16 -- a property of the arithmetic rather than a tolerance.
    assert!(
        stdout.contains("0.00390625"),
        "expected the uniform mean 1/256 = 0.00390625:\n{stdout}"
    );
    // Scored: a genuinely non-uniform softmax, exp(1)/(exp(1)+255). This is the
    // one that fails if the dispatcher drops Vx's scale -- CoreML's
    // scaled_dot_product_attention always divides by sqrt(head_dim) and takes no
    // scale argument, so q has to be pre-multiplied to cancel it. The uniform
    // case above cannot see that bug, because a uniform softmax does not depend
    // on the scale.
    assert!(
        stdout.contains("0.010543823"),
        "expected exp(1)/(exp(1)+255) = 0.010543823; a wrong value near 0.00395 \
         means the scale was dropped:\n{stdout}"
    );
}
