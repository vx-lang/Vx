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
    // Varied: every row and several head positions distinct, so a row-indexing
    // or transposition error shows. It has no closed form, so the reference is
    // the host fallback -- 0.2064209 for o[255][7], obtained by running the same
    // program with the CoreML model moved aside. The tolerance is real rather
    // than cosmetic: the Neural Engine accumulates in f16 and the host does not,
    // so the two agree to about 1e-4 and not to the bit.
    // The program's own output is the last non-empty line; stdout also carries
    // the JIT's progress messages.
    let varied: f64 = stdout
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .and_then(|l| l.split_whitespace().nth(2))
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("expected a third value in:\n{stdout}"));
    assert!(
        (varied - 0.2064209).abs() < 1e-3,
        "varied attention drifted from the host fallback's 0.2064209: {varied}"
    );
}
