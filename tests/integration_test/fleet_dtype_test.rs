//===- fleet_dtype_test.rs - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// What the shipped machine files actually admit, per element type.
//
// The `dtypes:` lists in fleet/ are quoted from vendor documents and the table
// they produce is a result, not a convenience -- so it is pinned here rather than
// left to whoever next edits a machine file. Each row below is a hardware claim
// with a citation in fleet/README.md.
//
// The interesting column is int4. It goes from admitted on A100 to refused on
// H100: Ampere's Tensor Cores take INT4 and Hopper's do not, deprecated after
// Ampere and emulated over int8 MMA on sm_90. Support is not a version number
// that only grows, which is the reason the vocabulary is declared per part
// instead of derived from a generation.
//
// Only E6026 is asserted on. Compiling a bare GPU machine file raises unrelated
// diagnostics (no host is declared in it, so a staged transfer has an end nothing
// describes); those say nothing about the element type and are not this test's
// business.

use std::path::PathBuf;
use std::process::Command;

/// (machine file, element type, admitted?)
const MATRIX: &[(&str, &str, bool)] = &[
    // fp8 arrives with Hopper.
    ("a100-80", "f8e4m3", false),
    ("h100-sxm", "f8e4m3", true),
    ("b200", "f8e4m3", true),
    ("mi300x", "f8e4m3", true),
    // fp4 arrives with Blackwell, and only there.
    ("a100-80", "f4e2m1", false),
    ("h100-sxm", "f4e2m1", false),
    ("b200", "f4e2m1", true),
    ("mi300x", "f4e2m1", false),
    // int4 goes the other way: Ampere has it, Hopper and Blackwell do not.
    ("a100-80", "i4", true),
    ("h100-sxm", "i4", false),
    ("b200", "i4", false),
    // Metal has no double, so the Apple part is the one row with no f64.
    ("m4-uma", "f64", false),
    ("a100-80", "f64", true),
    ("h100-sxm", "f64", true),
    // Every part takes f32 and f16; a machine that refused those would be wrong
    // in a way the rows above could not distinguish from a working check.
    ("a100-80", "f32", true),
    ("h100-sxm", "f32", true),
    ("b200", "f32", true),
    ("mi300x", "f32", true),
    ("m4-uma", "f32", true),
    ("m4-uma", "f16", true),
];

#[test]
fn the_fleet_admits_exactly_the_element_types_its_sources_claim() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("vx-fleet-dtype-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create the scratch directory");

    let mut wrong = Vec::new();
    for (machine, elem, admitted) in MATRIX {
        let machine_path = root.join(format!("fleet/{machine}.vx"));
        assert!(machine_path.is_file(), "missing {}", machine_path.display());

        let src = dir.join(format!("{machine}_{elem}.vx"));
        std::fs::write(
            &src,
            format!(
                "fn main() -> i32 {{\n  \
                 let t = Tensor<{elem}, [8, 8]>::uninit();\n  \
                 let _s = transfer(t, Memory::HBM);\n  \
                 return 0;\n}}\n"
            ),
        )
        .expect("failed to write the probe program");

        let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
            .current_dir(&root)
            .args([
                src.to_str().unwrap(),
                "--machine",
                machine_path.to_str().unwrap(),
                "--action",
                "print-ast",
            ])
            .output()
            .expect("failed to run vxc");
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let refused = log.contains("E6026");
        if refused == *admitted {
            wrong.push(format!(
                "  {machine} {elem}: expected {}, got {}",
                if *admitted { "admitted" } else { "refused" },
                if refused { "refused" } else { "admitted" },
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        wrong.is_empty(),
        "the fleet's element-type admissions no longer match their sources:\n{}",
        wrong.join("\n")
    );
}
