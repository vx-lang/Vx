//===- missing_solver_test.rs - Vx Compiler --------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// What the compiler does when there is no SMT solver. E6024, and the
// VX_ALLOW_UNVERIFIED downgrade to W1031.
//
// This is the fail-closed rule from Vx#374, and it is a claim about an absence:
// an obligation that could not be discharged must not be treated as one that
// was. It used to be. Every caller reached the solver through its own
// `Command::new("z3").spawn()`, and an `Err` -- including "no solver available"
// -- fell through a `if let Ok(Reject)` pattern, so a machine without z3
// certified every seam in silence. A developer without z3 saw a clean build and
// CI without z3 saw a green run; the only symptom was a diagnostic that should
// have fired and did not.
//
// A `.vx` fail test cannot pin that, because it runs in whatever environment the
// suite has and z3 is normally present -- the test would assert the diagnostic
// on a machine where the code path is never taken. So the environment is the
// fixture here: the compiler is run with a PATH that has no z3 on it, which is
// the situation being described.
//
// The z3-present control matters as much as the absence. Without it a suite that
// somehow never found z3 would report these as passing while proving nothing,
// which is the same failure mode the rule exists to prevent.

use std::path::PathBuf;
use std::process::Command;

/// A relaxed transfer edge carries a payload, which is a visibility obligation:
/// a consumer may read the buffer before the producer's write is visible. That
/// obligation is what needs a solver.
const RELAXED_SEAM: &str = "tests/frontend/fail/topology_declared_relaxed_seam.vx";

/// The *other* place an obligation is discharged, and a different code path: the
/// declaration check above runs over topology declarations, while this one runs per
/// transfer under `--verify-seams`. With z3 present this program is rejected (E6004),
/// which is what makes it a usable probe -- a machine that accepts it has stopped
/// checking.
const TRANSFER_SEAM: &str = "tests/frontend/fail/memory_relaxed_into_explicit.vx";

struct Run {
    stdout: String,
    ok: bool,
}

/// A directory with nothing in it, to be the whole of PATH when the point is that
/// there is no solver.
///
/// The first version of this used `/usr/bin:/bin`, on the reasoning that replacing
/// PATH outright does not depend on where z3 is installed. It depends on it
/// completely: that is exactly where a Linux `apt install z3` puts it, so on CI the
/// absence fixture had a solver in it and the runs it was meant to starve came back
/// with real verdicts. An empty directory cannot contain z3 on any platform.
fn solverless_path() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vx-no-solver-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create the solverless PATH directory");
    dir
}

/// The absence has to be real for anything below it to mean anything. A decided
/// verdict in a run that was supposed to have no solver is a broken fixture, and it
/// should say so rather than surfacing as a confusing assertion about E6024.
fn assert_no_solver_was_reachable(run: &Run, case: &str) {
    assert!(
        !run.stdout.contains("violates the boundary contract"),
        "{case}: the compiler reached a solver and decided the obligation, so this \
         run does not test the missing-solver path at all:\n{}",
        run.stdout
    );
}

/// Run the compiler over the relaxed-seam program. `solver` chooses whether z3
/// is reachable.
fn compile(solver: bool, allow_unverified: bool) -> Run {
    compile_file(RELAXED_SEAM, &[], solver, allow_unverified)
}

fn compile_file(src: &str, extra: &[&str], solver: bool, allow_unverified: bool) -> Run {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.current_dir(&root).args([src, "--action", "print-ast"]);
    cmd.args(extra);
    if solver {
        // The inherited PATH, which the suite's own environment provides.
        cmd.env("PATH", std::env::var("PATH").unwrap_or_default());
    } else {
        cmd.env("PATH", solverless_path());
    }
    if allow_unverified {
        cmd.env("VX_ALLOW_UNVERIFIED", "1");
    } else {
        cmd.env_remove("VX_ALLOW_UNVERIFIED");
    }
    let out = cmd.output().expect("failed to run vxc");
    Run {
        stdout: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        ok: out.status.success(),
    }
}

fn z3_on_path() -> bool {
    Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn an_undischarged_obligation_fails_the_build() {
    // Absent: refused, and the diagnostic names the tool and the escape hatch.
    let none = compile(false, false);
    assert_no_solver_was_reachable(&none, "declaration seam, no solver");
    assert!(
        none.stdout.contains("E6024"),
        "no solver must produce E6024, got:\n{}",
        none.stdout
    );
    assert!(
        !none.ok,
        "an undischarged obligation must fail the build, not warn:\n{}",
        none.stdout
    );
    assert!(
        none.stdout.contains("SMEM -> TMEM"),
        "the diagnostic must name the edge it could not verify:\n{}",
        none.stdout
    );
    assert!(
        none.stdout.contains("VX_ALLOW_UNVERIFIED"),
        "the diagnostic must name the way out of it:\n{}",
        none.stdout
    );

    // Absent, but accepted explicitly: compiles, and still says so every time.
    let allowed = compile(false, true);
    assert!(
        allowed.stdout.contains("W1031"),
        "VX_ALLOW_UNVERIFIED must downgrade to W1031, got:\n{}",
        allowed.stdout
    );
    assert!(
        !allowed.stdout.contains("E6024"),
        "the downgrade must replace the error, not accompany it:\n{}",
        allowed.stdout
    );
    assert!(
        allowed.ok,
        "the downgrade must let the build through:\n{}",
        allowed.stdout
    );

    // The control. Without it the two cases above would pass on a machine where
    // the solver is never found and prove nothing about the fail-closed rule.
    if !z3_on_path() {
        eprintln!("skipping the z3-present control: no z3 on PATH");
        return;
    }
    let present = compile(true, false);
    assert!(
        !present.stdout.contains("E6024"),
        "with a solver on PATH the obligation is discharged, so E6024 must not \
         appear -- if it does, the absence cases above are not testing absence:\n{}",
        present.stdout
    );
    assert!(
        present.stdout.contains("W1027"),
        "this edge does not preserve visibility, so a solver that ran must \
         report W1027 -- the verdict, not the missing-solver code:\n{}",
        present.stdout
    );
}

#[test]
fn a_transfer_site_obligation_also_fails_the_build() {
    // Vx#374 closed the fail-open at the declaration site and left it open here, at the
    // site `--verify-seams` actually drives. The symptom was exact: this program compiled
    // clean (exit 0) on a machine with no z3 and was rejected on one with it, so whether
    // an unsound program built depended on what happened to be installed. The old code
    // said so in a warning and carried on -- under E6004, a code meaning the contract was
    // shown to be violated, when nothing had been shown at all.
    let none = compile_file(TRANSFER_SEAM, &["--verify-seams"], false, false);
    assert_no_solver_was_reachable(&none, "transfer seam, no solver");
    assert!(
        none.stdout.contains("E6024"),
        "no solver must produce E6024 at a transfer seam too, got:\n{}",
        none.stdout
    );
    assert!(
        !none.ok,
        "an undischarged transfer obligation must fail the build, not warn:\n{}",
        none.stdout
    );
    assert!(
        none.stdout.contains("hop was NOT verified"),
        "the diagnostic must say the obligation was not verified, and name the hop:\n{}",
        none.stdout
    );

    let allowed = compile_file(TRANSFER_SEAM, &["--verify-seams"], false, true);
    assert!(
        allowed.stdout.contains("W1031") && !allowed.stdout.contains("E6024"),
        "VX_ALLOW_UNVERIFIED must downgrade to W1031 here as well:\n{}",
        allowed.stdout
    );
    assert!(
        allowed.ok,
        "the downgrade must let the build through:\n{}",
        allowed.stdout
    );

    // The control, and the whole point: with a solver the obligation is *decided*, and
    // this program is refused for the reason it should be -- E6004, the violated contract.
    if !z3_on_path() {
        eprintln!("skipping the z3-present control: no z3 on PATH");
        return;
    }
    let present = compile_file(TRANSFER_SEAM, &["--verify-seams"], true, false);
    assert!(
        !present.stdout.contains("E6024"),
        "with a solver the obligation is discharged, so E6024 must not appear:\n{}",
        present.stdout
    );
    assert!(
        present.stdout.contains("E6004") && !present.ok,
        "a decided obligation that fails must be reported as the violated contract:\n{}",
        present.stdout
    );
}
