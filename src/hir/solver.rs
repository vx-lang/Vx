//===- solver.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Whether an SMT solver is available, and what happens when it is not (Vx#374).
//
// The compiler discharges proof obligations by running `z3`: `comptime` checks in `prover.rs`,
// and `relaxed` transfer-edge visibility in `seam.rs`. Both used to answer "the solver could not
// be started" with the verdict that means *proved*, and carry on. A machine without z3 therefore
// compiled programs whose obligations were never checked, emitted nothing, and exited 0.
//
// That failure is invisible exactly where it is most likely. A developer without z3 sees a clean
// build; CI without z3 sees a green run; and the most probable way to end up with a Vx build and
// no z3 is a prebuilt binary on a machine that never installed one (Vx#368). The only symptom is
// a diagnostic that should have fired and did not, which reads as a compiler regression.
//
// So the answer is fail *closed*: an obligation that cannot be discharged is reported. Compiling
// without verification stays possible, because there are legitimate reasons to want it, but it
// has to be asked for and it is never silent.
//
//===----------------------------------------------------------------------===//

use std::sync::OnceLock;

/// The environment variable that permits compiling with obligations left undischarged.
pub const ALLOW_UNVERIFIED: &str = "VX_ALLOW_UNVERIFIED";

/// What probing for the solver found. Resolved once per process: the answer cannot change
/// mid-compilation, and every obligation would otherwise pay for a spawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// `z3 --version` ran; the string is its reported version, recorded so an artifact can say
    /// which solver discharged its obligations (Vx#367).
    Present { version: String },
    /// The solver could not be run, with the reason as the OS reported it.
    Missing { reason: String },
}

impl Availability {
    pub fn is_present(&self) -> bool {
        matches!(self, Availability::Present { .. })
    }

    /// The version string, for provenance. `None` when the solver is absent.
    pub fn version(&self) -> Option<&str> {
        match self {
            Availability::Present { version } => Some(version),
            Availability::Missing { .. } => None,
        }
    }
}

/// Probe for the solver, once per process.
pub fn availability() -> &'static Availability {
    static AVAIL: OnceLock<Availability> = OnceLock::new();
    AVAIL.get_or_init(probe)
}

fn probe() -> Availability {
    match std::process::Command::new("z3").arg("--version").output() {
        Ok(out) if out.status.success() => Availability::Present {
            version: String::from_utf8_lossy(&out.stdout).trim().to_string(),
        },
        Ok(out) => Availability::Missing {
            reason: format!("`z3 --version` exited with {}", out.status),
        },
        Err(e) => Availability::Missing {
            reason: e.to_string(),
        },
    }
}

/// Whether the user has asked to compile with obligations left undischarged.
///
/// Read through a `OnceLock` so a mid-compilation change to the environment cannot make one
/// obligation strict and the next lax.
pub fn unverified_allowed() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        std::env::var(ALLOW_UNVERIFIED)
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// The message shown when an obligation cannot be discharged because no solver is available.
///
/// Names the tool, why it is needed, and the escape hatch — a diagnostic that says only
/// "z3 not found" leaves the reader to guess whether it mattered.
pub fn missing_message(reason: &str) -> String {
    format!(
        "no SMT solver available, so this proof obligation could not be discharged ({reason}). \
         Vx runs `z3` to verify comptime checks and `relaxed` transfer visibility; install it \
         (macOS: `brew install z3`, Ubuntu: `apt install z3`) or set {ALLOW_UNVERIFIED}=1 to \
         compile with obligations left unverified"
    )
}

/// `Err(message)` when an obligation needs the solver, none is available, and the user has not
/// opted out. `Ok(())` when the solver is present, or when the opt-out is set — in which case
/// the caller reports the obligation as unverified rather than as proved.
pub fn require() -> Result<(), String> {
    match availability() {
        Availability::Present { .. } => Ok(()),
        Availability::Missing { reason } => {
            if unverified_allowed() {
                Ok(())
            } else {
                Err(missing_message(reason))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe answers, and answers the same way twice. Which answer is correct depends on the
    /// machine, so this asserts the shape rather than the verdict.
    #[test]
    fn availability_is_resolved_once_and_is_stable() {
        let a = availability();
        let b = availability();
        assert_eq!(a, b, "availability must not change within a process");
        match a {
            Availability::Present { version } => {
                assert!(
                    !version.is_empty(),
                    "a present solver must report a version"
                );
                assert!(a.is_present());
                assert_eq!(a.version(), Some(version.as_str()));
            }
            Availability::Missing { reason } => {
                assert!(!reason.is_empty(), "an absent solver must say why");
                assert!(!a.is_present());
                assert_eq!(a.version(), None);
            }
        }
    }

    /// The message has to name the tool and the escape hatch. A diagnostic that says only
    /// "z3 not found" is how this went unnoticed in the first place.
    #[test]
    fn the_missing_message_names_the_tool_and_the_way_out() {
        let m = missing_message("No such file or directory");
        assert!(m.contains("z3"), "must name the tool: {m}");
        assert!(m.contains(ALLOW_UNVERIFIED), "must name the opt-out: {m}");
        assert!(
            m.contains("brew install z3"),
            "must say how to install: {m}"
        );
        assert!(
            m.contains("No such file or directory"),
            "must carry the underlying reason: {m}"
        );
    }

    /// `require()` never errors when the solver is present, and never errors under the opt-out.
    /// The interesting case -- absent and not opted out -- cannot be forced here without
    /// unsetting PATH for the whole process, so it is covered by the integration test.
    #[test]
    fn require_agrees_with_availability() {
        if availability().is_present() {
            assert!(require().is_ok(), "a present solver must satisfy require()");
        } else if unverified_allowed() {
            assert!(require().is_ok(), "the opt-out must satisfy require()");
        } else {
            assert!(require().is_err(), "an absent solver must fail require()");
        }
    }
}
