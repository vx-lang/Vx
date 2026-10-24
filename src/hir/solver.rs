//===- solver.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// There is no separate "is z3 installed" probe here, and there was one. Every caller reaches this
// module from the `ErrorKind::NotFound` arm of its OWN `Command::new("z3").spawn()`, so by the
// time it asks, the answer is already established and the OS's own error is in hand -- a second
// `z3 --version` spawn could only confirm it, less accurately. The probe's one virtue was that it
// ran once per process, and buying that needed a cached global, which is the thing this compiler
// does not have (Vx#381).
//
//===----------------------------------------------------------------------===//

/// The environment variable that permits compiling with obligations left undischarged.
pub const ALLOW_UNVERIFIED: &str = "VX_ALLOW_UNVERIFIED";

/// Whether the user has asked to compile with obligations left undischarged.
///
/// Read straight from the environment, with nothing cached. There is nothing to cache: this is
/// reached only after a `z3` spawn has already failed, or once per relaxed transfer edge, so it is
/// never on a hot path -- and the compiler admits no locking primitive and no process-global
/// atomic anywhere in `src/`, which CI enforces by name
/// (docs/parallel_compiler_architecture.md 2.7).
///
/// An earlier version memoized this, and a probe beside it, through a once-cell. Both were
/// removed: the memo bought nothing, and it turned a read of the environment into shared mutable
/// state that every worker thread reached into.
pub fn unverified_allowed() -> bool {
    std::env::var(ALLOW_UNVERIFIED)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// The message shown when a `relaxed` transfer obligation cannot be discharged because no
/// solver is available.
///
/// Names the tool, why it is needed, and the escape hatch. A diagnostic that says only
/// "z3 not found" leaves the reader to guess whether it mattered. Contracts use
/// [`contracts_missing_message`] instead: they do not read `VX_ALLOW_UNVERIFIED` (Vx#2083).
pub fn missing_message(reason: &str) -> String {
    format!(
        "no SMT solver available, so this proof obligation could not be discharged ({reason}). \
         Vx runs `z3` to verify contracts and `relaxed` transfer visibility; install it \
         (macOS: `brew install z3`, Ubuntu: `apt install z3`) or set {ALLOW_UNVERIFIED}=1 to \
         compile with obligations left unverified"
    )
}

/// The message shown when a contract (`requires` / `ensures`) cannot be proved because no
/// solver is available.
///
/// Same shape as [`missing_message`], but without the `VX_ALLOW_UNVERIFIED` advice: that
/// variable only downgrades `relaxed` transfer checks, and naming it here sent readers to an
/// escape hatch that does nothing for contracts (Vx#2083).
pub fn contracts_missing_message(reason: &str) -> String {
    format!(
        "no SMT solver available, so this contract could not be discharged ({reason}). \
         Vx runs `z3` to verify contracts and `relaxed` transfer visibility; install it \
         (macOS: `brew install z3`, Ubuntu: `apt install z3`)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message has to name the tool and the escape hatch. A diagnostic that says only
    /// "z3 not found" is how this went unnoticed in the first place.
    #[test]
    fn the_missing_message_names_the_tool_and_the_way_out() {
        let m = missing_message("No such file or directory");
        assert!(m.contains("z3"), "must name the tool: {m}");
        assert!(m.contains(ALLOW_UNVERIFIED), "must name the opt-out: {m}");
        assert!(
            m.contains("contracts") && m.contains("relaxed"),
            "must name the checks z3 verifies: {m}"
        );
        assert!(
            m.contains("brew install z3"),
            "must say how to install: {m}"
        );
        assert!(
            m.contains("No such file or directory"),
            "must carry the underlying reason: {m}"
        );
    }

    /// Contracts must not advertise `VX_ALLOW_UNVERIFIED`: that variable does not apply to them.
    #[test]
    fn the_contracts_message_names_the_tool_but_not_the_transfer_opt_out() {
        let m = contracts_missing_message("No such file or directory");
        assert!(m.contains("z3"), "must name the tool: {m}");
        assert!(
            m.contains("contracts") && m.contains("relaxed"),
            "must name the checks z3 verifies: {m}"
        );
        assert!(
            !m.contains(ALLOW_UNVERIFIED),
            "must not name the transfer-only opt-out: {m}"
        );
        assert!(
            m.contains("brew install z3"),
            "must say how to install: {m}"
        );
        assert!(
            m.contains("No such file or directory"),
            "must carry the underlying reason: {m}"
        );
    }

    /// The opt-out is read from the environment every time, with nothing cached between calls.
    ///
    /// A cached answer is what this module used to have, and caching it is what put a once-cell
    /// -- shared mutable state reachable from every worker thread -- on a path that only ever
    /// needed an environment read.
    #[test]
    fn the_opt_out_is_read_from_the_environment_each_time() {
        let before = unverified_allowed();
        assert_eq!(
            before,
            unverified_allowed(),
            "two reads with nothing changed in between must agree"
        );
        assert_eq!(
            unverified_allowed(),
            std::env::var(ALLOW_UNVERIFIED)
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            "the answer must be the environment, not a remembered copy of it"
        );
    }
}
