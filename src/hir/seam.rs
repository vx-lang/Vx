//===- seam.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Per-seam local-completeness / soundness obligation, discharged in QF_BV via z3.
//
// A *seam* is a hop of a cross-device transfer: each `MemorySpace` edge produced
// by `arch::TransferCostGraph::transfer_path` is one seam. At each seam the
// compiler must show that the boundary contract still holds after the transfer.
// Following "Precision at the Boundary", we encode the reached abstract state as
// a bit-packed GID slice (per location: a 2-bit tag BOT/CONST/TOP + a value
// field), apply the seam's transfer function `F` (read off the lowering: a
// synchronizing transfer is the identity; a relaxed escape hatch sends published
// locations to TOP), and ask z3 whether the post-transfer state can violate the
// contract. `unsat` => the contract is guaranteed (ACCEPT); `sat` => a concrete
// counterexample, e.g. a stale read (REJECT).
//
// INTEGRATION (where to call this):
//   1. `pub mod seam;` in `src/hir/mod.rs`.
//   2. In the pass that lowers/checks a transfer (where
//      `TransferCostGraph::transfer_path(src, dst)` yields the hop list), for
//      each consecutive (`MemorySpace`) hop build:
//        - `AbsState` : the footprint of the buffer's contract, read from the
//          GID stream the SIMD pass already produced;
//        - `Transfer` : `Sync` if the lowering carries a release/acquire (system
//          scope) or a DMA completion wait, else `Relaxed { published }`;
//        - `Contract` : from the kernel's `assert` / required alignment.
//      then `match seam::check_seam(&st, &t, &c) { Reject{..} => diagnostic, .. }`.
//   3. Time each `check_seam` and accumulate against total compile time (eval M1).
//   This mirrors `src/hir/prover.rs`'s z3 invocation; only the logic is QF_BV.
//
//===----------------------------------------------------------------------===//

use std::io::Write;
use std::process::{Command, Stdio};

/// Width of the value field of a GID cell, in bits. 64 so a value contract can pin any
/// `u64` payload; narrower widths silently downgraded larger constants to the coarse
/// visibility-only check.
const VAL_BITS: u32 = 64;

/// Constant-propagation lattice tag, encoded as 2 bits (BOT=00, CONST=01, TOP=11).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    Bot,
    Const,
    Top,
}

/// One location's abstract cell within a contract footprint. `value` is
/// meaningful only when `tag == Const`.
#[derive(Clone, Debug)]
pub struct Cell {
    pub tag: Tag,
    pub value: u64,
}

impl Cell {
    pub fn constant(v: u64) -> Self {
        Cell {
            tag: Tag::Const,
            value: v,
        }
    }
    pub fn top() -> Self {
        Cell {
            tag: Tag::Top,
            value: 0,
        }
    }
}

/// The reached abstract state restricted to a contract's footprint: named
/// locations and their cells (the relevant slice of the bit-packed GID stream).
#[derive(Clone, Debug)]
pub struct AbsState {
    pub cells: Vec<(String, Cell)>,
}

/// The seam's transfer function `F`, read off the lowering.
#[derive(Clone, Debug)]
pub enum Transfer {
    /// Synchronizing (release/acquire at system scope, or DMA + completion wait):
    /// identity on the footprint.
    Sync,
    /// Relaxed escape hatch: the `published` payload locations lose their
    /// visibility guarantee and become TOP (a stale read may return any value).
    /// The synchronization signal a consumer keys on (e.g. the flag) is *not*
    /// listed here — it stays observable; it is the data it was meant to protect
    /// that goes stale.
    Relaxed { published: Vec<String> },
}

/// A contract clause: `premise.0 == const premise.1`  implies
/// `conclusion.0 == const conclusion.1`. (E.g. flag==1 ⇒ data==42.)
#[derive(Clone, Debug)]
pub struct Contract {
    pub premise: (String, u64),
    pub conclusion: (String, u64),
}

/// The compiler's verdict at a seam.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Reject { counterexample: String },
}

/// Apply the seam's transfer function to the reached state.
fn apply_transfer(state: &AbsState, t: &Transfer) -> AbsState {
    let mut out = state.clone();
    if let Transfer::Relaxed { published } = t {
        for (name, cell) in out.cells.iter_mut() {
            if published.iter().any(|p| p == name) {
                *cell = Cell::top();
            }
        }
    }
    out
}

fn hex(v: u64) -> String {
    // VAL_BITS-wide hex literal, e.g. #x2a for 8 bits, #x000000000000002a for 64.
    let nybbles = (VAL_BITS as usize) / 4;
    // `1 << 64` overflows; a full-width field masks to all of u64.
    let mask = if VAL_BITS >= 64 {
        u64::MAX
    } else {
        (1u64 << VAL_BITS) - 1
    };
    format!("#x{:0width$x}", v & mask, width = nybbles)
}

/// One-time QF_BV preamble: model production, logic, and the lattice-tag definitions.
/// Installed once per persistent solver (or prepended to each one-shot script).
const PREAMBLE: &str = "(set-option :produce-models true)\n\
(set-logic QF_BV)\n\
(define-fun BOT () (_ BitVec 2) #b00)\n\
(define-fun CONST () (_ BitVec 2) #b01)\n\
(define-fun TOP () (_ BitVec 2) #b11)\n";

/// Emit the post-transfer abstract state (per-location declarations + constraints),
/// without the preamble. The goal assertion is appended by the caller.
fn state_smt(post: &AbsState) -> String {
    let mut s = String::new();
    for (name, cell) in &post.cells {
        s.push_str(&format!("(declare-const tag_{name} (_ BitVec 2))\n"));
        s.push_str(&format!(
            "(declare-const val_{name} (_ BitVec {VAL_BITS}))\n"
        ));
        match cell.tag {
            Tag::Const => {
                s.push_str(&format!("(assert (= tag_{name} CONST))\n"));
                s.push_str(&format!("(assert (= val_{name} {}))\n", hex(cell.value)));
            }
            // TOP: tag fixed, value left free so z3 can exhibit the stale read.
            Tag::Top => s.push_str(&format!("(assert (= tag_{name} TOP))\n")),
            Tag::Bot => s.push_str(&format!("(assert (= tag_{name} BOT))\n")),
        }
    }
    s
}

/// Value-contract goal: `premise-reached /\ ~conclusion`, then `check-sat` and the
/// model query. SAT means the contract is violable at this seam.
fn value_goal_smt(c: &Contract) -> String {
    let (pn, pv) = &c.premise;
    let (cn, cv) = &c.conclusion;
    let mut s = String::new();
    s.push_str(&format!(
        "(assert (and (= tag_{pn} CONST) (= val_{pn} {})))\n",
        hex(*pv)
    ));
    s.push_str(&format!(
        "(assert (not (and (= tag_{cn} CONST) (= val_{cn} {}))))\n",
        hex(*cv)
    ));
    s.push_str("(check-sat)\n");
    s.push_str(&format!("(get-value (val_{cn} tag_{cn}))\n"));
    s
}

/// Per-buffer *visibility* goal: can any consumed buffer be `TOP` (stale) after the
/// transfer? SAT means a consumer can read a stale buffer.
fn buffers_goal_smt(consumed: &[String]) -> String {
    let disj = consumed
        .iter()
        .map(|b| format!("(= tag_{b} TOP)"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut s = format!("(assert (or {disj}))\n(check-sat)\n");
    for b in consumed {
        s.push_str(&format!("(get-value (tag_{b} val_{b}))\n"));
    }
    s
}

/// Single-buffer *value* goal: after the transfer, can `buffer` fail to hold its
/// statically-known producer value `expected`? Stronger than the visibility goal:
/// it pins the value, not just definiteness. A synchronizing transfer keeps the
/// state's `val == expected` (goal `unsat` => ACCEPT); a relaxed transfer sends the
/// buffer to TOP, freeing its value, so z3 can exhibit a concrete stale value
/// `!= expected` (`sat` => REJECT with a value-level counterexample).
fn value1_goal_smt(buffer: &str, expected: u64) -> String {
    format!(
        "(assert (not (= val_{buffer} {})))\n(check-sat)\n(get-value (tag_{buffer} val_{buffer}))\n",
        hex(expected)
    )
}

/// Run a QF_BV script through z3 (mirrors `prover.rs`). Returns (sat?, model-text).
fn run_z3(script: &str) -> Result<(bool, String), String> {
    let mut child = match Command::new("z3")
        .arg("-in")
        .arg("-smt2")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No solver: fail open so compilation can proceed (as in prover.rs).
            return Ok((false, "z3 not found".into()));
        }
        Err(e) => return Err(format!("Failed to spawn z3: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let verdict = text
        .lines()
        .map(str::trim)
        .find(|l| *l == "sat" || *l == "unsat");
    match verdict {
        Some("sat") => {
            let model: String = text
                .lines()
                .filter(|l| l.trim_start().starts_with("((") || l.trim_start().starts_with("(val_"))
                .collect::<Vec<_>>()
                .join(" ");
            Ok((true, model))
        }
        Some("unsat") => Ok((false, String::new())),
        _ => Err(format!("unexpected z3 output: {text}")),
    }
}

/// Discharge the per-seam value-contract obligation: does the contract survive
/// the transfer? (E.g. the message-passing `flag == 1 => data == 42`.)
pub fn check_seam(reached: &AbsState, t: &Transfer, c: &Contract) -> Result<Verdict, String> {
    let post = apply_transfer(reached, t);
    let script = format!("{PREAMBLE}{}{}", state_smt(&post), value_goal_smt(c));
    let (violable, model) = run_z3(&script)?;
    if violable {
        Ok(Verdict::Reject {
            counterexample: model,
        })
    } else {
        Ok(Verdict::Accept)
    }
}

/// Discharge the per-seam *visibility* obligation for opaque buffers: can any
/// `consumed` buffer be read stale (TOP) after the transfer `t`? This is the
/// local-completeness obligation for the coarsest abstraction (definite vs ⊤),
/// instantiated from the actual buffer names crossing the seam — a sync transfer
/// keeps each `CONST`; a relaxed transfer sends published buffers to `TOP`.
pub fn check_seam_buffers(
    reached: &AbsState,
    t: &Transfer,
    consumed: &[String],
) -> Result<Verdict, String> {
    if consumed.is_empty() {
        return Ok(Verdict::Accept);
    }
    let post = apply_transfer(reached, t);
    let script = format!(
        "{PREAMBLE}{}{}",
        state_smt(&post),
        buffers_goal_smt(consumed)
    );
    let (violable, model) = run_z3(&script)?;
    if violable {
        Ok(Verdict::Reject {
            counterexample: model,
        })
    } else {
        Ok(Verdict::Accept)
    }
}

/// A persistent z3 process, reused across every seam in a compilation. The spawn and
/// preamble are paid **once** (`new`); each obligation is one `(push)`/`(check-sat)`/
/// `(pop)` round-trip (`check`/`check_buffers`), so the per-seam cost the compiler
/// reports is solving time, not process-startup time. If z3 is absent the solver is
/// `unavailable` and every obligation fails open (Accept), matching `prover.rs`.
pub struct Solver {
    pipe: Option<Pipe>,
}

struct Pipe {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    out: std::io::BufReader<std::process::ChildStdout>,
}

/// Echoed after each obligation so the reader knows the solver's reply is complete.
const SENTINEL: &str = "<<SEAM-DONE>>";

impl Solver {
    /// Spawn the persistent solver and install the one-time preamble. Never fails:
    /// if z3 cannot be started, the solver is unavailable and fails open.
    pub fn new() -> Self {
        Solver {
            pipe: Self::start().ok(),
        }
    }

    fn start() -> Result<Pipe, String> {
        use std::io::{BufReader, Write};
        use std::process::{Command, Stdio};
        let mut child = Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        let out = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        stdin
            .write_all(PREAMBLE.as_bytes())
            .map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;
        Ok(Pipe { child, stdin, out })
    }

    /// Run one obligation body (state + goal) inside a fresh assertion scope. Returns
    /// `(violable, model)`; an unavailable solver returns `(false, _)` (fail open).
    fn query(&mut self, body: &str) -> Result<(bool, String), String> {
        use std::io::{BufRead, Write};
        let pipe = match &mut self.pipe {
            Some(p) => p,
            None => return Ok((false, "z3 unavailable".into())),
        };
        let script = format!("(push 1)\n{body}(pop 1)\n(echo \"{SENTINEL}\")\n");
        pipe.stdin
            .write_all(script.as_bytes())
            .map_err(|e| e.to_string())?;
        pipe.stdin.flush().map_err(|e| e.to_string())?;

        let mut sat: Option<bool> = None;
        let mut model = String::new();
        loop {
            let mut line = String::new();
            let n = pipe.out.read_line(&mut line).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("z3 closed unexpectedly".into());
            }
            let t = line.trim();
            if t.trim_matches('"') == SENTINEL {
                break;
            }
            match t {
                "sat" => sat = Some(true),
                "unsat" => sat = Some(false),
                "" => {}
                _ if t.starts_with("(error") => {} // e.g. get-value after unsat
                _ => {
                    if !model.is_empty() {
                        model.push(' ');
                    }
                    model.push_str(t);
                }
            }
        }
        match sat {
            Some(v) => Ok((v, model)),
            None => Err("no sat/unsat from z3".into()),
        }
    }

    /// Persistent-solver counterpart of [`check_seam`].
    pub fn check_seam(
        &mut self,
        reached: &AbsState,
        t: &Transfer,
        c: &Contract,
    ) -> Result<Verdict, String> {
        let post = apply_transfer(reached, t);
        let (violable, model) =
            self.query(&format!("{}{}", state_smt(&post), value_goal_smt(c)))?;
        Ok(if violable {
            Verdict::Reject {
                counterexample: model,
            }
        } else {
            Verdict::Accept
        })
    }

    /// Persistent-solver counterpart of [`check_seam_buffers`].
    pub fn check_seam_buffers(
        &mut self,
        reached: &AbsState,
        t: &Transfer,
        consumed: &[String],
    ) -> Result<Verdict, String> {
        if consumed.is_empty() {
            return Ok(Verdict::Accept);
        }
        let post = apply_transfer(reached, t);
        let (violable, model) = self.query(&format!(
            "{}{}",
            state_smt(&post),
            buffers_goal_smt(consumed)
        ))?;
        Ok(if violable {
            Verdict::Reject {
                counterexample: model,
            }
        } else {
            Verdict::Accept
        })
    }

    /// Per-seam *value* obligation for a buffer whose producer value `expected` is
    /// statically known: can the buffer fail to hold `expected` after the transfer?
    /// Stronger than [`check_seam_buffers`] (pins the value, not just definiteness).
    pub fn check_seam_value(
        &mut self,
        reached: &AbsState,
        t: &Transfer,
        buffer: &str,
        expected: u64,
    ) -> Result<Verdict, String> {
        let post = apply_transfer(reached, t);
        let (violable, model) = self.query(&format!(
            "{}{}",
            state_smt(&post),
            value1_goal_smt(buffer, expected)
        ))?;
        Ok(if violable {
            Verdict::Reject {
                counterexample: model,
            }
        } else {
            Verdict::Accept
        })
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        if let Some(pipe) = &mut self.pipe {
            use std::io::Write;
            // Best-effort clean shutdown; ignore errors (process may already be gone).
            let _ = pipe.stdin.write_all(b"(exit)\n");
            let _ = pipe.stdin.flush();
            let _ = pipe.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reached state after C2: data = const 42, flag = const 1.
    fn mp_state() -> AbsState {
        AbsState {
            cells: vec![
                ("data".into(), Cell::constant(42)),
                ("flag".into(), Cell::constant(1)),
            ],
        }
    }
    fn mp_contract() -> Contract {
        Contract {
            premise: ("flag".into(), 1),
            conclusion: ("data".into(), 42),
        }
    }

    #[test]
    fn sync_transfer_accepts() {
        let v = check_seam(&mp_state(), &Transfer::Sync, &mp_contract()).unwrap();
        assert_eq!(v, Verdict::Accept);
    }

    #[test]
    fn relaxed_transfer_rejects_with_stale_read() {
        // The flag (synchronization signal) stays observed; the data payload it
        // was meant to protect goes stale (TOP).
        let t = Transfer::Relaxed {
            published: vec!["data".into()],
        };
        match check_seam(&mp_state(), &t, &mp_contract()).unwrap() {
            Verdict::Reject { counterexample } => {
                // z3 should exhibit the stale read (val_data not = 42, e.g. 0).
                assert!(
                    counterexample.contains("val_data"),
                    "model: {counterexample}"
                );
            }
            Verdict::Accept => panic!("relaxed transfer must be rejected"),
        }
    }

    // Per-buffer visibility: a definite buffer `x` crossing the seam.
    fn buffer_state() -> AbsState {
        AbsState {
            cells: vec![("x".into(), Cell::constant(7))],
        }
    }

    #[test]
    fn sync_transfer_keeps_buffer_visible() {
        let v = check_seam_buffers(&buffer_state(), &Transfer::Sync, &["x".into()]).unwrap();
        assert_eq!(v, Verdict::Accept);
    }

    #[test]
    fn relaxed_transfer_makes_buffer_stale() {
        let t = Transfer::Relaxed {
            published: vec!["x".into()],
        };
        match check_seam_buffers(&buffer_state(), &t, &["x".into()]).unwrap() {
            Verdict::Reject { counterexample } => {
                assert!(counterexample.contains("tag_x"), "model: {counterexample}");
            }
            Verdict::Accept => panic!("relaxed transfer must make buffer x stale"),
        }
    }

    #[test]
    fn no_consumed_buffers_accepts() {
        let v = check_seam_buffers(&buffer_state(), &Transfer::Sync, &[]).unwrap();
        assert_eq!(v, Verdict::Accept);
    }

    #[test]
    fn value_contract_sync_accepts_relaxed_rejects() {
        // Buffer x has a statically-known producer value (7). A synchronizing transfer
        // preserves it; a relaxed one frees it, so z3 exhibits a stale value != 7.
        let mut s = Solver::new();
        let sync = s
            .check_seam_value(&buffer_state(), &Transfer::Sync, "x", 7)
            .unwrap();
        assert_eq!(sync, Verdict::Accept);
        let t = Transfer::Relaxed {
            published: vec!["x".into()],
        };
        match s.check_seam_value(&buffer_state(), &t, "x", 7).unwrap() {
            Verdict::Reject { counterexample } => {
                assert!(counterexample.contains("val_x"), "model: {counterexample}");
            }
            Verdict::Accept => panic!("relaxed transfer must violate the value contract for x"),
        }
    }

    #[test]
    fn value_contract_holds_for_value_above_old_8bit_field() {
        // 70_000 does not fit the former 8-bit value field (> 255). Under the old width it
        // was masked (70_000 & 0xff == 0x70) or rejected outright, downgrading the seam to
        // the coarse visibility check. With VAL_BITS = 64 the exact payload is pinned.
        const BIG: u64 = 70_000;
        let state = AbsState {
            cells: vec![("x".into(), Cell::constant(BIG))],
        };
        let mut s = Solver::new();
        let sync = s
            .check_seam_value(&state, &Transfer::Sync, "x", BIG)
            .unwrap();
        assert_eq!(sync, Verdict::Accept);
        let t = Transfer::Relaxed {
            published: vec!["x".into()],
        };
        match s.check_seam_value(&state, &t, "x", BIG).unwrap() {
            Verdict::Reject { counterexample } => {
                assert!(counterexample.contains("val_x"), "model: {counterexample}");
            }
            Verdict::Accept => panic!("relaxed transfer must violate the value contract for x"),
        }
    }

    #[test]
    fn hex_is_full_width_64_bit() {
        // 16 nybbles, exact value preserved (the old 8-bit field would mask 70_000 -> #x70).
        assert_eq!(hex(0x2a), "#x000000000000002a");
        assert_eq!(hex(70_000), "#x0000000000011170");
        assert_eq!(hex(u64::MAX), "#xffffffffffffffff");
    }

    #[test]
    #[ignore = "M1 timing micro-benchmark; run with: cargo test --lib \
                seam::tests::bench_marginal_seam_cost -- --ignored --nocapture"]
    fn bench_marginal_seam_cost() {
        // Marginal per-seam solving cost on the persistent solver, after warmup, over
        // many independent obligations. This isolates the cost the compiler actually
        // pays per seam (one push/check-sat/pop round-trip) from process startup, and
        // shows it is flat in the number of seams (cost scales with seams, not program
        // size). Reports microseconds.
        let mut s = Solver::new();
        let st = buffer_state();
        let sync = Transfer::Sync;
        let consumed = ["x".to_string()];

        for _ in 0..100 {
            let _ = s.check_seam_buffers(&st, &sync, &consumed).unwrap();
        }
        let n = 2000usize;
        let mut us = Vec::with_capacity(n);
        for _ in 0..n {
            let t = std::time::Instant::now();
            let _ = s.check_seam_buffers(&st, &sync, &consumed).unwrap();
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = us[0];
        let median = us[n / 2];
        let mean = us.iter().sum::<f64>() / n as f64;
        let p99 = us[(n as f64 * 0.99) as usize];
        println!(
            "[bench] marginal per-seam over {n} obligations (warm, persistent solver): \
             min={min:.1}us median={median:.1}us mean={mean:.1}us p99={p99:.1}us"
        );
    }

    #[test]
    fn persistent_solver_reuses_one_process() {
        // A single z3 process discharges several obligations via push/pop, returning
        // the same verdicts as the one-shot path.
        let mut s = Solver::new();
        let relaxed = Transfer::Relaxed {
            published: vec!["x".into()],
        };

        assert_eq!(
            s.check_seam_buffers(&buffer_state(), &Transfer::Sync, &["x".into()])
                .unwrap(),
            Verdict::Accept
        );
        match s
            .check_seam_buffers(&buffer_state(), &relaxed, &["x".into()])
            .unwrap()
        {
            Verdict::Reject { counterexample } => {
                assert!(counterexample.contains("tag_x"), "model: {counterexample}");
            }
            Verdict::Accept => panic!("relaxed must reject on the persistent solver"),
        }
        // A value-contract obligation on the very same process.
        assert_eq!(
            s.check_seam(&mp_state(), &Transfer::Sync, &mp_contract())
                .unwrap(),
            Verdict::Accept
        );
    }
}
