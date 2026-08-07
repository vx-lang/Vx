//===- intern_mode.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// How a generic instantiation gets its identity, plus the two measurement hooks the parallel
// frontend needs to be studied at all: per-phase timing (#297) and a gate on the pipeline's
// progress chatter (#306).
//
// Everything here is lock-free by construction. The frontend's design rule is that no lock
// primitive appears on the compilation path, and CI greps `src/` to enforce it -- so the phase
// timer is a fixed array of atomics rather than the obvious lock-guarded vector, and the quiet
// gate is a tri-state atomic rather than a one-shot lazy cell. A measurement hook that violated the
// property it exists to measure would be self-defeating.
//
// (The lint greps text, not code, so it also matches prose -- see #272. Naming those primitives in
// a comment here would fail CI, which is why this paragraph describes them instead.)
//
//===----------------------------------------------------------------------===//

use crate::gid::TypeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// How a generic instantiation's identity is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternMode {
    /// The default: mint a GID carrying a *local* arena index plus `LOCAL_DEFERRED_BIT`, then
    /// reconcile local arenas into the global one at a barrier (`deduplication_phase`) and rewrite
    /// the indices (`simd_patch_phase`).
    Deferred,
    /// Content-addressed: word 2 carries a digest of the argument GIDs, computed locally. Identity
    /// is final at mint time, so there is no arena entry, no deferred bit, nothing to reconcile and
    /// nothing to patch -- the barrier is *absent* rather than cheap.
    ///
    /// Opt-in (`--intern-mode=content`). Determinism becomes structural rather than earned by a
    /// canonical-order walk at a barrier, so it holds across any scheduling, partitioning, worker
    /// count, or process boundary -- which a compilation-local arena index cannot. See #307.
    Content,
}

static MODE: AtomicU8 = AtomicU8::new(0);

pub fn set_mode(mode: InternMode) {
    MODE.store(
        match mode {
            InternMode::Deferred => 0,
            InternMode::Content => 2,
        },
        Ordering::SeqCst,
    );
}

pub fn mode() -> InternMode {
    match MODE.load(Ordering::Relaxed) {
        2 => InternMode::Content,
        _ => InternMode::Deferred,
    }
}

pub fn is_content() -> bool {
    mode() == InternMode::Content
}

/// Rewrite a type stream into a canonical form so two interning strategies can be compared for
/// *semantic* equality despite assigning different arena indices.
///
/// Each generic-instantiation GID's word 2 is replaced by its rank in first-appearance order
/// within the stream, and the deferred bit is cleared. Two streams are equivalent iff their
/// canonical forms are equal: same GIDs in the same order, with instantiations referring to the
/// same argument lists under a consistent renaming.
///
/// `resolve` maps an arena index to its argument list in whichever arena the mode used.
pub fn canonicalise_stream<F>(stream: &[TypeId], resolve: F) -> Vec<TypeId>
where
    F: Fn(u64) -> Option<Vec<TypeId>>,
{
    use crate::gid::{Word2, Word2Arena, Word2Scope};
    // Both modes rank into one space: `deferred` resolves an arena index to its argument list,
    // `content` has no arena and ranks the digest itself via a synthetic key. Either way the
    // equivalence classes are the same -- which positions in the stream denote the same
    // instantiation -- which is what equality of programs actually means here.
    let mut rank: HashMap<Vec<TypeId>, u64> = HashMap::new();
    let mut out = Vec::with_capacity(stream.len());
    for id in stream {
        let mut c = *id;
        // Decode through `classify_word2`, the single word-2 decoder, rather than masking the
        // words here -- a second decoder is how the two index spaces re-diverge.
        let key = match id.classify_word2() {
            Word2::Index {
                index,
                arena: Word2Arena::Generics,
                ..
            } => resolve(index),
            // A digest is already a function of the argument list, so it is its own key. Wrapped in
            // a distinctive GID rather than used raw so it cannot alias a real argument vector.
            Word2::GenericDigest(d) => Some(vec![TypeId::new(u64::MAX, d, 0, 0)]),
            _ => None,
        };
        if let Some(k) = key {
            let next = rank.len() as u64;
            let r = *rank.entry(k).or_insert(next);
            // Rewrite to the canonical rank, always as a `Global` index: the deferred bit and the
            // digest-vs-index choice record *how* identity was assigned, not what the program
            // means, so neither may participate in an equivalence comparison.
            c.set_arena_index(r, Word2Arena::Generics, Word2Scope::Global);
        }
        out.push(c);
    }
    out
}

// ---- Progress chatter (#306) ------------------------------------------------------------------

/// Suppress the pipeline's progress output, via `VX_PIPELINE_QUIET=1`.
///
/// `parse_phase` opens its per-module closure with a `println!`, *inside* a rayon parallel-for.
/// `println!` takes the global stdout mutex, so every worker serialises on it once per module. For
/// a measurement that is fatal twice over: it puts a lock in the middle of the region whose scaling
/// is being measured, and a lock profile then reports contention on stdout rather than on anything
/// in the compiler. At a 512-module corpus it is 512 lock acquisitions per run.
///
/// Tri-state atomic rather than a one-shot lazy cell: 0 = not yet read, 1 = quiet, 2 = loud. A benign
/// race just re-reads the environment and stores the same answer.
pub fn quiet() -> bool {
    static QUIET: AtomicU8 = AtomicU8::new(0);
    match QUIET.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let q = std::env::var("VX_PIPELINE_QUIET").is_ok_and(|v| v != "0");
            QUIET.store(if q { 1 } else { 2 }, Ordering::Relaxed);
            q
        }
    }
}

// ---- Per-phase timing (#297) ------------------------------------------------------------------

/// The pipeline's phases, in the order they run. Fixed at compile time so the timer can be a plain
/// array of atomics indexed by position -- no map, no allocation, no lock.
/// `codegen` is split into its serial prologue and its parallel emit because a phase that is 59% of
/// a compile deserves to be attributed at finer grain than "codegen". The prologue builds the emit
/// context from the frozen registry and then walks every function's signature -- work proportional
/// to the whole program, on the critical path, inside what the phase table otherwise presents as a
/// parallel-for. Its cost is included in `codegen`, so the three do not sum independently.
pub const PHASES: [&str; 15] = [
    "parse",
    "macro_expand",
    "name_resolution",
    "registry_freeze",
    "sig_clone",
    "env_build",
    "return_prov",
    "type_check",
    "dedup_barrier",
    "stream_extract",
    "simd_patch",
    "codegen",
    "  codegen:setup",
    "  codegen:emit",
    "teardown",
];

static PHASE_NANOS: [AtomicU64; PHASES.len()] = [const { AtomicU64::new(0) }; PHASES.len()];

fn phase_index(name: &str) -> Option<usize> {
    PHASES.iter().position(|p| *p == name)
}

/// Time `f`, accumulating it under `name`.
///
/// Phases inside the parallel region record wall time for the whole region, not per-thread CPU
/// time: the question these numbers answer is how much of the *critical path* each phase owns.
/// An unknown name is timed into nothing rather than panicking -- a typo in an instrumentation
/// call should not take down a compile.
pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let out = f();
    record(name, t.elapsed());
    out
}

/// Accumulate an already-measured duration under `name`, for a region that does not fit a closure
/// — a stretch between two points inside a function whose value is produced later.
pub fn record(name: &str, elapsed: std::time::Duration) {
    if let Some(i) = phase_index(name) {
        PHASE_NANOS[i].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Drain the accumulated per-phase times, in pipeline order. Resets the counters, so successive
/// calls report successive runs -- which is what a repetition-based harness wants.
pub fn take_phases() -> Vec<(&'static str, std::time::Duration)> {
    PHASES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let nanos = PHASE_NANOS[i].swap(0, Ordering::Relaxed);
            (*name, std::time::Duration::from_nanos(nanos))
        })
        .collect()
}

/// One machine-readable line per phase: `phase,<name>,<milliseconds>`. Written to stdout so a
/// harness can parse a run without scraping the human report (#297).
pub fn phases_csv() -> String {
    take_phases()
        .into_iter()
        .map(|(n, d)| format!("phase,{n},{:.4}\n", d.as_secs_f64() * 1e3))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `timed` passes the closure's value through, and only names in [`PHASES`] are recorded --
    /// a typo in an instrumentation call must not panic a compile, but it must also not be
    /// silently attributed to the wrong phase.
    ///
    /// Deliberately asserts nothing about the accumulated counters: they are process-global, and
    /// `cargo test` runs tests in parallel, so any assertion on their value races every other test
    /// that compiles anything.
    #[test]
    fn timed_passes_values_through_and_only_knows_declared_phases() {
        assert_eq!(timed("parse", || 41 + 1), 42);
        assert_eq!(timed("not_a_phase", || "ok"), "ok");
        assert_eq!(phase_index("type_check"), Some(7));
        assert_eq!(phase_index("not_a_phase"), None);
        // Every name the pipeline instruments must be declared, or its time vanishes.
        assert_eq!(PHASES.len(), 15);
    }
}
