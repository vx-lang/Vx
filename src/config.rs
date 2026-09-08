//===- config.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Settings that describe *how* a compile runs, as opposed to what it compiles.
//
// This module has no imports on purpose. Both the orchestrator and the backend need to agree on
// these, so anything that lives here must be reachable from either without one of them having to
// depend on the other.
//
//===----------------------------------------------------------------------===//

/// How a compile iterates: with rayon, or without it at all.
///
/// [`Schedule::Sequential`] is **not** "rayon with one thread". It takes rayon off the path
/// entirely — plain `iter()` where the parallel form uses `par_iter()` — because otherwise the
/// ladder's 1-thread column is both the baseline *and* a rayon run, so whatever the parallel
/// machinery costs is charged to both sides and cancels out of every ratio. A speedup measured that
/// way answers "does more threads help this design", never "is this design faster than not doing it
/// at all", and only the second is a claim about compilers.
///
/// Everything else is identical: same phases, same order, same per-item work, same output. The only
/// difference is the iterator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Schedule {
    Parallel,
    Sequential,
}

impl Schedule {
    pub fn is_seq(self) -> bool {
        self == Schedule::Sequential
    }
}
