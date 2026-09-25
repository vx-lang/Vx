//===- imported_flaws.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An imported function whose body does not check. Its errors are not this compile's to report
// while nothing the compile emits calls it, and it is then left out of code generation, which
// cannot compile it. Once anything emitted calls it, its errors are reported.
//
//===----------------------------------------------------------------------===//
use crate::diagnostic::{Diagnostic, DiagnosticLevel};
use crate::hir::check::capacity_fold::FnCapacitySummary;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

/// The flawed functions something emitted calls, directly or through another flawed one it
/// calls. Every function checked without errors is emitted, so each is a starting point.
pub fn reached<'a>(
    summaries: impl IntoIterator<Item = &'a FnCapacitySummary>,
    flawed: &HashSet<String>,
) -> BTreeSet<String> {
    let mut calls_of_flawed: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut work: Vec<&str> = Vec::new();
    for s in summaries {
        let calls = s.calls.iter().map(|c| c.callee.as_str());
        if flawed.contains(&s.name) {
            calls_of_flawed
                .entry(s.name.as_str())
                .or_default()
                .extend(calls);
        } else {
            work.extend(calls);
        }
    }
    let mut out = BTreeSet::new();
    while let Some(name) = work.pop() {
        if flawed.contains(name) && out.insert(name.to_string()) {
            if let Some(next) = calls_of_flawed.get(name) {
                work.extend(next.iter().copied());
            }
        }
    }
    out
}

/// The errors of an imported function, each saying where it is, since its line numbers are in
/// another file than the program's.
pub fn errors_of(found: Vec<Diagnostic>, module: &str, function: &str) -> Vec<Diagnostic> {
    found
        .into_iter()
        .filter(|d| d.level == DiagnosticLevel::Error)
        .map(|mut d| {
            d.message = Cow::Owned(format!(
                "in `{function}` of the imported module `{module}`, which the compiled code calls: {}",
                d.message
            ));
            d
        })
        .collect()
}
