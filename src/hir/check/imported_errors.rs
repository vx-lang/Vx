//===- imported_errors.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type errors in the functions of an imported module.
//
// The program is compiled together with the modules it imports, and their functions are
// type-checked too. A module outside the standard library reports every error, as a Rust crate
// does: a package that `--package` names, or a file of the program. In a module of the standard
// library, an error is reported only if the program uses that function. Such a function with an
// error that nothing uses is not compiled either, because the code generator cannot compile it.
//
//===----------------------------------------------------------------------===//
use crate::diagnostic::{Diagnostic, DiagnosticLevel};
use crate::hir::check::capacity_fold::FnCapacitySummary;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Which of the functions in `with_errors` the program uses.
///
/// Every function without an error is compiled, so each of them counts as used, and so does
/// everything it calls. A function with an error is used if a used function calls it. The calls
/// come from what the type checker recorded for each function it checked.
pub fn used_functions_with_errors<'a>(
    checked: impl IntoIterator<Item = &'a FnCapacitySummary>,
    with_errors: &HashSet<String>,
) -> BTreeSet<String> {
    // What each function with an error calls, and a list of calls still to follow.
    let mut calls_from_error_functions: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut to_visit: Vec<&str> = Vec::new();
    for function in checked {
        let calls = function.calls.iter().map(|c| c.callee.as_str());
        if with_errors.contains(&function.name) {
            calls_from_error_functions
                .entry(function.name.as_str())
                .or_default()
                .extend(calls);
        } else {
            to_visit.extend(calls);
        }
    }
    let mut used = BTreeSet::new();
    while let Some(name) = to_visit.pop() {
        if with_errors.contains(name) && used.insert(name.to_string()) {
            if let Some(calls) = calls_from_error_functions.get(name) {
                to_visit.extend(calls.iter().copied());
            }
        }
    }
    used
}

/// The errors found in an imported function, with the function and module named in each, since
/// the line numbers refer to the module's file, not the program's. `used` says whether the
/// program uses the function, and the message says so too.
pub fn errors_to_report(
    found: Vec<Diagnostic>,
    module: &str,
    function: &str,
    used: bool,
) -> Vec<Diagnostic> {
    let used = if used {
        " and used by this program"
    } else {
        ""
    };
    found
        .into_iter()
        .filter(|d| d.level == DiagnosticLevel::Error)
        .map(|mut d| {
            d.message = Cow::Owned(format!(
                "in `{function}`, imported from `{module}`{used}: {}",
                d.message
            ));
            d
        })
        .collect()
}
