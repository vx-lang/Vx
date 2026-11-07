//===- abi_of_the_dispatch_library.rs - the whole ABI ----------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 WITH LLVM Exceptions.
// See LICENSE for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! The compiler emits calls to `vx_plugin_*`, `vx_collective_*` and the two
//! worker calls without knowing which backend answers them, so a backend that
//! misses one makes every program that uses it fail to link.
//!
//! The CI job builds the CPU backend, which has all of them. The Intel GPU
//! library is built only where oneAPI is, and it was missing the collective and
//! arrival calls for a while without anything noticing -- a program that joins
//! a job or has code after a nest of blocks linked everywhere except on the
//! machines that library is for. This reads the dynamic symbols of whichever
//! library this build selected, so the machine that builds a backend is the
//! one that checks it.
//!
//! Linux only, because it reads an ELF symbol table.

use std::path::Path;
use std::process::Command;

/// The header the compiler includes to call the ABI, which is where the
/// `vx_plugin_*` half of the list comes from.
const ABI_HEADER: &str = include_str!("../../include/vx_hardware_runtime.h");

/// Every `vx_plugin_*` function the header declares, in the order it declares
/// them.
///
/// Read from the header rather than written out here: a new call is added to
/// the header first, and a list that has to be updated by hand is a list that
/// falls behind without anyone noticing.
fn plugin_calls_from_the_header() -> Vec<&'static str> {
    let mut found = Vec::new();
    for line in ABI_HEADER.lines() {
        let start = match line.find(|c: char| c.is_ascii_alphabetic() || c == '_') {
            Some(at) => at,
            None => continue,
        };
        let line = &line[start..];
        // A declaration, not a definition, a typedef or a call of one.
        if line.starts_with("static") || line.starts_with("typedef") {
            continue;
        }
        let at = match line.find("vx_plugin_") {
            Some(at) => at,
            None => continue,
        };
        let rest = &line[at..];
        let end = match rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            Some(end) => end,
            None => continue,
        };
        if rest[end..].starts_with('(') {
            found.push(&rest[..end]);
        }
    }
    found
}

/// The calls that have no declaring header, and so cannot be read off one.
///
/// `vx_cuda_set_stream` is defined by each backend rather than declared in the
/// public header, and the collective and worker calls live in headers whose
/// contents are not all part of this interface. Keeping those here is the
/// honest record; the plugin half needs no list.
const CALLS_WITHOUT_A_HEADER: &[&str] = &[
    "vx_collective_all_gather_bf16",
    "vx_collective_all_gather_f32",
    "vx_collective_all_reduce_sum_bf16",
    "vx_collective_all_reduce_sum_f32",
    "vx_collective_finish",
    "vx_collective_init",
    "vx_collective_rank",
    "vx_collective_ranks",
    "vx_cuda_get_stream",
    "vx_cuda_set_stream",
    "vx_host_worker_count",
    "vx_host_worker_id",
];

/// Every call a program can carry.
fn calls() -> Vec<&'static str> {
    let mut all = plugin_calls_from_the_header();
    all.extend_from_slice(CALLS_WITHOUT_A_HEADER);
    all.sort_unstable();
    all.dedup();
    all
}

#[test]
fn the_dispatch_library_exports_every_call_the_compiler_emits() -> Result<(), String> {
    let library = Path::new(env!("NPU_SHARED_LIB_PATH"));
    if !library.exists() {
        return Err(format!(
            "the build says it selected {}, and there is no such library",
            library.display()
        ));
    }

    let output = match Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(library)
        .output()
    {
        Ok(output) => output,
        // No `nm` on this machine, so there is no other way to read the
        // symbols. A test that cannot look must not report that it did.
        Err(e) => {
            return Err(format!(
                "cannot run nm ({e}), so {} was not checked",
                library.display()
            ))
        }
    };
    if !output.status.success() {
        return Err(format!(
            "nm could not read {}: {}",
            library.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let symbols = String::from_utf8_lossy(&output.stdout);
    let exported: Vec<&str> = symbols
        .lines()
        .filter_map(|line| line.split_whitespace().nth(2))
        .collect();
    let wanted = calls();
    let missing: Vec<&str> = wanted
        .iter()
        .copied()
        .filter(|call| !exported.contains(call))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{} does not export {} of the calls the compiler emits: {}.\n\
             A program that uses one of them fails to link on a machine that \
             builds this backend.",
            library.display(),
            missing.len(),
            missing.join(", ")
        ));
    }
    Ok(())
}

/// The list is built from the header, so a header that stops declaring a call
/// would shrink it silently and the check above would pass on less. This pins
/// the size, and names what it covers.
#[test]
fn the_list_covers_the_whole_interface() {
    let from_header = plugin_calls_from_the_header();
    assert_eq!(
        from_header.len(),
        11,
        "the header declares {} plugin calls, and this test was written \
         against 11: {from_header:?}",
        from_header.len()
    );
    assert!(
        from_header.contains(&"vx_plugin_arrivals_take")
            && from_header.contains(&"vx_plugin_arrivals_give"),
        "the two calls whose absence went unnoticed are not in the list"
    );
    assert_eq!(calls().len(), 23, "the interface is 23 calls today");
}
