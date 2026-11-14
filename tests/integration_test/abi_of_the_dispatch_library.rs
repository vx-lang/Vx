//===- abi_of_the_dispatch_library.rs - the whole ABI ----------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
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

/// The one-rank implementation of the collectives. It is the only header that
/// declares those eight calls: the CUDA backend defines its own in
/// `runtime/cuda_dispatch.cpp`, and `runtime/vx_collective.h` is a C++ class
/// the two implementations share rather than a list of the ABI.
const COLLECTIVE_HEADER: &str = include_str!("../../runtime/vx_collective_cpu.h");

/// Where the two worker calls are defined, for the same reason.
const HOST_CALL_HEADER: &str = include_str!("../../runtime/vx_host_call.h");

/// Every call named `prefix…(` that `header` declares, in the order it declares
/// them.
///
/// Read from a header rather than written out here: a new call is added to a
/// header first, and a list that has to be updated by hand is a list that falls
/// behind without anyone noticing.
///
/// Two rules keep this to the interface. Only `extern "C"` counts -- a block
/// opened with `extern "C" {`, or a line that starts with it -- because a
/// header that implements the interface also holds helpers whose names begin
/// the same way, and a helper is not a call a program can make. And the name
/// has to follow a return type on its line, so a line that *calls* a helper
/// with that prefix is not read as a declaration of one. Comments and `static`
/// helpers are skipped for the same reason.
fn calls_in(header: &'static str, prefix: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    let mut in_block = false;
    let mut depth = 0i32;
    for line in header.lines() {
        let text = line.trim_start();
        let opens_block = text.starts_with("extern \"C\" {");
        let inside = opens_block || text.starts_with("extern \"C\"") || in_block;
        if inside && !text.starts_with("//") && !text.starts_with('*') {
            if let Some(name) = declared(text, prefix) {
                found.push(name);
            }
        }
        if opens_block {
            in_block = true;
        }
        if in_block {
            depth += text.matches('{').count() as i32;
            depth -= text.matches('}').count() as i32;
            if depth <= 0 {
                in_block = false;
            }
        }
    }
    found
}

/// The call `text` declares, if it declares one with this prefix.
fn declared<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.starts_with("static") || text.starts_with("typedef") {
        return None;
    }
    let at = text.find(prefix)?;
    // A return type, not a call: the name cannot be the first thing on the line.
    if text[..at].trim().is_empty() {
        return None;
    }
    let rest = &text[at..];
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
    rest[end..].starts_with('(').then(|| &rest[..end])
}

/// Every `vx_plugin_*` function the public header declares.
fn plugin_calls_from_the_header() -> Vec<&'static str> {
    calls_in(ABI_HEADER, "vx_plugin_")
}

/// Every `vx_collective_*` function the one-rank header declares.
fn collective_calls_from_the_header() -> Vec<&'static str> {
    calls_in(COLLECTIVE_HEADER, "vx_collective_")
}

/// The two calls that ask which worker is running a kernel.
fn worker_calls_from_the_header() -> Vec<&'static str> {
    calls_in(HOST_CALL_HEADER, "vx_host_worker_")
}

/// The calls that have no declaring header, one at a time, because they have
/// no header for the same reason.
///
/// `vx_cuda_set_stream` and `vx_cuda_get_stream` are the pair a host program
/// written to `docs/lang/abi.md` section 3.6 calls to choose a CUDA stream.
/// There is no header to read them off: the CUDA backend defines them in
/// `runtime/cuda_dispatch.cpp`, and every build without CUDA defines do-nothing
/// versions of them in `runtime/host_dispatch_common.h` so that such a program
/// links either way. Neither file declares the interface, so a scan of either
/// one would be reading an implementation for names that happen to match. The
/// other twenty-one calls need no list.
const CALLS_WITHOUT_A_HEADER: &[&str] = &["vx_cuda_get_stream", "vx_cuda_set_stream"];

/// Every call a program can carry.
fn calls() -> Vec<&'static str> {
    let mut all = plugin_calls_from_the_header();
    all.extend(collective_calls_from_the_header());
    all.extend(worker_calls_from_the_header());
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

/// The list is built from headers, so a header that stops declaring a call
/// would shrink it silently and the check above would pass on less. This pins
/// each group's size and the total, and names what the total covers.
#[test]
fn the_list_covers_the_whole_interface() {
    let plugins = plugin_calls_from_the_header();
    assert_eq!(
        plugins.len(),
        11,
        "the public header declares {} plugin calls, and this test was written \
         against 11: {plugins:?}",
        plugins.len()
    );
    assert!(
        plugins.contains(&"vx_plugin_arrivals_take")
            && plugins.contains(&"vx_plugin_arrivals_give"),
        "the two calls whose absence went unnoticed are not in the list"
    );

    let collectives = collective_calls_from_the_header();
    assert_eq!(
        collectives.len(),
        8,
        "the one-rank header declares {} collective calls, and this test was \
         written against 8: {collectives:?}",
        collectives.len()
    );

    let workers = worker_calls_from_the_header();
    assert_eq!(
        workers,
        ["vx_host_worker_id", "vx_host_worker_count"],
        "the two worker calls are not what the header defines"
    );

    assert_eq!(calls().len(), 23, "the interface is 23 calls today");
}
