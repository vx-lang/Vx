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

/// Every call a program can carry, as `runtime/x86_dispatch.cpp` provides them.
const CALLS: &[&str] = &[
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
    "vx_plugin_alloc_and_transfer",
    "vx_plugin_arrivals_give",
    "vx_plugin_arrivals_take",
    "vx_plugin_await_future",
    "vx_plugin_control",
    "vx_plugin_dispatch_async",
    "vx_plugin_dispatch_async_flat",
    "vx_plugin_free",
    "vx_plugin_release_future",
    "vx_plugin_tensor_map",
    "vx_plugin_transfer_device_to_host",
    "vx_plugin_transfer_peer",
];

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
        // No `nm` on this machine: there is no other way to read the symbols,
        // and a test that cannot look must not claim the library is complete.
        Err(e) => {
            println!("skipped: cannot run nm ({e})");
            return Ok(());
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
    let missing: Vec<&str> = CALLS
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
