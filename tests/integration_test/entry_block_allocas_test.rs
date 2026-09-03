//===- entry_block_allocas_test.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Every stack slot is allocated in the entry block, never inside a loop.
//
// An `alloca` is given back when the function returns and not before, so one
// emitted inside a loop body takes a fresh slot every iteration. A nest running
// a couple of million times then exhausts the stack, and the program dies of
// SIGSEGV having compiled without a diagnostic -- which is what
// benchmarks/flash_attention_ane/flash_attention_split.vx did until the slots
// moved to the entry block.
//
// Checked structurally rather than by running something big. MLIR hoists the
// simple shapes on its own, so a small program with a local in a loop passes
// whether or not the emitter does the right thing; only a program with enough
// around it -- spawns, tensors, a three-deep nest -- kept its slots in the
// loops. Asserting the invariant on that program catches a regression that a
// smaller runtime test would sleep through.

use std::path::PathBuf;
use std::process::Command;

/// Allocation sites only. A `memref.store` naming an alloca mentions it without
/// creating one, and counting those reports slots in loops that are not there.
fn allocation_site(line: &str) -> bool {
    let t = line.trim_start();
    match t.find("= ") {
        Some(i) => {
            let rhs = t[i + 2..].trim_start();
            rhs.starts_with("memref.alloca") || rhs.starts_with("llvm.alloca")
        }
        None => false,
    }
}

fn block_label(line: &str) -> Option<&str> {
    let t = line.trim_start();
    if t.starts_with('^') && t.contains(':') {
        Some(t.split(':').next().unwrap_or(""))
    } else {
        None
    }
}

#[test]
fn stack_slots_are_allocated_once_not_per_iteration() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vxc = root.join("target/debug/vxc");
    if !vxc.is_file() {
        eprintln!("skipping: {} is not built", vxc.display());
        return;
    }
    let src = root.join("benchmarks/flash_attention_ane/flash_attention_split.vx");
    assert!(src.is_file(), "missing {}", src.display());

    let out = Command::new(&vxc)
        .current_dir(&root)
        .args([src.to_str().unwrap(), "--action", "emit-mlir"])
        .output()
        .expect("failed to run vxc");
    let mlir = String::from_utf8_lossy(&out.stdout);
    assert!(
        mlir.contains("func.func"),
        "vxc emitted no MLIR:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mut block = "entry".to_string();
    let mut offenders = Vec::new();
    let mut sites = 0usize;
    for line in mlir.lines() {
        if let Some(b) = block_label(line) {
            block = b.to_string();
            continue;
        }
        if allocation_site(line) {
            sites += 1;
            if block != "entry" {
                offenders.push(format!("{block}: {}", line.trim()));
            }
        }
    }

    assert!(sites > 0, "no allocation sites found; the check is vacuous");
    assert!(
        offenders.is_empty(),
        "{} of {} stack slots are allocated inside a block other than the entry \
         block, so they are taken again on every iteration of whatever loop \
         encloses them:\n{}",
        offenders.len(),
        sites,
        offenders.join("\n")
    );
}
