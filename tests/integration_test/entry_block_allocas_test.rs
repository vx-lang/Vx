//===- entry_block_allocas_test.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Every stack slot is allocated in the entry block, never inside a loop.
//
// An `alloca` is given back when the function returns and not before, so one
// emitted inside a loop body takes a fresh slot every iteration. A nest running
// a couple of million times then exhausts the stack, and the program dies of
// SIGSEGV having compiled without a diagnostic -- which is what a program with
// stack slots inside a loop did until the slots moved to the entry block.
//
// Checked structurally rather than by running something big. MLIR hoists the
// simple shapes on its own, so a small program with a local in a loop passes
// whether or not the emitter does the right thing; only a program with enough
// around it -- spawns, tensors, a three-deep nest -- kept its slots in the
// loops. Asserting the invariant on that program catches a regression that a
// smaller runtime test would sleep through. Its loops became affine loops, which
// need no slots, so a second program checks the slots of loops in affine loops.

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
    // cargo points this at the profile the suite is running in; a hardcoded
    // `target/debug/vxc` is absent under `cargo test --release` and this skipped.
    let vxc = PathBuf::from(env!("CARGO_BIN_EXE_vxc"));
    let mut offenders = Vec::new();
    let mut sites = 0usize;
    for file in [
        "tests/backend/pass/ane_matmul_f16.vx",
        "tests/optimizations/pass/counted_loops_become_affine_loops.vx",
    ] {
        let src = root.join(file);
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
        for line in mlir.lines() {
            if line.trim_start().starts_with("func.func") {
                block = "entry".to_string();
                continue;
            }
            if let Some(b) = block_label(line) {
                block = b.to_string();
                continue;
            }
            if allocation_site(line) {
                sites += 1;
                // A function's own operations are indented by four spaces. More is the inside
                // of a region, such as the body of an affine loop.
                let in_region = line.len() - line.trim_start().len() > 4;
                if block != "entry" || in_region {
                    offenders.push(format!("{file} {block}: {}", line.trim()));
                }
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
