//===- emit_obj_produces_objects.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `--action emit-obj` produces an object file for the backend corpus.
//
// AOT emission had no coverage at all: `emit-obj` appeared nowhere in tests/, CI or
// scripts/, and it had been broken for months in two independent ways. Fifteen
// programs died with SIGSEGV -- `llvm.emit_c_interface` on `main` made
// convert-func-to-llvm synthesize `_mlir_ciface_main` and copy main's location onto
// it, DISubprogram included, which a distinct DISubprogram may not be; the verifier
// rejected the module, the execution engine returned null and the driver
// dereferenced it. Another 27 wrote no object because the arm resolved MLIR's
// runtime libraries by bare filename against the process CWD.
//
// Neither was visible from the exit status, which is the reason this test asserts a
// FILE EXISTS rather than a status: `dump_to_object_file` returns nothing, and the
// arm returned `Ok(())` whatever happened. It reports failure now, and this holds
// that.
//
//===----------------------------------------------------------------------===//

use std::path::PathBuf;
use std::process::Command;

/// Programs that legitimately produce no object, with the reason.
///
/// Empty, and worth keeping rather than deleting: the check reads it in both directions, so an
/// entry that starts working fails this test too. That is how `plugin_npe.vx` was found to have
/// been fixed, and it is what stops the list becoming a place failures go to be forgotten.
const KNOWN_BROKEN: &[(&str, &str)] = &[];

#[test]
fn the_backend_corpus_emits_object_files() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("tests/backend/pass");
    let out = std::env::temp_dir().join("vx_emit_obj_gate");
    std::fs::create_dir_all(&out).unwrap();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("tests/backend/pass is missing")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vx"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 100,
        "expected the backend corpus, found {} files",
        files.len()
    );

    let mut no_object = Vec::new();
    let mut unexpectedly_fine = Vec::new();

    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let obj = out.join(format!("{}.o", f.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(&obj);
        let status = Command::new(env!("CARGO_BIN_EXE_vxc"))
            .current_dir(&root)
            .args([f.to_str().unwrap(), "--action", "emit-obj", "-o"])
            .arg(&obj)
            .output()
            .expect("failed to run vxc");
        // An object that exists and is non-empty is the only evidence that counts.
        let produced = std::fs::metadata(&obj)
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        let known = KNOWN_BROKEN.iter().find(|(p, _)| *p == name);

        match (produced, known) {
            (false, None) => no_object.push(format!(
                "  {name}: no object (exit {}){}",
                status.status,
                match String::from_utf8_lossy(&status.stderr)
                    .lines()
                    .find(|l| l.contains("error") || l.contains("Failed"))
                {
                    Some(l) => format!("\n      {}", l.trim()),
                    None => String::new(),
                }
            )),
            (true, Some((_, why))) => {
                unexpectedly_fine.push(format!("  {name}: was listed as broken: {why}"))
            }
            _ => {}
        }
    }

    assert!(
        no_object.is_empty(),
        "emit-obj produced no object for:\n{}",
        no_object.join("\n")
    );
    assert!(
        unexpectedly_fine.is_empty(),
        "these emit an object now -- remove them from KNOWN_BROKEN:\n{}",
        unexpectedly_fine.join("\n")
    );
}
