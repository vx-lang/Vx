//===- spirv_images_on_pocl.rs - Vx Compiler -------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Runs each test program that targets `spirv64` twice: once with its kernels
//! on the host, and once with each SPIR-V image on pocl, an OpenCL that runs
//! kernels on the CPU. The two runs must print the same thing. So a wrong
//! SPIR-V image shows up with no Intel GPU. tests/runtime/pocl_dispatch.cpp is
//! the dispatch library of the second run.
//!
//! A program with an image must run at least one kernel from it, and must not
//! run a SPIR-V kernel on the host. Floats may differ in the last digits,
//! because a device computes `exp` or `sin` with its own functions.
//!
//! pocl cannot run in a process that loads LLVM 22: its own LLVM aborts. Only
//! the MLIR runner libraries of a program list LLVM 22, and the program calls
//! nothing of it. So the test runs the program with copies of those libraries
//! that `patchelf` takes LLVM out of.
//!
//! The test needs pocl (`clinfo -l` names it) and `patchelf`. Without them it
//! passes and says that it ran nothing. With `VX_REQUIRE_POCL=1`, it fails then.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::integration_test::runtime_test_build;

fn has_pocl() -> bool {
    Command::new("clinfo")
        .arg("-l")
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("Portable Computing Language"))
}

fn has_patchelf() -> bool {
    Command::new("patchelf")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn patchelf(args: &[&str], lib: &Path) -> String {
    let out = Command::new("patchelf")
        .args(args)
        .arg(lib)
        .output()
        .expect("run patchelf");
    assert!(
        out.status.success(),
        "patchelf {args:?} {} failed:\n{}",
        lib.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Copies of the MLIR runner libraries that do not list libLLVM or libMLIR,
/// in a new folder for `LD_LIBRARY_PATH`.
fn runner_libraries_without_llvm(dir: &Path) -> PathBuf {
    let llvm_config =
        std::env::var("LLVM_CONFIG_PATH").unwrap_or_else(|_| "llvm-config".to_string());
    let out = Command::new(&llvm_config)
        .arg("--libdir")
        .output()
        .unwrap_or_else(|e| panic!("run {llvm_config}: {e}"));
    let libdir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let copies = dir.join("runner-libraries");
    std::fs::create_dir_all(&copies).unwrap();
    for name in [
        "mlir_c_runner_utils",
        "mlir_runner_utils",
        "mlir_float16_utils",
    ] {
        let lib = libdir.join(format!("lib{name}.so"));
        let soname = patchelf(&["--print-soname"], &lib);
        let copy = copies.join(soname.trim());
        std::fs::copy(&lib, &copy).unwrap();
        for needed in patchelf(&["--print-needed"], &copy).lines() {
            if ["libLLVM", "libMLIR", "libmlir_apfloat_wrappers"]
                .iter()
                .any(|p| needed.starts_with(p))
            {
                patchelf(&["--remove-needed", needed], &copy);
            }
        }
    }
    copies
}

fn build_pocl_library(root: &Path, dir: &Path) -> PathBuf {
    let source = root.join("tests/runtime/pocl_dispatch.cpp");
    let lib = dir.join("libvx_pocl_dispatch.so");
    let cxx = runtime_test_build::cxx();
    let out = runtime_test_build::command(&cxx, "c++17")
        .args(["-shared", "-fPIC"])
        .arg(format!("-I{}", root.join("include").display()))
        .arg(format!("-I{}", root.join("runtime").display()))
        .arg(&source)
        .args(["-lffi", "-lOpenCL", "-ldl", "-o"])
        .arg(&lib)
        .output()
        .unwrap_or_else(|e| panic!("run {cxx}: {e}"));
    assert!(
        out.status.success(),
        "compiling {} failed:\n{}",
        source.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    lib
}

/// The test programs with a `main` whose topology says `arch: spirv64`.
fn spirv_programs(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "vx") {
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                let spirv = text.lines().any(|l| {
                    let l = l.trim();
                    l.starts_with("arch:") && l.contains("spirv64")
                });
                if spirv && text.contains("fn main") {
                    out.push(path);
                }
            }
        }
    }
    let mut out = Vec::new();
    for dir in ["backend", "frontend", "middle_end", "optimizations"] {
        walk(&root.join("tests").join(dir).join("pass"), &mut out);
    }
    out.sort();
    out
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn vxc(root: &Path, program: &Path, args: &[&str], env: &[(&str, &Path)]) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.current_dir(root).arg(program).args(args);
    cmd.env_remove("VX_DISPATCH_LIB");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run vxc");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// What the program printed, without the compiler's own lines and without the
/// address of each buffer, which changes from run to run.
fn printed(stdout: &str) -> Vec<String> {
    let mut words = Vec::new();
    for line in stdout.lines() {
        if line.starts_with("[JIT]") || line.starts_with("[flat-codegen]") {
            continue;
        }
        let mut skip_address = false;
        for word in line.split(|c: char| c.is_whitespace() || "[],|".contains(c)) {
            if word.is_empty() {
                continue;
            }
            if skip_address {
                skip_address = false;
                if word.starts_with("0x") {
                    continue;
                }
            }
            skip_address = word == "=" && words.last().is_some_and(|w: &String| w == "base@");
            words.push(word.to_string());
        }
    }
    words
}

/// Two words are the same, or two floats that differ only in the last digits.
fn same_word(host: &str, device: &str) -> bool {
    if host == device {
        return true;
    }
    let float = |w: &str| w.contains(['.', 'e']);
    match (host.parse::<f64>(), device.parse::<f64>()) {
        (Ok(a), Ok(b)) if float(host) || float(device) => {
            (a - b).abs() <= 1e-6 * a.abs().max(b.abs())
        }
        _ => false,
    }
}

/// The two counts that pocl_dispatch.cpp prints at exit.
fn kernel_counts(stderr: &str) -> (u32, u32) {
    let Some(line) = stderr
        .lines()
        .find_map(|l| l.strip_prefix("[pocl] SPIR-V kernels run on OpenCL: "))
    else {
        return (0, 0);
    };
    let (ran, on_host) = line
        .split_once(", run on the host copy: ")
        .expect("the counts line of pocl_dispatch.cpp");
    (ran.trim().parse().unwrap(), on_host.trim().parse().unwrap())
}

#[test]
fn integers_must_match_and_floats_only_in_the_last_digits() {
    assert!(same_word("0.84147096", "0.841471"));
    assert!(!same_word("0.5", "0.6"));
    assert!(!same_word("2000000", "2000001"));
    assert_eq!(
        printed("Unranked Memref base@ = 0x5a99d83d8290 rank = 1\n[2,  4]"),
        ["Unranked", "Memref", "base@", "=", "rank", "=", "1", "2", "4"]
    );
}

#[test]
fn spirv_images_on_pocl_print_what_the_host_prints() {
    if !has_pocl() || !has_patchelf() {
        assert!(
            std::env::var("VX_REQUIRE_POCL").is_err(),
            "VX_REQUIRE_POCL is set, but pocl (clinfo -l) or patchelf is missing"
        );
        eprintln!("spirv_images_on_pocl: no pocl or no patchelf, so no program ran");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("spirv_images_on_pocl");
    std::fs::create_dir_all(&dir).unwrap();
    let lib = build_pocl_library(&root, &dir);
    let copies = runner_libraries_without_llvm(&dir);
    let programs = spirv_programs(&root);
    assert!(!programs.is_empty(), "no test program targets spirv64");

    let check = |program: &PathBuf| -> Result<bool, String> {
        let name = program.strip_prefix(&root).unwrap().display().to_string();
        let ir = vxc(&root, program, &["--emit-llvm"], &[]);
        let has_image = ir.stdout.contains("format=spirv");
        let host = vxc(&root, program, &[], &[]);
        // The default way of pocl to run the lanes of a block crashed on a
        // correct image: a loop, then a barrier, then a loop over a Workgroup
        // tile. `loops` ran it right.
        let pocl = vxc(
            &root,
            program,
            &[],
            &[
                ("VX_DISPATCH_LIB", &lib),
                ("LD_LIBRARY_PATH", &copies),
                ("POCL_WORK_GROUP_METHOD", Path::new("loops")),
            ],
        );
        let (ran, on_host) = kernel_counts(&pocl.stderr);
        let (h, p) = (printed(&host.stdout), printed(&pocl.stdout));
        let same = h.len() == p.len() && h.iter().zip(&p).all(|(a, b)| same_word(a, b));
        if host.code != pocl.code || !same {
            return Err(format!(
                "{name}: the run on pocl printed something else.\nhost (exit {:?}):\n{}\npocl (exit {:?}):\n{}\n{}",
                host.code, host.stdout, pocl.code, pocl.stdout, pocl.stderr
            ));
        }
        if has_image && (ran == 0 || on_host != 0) {
            return Err(format!(
                "{name}: it has a SPIR-V image, but {ran} kernels ran on pocl and {on_host} on the host copy.\n{}",
                pocl.stderr
            ));
        }
        Ok(has_image)
    };

    let results: Vec<Result<bool, String>> = std::thread::scope(|s| {
        programs
            .chunks(programs.len().div_ceil(4))
            .map(|chunk| s.spawn(move || chunk.iter().map(check).collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect()
    });
    let failures: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    assert!(
        failures.is_empty(),
        "{}",
        failures
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    );
    let with_images = results.iter().filter(|r| matches!(r, Ok(true))).count();
    assert!(
        with_images > 0,
        "no program had a SPIR-V image, so pocl ran nothing"
    );
    eprintln!(
        "spirv_images_on_pocl: {} programs, {with_images} with SPIR-V images that ran on pocl",
        programs.len()
    );
}
