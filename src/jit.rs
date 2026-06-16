//===- jit.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Just-In-Time (JIT) compilation and execution engine.
// It leverages the MLIR execution engine to compile lowered MLIR modules into
// machine code on the fly, enabling dynamic execution of Vx code without
// requiring a separate ahead-of-time compilation step.
//
//===----------------------------------------------------------------------===//
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static JIT_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn execute_mlir(
    mlir_src: &str,
    program_args: Vec<String>,
    opt_level: u8,
    disable_llvm_optimizations: bool,
) -> Result<String, String> {
    // Ensure target/jit directory exists
    let jit_dir = std::path::Path::new("target/jit");
    if !jit_dir.exists() {
        std::fs::create_dir_all(jit_dir).map_err(|e| e.to_string())?;
    }

    let pid = std::process::id();
    let counter = JIT_COUNTER.fetch_add(1, Ordering::SeqCst);
    let uid = format!("{}_{}", pid, counter);

    let temp_mlir = format!("target/jit/temp_{}.mlir", uid);
    let temp_ll = format!("target/jit/temp_{}.ll", uid);

    // 1. Write MLIR to temp file
    let mut mlir_file = File::create(&temp_mlir).map_err(|e| e.to_string())?;
    mlir_file
        .write_all(mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    // Runtime functions are now loaded via libvx_std_core.dylib

    let lib_npu = std::env!("NPU_SHARED_LIB_PATH").to_string();

    let mlir_translate_path =
        std::env::var("MLIR_TRANSLATE_PATH").unwrap_or_else(|_| "mlir-translate".to_string());
    println!("[JIT] Translating to LLVM IR...");
    let mlir_translate_out = Command::new(&mlir_translate_path)
        .args(["--mlir-to-llvmir", &temp_mlir])
        .output()
        .map_err(|e| format!("Failed to execute {}: {}", mlir_translate_path, e))?;

    if !mlir_translate_out.status.success() {
        let err_str = String::from_utf8_lossy(&mlir_translate_out.stderr);
        return Err(format!(
            "mlir-translate failed:
{}
",
            err_str
        ));
    }

    let mut llvmir_file = File::create(&temp_ll).map_err(|e| e.to_string())?;
    llvmir_file
        .write_all(&mlir_translate_out.stdout)
        .map_err(|e| e.to_string())?;

    let temp_opt_ll = format!("target/jit/temp_opt_{}.ll", uid);
    let mut opt_args = vec![];
    let actual_opt_level = if disable_llvm_optimizations {
        0
    } else {
        opt_level
    };

    let mut passes = format!("default<O{}>", actual_opt_level);

    if let Ok(enzyme_lib) = std::env::var("ENZYME_LIB") {
        opt_args.push(format!("-load-pass-plugin={}", enzyme_lib));
        passes.push_str(",enzyme");
    }

    opt_args.push(format!("-passes={}", passes));
    opt_args.push("-S".to_string());
    opt_args.push(temp_ll.clone());
    opt_args.push("-o".to_string());
    opt_args.push(temp_opt_ll.clone());

    let opt_path = std::env::var("OPT_PATH").unwrap_or_else(|_| "opt".to_string());
    println!("[JIT] Optimizing LLVM IR (-O{})...", actual_opt_level);
    let opt_out = Command::new(&opt_path)
        .args(&opt_args)
        .output()
        .map_err(|e| format!("Failed to execute {}: {}", opt_path, e))?;

    if !opt_out.status.success() {
        let err_str = String::from_utf8_lossy(&opt_out.stderr);
        return Err(format!(
            "opt failed:
{}",
            err_str
        ));
    }

    let llc_path = std::env::var("LLC_PATH").unwrap_or_else(|_| "llc".to_string());
    println!(
        "[JIT] Compiling to native object (-O{})...",
        actual_opt_level
    );
    let temp_obj = format!("target/jit/temp_opt_{}.o", uid);
    let llc_out = Command::new(&llc_path)
        .args([
            &format!("-O={}", actual_opt_level),
            "-filetype=obj",
            "-relocation-model=pic",
            &temp_opt_ll,
            "-o",
            &temp_obj,
        ])
        .output()
        .map_err(|e| format!("Failed to execute {}: {}", llc_path, e))?;

    if !llc_out.status.success() {
        let err_str = String::from_utf8_lossy(&llc_out.stderr);
        return Err(format!(
            "llc failed:
{}
",
            err_str
        ));
    }

    println!("[JIT] Linking native executable...");
    let temp_exe = format!("target/jit/temp_{}.out", uid);
    let current_dir = std::env::current_dir().unwrap();
    let profile_dir = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    let llvm_config_path =
        std::env::var("LLVM_CONFIG_PATH").unwrap_or_else(|_| "llvm-config".to_string());

    let llvm_libdir_out = Command::new(&llvm_config_path)
        .arg("--libdir")
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "Compiler toolchain error: '{}' was not found. Please ensure LLVM is installed and in your PATH, or set LLVM_CONFIG_PATH.",
                    llvm_config_path
                )
            } else {
                format!("Failed to run {}: {}", llvm_config_path, e)
            }
        })?;
    let llvm_libdir = String::from_utf8_lossy(&llvm_libdir_out.stdout)
        .trim()
        .to_string();

    let clang_path = std::env::var("CLANG_PATH").unwrap_or_else(|_| "clang".to_string());
    let mut clang_cmd = Command::new(&clang_path);

    // Rpaths
    clang_cmd.args([
        &format!("-Wl,-rpath,{}", llvm_libdir),
        &format!(
            "-Wl,-rpath,{}/target/{}",
            current_dir.display(),
            profile_dir
        ),
        &format!("-Wl,-rpath,{}/target/jit", current_dir.display()),
    ]);

    // Input obj and output exe
    clang_cmd.args([&temp_obj, "-o", &temp_exe]);

    // Libraries
    clang_cmd.args([
        &format!(
            "{}/libmlir_c_runner_utils{}",
            llvm_libdir,
            std::env::consts::DLL_SUFFIX
        ),
        &format!(
            "{}/libmlir_runner_utils{}",
            llvm_libdir,
            std::env::consts::DLL_SUFFIX
        ),
        &format!(
            "{}/target/{}/{}vx_std_core{}",
            current_dir.display(),
            profile_dir,
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ),
    ]);

    if cfg!(target_os = "macos") {
        clang_cmd.args([&lib_npu]);
    }

    let clang_out = clang_cmd.output().map_err(|e| e.to_string())?;

    if !clang_out.status.success() {
        let err_str = String::from_utf8_lossy(&clang_out.stderr);
        return Err(format!(
            "clang failed:
{}",
            err_str
        ));
    }

    println!("[JIT] Executing native binary...");
    let mut exe_cmd = Command::new(&temp_exe);
    exe_cmd.args(program_args);
    if std::env::var("RUST_BACKTRACE").is_err() {
        exe_cmd.env("RUST_BACKTRACE", "1");
    }

    let exe_out = exe_cmd.output().map_err(|e| e.to_string())?;

    if !exe_out.status.success() {
        let err_str = String::from_utf8_lossy(&exe_out.stderr);
        let out_str = String::from_utf8_lossy(&exe_out.stdout);
        let code = exe_out.status.code().unwrap_or(-1);
        println!("[JIT] Program exited with code: {}", code);
        if !out_str.is_empty() {
            println!("{}", out_str);
        }
        if !err_str.is_empty() {
            println!(
                "[JIT] Error output:
{}",
                err_str
            );
        }
        return Err(format!("Program exited with non-zero code: {}", code));
    }

    let output_str = format!(
        "{}{}",
        String::from_utf8_lossy(&exe_out.stdout),
        String::from_utf8_lossy(&exe_out.stderr)
    );

    Ok(output_str)
}
