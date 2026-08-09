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

fn run_cmd(mut cmd: Command, desc: &str) -> Result<std::process::Output, String> {
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run {}: {}", desc, e))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{} failed:\n{}", desc, err));
    }
    Ok(output)
}

pub fn execute_mlir(
    mlir_src: &str,
    program_args: Vec<String>,
    opt_level: u8,
    disable_llvm_optimizations: bool,
) -> Result<String, String> {
    let temp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;

    let temp_mlir = temp_dir
        .path()
        .join("input.mlir")
        .to_string_lossy()
        .into_owned();
    let temp_ll = temp_dir
        .path()
        .join("input.ll")
        .to_string_lossy()
        .into_owned();

    // 1. Write MLIR to temp file
    let mut mlir_file = File::create(&temp_mlir).map_err(|e| e.to_string())?;
    mlir_file
        .write_all(mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    // Runtime functions are now loaded via libvx_std_core.dylib

    // The dispatch backend built alongside this compiler, overridable at run
    // time. The compile-time path names a file in the build machine's OUT_DIR,
    // which is the right answer when the compiler runs where it was built and
    // no answer at all when it does not: a vxc copied onto a rented GPU box has
    // to be pointed at a backend built there, against that box's CUDA.
    let lib_npu = std::env::var("VX_DISPATCH_LIB")
        .unwrap_or_else(|_| std::env!("NPU_SHARED_LIB_PATH").to_string());

    let mlir_translate_path =
        std::env::var("MLIR_TRANSLATE_PATH").unwrap_or_else(|_| "mlir-translate".to_string());
    println!("[JIT] Translating to LLVM IR...");
    let mut mlir_translate_cmd = Command::new(&mlir_translate_path);
    mlir_translate_cmd.args(["--mlir-to-llvmir", &temp_mlir]);
    let mlir_translate_out = run_cmd(mlir_translate_cmd, "mlir-translate")?;

    let mut llvmir_file = File::create(&temp_ll).map_err(|e| e.to_string())?;
    llvmir_file
        .write_all(&mlir_translate_out.stdout)
        .map_err(|e| e.to_string())?;

    let temp_opt_ll = temp_dir
        .path()
        .join("opt.ll")
        .to_string_lossy()
        .into_owned();
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
    let mut opt_cmd = Command::new(&opt_path);
    opt_cmd.args(&opt_args);
    run_cmd(opt_cmd, "opt")?;

    let llc_path = std::env::var("LLC_PATH").unwrap_or_else(|_| "llc".to_string());
    println!(
        "[JIT] Compiling to native object (-O{})...",
        actual_opt_level
    );
    let temp_obj = temp_dir.path().join("opt.o").to_string_lossy().into_owned();
    let mut llc_cmd = Command::new(&llc_path);
    llc_cmd.args([
        &format!("-O={}", actual_opt_level),
        "-filetype=obj",
        "-relocation-model=pic",
        &temp_opt_ll,
        "-o",
        &temp_obj,
    ]);
    run_cmd(llc_cmd, "llc")?;

    println!("[JIT] Linking native executable...");
    let temp_exe = temp_dir
        .path()
        .join("temp.out")
        .to_string_lossy()
        .into_owned();
    let current_dir = std::env::current_dir().unwrap();
    let profile_dir = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    // Resolve through PATH by default (config.local puts the intended LLVM
    // first), matching how build.rs locates the toolchain. Hardcoding a
    // Homebrew prefix here made the JIT unusable on Linux.
    let llvm_config_path =
        std::env::var("LLVM_CONFIG_PATH").unwrap_or_else(|_| "llvm-config".to_string());

    let mut llvm_config_cmd = Command::new(&llvm_config_path);
    llvm_config_cmd.arg("--libdir");
    let llvm_libdir_out = llvm_config_cmd.output().map_err(|e| {
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
        // No longer need target/jit in rpath
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

    // The dispatch runtime provides vx_plugin_dispatch_async, which any program
    // containing a non-CPU `spawn on` calls. macOS gets the ANE/AMX backend
    // (runtime/npu_dispatch.mm); other platforms get the portable host shim
    // (runtime/host_dispatch.cpp). Both call outlined kernels through libffi.
    if !lib_npu.is_empty() {
        clang_cmd.args([&lib_npu]);
        clang_cmd.arg("-lffi");

        // The dispatcher resolves the outlined kernel with
        // dlsym(RTLD_DEFAULT, "_mlir_ciface_..."), which searches the running
        // executable's dynamic symbol table. Mach-O exports those symbols
        // anyway; ELF does not unless asked, so without this the lookup fails
        // at run time on Linux.
        if cfg!(target_os = "linux") {
            clang_cmd.arg("-rdynamic");
        }
    }

    clang_cmd.arg("-lm");

    run_cmd(clang_cmd, "clang")?;

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
