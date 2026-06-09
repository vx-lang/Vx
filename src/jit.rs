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
use std::sync::Once;

static JIT_COUNTER: AtomicUsize = AtomicUsize::new(0);
static COMPILE_NPU_ONCE: Once = Once::new();

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

    let lib_npu = "target/jit/libnpu_shared.dylib".to_string();

    // 1. Write MLIR to temp file
    let mut mlir_file = File::create(&temp_mlir).map_err(|e| e.to_string())?;
    mlir_file
        .write_all(mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    // Runtime functions are now loaded via libvx_std_core.dylib

    if cfg!(target_os = "macos") {
        COMPILE_NPU_ONCE.call_once(|| {
            println!("[JIT] Compiling Objective-C++ NPU Dispatcher (Shared)...");
            let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
            let cxxflags_env = std::env::var("CXXFLAGS").unwrap_or_else(|_| {
                "-shared -fPIC -fobjc-arc -O3 -Wno-deprecated-declarations".to_string()
            });
            let cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();

            let mut cxx_cmd = Command::new(&cxx);
            cxx_cmd.args(&cxxflags);
            cxx_cmd.args([
                "runtime/npu_dispatch.mm",
                "-framework",
                "Accelerate",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "CoreML",
                "-o",
                &lib_npu,
            ]);

            let npu_status = cxx_cmd.status().expect("Failed to execute clang++");
            if !npu_status.success() {
                panic!("Failed to compile Objective-C++ NPU Dispatcher");
            }
        });
    }

    println!("[JIT] Translating to LLVM IR...");
    let mlir_translate_out = Command::new("mlir-translate")
        .args(["--mlir-to-llvmir", &temp_mlir])
        .output()
        .map_err(|e| e.to_string())?;

    if !mlir_translate_out.status.success() {
        let err_str = String::from_utf8_lossy(&mlir_translate_out.stderr);
        return Err(format!("mlir-translate failed:\n{}", err_str));
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

    println!("[JIT] Optimizing LLVM IR (-O{})...", actual_opt_level);
    let opt_out = Command::new("opt")
        .args(&opt_args)
        .output()
        .map_err(|e| e.to_string())?;

    if !opt_out.status.success() {
        let err_str = String::from_utf8_lossy(&opt_out.stderr);
        return Err(format!("opt failed:\n{}", err_str));
    }

    println!("[JIT] Executing via LLI...");
    let current_dir = std::env::current_dir().unwrap();
    let mut lli_cmd = Command::new("lli");

    if cfg!(target_os = "macos") {
        lli_cmd.arg(format!("--load={}", lib_npu));
    }

    let profile_dir = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    lli_cmd.args([
        &format!(
            "--load=libmlir_c_runner_utils{}",
            std::env::consts::DLL_SUFFIX
        ),
        &format!(
            "--load=libmlir_runner_utils{}",
            std::env::consts::DLL_SUFFIX
        ),
        &format!(
            "--load={}/target/{}/{}vx_std_core{}",
            current_dir.display(),
            profile_dir,
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ),
        &temp_opt_ll,
    ]);

    lli_cmd.args(program_args);

    let lli_out = lli_cmd.output().map_err(|e| e.to_string())?;

    if !lli_out.status.success() {
        let err_str = String::from_utf8_lossy(&lli_out.stderr);
        let code = lli_out.status.code().unwrap_or(-1);
        println!("[JIT] Program exited with code: {}", code);
        if !err_str.is_empty() {
            println!("[JIT] Error output:\n{}", err_str);
        }
    }

    let output_str = format!(
        "{}{}",
        String::from_utf8_lossy(&lli_out.stdout),
        String::from_utf8_lossy(&lli_out.stderr)
    );

    Ok(output_str)
}
