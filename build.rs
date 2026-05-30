//===- build.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Vx Build Script (`build.rs`)
//!
//! This script is automatically executed by Cargo before compiling the `vxc` compiler.
//!
//! # Why does Vx need this?
//! Vx supports two execution modes: JIT Execution and AOT (Ahead-of-Time) Compilation.
//!
//! 1. **JIT Execution**: Managed dynamically by `src/jit.rs`, which shells out to `clang++` at
//!    runtime to build `.dylib` files for `lli`.
//! 2. **AOT Compilation**: If a user uses `vxc` to compile their Vx code into a standalone
//!    executable binary, the linker needs a static version of the Objective-C++ hardware dispatcher.
//!
//! This script ensures that `libnpu_dispatch.a` is pre-compiled into Cargo's `OUT_DIR` so that
//! the standalone AOT linker can statically bundle the Apple Accelerate AMX hardware dispatcher
//! directly into the final application.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Ensure llvm-config is in PATH because the `melior` and `tblgen` dependencies require it.
    let llvm_configs = [
        "llvm-config-22",
        "llvm-config-21",
        "llvm-config-20",
        "llvm-config-19",
        "llvm-config-18",
        "llvm-config-17",
        "llvm-config-16",
        "llvm-config-15",
        "llvm-config",
    ];
    let mut found = false;
    for cfg in llvm_configs.iter() {
        if Command::new(cfg).arg("--version").output().is_ok() {
            found = true;
            break;
        }
    }

    if !found {
        println!(
            "cargo:warning=⚠️  LLVM is not in PATH! `melior` and `tblgen` will fail to build."
        );
        println!("cargo:warning=On macOS with Homebrew, run: export PATH=\"/opt/homebrew/opt/llvm/bin:$PATH\"");
        println!("cargo:warning=On Ubuntu, run: sudo apt-get install llvm-22 llvm-22-dev");
        panic!("llvm-config not found in PATH. Make sure LLVM 15+ is installed.");
    }

    // Check if we're on macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rerun-if-changed=runtime/npu_dispatch.mm");
        println!("cargo:rerun-if-changed=runtime/npu_dispatch.h");

        let out_dir = env::var("OUT_DIR").unwrap();
        let obj_path = PathBuf::from(&out_dir).join("npu_dispatch.o");
        let lib_path = PathBuf::from(&out_dir).join("libnpu_dispatch.a");

        // Determine compiler and flags
        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
        let cxxflags_env =
            env::var("CXXFLAGS").unwrap_or_else(|_| "-O3 -Wno-deprecated-declarations".to_string());
        let cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();

        // Compile the Objective-C++ runtime file
        let mut clang_cmd = Command::new(&cxx);
        clang_cmd.args([
            "-c",
            "runtime/npu_dispatch.mm",
            "-o",
            obj_path.to_str().unwrap(),
            "-fobjc-arc",
        ]);
        clang_cmd.args(&cxxflags);

        let status = clang_cmd
            .status()
            .unwrap_or_else(|_| panic!("Failed to execute {}", cxx));

        assert!(status.success(), "clang++ compilation failed");

        // Determine archiver and flags
        let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
        let arflags_env = env::var("ARFLAGS").unwrap_or_else(|_| "rcs".to_string());
        let arflags: Vec<&str> = arflags_env.split_whitespace().collect();

        // Create the static archive
        let mut ar_cmd = Command::new(&ar);
        ar_cmd.args(&arflags);
        ar_cmd.args([lib_path.to_str().unwrap(), obj_path.to_str().unwrap()]);

        let status = ar_cmd
            .status()
            .unwrap_or_else(|_| panic!("Failed to execute {}", ar));

        assert!(status.success(), "{} archiving failed", ar);

        // Tell cargo to link against the generated library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=npu_dispatch");

        // Link required Apple frameworks and C++ standard library
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=CoreML");
    } else {
        println!("cargo:warning=Vx v2.0 hardware dispatch requires macOS Apple Silicon (AMX). Skipping NPU dispatcher compilation on this OS.");
    }

    // --- Compile MLIR Pass Plugin Loader Wrapper ---
    println!("cargo:rerun-if-changed=src/plugin_loader.cpp");

    let out_dir = env::var("OUT_DIR").unwrap();
    let plugin_obj_path = PathBuf::from(&out_dir).join("plugin_loader.o");
    let plugin_lib_path = PathBuf::from(&out_dir).join("libplugin_loader.a");

    let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    // Get LLVM CXXFLAGS via llvm-config
    let llvm_cxxflags_out = Command::new("llvm-config")
        .arg("--cxxflags")
        .output()
        .expect("Failed to get llvm-config cxxflags");
    let llvm_cxxflags = String::from_utf8_lossy(&llvm_cxxflags_out.stdout)
        .trim()
        .to_string();
    let llvm_cxxflags_vec: Vec<&str> = llvm_cxxflags.split_whitespace().collect();

    let mut plugin_cmd = Command::new(&cxx);
    plugin_cmd.args([
        "-c",
        "src/plugin_loader.cpp",
        "-o",
        plugin_obj_path.to_str().unwrap(),
        "-std=c++17",
    ]);
    plugin_cmd.args(&llvm_cxxflags_vec);

    let status = plugin_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute {} for plugin_loader", cxx));
    assert!(
        status.success(),
        "clang++ compilation failed for plugin_loader"
    );

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let arflags_env = env::var("ARFLAGS").unwrap_or_else(|_| "rcs".to_string());
    let arflags: Vec<&str> = arflags_env.split_whitespace().collect();

    let mut ar_cmd = Command::new(&ar);
    ar_cmd.args(&arflags);
    ar_cmd.args([
        plugin_lib_path.to_str().unwrap(),
        plugin_obj_path.to_str().unwrap(),
    ]);
    let status = ar_cmd
        .status()
        .unwrap_or_else(|_| panic!("Failed to archive libplugin_loader.a"));
    assert!(status.success(), "ar failed for libplugin_loader");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=plugin_loader");

    // --- Compile Vx MLIR Dialect ---
    println!("cargo:rerun-if-changed=include/VxDialect.td");
    println!("cargo:rerun-if-changed=include/VxDialect.h");
    println!("cargo:rerun-if-changed=src/dialect/VxDialect.cpp");

    let llvm_bindir_out = Command::new("llvm-config")
        .arg("--bindir")
        .output()
        .expect("Failed to get llvm-config bindir");
    let llvm_bindir = String::from_utf8_lossy(&llvm_bindir_out.stdout)
        .trim()
        .to_string();
    let tblgen = PathBuf::from(&llvm_bindir).join("mlir-tblgen");
    let tblgen_str = tblgen.to_str().unwrap();

    let llvm_includedir_out = Command::new("llvm-config")
        .arg("--includedir")
        .output()
        .expect("Failed to get llvm-config includedir");
    let llvm_include = String::from_utf8_lossy(&llvm_includedir_out.stdout)
        .trim()
        .to_string();

    // 1. Generate Dialect Declarations
    let status = Command::new(tblgen_str)
        .args([
            "-gen-dialect-decls",
            "-I",
            &llvm_include,
            "include/VxDialect.td",
            "-o",
            &format!("{}/VxDialect.h.inc", out_dir),
        ])
        .status()
        .expect("Failed to run mlir-tblgen for dialect decls");
    assert!(status.success(), "mlir-tblgen failed");

    // 2. Generate Dialect Definitions
    let status = Command::new(tblgen_str)
        .args([
            "-gen-dialect-defs",
            "-I",
            &llvm_include,
            "include/VxDialect.td",
            "-o",
            &format!("{}/VxDialect.cpp.inc", out_dir),
        ])
        .status()
        .expect("Failed to run mlir-tblgen for dialect defs");
    assert!(status.success(), "mlir-tblgen failed");

    // 3. Generate Operation Declarations
    let status = Command::new(tblgen_str)
        .args([
            "-gen-op-decls",
            "-I",
            &llvm_include,
            "include/VxDialect.td",
            "-o",
            &format!("{}/VxOps.h.inc", out_dir),
        ])
        .status()
        .expect("Failed to run mlir-tblgen for op decls");
    assert!(status.success(), "mlir-tblgen failed");

    // 4. Generate Operation Definitions
    let status = Command::new(tblgen_str)
        .args([
            "-gen-op-defs",
            "-I",
            &llvm_include,
            "include/VxDialect.td",
            "-o",
            &format!("{}/VxOps.cpp.inc", out_dir),
        ])
        .status()
        .expect("Failed to run mlir-tblgen for op defs");
    assert!(status.success(), "mlir-tblgen failed");
    println!("cargo:rerun-if-changed=src/dialect/VxLowering.cpp");
    println!("cargo:rerun-if-changed=src/dialect/vx-opt.cpp");

    // Compile the dialect
    let dialect_obj_path = PathBuf::from(&out_dir).join("VxDialect.o");
    let lowering_obj_path = PathBuf::from(&out_dir).join("VxLowering.o");
    let vx_opt_obj_path = PathBuf::from(&out_dir).join("vx-opt.o");
    let dialect_lib_path = PathBuf::from(&out_dir).join("libvx_dialect.a");

    let mut dialect_cmd = Command::new(&cxx);
    dialect_cmd.args([
        "-c",
        "src/dialect/VxDialect.cpp",
        "-o",
        dialect_obj_path.to_str().unwrap(),
        "-std=c++17",
        &format!("-I{}", out_dir), // to find the generated .inc files
        "-Iinclude",               // to find VxDialect.h
    ]);
    dialect_cmd.args(&llvm_cxxflags_vec);
    let status = dialect_cmd
        .status()
        .expect("Failed to execute cxx for VxDialect");
    assert!(status.success(), "clang++ compilation failed for VxDialect");

    let mut lowering_cmd = Command::new(&cxx);
    lowering_cmd.args([
        "-c",
        "src/dialect/VxLowering.cpp",
        "-o",
        lowering_obj_path.to_str().unwrap(),
        "-std=c++17",
        &format!("-I{}", out_dir), // to find the generated .inc files
        "-Iinclude",               // to find VxDialect.h
    ]);
    lowering_cmd.args(&llvm_cxxflags_vec);
    let status = lowering_cmd
        .status()
        .expect("Failed to execute cxx for VxLowering");
    assert!(
        status.success(),
        "clang++ compilation failed for VxLowering"
    );

    let mut vxopt_cmd = Command::new(&cxx);
    vxopt_cmd.args([
        "-c",
        "src/dialect/vx-opt.cpp",
        "-o",
        vx_opt_obj_path.to_str().unwrap(),
        "-std=c++17",
        &format!("-I{}", out_dir), // to find the generated .inc files
        "-Iinclude",               // to find VxDialect.h
    ]);
    vxopt_cmd.args(&llvm_cxxflags_vec);
    let status = vxopt_cmd
        .status()
        .expect("Failed to execute cxx for vx-opt");
    assert!(status.success(), "clang++ compilation failed for vx-opt");

    let mut ar_cmd = Command::new(&ar);
    ar_cmd.args(&arflags);
    ar_cmd.args([
        dialect_lib_path.to_str().unwrap(),
        dialect_obj_path.to_str().unwrap(),
        lowering_obj_path.to_str().unwrap(),
        vx_opt_obj_path.to_str().unwrap(),
    ]);
    let status = ar_cmd.status().expect("Failed to archive libvx_dialect.a");
    assert!(status.success(), "ar failed for libvx_dialect");

    println!("cargo:rustc-link-lib=static=vx_dialect");
}
