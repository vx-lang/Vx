//! Akar Build Script (`build.rs`)
//!
//! This script is automatically executed by Cargo before compiling the `akarc` compiler.
//!
//! # Why does Akar need this?
//! Akar supports two execution modes: JIT Execution and AOT (Ahead-of-Time) Compilation.
//!
//! 1. **JIT Execution**: Managed dynamically by `src/jit.rs`, which shells out to `clang++` at
//!    runtime to build `.dylib` files for `lli`.
//! 2. **AOT Compilation**: If a user uses `akarc` to compile their Akar code into a standalone
//!    executable binary, the linker needs a static version of the Objective-C++ hardware dispatcher.
//!
//! This script ensures that `libnpu_dispatch.a` is pre-compiled into Cargo's `OUT_DIR` so that
//! the standalone AOT linker can statically bundle the Apple Accelerate AMX hardware dispatcher
//! directly into the final application.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
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
        panic!("Akar v2.0 hardware dispatch requires macOS Apple Silicon (AMX). Other operating systems are not currently supported.");
    }
}
