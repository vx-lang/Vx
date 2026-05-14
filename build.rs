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

        // Compile the Objective-C++ runtime file
        let status = Command::new("clang++")
            .args([
                "-c",
                "runtime/npu_dispatch.mm",
                "-o",
                obj_path.to_str().unwrap(),
                "-fobjc-arc",
                "-O3",
                "-Wno-deprecated-declarations",
            ])
            .status()
            .expect("Failed to execute clang++");

        assert!(status.success(), "clang++ compilation failed");

        // Create the static archive
        let status = Command::new("ar")
            .args([
                "rcs",
                lib_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute ar");

        assert!(status.success(), "ar archiving failed");

        // Tell cargo to link against the generated library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=npu_dispatch");

        // Link required Apple frameworks and C++ standard library
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        panic!("Akar v2.0 hardware dispatch requires macOS Apple Silicon (AMX). Other operating systems are not currently supported.");
    }
}
