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

/// Where a usable CUDA toolkit lives, as (root, library directory), or `None`
/// to build the portable host shim instead.
///
/// A GPU is not required here and deliberately not looked for: the build host
/// and the run host are different machines in this project's workflow, and the
/// backend decides at run time whether a device is present (see
/// `cuda_available()` in runtime/cuda_dispatch.cpp). What must be present to
/// build is the toolkit -- headers plus libcudart and libcublas.
///
/// Set VX_DISABLE_CUDA to build the host shim on a machine that has CUDA.
fn cuda_root() -> Option<(PathBuf, PathBuf)> {
    if env::var_os("VX_DISABLE_CUDA").is_some() {
        return None;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(dir) = env::var_os(var) {
            candidates.push(PathBuf::from(dir));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/usr"));

    for root in candidates {
        if !root.join("include/cuda_runtime.h").exists() {
            continue;
        }
        for lib in ["lib64", "lib/x86_64-linux-gnu", "lib"] {
            let libdir = root.join(lib);
            if libdir.join("libcudart.so").exists() {
                return Some((root, libdir));
            }
        }
    }

    None
}

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

    // The fleet headers, and the two the dispatch plan is built from. Every
    // backend compiles these -- npu_dispatch.mm as much as cuda_dispatch.cpp --
    // so they are listed here, above the platform split, rather than inside one
    // arm of it.
    //
    // They were listed only in the `else`. On macOS that made editing
    // vx_remote_client.h rebuild nothing: cargo reported "Finished" in a tenth
    // of a second, the JIT linked the previous dispatch library, and the change
    // appeared to have no effect. Which is an hour spent debugging a binary
    // that does not contain the fix, with no sign of it anywhere -- and it
    // happened twice, because the first fix was written into whichever branch
    // was open at the time (#348).
    println!("cargo:rerun-if-changed=runtime/vx_remote_routing.h");
    println!("cargo:rerun-if-changed=runtime/vx_remote_client.h");
    println!("cargo:rerun-if-changed=runtime/vx_remote_region.h");
    println!("cargo:rerun-if-changed=runtime/vx_manifest.h");
    println!("cargo:rerun-if-changed=runtime/vx_transport.h");
    println!("cargo:rerun-if-changed=runtime/vx_wire.h");
    println!("cargo:rerun-if-changed=runtime/vx_agent.h");
    println!("cargo:rerun-if-changed=runtime/vx_device_pool.h");
    println!("cargo:rerun-if-changed=runtime/vx_dispatch_plan.h");
    println!("cargo:rerun-if-changed=runtime/vx_host_call.h");
    println!("cargo:rerun-if-changed=include/vx_hardware_runtime.h");

    // Check if we're on macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rerun-if-changed=runtime/npu_dispatch.mm");
        println!("cargo:rerun-if-changed=runtime/npu_dispatch.h");
        println!("cargo:rerun-if-changed=scripts/generate_ane_primitives.py");

        let out_dir = env::var("OUT_DIR").unwrap();
        let obj_path = PathBuf::from(&out_dir).join("npu_dispatch.o");
        let lib_path = PathBuf::from(&out_dir).join("libnpu_dispatch.a");

        // --- Automate ANE Primitive Generation ---
        println!("cargo:warning=Building ANE primitive models via CoreML...");

        // The interpreter has to be one that can actually *build* a model, which
        // the first `python3` on PATH generally cannot: this checkout keeps
        // coremltools in its own venv. Picking by name rather than by capability
        // meant the models were never generated on a machine provisioned for
        // them, and the warning was the same one a machine with no coremltools at
        // all prints.
        //
        // The probe imports `libmilstoragepython`, not just `coremltools`.
        // coremltools is part pure Python and part compiled extension, and only
        // the pure half installs on an interpreter it ships no wheel for -- so
        // `import coremltools` succeeds on a Python too new for it and the model
        // build then dies much later with `RuntimeError: BlobWriter not loaded`.
        // Probing the half that is missing is what makes the fallback work.
        let python = ["VX_PYTHON"]
            .iter()
            .filter_map(|k| env::var(k).ok())
            .chain(
                [
                    "venv/bin/python3",
                    "python3",
                    "python3.13",
                    "python3.12",
                    "python3.11",
                ]
                .iter()
                .map(|s| s.to_string()),
            )
            .find(|p| {
                Command::new(p)
                    .args(["-c", "import coremltools, coremltools.libmilstoragepython"])
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            });

        match &python {
            Some(p) => println!("cargo:warning=ANE models: generating with {p}"),
            None => println!(
                "cargo:warning=No interpreter can build the ANE models: none of $VX_PYTHON, \
                 venv/bin/python3, python3, python3.13, python3.12 or python3.11 has a working \
                 coremltools (the compiled half, libmilstoragepython, is what is checked). \
                 Dispatch will use the CPU shim. Set VX_PYTHON to one that does."
            ),
        }

        let py_status = python.as_ref().map(|p| {
            Command::new(p)
                .args([
                    "scripts/generate_ane_primitives.py",
                    "--out-dir",
                    &out_dir,
                    "--dim",
                    "4",
                ])
                .status()
        });

        if let Some(Ok(status)) = py_status {
            if status.success() {
                // Compile the .mlpackage into .mlmodelc
                // The fp16 512x512 is the one the Neural Engine will actually take:
                // CoreML prefers the CPU for every fp32 matmul at every size, and
                // for fp16 below 512. The 4x4 pair stays for the affine path and
                // the tests written to it.
                for model_name in &["matmul_4x4", "affine_4", "matmul_512x512_fp16"] {
                    let pkg_path =
                        PathBuf::from(&out_dir).join(format!("{}.mlpackage", model_name));
                    let modelc_path =
                        PathBuf::from(&out_dir).join(format!("{}.mlmodelc", model_name));

                    // coremlc compile <pkg> <out_dir>
                    //
                    // `coremlc` ships with Xcode and not with the Command Line
                    // Tools, so `xcrun` cannot find it when `xcode-select` points
                    // at the CLT -- which is the default on a machine that has
                    // both. Pointing DEVELOPER_DIR at Xcode for this one call
                    // fixes that without `sudo xcode-select -s`, which is not a
                    // build script's business.
                    let mut coremlc = Command::new("xcrun");
                    if Command::new("xcrun")
                        .args(["--find", "coremlc"])
                        .output()
                        .map(|o| !o.status.success())
                        .unwrap_or(true)
                    {
                        let xcode = "/Applications/Xcode.app/Contents/Developer";
                        if PathBuf::from(xcode).exists() {
                            coremlc.env("DEVELOPER_DIR", xcode);
                        }
                    }
                    let coremlc_status = coremlc
                        .args(["coremlc", "compile", pkg_path.to_str().unwrap(), &out_dir])
                        .status();

                    if let Ok(c_status) = coremlc_status {
                        if c_status.success() {
                            // Copy to project root so tests/runtime can easily find them
                            let root_dir = env::current_dir().unwrap();
                            let target_modelc = root_dir.join(format!("{}.mlmodelc", model_name));
                            // Delete old one if exists
                            let _ = std::fs::remove_dir_all(&target_modelc);
                            let cp_status = Command::new("cp")
                                .args([
                                    "-R",
                                    modelc_path.to_str().unwrap(),
                                    target_modelc.to_str().unwrap(),
                                ])
                                .status();

                            if cp_status.is_ok() && cp_status.unwrap().success() {
                                println!(
                                    "cargo:warning=Successfully compiled {}.mlmodelc",
                                    model_name
                                );
                            }
                        } else {
                            println!(
                                "cargo:warning=Failed to compile {} with coremlc",
                                model_name
                            );
                        }
                    }
                }
            } else {
                println!("cargo:warning=Python script failed to generate ANE models.");
            }
        } else if python.is_some() {
            println!(
                "cargo:warning=Failed to invoke the ANE model generator; \
                 dispatch will use the CPU shim"
            );
        }

        // Determine compiler and flags
        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
        let cxxflags_env =
            env::var("CXXFLAGS").unwrap_or_else(|_| "-O3 -Wno-deprecated-declarations".to_string());
        let mut cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();
        // Fall back to running the outlined kernel on the CPU (via libffi) when the ANE/CoreML
        // path is unavailable -- no accelerator, or the CoreML models were not built (no
        // coremltools/coremlc). Without this the dispatcher aborts on any unhandled kernel.
        cxxflags.push("-DVX_ENABLE_CPU_FALLBACK");

        // Compile the Objective-C++ runtime file for AOT (Static Archive)
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

        // --- Compile the Objective-C++ runtime file for JIT (Shared Library) ---
        let lib_shared_path = PathBuf::from(&out_dir).join("libnpu_shared.dylib");
        let mut clang_shared_cmd = Command::new(&cxx);
        clang_shared_cmd.args([
            "-shared",
            "-fPIC",
            "-fobjc-arc",
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
            "-lffi",
            "-o",
            lib_shared_path.to_str().unwrap(),
        ]);
        clang_shared_cmd.args(&cxxflags);

        let status = clang_shared_cmd
            .status()
            .unwrap_or_else(|_| panic!("Failed to execute clang++ for shared library"));

        assert!(
            status.success(),
            "clang++ compilation failed for shared library"
        );

        // Pass the path to the shared library to the Rust compiler
        println!(
            "cargo:rustc-env=NPU_SHARED_LIB_PATH={}",
            lib_shared_path.display()
        );

        // Tell cargo to link against the generated static library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=npu_dispatch");

        // Link required Apple frameworks and C++ standard library
        println!("cargo:rustc-link-lib=dylib=c++");
        // libffi: used by vx_plugin_dispatch_async to call JIT kernels per the C ABI
        println!("cargo:rustc-link-lib=dylib=ffi");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=CoreML");
    } else {
        // A program containing a non-CPU `spawn on` needs a provider for the
        // vx_plugin_* ABI or it will not link, so one is always built. Which
        // one depends on what the machine has: the CUDA backend where a toolkit
        // is installed (runtime/cuda_dispatch.cpp), and otherwise the portable
        // shim that runs outlined kernels on the CPU through libffi
        // (runtime/host_dispatch.cpp). Both fall back to the host for anything
        // they cannot route, so the choice affects speed, not results.
        println!("cargo:rerun-if-changed=runtime/host_dispatch.cpp");
        println!("cargo:rerun-if-changed=runtime/host_dispatch_common.h");
        println!("cargo:rerun-if-changed=runtime/x86_dispatch.cpp");
        println!("cargo:rerun-if-changed=runtime/arm64_dispatch.cpp");
        println!("cargo:rerun-if-changed=runtime/cuda_dispatch.cpp");

        // Which backend answers the plugin ABI. CUDA wins where a toolkit is
        // installed, since it is the only one of these that can reach an
        // accelerator; otherwise the target's own CPU backend, and the portable
        // one for an architecture nobody has looked at yet.
        //
        // Every target gets a backend, which is what lets the compiler emit
        // allocate/transfer/free unconditionally and never name a vendor.
        let cuda = cuda_root();
        let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let (source, lib_stem) = match (&cuda, arch.as_str()) {
            (Some(_), _) => ("runtime/cuda_dispatch.cpp", "vx_cuda_dispatch"),
            (None, "x86_64") => ("runtime/x86_dispatch.cpp", "vx_x86_dispatch"),
            (None, "aarch64") => ("runtime/arm64_dispatch.cpp", "vx_arm64_dispatch"),
            (None, _) => ("runtime/host_dispatch.cpp", "vx_host_dispatch"),
        };

        let out_dir = env::var("OUT_DIR").unwrap();
        let lib_shared_path = PathBuf::from(&out_dir).join(format!(
            "{}{}{}",
            std::env::consts::DLL_PREFIX,
            lib_stem,
            std::env::consts::DLL_SUFFIX
        ));

        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
        let cxxflags_env = env::var("CXXFLAGS").unwrap_or_else(|_| "-O3".to_string());
        let cxxflags: Vec<&str> = cxxflags_env.split_whitespace().collect();

        let mut clang_shared_cmd = Command::new(&cxx);
        clang_shared_cmd.args([
            "-shared",
            "-fPIC",
            source,
            "-lffi",
            "-o",
            lib_shared_path.to_str().unwrap(),
        ]);
        clang_shared_cmd.args(&cxxflags);

        if let Some((root, libdir)) = &cuda {
            // The rpath matters because this library is loaded by the programs
            // vxc links, not by vxc itself: without it a program would have to
            // be run with the toolkit on LD_LIBRARY_PATH.
            clang_shared_cmd.arg(format!("-I{}/include", root.display()));
            clang_shared_cmd.arg(format!("-L{}", libdir.display()));
            clang_shared_cmd.arg(format!("-Wl,-rpath,{}", libdir.display()));
            // `-lcuda` is the driver API, for loading a device image the
            // compiler emitted (#251). It is linked from the toolkit's stubs
            // and resolved at run time against the real driver -- which is why
            // the stub directory is on -L and deliberately not on the rpath: an
            // rpath to it would find the stub first, and the stub's entry
            // points do nothing.
            clang_shared_cmd.arg(format!("-L{}/stubs", libdir.display()));
            clang_shared_cmd.args(["-lcudart", "-lcublas", "-lcuda"]);
        }

        let status = clang_shared_cmd
            .status()
            .unwrap_or_else(|_| panic!("Failed to execute {} for {}", cxx, source));
        assert!(status.success(), "{} failed to build {}", cxx, source);

        println!(
            "cargo:rustc-env=NPU_SHARED_LIB_PATH={}",
            lib_shared_path.display()
        );
        match &cuda {
            Some((root, _)) => println!(
                "cargo:warning=Built the CUDA dispatch backend against {}; \
                 recognised matmuls run on the GPU, everything else on the CPU.",
                root.display()
            ),
            None => println!(
                "cargo:warning=Built the {} dispatch backend; kernels run on the CPU via libffi.",
                source
            ),
        }
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
