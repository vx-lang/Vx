use std::fs::File;
use std::io::Write;
use std::process::Command;

pub fn execute_mlir(mlir_src: &str) -> Result<String, String> {
    // Ensure target/jit directory exists
    let jit_dir = std::path::Path::new("target/jit");
    if !jit_dir.exists() {
        std::fs::create_dir_all(jit_dir).map_err(|e| e.to_string())?;
    }

    // 1. Write MLIR to temp file
    let mut mlir_file = File::create("target/jit/temp.mlir").map_err(|e| e.to_string())?;
    mlir_file
        .write_all(mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    println!("[JIT] Compiling C Runtime...");
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let cflags_env = std::env::var("CFLAGS").unwrap_or_else(|_| "-shared -fPIC".to_string());
    let cflags: Vec<&str> = cflags_env.split_whitespace().collect();

    let mut clang_cmd = Command::new(&cc);
    clang_cmd.args(&cflags);
    clang_cmd.args(["src/runtime/akar_rt.c", "-o", "target/jit/libakar_rt.dylib"]);

    let clang_status = clang_cmd.status().map_err(|e| e.to_string())?;

    if !clang_status.success() {
        return Err("Failed to compile C runtime".to_string());
    }

    if cfg!(target_os = "macos") {
        println!("[JIT] Compiling Objective-C++ NPU Dispatcher...");
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
            "-o",
            "target/jit/libnpu_dispatch.dylib",
        ]);

        let npu_status = cxx_cmd.status().map_err(|e| e.to_string())?;

        if !npu_status.success() {
            return Err("Failed to compile Objective-C++ NPU Dispatcher".to_string());
        }
    } else {
        return Err(
            "Akar JIT execution currently requires macOS Apple Silicon for hardware dispatch."
                .to_string(),
        );
    }

    println!("[JIT] Lowering to LLVM Dialect...");
    let mlir_opt_out = Command::new("/opt/homebrew/opt/llvm/bin/mlir-opt")
        .args([
            "--convert-scf-to-cf",
            "--expand-strided-metadata",
            "--lower-affine",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "target/jit/temp.mlir",
        ])
        .output()
        .map_err(|e| e.to_string())?;

    if !mlir_opt_out.status.success() {
        let err_str = String::from_utf8_lossy(&mlir_opt_out.stderr);
        return Err(format!("mlir-opt failed:\n{}", err_str));
    }

    let mut lowered_mlir_file =
        File::create("target/jit/temp_llvm.mlir").map_err(|e| e.to_string())?;
    lowered_mlir_file
        .write_all(&mlir_opt_out.stdout)
        .map_err(|e| e.to_string())?;

    println!("[JIT] Translating to LLVM IR...");
    let mlir_translate_out = Command::new("/opt/homebrew/opt/llvm/bin/mlir-translate")
        .args(["--mlir-to-llvmir", "target/jit/temp_llvm.mlir"])
        .output()
        .map_err(|e| e.to_string())?;

    if !mlir_translate_out.status.success() {
        let err_str = String::from_utf8_lossy(&mlir_translate_out.stderr);
        return Err(format!("mlir-translate failed:\n{}", err_str));
    }

    let mut llvmir_file = File::create("target/jit/temp.ll").map_err(|e| e.to_string())?;
    llvmir_file
        .write_all(&mlir_translate_out.stdout)
        .map_err(|e| e.to_string())?;

    println!("[JIT] Executing via LLI...");
    let lli_out = Command::new("/opt/homebrew/opt/llvm/bin/lli")
        .args([
            "--load=target/jit/libakar_rt.dylib",
            "--load=target/jit/libnpu_dispatch.dylib",
            "--load=/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
            "--load=/opt/homebrew/opt/llvm/lib/libmlir_runner_utils.dylib",
            "target/jit/temp.ll",
        ])
        .output()
        .map_err(|e| e.to_string())?;

    if !lli_out.status.success() {
        let err_str = String::from_utf8_lossy(&lli_out.stderr);
        return Err(format!("lli execution failed:\n{}", err_str));
    }

    let output_str = String::from_utf8_lossy(&lli_out.stdout);

    Ok(output_str.to_string())
}
