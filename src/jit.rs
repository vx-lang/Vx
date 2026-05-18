use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

static JIT_COUNTER: AtomicUsize = AtomicUsize::new(0);
static COMPILE_NPU_ONCE: Once = Once::new();

pub fn execute_mlir(mlir_src: &str) -> Result<String, String> {
    // Ensure target/jit directory exists
    let jit_dir = std::path::Path::new("target/jit");
    if !jit_dir.exists() {
        std::fs::create_dir_all(jit_dir).map_err(|e| e.to_string())?;
    }

    let pid = std::process::id();
    let counter = JIT_COUNTER.fetch_add(1, Ordering::SeqCst);
    let uid = format!("{}_{}", pid, counter);

    let temp_mlir = format!("target/jit/temp_{}.mlir", uid);
    let temp_llvm = format!("target/jit/temp_llvm_{}.mlir", uid);
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
    } else {
        return Err(
            "Vx JIT execution currently requires macOS Apple Silicon for hardware dispatch."
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
            "--convert-vector-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            &temp_mlir,
        ])
        .output()
        .map_err(|e| e.to_string())?;

    if !mlir_opt_out.status.success() {
        let err_str = String::from_utf8_lossy(&mlir_opt_out.stderr);
        return Err(format!("mlir-opt failed:\n{}", err_str));
    }

    let mut lowered_mlir_file = File::create(&temp_llvm).map_err(|e| e.to_string())?;
    lowered_mlir_file
        .write_all(&mlir_opt_out.stdout)
        .map_err(|e| e.to_string())?;

    println!("[JIT] Translating to LLVM IR...");
    let mlir_translate_out = Command::new("/opt/homebrew/opt/llvm/bin/mlir-translate")
        .args(["--mlir-to-llvmir", &temp_llvm])
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

    println!("[JIT] Executing via LLI...");
    let current_dir = std::env::current_dir().unwrap();
    let lli_out = Command::new("/opt/homebrew/opt/llvm/bin/lli")
        .args([
            &format!("--load={}", lib_npu),
            "--load=/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
            "--load=/opt/homebrew/opt/llvm/lib/libmlir_runner_utils.dylib",
            &format!(
                "--load={}/target/debug/libvx_std_core.dylib",
                current_dir.display()
            ),
            &temp_ll,
        ])
        .output()
        .map_err(|e| e.to_string())?;

    if !lli_out.status.success() {
        let err_str = String::from_utf8_lossy(&lli_out.stderr);
        return Err(format!("lli execution failed:\n{}", err_str));
    }

    let output_str = format!(
        "{}{}",
        String::from_utf8_lossy(&lli_out.stdout),
        String::from_utf8_lossy(&lli_out.stderr)
    );

    Ok(output_str)
}
