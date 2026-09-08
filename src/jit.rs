//===- jit.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Where `libvx_std_core` sits for the profile this compiler was built with.
///
/// Its absence is worth its own message: `cargo test` never emits the shared library, because a
/// dependency edge only asks for an rlib, so a checkout that has only been tested reaches the
/// The directory LLVM installs its runtime libraries in.
///
/// Resolved through PATH by default (config.local puts the intended LLVM first), matching how
/// build.rs locates the toolchain. Hardcoding a Homebrew prefix made this unusable on Linux.
pub fn llvm_libdir() -> Result<String, String> {
    let llvm_config_path =
        std::env::var("LLVM_CONFIG_PATH").unwrap_or_else(|_| "llvm-config".to_string());
    let out = Command::new(&llvm_config_path)
        .arg("--libdir")
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "Compiler toolchain error: '{llvm_config_path}' was not found. Please ensure LLVM is installed and in your PATH, or set LLVM_CONFIG_PATH."
                )
            } else {
                format!("Failed to run {llvm_config_path}: {e}")
            }
        })?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The shared libraries a compiled program needs at load time: MLIR's two runner utils, the Vx
/// standard-library core, and the dispatch plugin.
///
/// One resolution shared by the JIT and by `--action emit-obj`. The object path used to build its
/// own list from BARE FILENAMES, which MLIR resolves against the process CWD, so it looked for
/// `libmlir_runner_utils` in the repo root and silently produced no object; and it looked for the
/// dispatch plugin under `target/jit/`, a directory the build never writes.
pub fn shared_library_paths() -> Result<Vec<String>, String> {
    let libdir = llvm_libdir()?;
    let mut libs = vec![
        format!(
            "{libdir}/libmlir_c_runner_utils{}",
            std::env::consts::DLL_SUFFIX
        ),
        format!(
            "{libdir}/libmlir_runner_utils{}",
            std::env::consts::DLL_SUFFIX
        ),
        runtime_library_path()?,
    ];
    let npu = std::env::var("VX_DISPATCH_LIB")
        .unwrap_or_else(|_| std::env!("NPU_SHARED_LIB_PATH").to_string());
    if !npu.is_empty() && std::path::Path::new(&npu).exists() {
        libs.push(npu);
    }
    // Supplies the plain `printMemrefBF16`/`printMemrefF16` MLIR exports only packed.
    //
    // Overridable at run time for the same reason the dispatch backend is: the compile-time value
    // is a path into the build machine's OUT_DIR, which an installed toolchain ships its own copy
    // of somewhere else. Without an override the shims are simply never found, and half-precision
    // memrefs print nothing, with no diagnostic to explain why.
    let shims = std::env::var("VX_MLIR_SHIMS")
        .unwrap_or_else(|_| std::env!("VX_MLIR_SHIMS_PATH").to_string());
    if !shims.is_empty() && std::path::Path::new(&shims).exists() {
        libs.push(shims);
    }
    Ok(libs)
}

/// linker without it and clang reports a missing file with no hint of which build produces it.
pub fn runtime_library_path() -> Result<String, String> {
    let file_name = format!(
        "{}vx_std_core{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );

    // Three places, most specific first. The last one is the source checkout, which is where
    // this used to look and nowhere else -- a path relative to the process working directory,
    // so an installed compiler run from a user's own project could never find its own runtime.
    let mut tried = Vec::new();

    // 1. Named outright. The escape hatch for a layout neither of the others describes.
    if let Ok(dir) = std::env::var("VX_RUNTIME_LIB_DIR") {
        let path = std::path::PathBuf::from(dir).join(&file_name);
        if path.exists() {
            return Ok(path.to_string_lossy().into_owned());
        }
        tried.push(path);
    }

    // 2. Next to the compiler, as an installed toolchain lays it out: `bin/vxc` and
    // `lib/libvx_std_core.*` under one prefix. Found without any environment variable, so a
    // downloaded toolchain works on the first run.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            for relative in ["../lib", "."] {
                let path = bin_dir.join(relative).join(&file_name);
                if path.exists() {
                    return Ok(path.to_string_lossy().into_owned());
                }
                tried.push(path);
            }
        }
    }

    // 3. The cargo target directory of a source checkout, relative to the working directory.
    let profile_dir = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let current_dir = std::env::current_dir().map_err(|e| e.to_string())?;
    let path = current_dir
        .join("target")
        .join(profile_dir)
        .join(&file_name);
    if path.exists() {
        return Ok(path.to_string_lossy().into_owned());
    }
    tried.push(path);

    let looked = tried
        .iter()
        .map(|p| format!("  {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!(
        "the Vx runtime library is missing. Looked in:\n{looked}\nIn a source checkout, build it \
         with `cargo build`, which now covers stdlib/rust_core. `cargo test` alone never produces \
         it. In an installed toolchain, the library belongs in `lib/` beside the compiler's \
         `bin/`, or point VX_RUNTIME_LIB_DIR at it."
    ))
}

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

/// What the compiled program wrote, with its two streams kept apart.
///
/// They used to be concatenated into one string before anyone saw them, which is lossy in a way
/// that matters: `vxc --run` printed the whole thing to its own stdout, so a diagnostic the program
/// wrote to stderr landed *after* everything it wrote to stdout, and a caller taking the last line
/// of stdout got the diagnostic instead of the answer. That is what broke two `remote_client_test`
/// cases on a Linux box with the CUDA toolkit and no device: the dispatch backend's "no CUDA
/// device" notice trailed the real output.
///
/// The test harnesses do want both streams -- backend fixtures assert on `Hello Stderr!` and on
/// panic text -- so the streams are returned rather than one being dropped. `execute_mlir` keeps
/// the old concatenated form for those callers; the driver uses this and routes each stream to the
/// matching one of its own.
pub struct ProgramOutput {
    pub stdout: String,
    pub stderr: String,
}

/// Run a program and return its streams concatenated, stdout first.
///
/// Kept for callers that only want "what did it print" and do not care which stream it came from,
/// notably the fixture harnesses whose EXPECT lines match against either.
pub fn execute_mlir(
    mlir_src: &str,
    program_args: Vec<String>,
    opt_level: u8,
    disable_llvm_optimizations: bool,
) -> Result<String, String> {
    let out = execute_mlir_streams(
        mlir_src,
        program_args,
        opt_level,
        disable_llvm_optimizations,
    )?;
    Ok(format!("{}{}", out.stdout, out.stderr))
}

pub fn execute_mlir_streams(
    mlir_src: &str,
    program_args: Vec<String>,
    opt_level: u8,
    disable_llvm_optimizations: bool,
) -> Result<ProgramOutput, String> {
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
        &runtime_library_path()?,
    ]);

    // The plain half-precision printers MLIR exports only packed. Linked here as well as handed to
    // the execution engine, because this path builds a native executable with clang rather than
    // loading shared libraries into a JIT.
    let shims = std::env!("VX_MLIR_SHIMS_PATH");
    if !shims.is_empty() && std::path::Path::new(shims).exists() {
        clang_cmd.arg(shims);
    }

    // The dispatch runtime provides vx_plugin_dispatch_async, which any program
    // containing a non-CPU `spawn on` calls. macOS gets the ANE/AMX backend
    // (runtime/npu_dispatch.mm); other platforms get the portable host shim
    // (runtime/host_dispatch.cpp). Both call outlined kernels through libffi.
    // Existence, not just non-emptiness. The default is an absolute path into the OUT_DIR of the
    // machine that BUILT this compiler, so in an installed toolchain it names a file that was
    // never shipped -- and handing clang a path that is not there fails the link outright with
    // "no such file or directory", rather than falling back to host execution. The sibling
    // resolution in `shared_library_paths` has always checked this; the link path had not.
    if !lib_npu.is_empty() && std::path::Path::new(&lib_npu).exists() {
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
        // A process killed by a signal has no exit code, and reporting the
        // `unwrap_or` fallback as if it were one turned every crash into
        // "exited with code -1" -- which reads like the program returned -1 and
        // says nothing about a segfault, an abort or an out-of-memory kill.
        //
        // Only the signal case is new wording. Both exit-code messages are
        // byte-identical to what they were, because callers match on them: the
        // differential harness parses the number off the end of the error, and
        // a monomorphization test looks for "exited with code: 8".
        let killed = killed_by_signal(&exe_out.status);
        match &killed {
            Some(how) => println!("[JIT] Program {}", how),
            None => println!(
                "[JIT] Program exited with code: {}",
                exe_out.status.code().unwrap_or(-1)
            ),
        }
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
        return Err(match killed {
            Some(how) => format!("Program {how}"),
            None => format!(
                "Program exited with non-zero code: {}",
                exe_out.status.code().unwrap_or(-1)
            ),
        });
    }

    Ok(ProgramOutput {
        stdout: String::from_utf8_lossy(&exe_out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&exe_out.stderr).into_owned(),
    })
}

/// The signal that killed a run, in words, or `None` for an ordinary exit.
///
/// A signal is not an exit code: `ExitStatus::code` answers `None` for one, so
/// the usual `unwrap_or(-1)` turns every crash into "exited with code -1" --
/// which reads as though the program returned -1 and hides the difference
/// between a wrong answer and a segfault.
fn killed_by_signal(status: &std::process::ExitStatus) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                4 => " (SIGILL)",
                6 => " (SIGABRT, often a failed assert)",
                8 => " (SIGFPE)",
                9 => " (SIGKILL, killed by the system rather than crashing)",
                10 => " (SIGBUS)",
                11 => " (SIGSEGV, a bad memory access; a stack overflow reaches here too)",
                _ => "",
            };
            return Some(format!("was killed by signal {sig}{name}"));
        }
    }
    let _ = status;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the shared library every test that runs a program fails, and each one reports
    /// only a clang error naming a file it has never heard of. This one names the cause.
    #[test]
    fn the_runtime_library_the_jit_links_is_present() {
        if let Err(why) = runtime_library_path() {
            panic!("{why}");
        }
    }
}
