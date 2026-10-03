// This test expands the same raw-pointer FFI macro as the production library.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

// The Vx declaration of file close must agree with the Rust function's C ABI.
include!("../stdlib/rust_core/src/ffi/macros.rs");
instantiate_file_ffi!();

#[test]
fn file_close_has_no_return_value_on_both_sides_of_the_ffi() -> Result<(), String> {
    let _: extern "C" fn(*mut std::ffi::c_void) = vx_file_drop;

    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let program = dir.path().join("close.vx");
    std::fs::write(
        &program,
        "import std::fs;\n\
         fn main() -> i32 {\n\
           unsafe {\n\
             let mut file = File::open(\"/tmp/vx-close-abi\", 0);\n\
             file_drop(&mut file);\n\
           }\n\
           return 0;\n\
         }\n",
    )
    .map_err(|error| format!("write Vx program: {error}"))?;

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&program)
        .args(["--action", "emit-mlir"])
        .output()
        .map_err(|error| format!("run Vx compiler: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Vx program did not compile:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let mlir = String::from_utf8_lossy(&output.stdout);
    let declaration = mlir
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("func.func private @vx_file_drop("));
    let expected = Some("func.func private @vx_file_drop(!llvm.ptr)");
    if declaration != expected {
        return Err(format!(
            "Vx's file-close declaration must match Rust's void return: \
             expected {expected:?}, got {declaration:?}"
        ));
    }
    Ok(())
}
