// A signed negative byte count must be rejected before it can reach Rust's usize parameter.
#[test]
fn file_read_rejects_a_negative_length_before_execution() -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let program = dir.path().join("negative_length.vx");
    std::fs::write(
        &program,
        "import std::fs;\n\
         extern {\n\
           fn malloc(size : i64) -> *mut u8;\n\
           fn free(ptr : *mut u8) -> void;\n\
         }\n\
         fn main() -> i32 {\n\
           unsafe {\n\
             let mut file = File::open(\"/tmp/vx-negative-file-length\", 0);\n\
             let buffer = malloc(1);\n\
             let negative : i64 = 0 - 1;\n\
             let _count = file.read(buffer, negative);\n\
             file_drop(&mut file);\n\
             free(buffer);\n\
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
    let mlir = String::from_utf8_lossy(&output.stdout);
    let read_call = mlir
        .lines()
        .map(str::trim)
        .find(|line| line.contains("call @File$read("))
        .unwrap_or("no file-read call in the compiler output");
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    if output.status.success()
        || !diagnostic.contains("Error[E3003]")
        || !diagnostic.contains("Expected u64, got i64")
    {
        return Err(format!(
            "Vx must reject an i64 file-read length as a u64 type mismatch before execution.\n\
             Compiler exit: {}\nCompiler diagnostic: {diagnostic}\nCompiler call: {read_call}",
            output.status
        ));
    }
    Ok(())
}

#[test]
fn file_write_rejects_a_negative_length_before_execution() -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let program = dir.path().join("negative_write_length.vx");
    std::fs::write(
        &program,
        "import std::fs;\n\
         fn main() -> i32 {\n\
           unsafe {\n\
             let mut file = File::open(\"/tmp/vx-negative-file-write\", 1);\n\
             let negative : i64 = 0 - 1;\n\
             let _count = file.write(0 as *const u8, negative);\n\
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
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    if output.status.success()
        || !diagnostic.contains("Error[E3003]")
        || !diagnostic.contains("Expected u64, got i64")
    {
        return Err(format!(
            "Vx must reject an i64 file-write length as a u64 type mismatch before execution.\n\
             Compiler exit: {}\nCompiler diagnostic: {diagnostic}",
            output.status
        ));
    }
    Ok(())
}
