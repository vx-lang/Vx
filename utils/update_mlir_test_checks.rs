use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
  let args: Vec<String> = env::args().collect();
  if args.len() < 2 {
    eprintln!(
    "Usage: rustc utils/update_mlir_test_checks.rs && ./update_mlir_test_checks <test_files...>"
    );
    std::process::exit(1);
  }

  let mut project_root = env::current_dir().unwrap();
  if !project_root.join("target/debug/vxc").exists() {
    // Fallback or handle differently, but usually it's run from project root
    eprintln!("Could not find target/debug/vxc. Are you running from the project root?");
  }

  let vxc_path = project_root.join("target/debug/vxc");

  for file_path in args.iter().skip(1) {
    println!("Updating {}...", file_path);

    let source = match fs::read_to_string(file_path) {
      Ok(s) => s,
      Err(e) => {
        eprintln!("Failed to read {}: {}", file_path, e);
        continue;
      }
    };

    // Run vxc --emit-mlir
    let output = Command::new(&vxc_path)
    .arg("--emit-mlir")
    .arg(file_path)
    .output()
    .expect("Failed to execute vxc");

    if !output.status.success() {
      let stderr = String::from_utf8_lossy(&output.stderr);
      eprintln!("vxc failed on {}:
{}", file_path, stderr);
      continue;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Extract mlir, skipping the debug prints before the module
    let mut mlir_lines = Vec::new();
    let mut in_module = false;
    for line in stdout.lines() {
      if line.starts_with("module {") {
        in_module = true;
      }
      if in_module {
        // To avoid overly brittle SSA matching, we just write the exact line.
        // Note: FileCheck will exact-match these lines. 
        mlir_lines.push(format!("// CHECK: {}", line));
      }
    }

    if mlir_lines.is_empty() {
      eprintln!("Warning: No MLIR module output found for {}", file_path);
      continue;
    }

    let mut new_lines = Vec::new();
    let mut insert_idx = 0;

    for line in source.lines() {
      if line.trim_start().starts_with("// CHECK:") {
        continue;
      }
      new_lines.push(line.to_string());
      if line.trim_start().starts_with("// RUN:") {
        insert_idx = new_lines.len();
      }
    }

    // Insert new check lines at insert_idx
    for check_line in mlir_lines.into_iter().rev() {
      new_lines.insert(insert_idx, check_line);
    }

    // Write back
    let new_content = new_lines.join("
") + "
";
    if let Err(e) = fs::write(file_path, new_content) {
      eprintln!("Failed to write {}: {}", file_path, e);
    } else {
      println!("Updated {}", file_path);
    }
  }
}
