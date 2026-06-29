use std::fs;
use std::path::Path;

#[test]
fn test_no_panics_or_asserts_in_tests() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut violations = Vec::new();

    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            // Do not lint the lint test itself!
            if file_name == "lint_test.rs" {
                continue;
            }

            let content = fs::read_to_string(&path).unwrap();
            for (i, line) in content.lines().enumerate() {
                let trimmed = line.trim();
                // Ignore commented-out lines
                if trimmed.starts_with("//") {
                    continue;
                }

                // Check for macros
                if trimmed.contains("panic!(")
                    || trimmed.contains("assert!(")
                    || trimmed.contains("assert_eq!(")
                    || trimmed.contains("assert_ne!(")
                {
                    violations.push(format!(
                        "{}:{}: prohibited macro usage: `{}`",
                        file_name,
                        i + 1,
                        trimmed
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "Found prohibited panic! or assert! macros in tests directory:\n{}",
            violations.join("\n")
        ));
    }

    Ok(())
}
