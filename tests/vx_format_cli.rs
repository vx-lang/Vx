use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const SOURCE: &str = "fn main() {\nlet value = 1;\n}\n";

fn project(parent: &Path, name: &str, config: Option<&str>) -> std::path::PathBuf {
    let directory = parent.join(name);
    fs::create_dir_all(directory.join(".git")).unwrap();
    fs::write(directory.join("main.vx"), SOURCE).unwrap();
    if let Some(config) = config {
        fs::write(directory.join("vx-format.toml"), config).unwrap();
    }
    directory
}

fn formatter(directory: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vx-format"))
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn prints_defaults_without_reading_project_config_or_changing_files() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", Some("invalid TOML ["));
    let output = formatter(&project, &["--print-config", "default"]);
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text, "[format]\nindent = 2\nfn_signature_width = 100\n");
    assert_eq!(
        vx_config::parse_format_config(Path::new("vx-format.toml"), &text).unwrap(),
        vx_config::FormatConfig::default()
    );
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);
    assert_eq!(
        fs::read_to_string(project.join("vx-format.toml")).unwrap(),
        "invalid TOML ["
    );
}

#[test]
fn rejects_unsupported_print_modes_and_conflicting_arguments() {
    let directory = tempfile::tempdir().unwrap();
    for args in [
        vec![],
        vec!["--print-config"],
        vec!["--print-config", "unknown"],
        vec!["--print-config", "default", "--indent", "4"],
        vec!["--print-config", "default", "--config", "vx-format.toml"],
        vec!["--print-config", "default", "main.vx"],
        vec!["--print-config", "default", "--check"],
    ] {
        let output = formatter(directory.path(), &args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
    }
}

#[test]
fn check_reports_each_changed_file_without_writing() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    fs::write(project.join("other.vx"), SOURCE).unwrap();
    let output = formatter(&project, &["--check", "main.vx", "other.vx"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report = String::from_utf8(output.stdout).unwrap();
    let expected = |name| {
        format!("--- {name}\n+++ {name} (formatted)\n@@ -1,3 +1,3 @@\n fn main() {{\n-let value = 1;\n+  let value = 1;\n }}\n")
    };
    let first = expected("main.vx");
    let second = expected("other.vx");
    assert!(
        report == format!("{first}{second}") || report == format!("{second}{first}"),
        "{report}"
    );
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);
    assert_eq!(
        fs::read_to_string(project.join("other.vx")).unwrap(),
        SOURCE
    );
}

#[test]
fn check_shows_unicode_and_missing_final_newlines() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    let source = "fn message() -> i32 {\nreturn \"café\";\n}";
    fs::write(project.join("main.vx"), source).unwrap();
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "--- main.vx\n+++ main.vx (formatted)\n@@ -1,3 +1,3 @@\n fn message() -> i32 {\n-return \"café\";\n+  return \"café\";\n }\n\\ No newline at end of file\n"
    );
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), source);
}

#[test]
fn signature_width_controls_wrapping_and_check_mode() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(
        directory.path(),
        "project",
        Some("[format]\nindent = 4\nfn_signature_width = 39\n"),
    );
    let source = "fn add(first: i32, second: i32) -> i32 {\n  first + second\n}\n";
    fs::write(project.join("main.vx"), source).unwrap();
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let diff = String::from_utf8(output.stdout).unwrap();
    assert!(diff.contains("@@ -1,3 +1,6 @@"), "{diff}");
    assert!(diff.contains("+    second: i32\n+) -> i32 {\n"), "{diff}");
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), source);
    let output = formatter(&project, &["main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        fs::read_to_string(project.join("main.vx")).unwrap(),
        "fn add(\n    first: i32,\n    second: i32\n) -> i32 {\n    first + second\n}\n"
    );
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
}

#[test]
fn check_uses_configuration_and_overrides() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", Some("[format]\nindent = 0\n"));
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");

    let output = formatter(&project, &["--check", "--indent", "4", "main.vx"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);

    let output = formatter(&project, &["--indent", "4", "main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let formatted = fs::read_to_string(project.join("main.vx")).unwrap();
    let output = formatter(&project, &["--check", "--indent", "4", "main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert_eq!(
        fs::read_to_string(project.join("main.vx")).unwrap(),
        formatted
    );
}

#[test]
fn check_honors_file_and_region_exclusions() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    let disabled_file = "//! vx-format: OFF\nfn   main(){let x=1;}\n";
    let disabled_region =
        "// vx-format-begin: OFF\nconst   value : i32 = 1;\n// vx-format-end: OFF\n";
    fs::write(project.join("main.vx"), disabled_file).unwrap();
    fs::write(project.join("region.vx"), disabled_region).unwrap();
    let output = formatter(&project, &["--check", "main.vx", "region.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        fs::read_to_string(project.join("main.vx")).unwrap(),
        disabled_file
    );
    assert_eq!(
        fs::read_to_string(project.join("region.vx")).unwrap(),
        disabled_region
    );
}

#[test]
fn check_distinguishes_errors_from_formatting_changes() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", Some("invalid TOML ["));
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("vx-format.toml"));
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);

    fs::remove_file(project.join("vx-format.toml")).unwrap();
    let output = formatter(&project, &["--check", "main.vx", "missing.vx"]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing.vx"));
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);
}

#[test]
fn uses_project_indentation_and_explicit_overrides() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", Some("[format]\nindent = 4\n"));
    let output = formatter(&project, &["main.vx"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(project.join("main.vx")).unwrap(),
        "fn main() {\n    let value = 1;\n}\n"
    );

    let output = formatter(&project, &["--indent", "0", "main.vx"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), SOURCE);
}

#[test]
fn defaults_without_config_and_ignores_lint_config() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    fs::write(project.join("vx-lint.toml"), "not valid TOML [").unwrap();
    let output = formatter(&project, &["main.vx"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(project.join("main.vx")).unwrap(),
        "fn main() {\n  let value = 1;\n}\n"
    );
}

#[test]
fn resolves_each_project_and_explicit_config_paths() {
    let directory = tempfile::tempdir().unwrap();
    let first = project(directory.path(), "first", Some("[format]\nindent = 4\n"));
    let second = project(directory.path(), "second", Some("[format]\nindent = 0\n"));
    let output = formatter(directory.path(), &["first/main.vx", "second/main.vx"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(first.join("main.vx")).unwrap(),
        "fn main() {\n    let value = 1;\n}\n"
    );
    assert_eq!(fs::read_to_string(second.join("main.vx")).unwrap(), SOURCE);

    let output = formatter(
        directory.path(),
        &["--config", "first/vx-format.toml", "second/main.vx"],
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(second.join("main.vx")).unwrap(),
        "fn main() {\n    let value = 1;\n}\n"
    );
}

#[test]
fn configuration_errors_prevent_writes_to_every_input() {
    let directory = tempfile::tempdir().unwrap();
    let first = project(directory.path(), "first", Some("[format]\nindent = 4\n"));
    let second = project(directory.path(), "second", None);
    for invalid in [
        "",
        "[format]",
        "[format]\nindent = -1",
        "[format]\nindent = 2\nwidth = 80",
        "[",
    ] {
        fs::write(second.join("vx-format.toml"), invalid).unwrap();
        let output = formatter(
            directory.path(),
            &["--indent", "4", "first/main.vx", "second/main.vx"],
        );
        assert!(!output.status.success(), "accepted {invalid:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("vx-format.toml"));
        assert_eq!(fs::read_to_string(first.join("main.vx")).unwrap(), SOURCE);
        assert_eq!(fs::read_to_string(second.join("main.vx")).unwrap(), SOURCE);
    }
    let output = formatter(&first, &["--config", "missing.toml", "main.vx"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing.toml"));
    assert_eq!(fs::read_to_string(first.join("main.vx")).unwrap(), SOURCE);
}

#[test]
fn unsupported_syntax_warns_without_failing_or_writing() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    let source = "fn future(value: i32) -> i32 effects Read { value }\n";
    fs::write(project.join("main.vx"), source).unwrap();
    for args in [vec!["--check", "main.vx"], vec!["main.vx"]] {
        let output = formatter(&project, &args);
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
        let warnings = String::from_utf8(output.stderr).unwrap();
        assert!(warnings.contains("WARN: main.vx:1:"), "{warnings}");
        assert!(warnings.contains("preserve"), "{warnings}");
        assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), source);
    }
}

#[test]
fn warnings_do_not_hide_supported_differences_in_the_same_file() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    let source =
        "fn future() -> i32 {\nfuture_binding value;\n}\nfn known(value: i32) -> i32 {\nvalue\n}\n";
    fs::write(project.join("main.vx"), source).unwrap();
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("WARN: main.vx:1:"));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("--- main.vx\n+++ main.vx (formatted)\n")
    );
    assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), source);
    let output = formatter(&project, &["main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let formatted = fs::read_to_string(project.join("main.vx")).unwrap();
    assert!(formatted.starts_with("fn future() -> i32 {\nfuture_binding value;\n}\n"));
    assert!(
        formatted.ends_with("fn known(value: i32) -> i32 {\n  value\n}\n"),
        "{formatted}"
    );
    let output = formatter(&project, &["--check", "main.vx"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("WARN:"));
}

#[test]
fn exclusions_do_not_hide_diffs_in_check_mode() {
    let directory = tempfile::tempdir().unwrap();
    let project = project(directory.path(), "project", None);
    let excluded = "// vx-format-begin: OFF\nfuture_binding value;\n// vx-format-end: OFF\n";
    for source in [
        format!("{excluded}fn   known(value:i32)->i32 {{\nreturn value;\n}}\n"),
        format!("fn   known(value:i32)->i32 {{\n{excluded}return value;\n}}\n"),
    ] {
        fs::write(project.join("main.vx"), &source).unwrap();
        let output = formatter(&project, &["--check", "main.vx"]);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stderr.is_empty(), "{output:?}");
        let diff = String::from_utf8(output.stdout).unwrap();
        assert!(diff.contains("+fn known(value: i32) -> i32 {"), "{diff}");
        assert!(diff.contains("+  return value;"), "{diff}");
        assert_eq!(fs::read_to_string(project.join("main.vx")).unwrap(), source);
    }
}
