use anyhow::{ensure, Context, Result};
use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const SOURCE: &str = "fn main() {\nlet value = 1;\n}\n";

fn project(parent: &Path, name: &str, config: Option<&str>) -> Result<std::path::PathBuf> {
    let directory = parent.join(name);
    fs::create_dir_all(directory.join(".git"))?;
    fs::write(directory.join("main.vx"), SOURCE)?;
    if let Some(config) = config {
        fs::write(directory.join("vx-format.toml"), config)?;
    }
    Ok(directory)
}

fn formatter(directory: &Path, args: &[&str]) -> Result<Output> {
    Ok(Command::new(env!("CARGO_BIN_EXE_vx-format"))
        .current_dir(directory)
        .args(args)
        .output()?)
}

#[track_caller]
fn check_equal<T: Debug + PartialEq<U> + ?Sized, U: Debug + ?Sized>(
    actual: &T,
    expected: &U,
) -> Result<()> {
    let location = std::panic::Location::caller();
    ensure!(
        actual == expected,
        "{}:{}: expected {expected:?}, got {actual:?}",
        location.file(),
        location.line()
    );
    Ok(())
}

#[test]
fn prints_defaults_without_reading_project_config_or_changing_files() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", Some("invalid TOML ["))?;
    let output = formatter(&project, &["--print-config", "default"])?;
    ensure!(output.status.success(), "{output:?}");
    ensure!(output.stderr.is_empty(), "{output:?}");
    let text = String::from_utf8(output.stdout)?;
    check_equal(&text, &"[format]\nindent = 2\nfn_signature_width = 100\nspace_before_colon = false\nformat_function_headers = true\n")?;
    check_equal(
        &vx_config::parse_format_config(Path::new("vx-format.toml"), &text)?,
        &vx_config::FormatConfig::default(),
    )?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;
    check_equal(
        &fs::read_to_string(project.join("vx-format.toml"))?,
        &"invalid TOML [",
    )?;
    Ok(())
}

#[test]
fn rejects_unsupported_print_modes_and_conflicting_arguments() -> Result<()> {
    let directory = tempfile::tempdir()?;
    for args in [
        vec![],
        vec!["--print-config"],
        vec!["--print-config", "unknown"],
        vec!["--print-config", "default", "--indent", "4"],
        vec!["--print-config", "default", "--space-before-colon", "true"],
        vec!["--print-config", "default", "--config", "vx-format.toml"],
        vec!["--print-config", "default", "main.vx"],
        vec!["--print-config", "default", "--check"],
        vec![
            "--print-config",
            "default",
            "--format-function-headers",
            "false",
        ],
    ] {
        let output = formatter(directory.path(), &args)?;
        ensure!(!output.status.success(), "accepted {args:?}");
        ensure!(output.stdout.is_empty(), "{output:?}");
    }
    Ok(())
}

#[test]
fn check_reports_each_changed_file_without_writing() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    fs::write(project.join("other.vx"), SOURCE)?;
    let output = formatter(&project, &["--check", "main.vx", "other.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    let report = String::from_utf8(output.stdout)?;
    let expected = |name| {
        format!("--- {name}\n+++ {name} (formatted)\n@@ -1,3 +1,3 @@\n fn main() {{\n-let value = 1;\n+  let value = 1;\n }}\n")
    };
    let first = expected("main.vx");
    let second = expected("other.vx");
    ensure!(
        report == format!("{first}{second}") || report == format!("{second}{first}"),
        "{report}"
    );
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;
    check_equal(&fs::read_to_string(project.join("other.vx"))?, &SOURCE)?;
    Ok(())
}

#[test]
fn check_shows_unicode_and_missing_final_newlines() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let source = "fn message() -> i32 {\nreturn \"café\";\n}";
    fs::write(project.join("main.vx"), source)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    check_equal(&String::from_utf8(output.stdout)?, &"--- main.vx\n+++ main.vx (formatted)\n@@ -1,3 +1,3 @@\n fn message() -> i32 {\n-return \"café\";\n+  return \"café\";\n }\n\\ No newline at end of file\n")?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    Ok(())
}

#[test]
fn signature_width_controls_wrapping_and_check_mode() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(
        directory.path(),
        "project",
        Some("[format]\nindent = 4\nfn_signature_width = 39\n"),
    )?;
    let source = "fn add(first: i32, second: i32) -> i32 {\n  first + second\n}\n";
    fs::write(project.join("main.vx"), source)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    let diff = String::from_utf8(output.stdout)?;
    ensure!(diff.contains("@@ -1,3 +1,6 @@"), "{diff}");
    ensure!(diff.contains("+    second: i32\n+) -> i32 {\n"), "{diff}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    let output = formatter(&project, &["main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    check_equal(
        &fs::read_to_string(project.join("main.vx"))?,
        &"fn add(\n    first: i32,\n    second: i32\n) -> i32 {\n    first + second\n}\n",
    )?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    Ok(())
}

#[test]
fn colon_spacing_configuration_and_cli_override_apply_in_both_modes() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(
        directory.path(),
        "project",
        Some("[format]\nindent = 2\nspace_before_colon = true\n"),
    )?;
    let spaced = "fn value(x : i32) -> i32 {\n  x\n}\n";
    let compact = "fn value(x: i32) -> i32 {\n  x\n}\n";
    fs::write(project.join("main.vx"), compact)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &compact)?;
    let output = formatter(&project, &["main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &spaced)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;

    let output = formatter(
        &project,
        &["--check", "--space-before-colon", "false", "main.vx"],
    )?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &spaced)?;
    let output = formatter(&project, &["--space-before-colon", "false", "main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &compact)?;

    fs::write(
        project.join("vx-format.toml"),
        "[format]\nindent = 2\nspace_before_colon = false\n",
    )?;
    let output = formatter(&project, &["--space-before-colon", "true", "main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &spaced)?;
    Ok(())
}

#[test]
fn check_uses_configuration_and_overrides() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", Some("[format]\nindent = 0\n"))?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");

    let output = formatter(&project, &["--check", "--indent", "4", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;

    let output = formatter(&project, &["--indent", "4", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    let formatted = fs::read_to_string(project.join("main.vx"))?;
    let output = formatter(&project, &["--check", "--indent", "4", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &formatted)?;
    Ok(())
}

#[test]
fn check_honors_file_and_region_exclusions() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let disabled_file = "//! vx-format: OFF\nfn   main(){let x=1;}\n";
    let disabled_region =
        "// vx-format-begin: OFF\nconst   value : i32 = 1;\n// vx-format-end: OFF\n";
    fs::write(project.join("main.vx"), disabled_file)?;
    fs::write(project.join("region.vx"), disabled_region)?;
    let output = formatter(&project, &["--check", "main.vx", "region.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    check_equal(
        &fs::read_to_string(project.join("main.vx"))?,
        &disabled_file,
    )?;
    check_equal(
        &fs::read_to_string(project.join("region.vx"))?,
        &disabled_region,
    )?;
    Ok(())
}

#[test]
fn check_distinguishes_errors_from_formatting_changes() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", Some("invalid TOML ["))?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(2)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    ensure!(String::from_utf8_lossy(&output.stderr).contains("vx-format.toml"));
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;

    fs::remove_file(project.join("vx-format.toml"))?;
    let output = formatter(&project, &["--check", "main.vx", "missing.vx"])?;
    check_equal(&output.status.code(), &Some(2)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    ensure!(String::from_utf8_lossy(&output.stderr).contains("missing.vx"));
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;
    Ok(())
}

#[test]
fn uses_project_indentation_and_explicit_overrides() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", Some("[format]\nindent = 4\n"))?;
    let output = formatter(&project, &["main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(
        &fs::read_to_string(project.join("main.vx"))?,
        &"fn main() {\n    let value = 1;\n}\n",
    )?;

    let output = formatter(&project, &["--indent", "0", "main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &SOURCE)?;
    Ok(())
}

#[test]
fn defaults_without_config_and_ignores_lint_config() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    fs::write(project.join("vx-lint.toml"), "not valid TOML [")?;
    let output = formatter(&project, &["main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(
        &fs::read_to_string(project.join("main.vx"))?,
        &"fn main() {\n  let value = 1;\n}\n",
    )?;
    Ok(())
}

#[test]
fn resolves_each_project_and_explicit_config_paths() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let first = project(directory.path(), "first", Some("[format]\nindent = 4\n"))?;
    let second = project(directory.path(), "second", Some("[format]\nindent = 0\n"))?;
    let output = formatter(directory.path(), &["first/main.vx", "second/main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(
        &fs::read_to_string(first.join("main.vx"))?,
        &"fn main() {\n    let value = 1;\n}\n",
    )?;
    check_equal(&fs::read_to_string(second.join("main.vx"))?, &SOURCE)?;

    let output = formatter(
        directory.path(),
        &["--config", "first/vx-format.toml", "second/main.vx"],
    )?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(
        &fs::read_to_string(second.join("main.vx"))?,
        &"fn main() {\n    let value = 1;\n}\n",
    )?;
    Ok(())
}

#[test]
fn configuration_errors_prevent_writes_to_every_input() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let first = project(directory.path(), "first", Some("[format]\nindent = 4\n"))?;
    let second = project(directory.path(), "second", None)?;
    for invalid in [
        "",
        "[format]",
        "[format]\nindent = -1",
        "[format]\nindent = 2\nwidth = 80",
        "[",
    ] {
        fs::write(second.join("vx-format.toml"), invalid)?;
        let output = formatter(
            directory.path(),
            &["--indent", "4", "first/main.vx", "second/main.vx"],
        )?;
        ensure!(!output.status.success(), "accepted {invalid:?}");
        ensure!(String::from_utf8_lossy(&output.stderr).contains("vx-format.toml"));
        check_equal(&fs::read_to_string(first.join("main.vx"))?, &SOURCE)?;
        check_equal(&fs::read_to_string(second.join("main.vx"))?, &SOURCE)?;
    }
    let output = formatter(&first, &["--config", "missing.toml", "main.vx"])?;
    ensure!(!output.status.success());
    ensure!(String::from_utf8_lossy(&output.stderr).contains("missing.toml"));
    check_equal(&fs::read_to_string(first.join("main.vx"))?, &SOURCE)?;
    Ok(())
}

#[test]
fn unsupported_syntax_warns_without_failing_or_writing() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let source = "fn future(value: i32) -> i32 effects Read { value }\n";
    fs::write(project.join("main.vx"), source)?;
    for args in [vec!["--check", "main.vx"], vec!["main.vx"]] {
        let output = formatter(&project, &args)?;
        check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
        ensure!(output.stdout.is_empty(), "{output:?}");
        let warnings = String::from_utf8(output.stderr)?;
        ensure!(warnings.contains("WARN: main.vx:1:"), "{warnings}");
        ensure!(warnings.contains("preserve"), "{warnings}");
        check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    }
    Ok(())
}

#[test]
fn warnings_do_not_hide_supported_differences_in_the_same_file() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let source =
        "fn future() -> i32 {\nfuture_binding value;\n}\nfn known(value: i32) -> i32 {\nvalue\n}\n";
    fs::write(project.join("main.vx"), source)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    ensure!(String::from_utf8_lossy(&output.stderr).contains("WARN: main.vx:1:"));
    ensure!(
        String::from_utf8_lossy(&output.stdout).contains("--- main.vx\n+++ main.vx (formatted)\n")
    );
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    let output = formatter(&project, &["main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    let formatted = fs::read_to_string(project.join("main.vx"))?;
    ensure!(formatted.starts_with("fn future() -> i32 {\nfuture_binding value;\n}\n"));
    ensure!(
        formatted.ends_with("fn known(value: i32) -> i32 {\n  value\n}\n"),
        "{formatted}"
    );
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    ensure!(String::from_utf8_lossy(&output.stderr).contains("WARN:"));
    Ok(())
}

#[test]
fn exclusions_do_not_hide_diffs_in_check_mode() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let excluded = "// vx-format-begin: OFF\nfuture_binding value;\n// vx-format-end: OFF\n";
    for source in [
        format!("{excluded}fn   known(value:i32)->i32 {{\nreturn value;\n}}\n"),
        format!("fn   known(value:i32)->i32 {{\n{excluded}return value;\n}}\n"),
    ] {
        fs::write(project.join("main.vx"), &source)?;
        let output = formatter(&project, &["--check", "main.vx"])?;
        check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
        ensure!(output.stderr.is_empty(), "{output:?}");
        let diff = String::from_utf8(output.stdout)?;
        ensure!(diff.contains("+fn known(value: i32) -> i32 {"), "{diff}");
        ensure!(diff.contains("+  return value;"), "{diff}");
        check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    }
    Ok(())
}

#[test]
fn disabled_headers_preserve_their_style_and_still_format_bodies() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(
        directory.path(),
        "project",
        Some("[format]\nindent = 2\nfn_signature_width = 20\nformat_function_headers = false\n"),
    )?;
    let original = "fn   keep(a: i32, b : i32) -> i32 {\nreturn a;\n}\n";
    let preserved = "fn   keep(a: i32, b : i32) -> i32 {\n  return a;\n}\n";
    fs::write(project.join("main.vx"), original)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &original)?;
    let output = formatter(&project, &["--space-before-colon", "true", "main.vx"])?;
    ensure!(output.status.success(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &preserved)?;
    let output = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    let output = formatter(
        &project,
        &["--check", "--format-function-headers", "true", "main.vx"],
    )?;
    check_equal(&output.status.code(), &Some(1)).with_context(|| format!("{output:?}"))?;
    ensure!(String::from_utf8_lossy(&output.stdout).contains("+fn keep("));
    fs::write(
        project.join("vx-format.toml"),
        "[format]\nindent = 2\nformat_function_headers = true\n",
    )?;
    let output = formatter(
        &project,
        &["--check", "--format-function-headers", "false", "main.vx"],
    )?;
    check_equal(&output.status.code(), &Some(0)).with_context(|| format!("{output:?}"))?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &preserved)?;
    Ok(())
}

#[test]
fn repository_configuration_matches_hook_and_ci_formatting() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(
        directory.path(),
        "project",
        Some(include_str!("../vx-format.toml")),
    )?;
    let source = "fn   keep(a: i32, b : i32) -> i32\nrequires a > b {\n  return a;\n}\n";
    fs::write(project.join("main.vx"), source)?;
    let check = formatter(&project, &["--check", "main.vx"])?;
    check_equal(&check.status.code(), &Some(0)).with_context(|| format!("{check:?}"))?;
    ensure!(check.stdout.is_empty(), "{check:?}");
    ensure!(check.stderr.is_empty(), "{check:?}");
    let write = formatter(&project, &["main.vx"])?;
    ensure!(write.status.success(), "{write:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    Ok(())
}

#[test]
fn enabled_headers_preserve_contract_layout_in_check_and_write_modes() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let project = project(directory.path(), "project", None)?;
    let source = "fn   check(n : i32) -> i32\nrequires n > 0\nensures return > 0 {\nreturn n;\n}\n";
    let expected =
        "fn check(n: i32) -> i32\nrequires n > 0\nensures return > 0 {\n  return n;\n}\n";
    fs::write(project.join("main.vx"), source)?;
    let check = formatter(
        &project,
        &["--check", "--format-function-headers", "true", "main.vx"],
    )?;
    check_equal(&check.status.code(), &Some(1)).with_context(|| format!("{check:?}"))?;
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &source)?;
    let write = formatter(&project, &["--format-function-headers", "true", "main.vx"])?;
    ensure!(write.status.success(), "{write:?}");
    check_equal(&fs::read_to_string(project.join("main.vx"))?, &expected)?;
    let check = formatter(
        &project,
        &["--check", "--format-function-headers", "true", "main.vx"],
    )?;
    check_equal(&check.status.code(), &Some(0)).with_context(|| format!("{check:?}"))?;
    ensure!(check.stdout.is_empty(), "{check:?}");
    ensure!(check.stderr.is_empty(), "{check:?}");
    Ok(())
}
