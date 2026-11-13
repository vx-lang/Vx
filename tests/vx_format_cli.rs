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
    check_equal(&text, &"[format]\nindent = 2\n")?;
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
        vec!["--print-config", "default", "--config", "vx-format.toml"],
        vec!["--print-config", "default", "main.vx"],
        vec!["--print-config", "default", "--check"],
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
fn input_read_errors_prevent_writes_to_every_input() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let first = project(directory.path(), "first", Some("[format]\nindent = 4\n"))?;
    let second = project(directory.path(), "second", None)?;
    let invalid_utf8 = b"fn second() {}\n\xff";
    fs::write(second.join("main.vx"), invalid_utf8)?;

    let needs_formatting = formatter(directory.path(), &["--check", "first/main.vx"])?;
    check_equal(&needs_formatting.status.code(), &Some(1))
        .with_context(|| format!("{needs_formatting:?}"))?;

    // One worker makes deferred reads process the valid file before the invalid file.
    let output = Command::new(env!("CARGO_BIN_EXE_vx-format"))
        .current_dir(directory.path())
        .env("RAYON_NUM_THREADS", "1")
        .args(["first/main.vx", "second/main.vx"])
        .output()?;
    check_equal(&output.status.code(), &Some(2)).with_context(|| format!("{output:?}"))?;
    check_equal(&fs::read_to_string(first.join("main.vx"))?, &SOURCE)?;
    check_equal(&fs::read(second.join("main.vx"))?, &invalid_utf8.to_vec())?;
    ensure!(output.stdout.is_empty(), "{output:?}");
    ensure!(String::from_utf8_lossy(&output.stderr).contains("second/main.vx"));
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
