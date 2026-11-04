//! The analyzer's command line, driven as a real process.
//!
//! The binary is a server that reads stdin and does not exit on its own, so a regression here
//! hangs the job instead of failing an assertion. Every case below waits with a bound.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The server, built by Cargo alongside this test.
const ANALYZER: &str = env!("CARGO_BIN_EXE_vx-analyzer");

/// How long an argument-only run may take before it is called a hang.
const LIMIT: Duration = Duration::from_secs(30);

/// Run the analyzer with `args` and return `(status, stdout, stderr)`.
///
/// Stdin is closed so the server sees end-of-file; the child is killed if it outlives `LIMIT`.
fn run(args: &[&str]) -> (Option<std::process::ExitStatus>, String, String) {
    let mut child = Command::new(ANALYZER)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("start {ANALYZER}: {error}"));

    // Close stdin so the server sees end-of-file and exits.
    drop(child.stdin.take());

    let deadline = Instant::now() + LIMIT;
    let status = loop {
        match child
            .try_wait()
            .unwrap_or_else(|e| panic!("wait for {ANALYZER}: {e}"))
        {
            Some(status) => break Some(status),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    };

    // The process has exited or been killed, so these reads drain and return.
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    (status, stdout, stderr)
}

#[test]
fn version_prints_the_workspace_version_and_exits() {
    let (status, stdout, stderr) = run(&["--version"]);
    let status = status.expect("`vx-analyzer --version` did not exit in 30s: it hung");
    assert!(status.success(), "exit {status}: stderr was {stderr:?}");
    let printed = stdout.trim();
    assert_eq!(
        printed,
        concat!("vx-analyzer ", env!("CARGO_PKG_VERSION")),
        "unexpected --version output"
    );
}

#[test]
fn help_lists_the_binary_without_advertising_stdio() {
    let (status, stdout, _stderr) = run(&["--help"]);
    let status = status.expect("`vx-analyzer --help` did not exit in 30s: it hung");
    assert!(status.success(), "exit {status}");
    assert!(
        stdout.contains("vx-analyzer"),
        "help did not name the binary: {stdout:?}"
    );
    assert!(
        !stdout.contains("--stdio"),
        "the hidden --stdio flag should not appear in --help: {stdout:?}"
    );
}

#[test]
fn the_hidden_stdio_flag_is_accepted() {
    // An editor that passes `--stdio` must be served, not stopped by an unknown-argument error.
    let (status, _stdout, stderr) = run(&["--stdio"]);
    let status = status.expect("`vx-analyzer --stdio` did not exit in 30s: it hung");
    assert!(
        !stderr.contains("unexpected argument") && !stderr.contains("--stdio"),
        "the hidden flag was rejected: {stderr:?}"
    );
    assert!(status.success(), "exit {status}: stderr was {stderr:?}");
}

#[test]
fn an_unknown_flag_is_still_rejected() {
    // Parsing is not dropped: a genuine typo still fails.
    let (status, _stdout, stderr) = run(&["--not-a-flag"]);
    let status = status.expect("`vx-analyzer --not-a-flag` did not exit in 30s: it hung");
    assert!(!status.success(), "an unknown flag was accepted");
    assert!(
        stderr.contains("unexpected argument") || stderr.contains("--not-a-flag"),
        "the rejection did not name the argument: {stderr:?}"
    );
}
