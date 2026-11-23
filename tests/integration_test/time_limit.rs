//! Time limits for the programs that the tests run. Without a limit, one program that hangs
//! stops the whole test run with no message.

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// The time limit of one compile or run of a test program.
pub const SECONDS: u32 = 300;

/// Whether the machine has the `timeout` command.
pub fn has_timeout() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        Command::new("timeout")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// A command that runs the vxc under test, and stops it after `SECONDS`. `timeout` signals its
/// whole process group, so the program that vxc runs stops too. A machine with no `timeout`
/// (macOS) runs vxc with no limit. A run that `timeout` stopped exits with 124.
pub fn vxc() -> Command {
    let vxc = env!("CARGO_BIN_EXE_vxc");
    if has_timeout() {
        let mut cmd = Command::new("timeout");
        cmd.args(["-k", "5", &SECONDS.to_string(), vxc]);
        cmd
    } else {
        Command::new(vxc)
    }
}

/// Runs `cmd` and returns its output, as `Command::output` does. The command runs in a process
/// group of its own, and this stops every process left in that group when the command exits. A
/// process that a shell started in the background can hold the output pipe open, and otherwise
/// the test waits for the end of the output forever. `timeout` makes a new process group, so a
/// shell that starts it must `exec` it.
pub fn output(cmd: &mut Command) -> std::io::Result<Output> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()?;
    let stdout = read_in_a_thread(child.stdout.take().unwrap());
    let stderr = read_in_a_thread(child.stderr.take().unwrap());
    let status = child.wait()?;
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{}", child.id())])
        .stderr(Stdio::null())
        .status();
    Ok(Output {
        status,
        stdout: stdout.join().unwrap()?,
        stderr: stderr.join().unwrap()?,
    })
}

fn read_in_a_thread(
    mut pipe: impl Read + Send + 'static,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).map(|_| bytes)
    })
}

/// Stops the test process if it is not dropped within `SECONDS`. Use it for work that runs
/// inside the test process, where `timeout` cannot reach.
/// Dropping it drops the sender, which ends the wait of the watchdog thread.
pub struct Watchdog {
    _done: mpsc::Sender<()>,
}

impl Watchdog {
    pub fn new(what: String) -> Self {
        let (done, wait) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) =
                wait.recv_timeout(Duration::from_secs(SECONDS.into()))
            {
                // libtest captures `eprintln!` and loses it at `exit`, so write to the handle.
                let _ = writeln!(
                    std::io::stderr(),
                    "{what} did not finish in {SECONDS} s. The test run stops here."
                );
                std::process::exit(124);
            }
        });
        Watchdog { _done: done }
    }
}
