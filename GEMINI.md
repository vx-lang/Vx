## Instructions

Keep all your attention on this repository and the tasks related to it. If I need help with other repositories, I will open a separate chat session.

## Git

- **CRITICAL** You are not allowed to `git push`
- **CRITICAL** You are not allowed to edit .git/config
- Commit changes whenever you make a meaningful change and it builds cleanly.
- Always write detailed commit messages with a commit message body. If the change fixes a bug, indicate that this bug is fixed by the commit using 'Fixes: #<BUG-ID>' in the commit message body.
- Always run formatters after doing `git add` and before `git commit`:
  - 'clang-format' for C++ code
  - 'cargo run --bin vx-format -- <file>' for .vx code.
    Do NOT run clang-format on .vx files. Fix all the warnings/errors before commiting.
- Try to commit as often whenever you think that there is some logical completion of the task.
- Make sure there is a github issue ID attached to each commit unless the commit is cleanup (formatting, minor fixes). If the issue is fixed by the issue add 'Fixes: #\<ISSUE_ID>' to auto-close the issue on github.
- If you have finished a walkthrough then save the Walkthrough, Task.md, and Implementation Plan in the docs/discussions/ directory. Create a new file, do not overwrite an existing file.
- When planning to add TODO to make progress: better add a `assert` to prevent accidental usage of the feature (prefer adding a github issue ID to it as well).

## Build instructions

- Never use all the cores. Leave 2 cores idle. For example if there are 10 cores available, use 8 cores at most.
- The LLVM build tools can be found with the help of `/opt/homebrew/opt/llvm/bin/llvm-config` script.
- Create a build log in the respective build directory.

## Virtual Environment

**CRITICAL**
Use the virtualenv created inside venv. Stop you dont find a venv virtual environment subdirectory in this project. If the venv looks corrupt, stop and ask me to make a fresh one for you.

- Make sure to terminate the virtualenv session after you are done with work.

## General coding guidelines

- Remove trailing whitespaces
- Remove redundant files/scripts that you create for making code changes

## Testing

After writing test, use `utils/update_mlir_test_checks.rs` to update the test checks.

## Rust Toolchain Environment

**CRITICAL**
Always prepend the following environment variables to EVERY `cargo` and `rustup` command you run (e.g. `cargo build`, `cargo test`, `cargo fmt`, `rustup default stable`):

```
source config.local
```
