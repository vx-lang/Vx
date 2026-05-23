## Instructions
Keep all your attention on this repository and the tasks related to it. If I need help with other repositories, I will open a separate chat session.

## Git
- Commit changes whenever you make a meaningful change and it builds cleanly.
- Try to commit as often whenever you think that there is some logical completion of the task.
- Always write detailed commit messages.
- Always run formatters: 'clang-format', fix all the warnings/errors before commiting.
- Make sure there is a github issue ID attached to each commit unless the commit is cleanup (formatting, minor fixes)
- If you have finished a walkthrough then save the Walkthrough, Task.md, and Implementation Plan in the docs/discussions/ directory.
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

## Rust Toolchain Environment
**CRITICAL**
Always prepend the following environment variables to EVERY `cargo` and `rustup` command you run (e.g. `cargo build`, `cargo test`, `cargo fmt`, `rustup default stable`):
```
export CARGO_HOME=/Users/adityak/go/Vx/.cargo && export RUSTUP_HOME=/Users/adityak/go/Vx/.rustup && export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```
