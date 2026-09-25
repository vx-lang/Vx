## Instructions

Keep all your attention on this repository and the tasks related to it. If I need help with other repositories, I will open a separate chat session.
Use simple English everywhere (comments, logs, chat sessions) not cryptic. It should be understood by a junior engineer.

This covers commit messages, PR descriptions, error messages, test file names and test
comments too. Vx is a programming language, and its tests are read by its users: an undergrad
should understand what a test checks from the test file alone, without reading the compiler.

- Use the words a programmer already knows: "a function with a type error", "a function the
  program uses", "generates code". Do not invent terms from the implementation ("flawed
  function", "reached", "emitted") and then use them as if the reader knew them.
- Explain a compiler term the first time the reader needs it, or use a plainer one: "a normal
  compile and a parallel compile (`-j`)" rather than "the sequential and parallel drivers".
- Name test files and the functions inside them for what they are: `wrong_field`, `add_one`,
  `unused_imported_functions_with_type_errors.vx`.
- Read the draft again as someone who has seen only that file. Another agent (for example
  `agy -p "<text and question>"`) can review the wording; it is good at catching double
  negatives, internal terms and ambiguous words like "output".

## Tool usage

These keep commands clean and avoid unnecessary permission prompts:

- Do NOT prefix Bash commands with `cd <repo root>` (e.g. `cd ~/Vx; ...`). The Bash tool's working directory is already the repository root.
- Use the dedicated `Grep`, `Glob`, and `Read` tools for searching and reading files instead of Bash `grep`, `rg`, `find`, `ls`, `cat`, `head`, `tail`, or `sed`. They are faster and never need approval.
- Keep each Bash call to a single logical operation. Do NOT pad commands with `echo "==="` separators or chain several unrelated commands together with `;`/`&&` just to batch them — that turns otherwise-allowed commands into compound shell that prompts.

## Git

- **CRITICAL** Never push to `main`, and never force-push anything. Push a feature branch
  and open a pull request; `main` moves through review. This replaced a blanket ban on
  pushing, which protected `main` by making every machine a dead end: work committed on one
  could only reach another by hand. Branch protection on GitHub is what actually enforces
  this — the rule here is so you do not have to discover it by being refused.
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
- Do not use cryptic nomenclature like `brick 3`, `C1.1` in comments. They are for your tracking but they do not belong in source code comments.
- Comments should not have bug-id or refer to other commit SHA. Those can be tracked via git.
- When commenting on issues do not put backticks on commit SHA like `7f2b387e`, this prevents github to link to the actual commit. Simply write the SHA like 7f2b387e.

## Build instructions

- Never use all the cores. Leave 2 cores idle. For example if there are 10 cores available, use 8 cores at most.
- The LLVM build tools are found by `./setup.sh`, which locates them and writes their
  location into `config.local`. Source that rather than hardcoding a path: it is
  `/opt/homebrew/opt/llvm/bin` on a Homebrew macOS machine and `/usr/lib/llvm-22/bin` on
  Ubuntu, and nothing you need -- not `llvm-config`, not `mlir-translate`, not even
  `cargo` -- is on PATH in a non-login shell until it is sourced.
- Create a build log in the respective build directory.

## Virtual Environment

**CRITICAL**
Use the virtualenv created inside venv. Stop you dont find a venv virtual environment subdirectory in this project. If the venv looks corrupt, stop and ask me to make a fresh one for you.

- Make sure to terminate the virtualenv session after you are done with work.

## General coding guidelines

- Remove trailing whitespaces
- Remove redundant files/scripts that you create for making code changes
- Prefer assert to escape hatches. This is a compiler, we better crash then fail silently.
- Keep comments in proportion to the code. A paragraph explaining a one-line change is
  worse than nothing: it buries the line and goes stale first. Say the non-obvious thing
  once, in a sentence or two, and stop. The same goes for commit messages, PR bodies and
  test file headers -- a fixture header should not be longer than the fixture.
  If the reasoning really needs several paragraphs, it belongs in the issue or in
  docs/discussions/, with the code pointing at it.

## Testing

- After writing test, use `utils/update_mlir_test_checks.rs` to update the test checks.
- Before trusting a new test, break the code it covers, watch it fail, and read **which
  assertion reported**. Seeing the test go red is not enough. A check can go red for a
  reason unrelated to the thing it names, and a check that happened to match an incidental
  value passes while testing nothing at all. So break each assertion separately rather than
  breaking the file once: one sabotage only ever proves the one assertion it happened to
  reach.
- Care does not substitute for this. Four fixtures were found claiming more than their
  checks established, and one of them was written the same day by someone specifically
  thinking about verification — prose describing two cases, checks covering one.
  `tests/optimizations/pass/space_access_reaches_the_backend.vx` records the three attempts
  that file took to become sound, and which trap each attempt fell into.
- A test's comment is a claim about what it asserts. If they disagree, the comment is what
  the next reader believes, so fix the checks or narrow the comment.

## Rust Toolchain Environment

**CRITICAL**
Always prepend the following environment variables to EVERY `cargo` and `rustup` command you run (e.g. `cargo build`, `cargo test`, `cargo fmt`, `rustup default stable`):

```
source config.local
```
