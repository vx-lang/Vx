# Vx formatter configuration and check mode

Part of [issue 2343](https://github.com/vx-lang/Vx/issues/2343).
This work covers formatter configuration and command-line behavior.
Compiler analysis and lint rules remain separate.

## Implemented

- [x] Add a standalone `vx-config` library without compiler or backend dependencies.
- [x] Read `[format] indent` from `vx-format.toml`.
- [x] Use two-space indentation when automatic discovery finds no configuration.
- [x] Discover the nearest configuration from each input's real parent directory through its Git boundary.
- [x] Reject invalid configuration before modifying source files.
- [x] Preserve `--indent` overrides and add `--config FILE`.
- [x] Print built-in defaults with `--print-config default`.
- [x] Add non-writing `--check` mode with exit statuses for success, formatting differences, and errors.
- [x] Use `--check` in CI instead of rewriting files and checking Git differences.
- [x] Add configuration and CLI test cases and document usage.

Only indentation is configurable.
The formatter ignores `vx-lint.toml`.
Existing files must contain the required `[format] indent` setting.
Zero indentation remains supported.

## Validation still required

- [ ] Run the configuration tests and formatter CLI tests.
- [ ] Run the existing formatter unit tests.
- [ ] Perform the required mutation checks for each important assertion.
- [ ] Check Markdown formatting when `mdformat` is available in the project environment.
- [ ] Run the updated CI formatting step.

The user requested no tests yet.
Tests were compiled without execution.
Formatter builds, Rust formatting checks, and whitespace checks passed before this branch split.
Logs are under `build/vx-format-config-*`, `build/vx-format-print-config-*`, and `build/vx-format-check-*`.

## Future work

- [ ] Define a complete Vx style guide with examples.
- [ ] Add file and directory exclusions through configuration.
- [ ] Document editor integration and publish the standard-format blog.
- [ ] Add further settings only when the formatter implements their behavior.
