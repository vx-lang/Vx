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

Indentation and function declaration width are configurable.
The formatter ignores `vx-lint.toml`.
Existing files must contain the required `[format] indent` setting.
Zero indentation remains supported.

## Function declaration layout

Added on 2026-10-29 for function declarations only.
This work does not change function-call layout.

- [x] Add optional `fn_signature_width`, defaulting to 100, to `vx-format.toml` and printed defaults.
- [x] Keep short signatures on one line and collapse existing multiline signatures when they fit.
- [x] Split longer declarations into one parameter per line with one additional indentation level.
- [x] Include indentation, modifiers, the return type, and the body delimiter in the width decision.
- [x] Keep nested type commas intact and preserve the existing no-trailing-comma grammar.
- [x] Leave declarations containing comments under the existing formatting policy.
- [x] Respect formatting exclusions and use the same layout in normal and check modes.
- [x] Add boundary, nested-type, declaration, exclusion, token-preservation, and repeat-formatting assertions.
- [ ] Execute the new tests and required mutation checks when authorized.

Existing configuration files containing only `indent` remain valid.
`fn_signature_width` must be positive.
It controls declaration layout rather than a global line-length limit.
Long names, types, and declarations without parameters can exceed the preferred width.

Validation: test sources compile without execution.
Signature-layout logs use the `build/vx-format-signatures-` prefix.

- `cargo build --offline --locked -p vxc --bin vx-format` passed.
- `cargo check --offline --locked -p vxc --lib --tests` passed without running tests.
- `cargo check --offline --locked -p vx-config --tests` passed without running tests.
- `cargo clippy --offline --locked -p vxc --lib --bin vx-format -- -D warnings` passed.
- Rust formatting and whitespace checks passed.
- Cargo commands sourced `config.local` and used eight workers for compilation.
- Test execution, mutation checks, and Markdown formatting remain pending.

## Rule framework

- [x] Add an internal `FormatRule` interface with stable names and typed configuration access.
- [x] Add an explicit registry with deterministic ordering and duplicate-name checks.
- [x] Share token context, delimiter matching, and formatting exclusions.
- [x] Represent rule output as original token references plus spaces and indented line breaks.
- [x] Reject overlapping layouts, token removal or reordering, and incomplete delimiter ranges.
- [x] Check lexical token preservation across replacement boundaries.
- [x] Move function-signature formatting into `rules/function_signatures.rs`.
- [x] Remove signature-specific dispatch from the main rendering loop.
- [x] Add extension-contract assertions and a contributor architecture guide.
- [ ] Execute framework, formatter, and CLI tests and their required mutation checks when authorized.

The framework is internal and does not load external plugins.
Existing normalization and fallback formatting remain in the original pipeline.
The public CLI and configuration schema are unchanged by this refactor.
Architecture logs use the `build/vx-format-rules-` prefix.

- `cargo build --offline --locked -p vxc --bin vx-format` passed.
- `cargo check --offline --locked -p vxc --lib --tests` compiled the tests without execution.
- `cargo clippy --offline --locked -p vxc --lib --bin vx-format --tests -- -D warnings` passed.
- Rust formatting and whitespace checks passed.
- Cargo commands sourced `config.local` and used eight workers for compilation.
- Test execution and mutation checks remain deferred at the user's request.
- Markdown formatting remains pending because the project environment lacks `mdformat`.

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

- [ ] Review the [syntax-aware architecture proposal](vx_format_syntax_architecture_proposal.md) before changing dispatch or adding further structural rules.
- [ ] Define a complete Vx style guide with examples.
- [ ] Add file and directory exclusions through configuration.
- [ ] Document editor integration and publish the standard-format blog.
- [ ] Add further settings only when the formatter implements their behavior.
