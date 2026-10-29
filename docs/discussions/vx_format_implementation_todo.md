# Vx formatter implementation

Part of [issue 2343](https://github.com/vx-lang/Vx/issues/2343).
Formatter configuration and rules remain independent from lint configuration and analysis.

## Configuration and commands

- [x] Add the standalone `vx-config` library.
- [x] Read `vx-format.toml` and use defaults when automatic discovery finds no file.
- [x] Discover configuration per input through its Git boundary.
- [x] Support `indent`, `fn_signature_width`, `--indent`, and `--config`.
- [x] Reject invalid configuration before writing any input.
- [x] Print complete defaults through `--print-config default`.
- [x] Implement non-writing `--check` and use it in CI.
- [x] Show unified diffs with file headers, numbered hunks, and surrounding context in check mode.
- [x] Report unsupported syntax with warnings and preserve affected source.
- [x] Return `0` for warning-only checks, `1` for supported differences, and `2` for command or input errors.

## Existing formatting behavior

- [x] Keep short function declarations on one line.
- [x] Wrap long declarations into one parameter per line using the configured width.
- [x] Include indentation, modifiers, complete header clauses, and the following delimiter in width decisions.
- [x] Retain compatible type normalization without splitting commas inside nested types.
- [x] Keep function-call layout unchanged.
- [x] Preserve headers containing comments rather than guessing their layout.

## Active foundations

- [x] Retain original source bytes, Unicode token ranges, delimiters, and source positions.
- [x] Discover opaque macro arguments, trailing blocks, matchers, and transcribers before ordinary declarations.
- [x] Reuse parser helpers for types, generics, topology, contracts, and transfer constraints.
- [x] Record root, implementation, trait, and foreign declaration contexts with conservative eligibility checks.
- [x] Inherit enclosing generic parameters during type recognition.
- [x] Probe body and remaining-file grammar without imports, semantic analysis, or backend work.
- [x] Dispatch through one registered owner per syntax kind.
- [x] Discover independent syntax roots through the registry rather than engine-specific branches.
- [x] Compose parent and child documents with shared width and indentation budgets.
- [x] Protect source ownership before running any legacy transformations.
- [x] Limit legacy block normalization, spacing, and indentation to permitted source intervals.
- [x] Carry enclosing delimiters, multiline-bracket state, indentation, line, and column across all intervals.
- [x] Remove whole-output cleanup that could damage literals or preserved text.
- [x] Validate final token order, spelling, grammatical adjacency, opaque bytes, exclusions, and macro columns.
- [x] Preserve a bounded unsupported declaration, or the whole file when boundaries are uncertain.
- [x] Activate the same pipeline for CLI formatting and check mode.
- [x] Remove the former token-position registry and duplicate signature scanner.
- [x] Document registration and syntax-layer extension boundaries.

The engine now supports additional registered owners without another syntax-specific rendering branch.
New language constructs still require shared syntax recognition when a suitable view does not exist.
Type views remain token-backed, without a complete recursive concrete syntax tree.
The internal registry does not load external plugins or promise a stable public extension API.

See [the architecture guide](../tools/vx-format-source-architecture.md) and [the contributor guide](../tools/vx-format-rules.md).
The [original proposal](vx_format_syntax_architecture_proposal.md) records the design rationale and migration contracts.

## Validation

The user authorized test execution during foundation activation.
Earlier compile-only reports did not establish behavioral correctness.
Baseline execution exposed five dormant source-range failures caused by lexer whitespace line metadata.
Token ranges now derive from cumulative original byte lengths.

Activation logs use `build/vx-format-activation-`.
The restored implementation passed these checks:

- Configuration tests: 6 passed.
- Formatter unit and contract tests: 48 passed.
- CLI tests: 13 passed.
- Deliberate regression checks: 20 failed at the intended assertions before code restoration.
- Clippy passed with warnings denied for the library, formatter binary, and CLI test target.
- The formatter binary built successfully.
- All 1,786 Vx corpus files completed formatting and check operations without crashes on temporary copies.
- Every corpus copy stayed unchanged after a second formatting pass and passed its subsequent check.
- Hash comparisons confirmed that repository corpus files remained untouched.
- The read-only repository check returned `1`, reporting 959 formatting differences and warnings for 118 files.

The corpus formatting gate is therefore not green.
This change does not rewrite those fixtures or claim complete formatting coverage.
Many fail-test fixtures deliberately contain invalid syntax and receive preservation warnings.

Commands used `source config.local`, eight build workers, and at most eight active test or corpus workers.
Focused Cargo targets were:

```sh
cargo test --offline --locked -p vx-config -- --test-threads=8
cargo test --offline --locked -p vxc --lib formatter:: -- --test-threads=8
cargo test --offline --locked -p vxc --test vx_format_cli -- --test-threads=8
cargo clippy --offline --locked -p vxc --lib --bin vx-format --test vx_format_cli -- -D warnings
cargo build --offline --locked -p vxc --bin vx-format
```

Mutation checks covered adjacency, macro bytes, exclusions, token layouts, duplicate owners, registered roots, legacy ownership, widths, preserved state, header clauses, type normalization, macro discovery, coverage, non-writing checks, and CLI warnings.
Logs under `build/vx-format-activation-mutations/` identify the intended failing assertion for each deliberate regression.
The complete workspace and backend suites were not run.

## Check-mode diff validation

Check-mode diff reporting uses the `similar` library for line comparison and unified hunks.
Each file's complete diff is written under one stdout lock to prevent mixed parallel reports.
Diff coverage includes exact file headers, context, line counts, additions, removals, Unicode, and missing final newlines.
The 14 CLI tests passed before five deliberate output regressions failed at their intended assertions.
Diff validation logs use the `build/vx-format-diff-` prefix.

## Future work

- [ ] Define the complete Vx style guide with examples.
- [ ] Add file and directory exclusions through configuration.
- [ ] Document editor integration and publish the standard-format blog.
- [ ] Add further settings only when the formatter implements their behavior.
- [ ] Add richer shared syntax views when future formatting rules need them.
- [ ] Measure parser and layout costs before selecting more elaborate syntax representations.

Markdown formatting remains unavailable because the project environment lacks `mdformat`.
