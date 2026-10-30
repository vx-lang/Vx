# Vx formatter

Run `vx-format file.vx` to format a file in place.
Pass multiple file paths to format several files.

## Project configuration

Create `vx-format.toml` in your project:

```toml
[format]
indent = 2
fn_signature_width = 100
space_before_colon = false
format_function_headers = true
```

`indent` sets the number of spaces per indentation level.
Zero disables indentation.
The formatter preserves its existing spacing and layout rules.

`fn_signature_width` sets the preferred maximum width of a function declaration.
It defaults to 100 when omitted and must be greater than zero.
Existing configuration files containing only `indent` remain valid.

`space_before_colon` controls single colons in formatted function headers, including parameters and generic bounds.
It defaults to `false`, which produces `value: i32`.
Set it to `true` for `value : i32` and `T : Copy`.
Both styles use one space after the colon.
Paths such as `module::Type` remain unchanged.
This setting does not change colon spacing in function bodies or struct declarations.
The added spaces count toward `fn_signature_width`.

`format_function_headers` defaults to `true`.
Set it to `false` to preserve existing function header spacing, type spelling, wrapping, and contract-clause layout.
Body formatting remains active.
Header width and colon-spacing settings do not change disabled headers.
The Vx repository's root `vx-format.toml` disables header formatting to preserve its existing mixed styles.
CI and `scripts/pre-commit.sh` both discover that configuration without command-line style overrides.
The built-in defaults above remain unchanged.

For each input, the formatter searches from the file's real parent directory upward.
It uses the nearest `vx-format.toml` and does not merge ancestor files.
The search includes the first directory containing a `.git` file or directory, then stops.
Without a Git boundary, the search stops at the filesystem root.
Inputs from different projects can use different configurations in one command.

When no configuration file exists, the formatter defaults to two spaces.
An existing file must contain `[format]` and `indent`.
Empty files, unknown settings, invalid types, and negative indentation values cause errors.
Invalid or unreadable files never cause a fallback to defaults.
The formatter ignores `vx-lint.toml`.

## Command-line overrides

Print the built-in defaults without reading project configuration or formatting source files:

```sh
vx-format --print-config default
```

The command writes a complete TOML configuration to standard output:

```toml
[format]
indent = 2
fn_signature_width = 100
space_before_colon = false
format_function_headers = true
```

Save it with `vx-format --print-config default > vx-format.toml`.
Print mode cannot be combined with input files, `--config`, `--indent`, `--space-before-colon`, `--format-function-headers`, or `--check`.

Use `--config FILE` to select one configuration file for all inputs.
Relative configuration paths use the command's working directory.
An explicitly selected missing file causes an error.

Use `--indent 4` to override indentation from a valid configuration or the built-in default.
Overrides do not hide errors in an existing configuration file.
Use `--space-before-colon true` or `--space-before-colon false` to override the configured colon style.
Use `--format-function-headers true` or `--format-function-headers false` to override the header-formatting setting.

```sh
vx-format --config path/to/vx-format.toml file.vx
vx-format --indent 4 file.vx
vx-format --space-before-colon true file.vx
vx-format --check --format-function-headers false file.vx
```

The formatter validates all configurations and reads all inputs before modifying any file.
Configuration and file-access errors return a failing process status.
Write errors can occur after other files were formatted.

## Check formatting

Use `--check` to check formatting without modifying files:

```sh
vx-format --check file.vx other.vx
vx-format --check --indent 4 file.vx
```

Check mode uses the same configuration, overrides, and formatting exclusions as normal formatting.
It prints a unified diff for each file that needs formatting.
The `---` header names the original file, and the `+++` header names the formatted version.
Numbered `@@` hunks show removed lines with `-`, added lines with `+`, and three surrounding context lines.
Unchanged files produce no diff.
Unsupported syntax produces a `WARN` message and preserves the affected source.
A warning alone does not fail check mode.
Supported differences elsewhere still return status `1`.
CI uses this mode to check Vx files in `tests` and `stdlib`.

| Exit status | Meaning |
| --- | --- |
| `0` | Formatting succeeded, or every checked file is already formatted. |
| `1` | At least one checked file needs formatting. |
| `2` | An argument, configuration, or file-access error prevented completion. |

## Formatting exclusions

Place `//! vx-format: OFF` as the first comment to preserve an entire file.
Surround a source region with these existing markers to preserve its text:

```vx
// vx-format-begin: OFF
let   value = 1;
// vx-format-end: OFF
```

The formatter preserves both marker lines and enclosed text exactly.
Unrecognized macro token trees also retain their original bytes and columns.
File exclusions through TOML are not implemented yet.

## Function declarations

Short declarations stay on one line, even when their parameters originally occupied several lines:

```vx
fn add(first: i32, second: i32) -> i32 {
  first + second
}
```

When a declaration exceeds `fn_signature_width`, each parameter occupies its own line.
For example, with `fn_signature_width = 39`:

```vx
fn add(
  first: i32,
  second: i32
) -> i32 {
  first + second
}
```

The width includes indentation, visibility modifiers, the return type, and the opening brace or semicolon.
The closing parenthesis aligns with the declaration's indentation.
Methods and foreign declarations follow the same rule.
Commas inside generic, array, and tuple types do not separate parameters.
The formatter does not insert trailing commas because Vx does not accept them in parameter lists.

This rule follows Rustfmt's preference for short signatures on one line and long signatures with vertical parameters.
See [Rustfmt's default parameter layout](https://github.com/rust-lang/rustfmt/blob/main/Configurations.md#fn_params_layout).

Declarations containing comments retain their original header text.
Supported surrounding code can still format.
Formatting-disabled declarations remain excluded from this rule.
Function calls retain their existing formatting policy.
Formatting a header preserves the existing text and line breaks of its `requires`, `ensures`, and `where` clauses.
Inline clauses count toward the available signature width.
Clauses on following lines do not force short parameter lists to wrap.

The width is a preference: a single long type or function name can still exceed it.
Declarations without parameters remain on one line.

## Extending the formatter

Built-in rules use an internal registry and a shared layout renderer.
See [adding a formatting rule](vx-format-rules.md) for the architecture and contributor checklist.
