# Vx formatter

Run `vx-format file.vx` to format a file in place.
Pass multiple file paths to format several files.

## Project configuration

Create `vx-format.toml` in your project:

```toml
[format]
style_edition = "2026"
indent = 2
max_width = 100
fn_signature_width = 100
space_before_colon = false

[format.rules]
function_headers = true
```

Only `indent` is required.
Each other setting takes its default from the style edition when the file does not set it.

`style_edition` selects a set of defaults.
It defaults to `"2026"`, the only edition so far.
A later edition can change a default.
A project that names an older edition keeps that edition's defaults, so its formatting does not change.

`indent` sets the number of spaces per indentation level.
Zero disables indentation.

`max_width` sets the preferred maximum width of a line, including indentation.
It defaults to 100 and must be greater than zero.

`fn_signature_width` sets a narrower width for function declarations.
It defaults to `max_width` and must not be greater than `max_width`.
A single long type or function name can still make a declaration wider.

`space_before_colon` controls single colons in formatted function headers, including parameters and generic bounds.
It defaults to `false`, which produces `value: i32`.
Set it to `true` for `value : i32` and `T : Copy`.
Both styles use one space after the colon.
Paths such as `module::Type` remain unchanged.
This setting does not change colon spacing in function bodies or struct declarations.
The added spaces count toward `fn_signature_width`.

## Formatting rules

`[format.rules]` switches each formatting rule on or off by name.
A rule that the table does not name uses its default.
An unknown rule name is an error.
`vx-format --print-config default` lists every rule with its default.

| Rule | Default | What it formats |
| --- | --- | --- |
| `function_headers` | `true` | Spacing, type spelling, and wrapping of function declarations. |

A switched-off rule keeps the original text of the code it formats.
For example, `function_headers = false` keeps each header's spacing, wrapping, and contract-clause layout.
Body formatting stays active, and width and colon settings do not change those headers.

The Vx repository's root `vx-format.toml` switches off `function_headers` to keep its existing mixed styles.
CI and `scripts/pre-commit.sh` both use that file.

## Configuration files

For each input, the formatter searches from the file's real parent directory upward.
It uses the nearest `vx-format.toml` and does not merge ancestor files.
The search includes the first directory containing a `.git` file or directory, then stops.
Without a Git boundary, the search stops at the filesystem root.
Inputs from different projects can use different configurations in one command.

When no configuration file exists, the formatter uses the built-in defaults.
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
style_edition = "2026"
indent = 2
max_width = 100
fn_signature_width = 100
space_before_colon = false

[format.rules]
function_headers = true
```

Save it with `vx-format --print-config default > vx-format.toml`.
Print mode cannot be combined with input files, `--config`, `--indent`, `--set`, or `--check`.

Use `--config FILE` to select one configuration file for all inputs.
Relative configuration paths use the command's working directory.
An explicitly selected missing file causes an error.

Use `--set KEY=VALUE` to override one setting of `[format]` for this command.
Use a dotted key for a rule, such as `rules.function_headers`.
Repeat `--set` to override several settings.
`--indent 4` is short for `--set indent=4`.
The formatter checks each override like a setting in the file.
Overrides do not hide errors in an existing configuration file.

```sh
vx-format --config path/to/vx-format.toml file.vx
vx-format --indent 4 file.vx
vx-format --set max_width=80 --set space_before_colon=true file.vx
vx-format --check --set rules.function_headers=false file.vx
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
