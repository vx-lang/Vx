# Vx formatter

Run `vx-format file.vx` to format a file in place.
Pass multiple file paths to format several files.

## Project configuration

Create `vx-format.toml` in your project:

```toml
[format]
indent = 2
```

`indent` is the only supported setting.
It sets the number of spaces per indentation level.
Zero disables indentation.
The formatter preserves its existing spacing and layout rules.

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
```

Save it with `vx-format --print-config default > vx-format.toml`.
Print mode cannot be combined with input files, `--config`, `--indent`, or `--check`.

Use `--config FILE` to select one configuration file for all inputs.
Relative configuration paths use the command's working directory.
An explicitly selected missing file causes an error.

Use `--indent 4` to override indentation from a valid configuration or the built-in default.
Overrides do not hide errors in an existing configuration file.

```sh
vx-format --config path/to/vx-format.toml file.vx
vx-format --indent 4 file.vx
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
It reports each file that needs formatting and stays silent for unchanged files.
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

File exclusions through TOML are not implemented yet.
