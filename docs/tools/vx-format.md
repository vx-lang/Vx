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
struct_lit_width = 40
single_line_if_else_max_width = 100

[format.rules]
function_headers = true
struct_literals = true
match_arms = true
if_values = true
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

`struct_lit_width` sets how wide the fields of a struct literal may be, for the literal to stay on one line.
It defaults to 40% of `max_width` and must not be greater than `max_width`.

`single_line_if_else_max_width` sets how wide an `if` value may be, for it to stay on one line.
It defaults to `max_width`, so by default only the width of the line limits an `if` value.

## Formatting rules

`[format.rules]` switches each formatting rule on or off by name.
A rule that the table does not name uses its default.
An unknown rule name is an error.
`vx-format --print-config default` lists every rule with its default.

| Rule | Default | What it formats |
| --- | --- | --- |
| `function_headers` | `true` | Spacing, type spelling, and wrapping of function declarations. |
| `struct_literals` | `true` | Puts a short struct literal on one line, and a long one with one field on each line. |
| `match_arms` | `true` | Starts each arm of a `match` on a line of its own. |
| `if_values` | `true` | Puts a short `if` value, such as `if c { 1 } else { 0 }`, on one line. |

A switched-off rule leaves its code to the base formatting, as if the rule did not exist.
The base formatting changes only indentation and the line breaks around blocks.
For example, `function_headers = false` keeps each header's spacing, wrapping, and contract-clause layout.
Width and colon settings then do not change those headers.

The Vx repository's root `vx-format.toml` switches off all four rules to keep its existing layout.
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

The command writes the defaults as TOML to standard output:

```toml
[format]
style_edition = "2026"
indent = 2
max_width = 100
space_before_colon = false
# Unset, these widths follow max_width: fn_signature_width is max_width,
# struct_lit_width is 40% of it, and single_line_if_else_max_width is 100% of it.

[format.rules]
function_headers = true
struct_literals = true
match_arms = true
if_values = true
```

The output leaves out the three narrower widths, so a saved copy still follows a new `max_width`.
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
A macro call keeps its bytes exactly as written, but its first line gets the indentation of the code around it.
A `macro_rules!` definition keeps its lines exactly as written.
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

## Struct literals

A struct literal stays on one line when its fields fit in `struct_lit_width` columns and the line fits in `max_width`:

```text
let p = Point { x: 1, y: 2 };
```

Otherwise each field goes on a line of its own:

```text
return Wide {
  negative: d.negative,
  exponent: d.exponent
};
```

The code after the literal on the same line counts toward `max_width`.
A literal on one line has no comma after its last field.
A literal with one field on each line keeps the comma after its last field, if it has one.
A literal stays on several lines when it holds a `{` that does not start a struct literal with fields.
Examples are a block such as `unsafe { ... }` and an empty literal such as `PhantomData<T> {}`.
A literal with a comment between its fields gets one field on each line, and each comment stays on its line.
A literal with a comment before a comma, or before its `{`, gets the base formatting.
A short `unsafe` block that holds a struct literal stays on one line, as in `unsafe { *p = Pair { a: 1, b: 2 }; }`.
This happens when the literal goes on one line and the block fits in 80 columns and in `max_width`.
The rule formats a struct literal inside a field value the same way.
The base formatting formats the field values themselves.

## Match arms

Each arm of a `match` starts a line of its own:

```text
match dot {
  Shape::Dot(d) => {
    n = d.x;
  },
  Shape::Line(u, _) => {
    n = u.y;
  }
}
```

A comment between arms stays on its line, and so does a comment at the end of an arm's line.
One blank line between arms stays.
A comma after an arm goes on the line of the arm, as in `},`.
After an arm that ends with `;`, a comma on the next line stays there.
A match with a comment before its `{`, or between an arm and its comma, gets the base formatting.

## If values

An `if` where a value goes stays on one line when each branch holds one value, the `if` fits in `single_line_if_else_max_width`, and the line fits in `max_width`:

```text
let a = if c > 0 { 1 } else { 0 };
```

A value goes after `=`, `(`, `,`, `:`, `return`, or `=>`.
A macro call such as `vec![1]` can be the value of a branch.
An `if comptime` value follows the same rules.
The base formatting gives each branch lines of its own in these cases:

- an `if` statement,
- an `if` with a statement in a branch,
- an `else if`,
- an `if` with a comment,
- an `if` with a block, an `if`, a `match` or a `return` in a branch.

## Extending the formatter

Built-in rules use an internal registry and a shared layout renderer.
See [adding a formatting rule](vx-format-rules.md) for the architecture and contributor checklist.
