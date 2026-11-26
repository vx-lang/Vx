# Adding a Vx formatting rule

The CLI uses syntax-aware dispatch through `NodeRegistry`.
One `NodeFormatter` owns each `SyntaxKind`.
Parents delegate child layouts through `NodeContext::child`.
These interfaces are internal Rust interfaces, without a stable external plugin API.

## Pipeline

1. Read configuration and retain the original source bytes.
1. Discover opaque macros and exclusions before recognizing ordinary Vx declarations.
1. Recognize supported syntax with shared parser helpers and record coverage warnings.
1. Protect preserved ranges before selecting registered syntax roots.
1. Render roots through the registry and render remaining intervals through the bounded legacy adapter.
1. Check the final output for token, adjacency, and exact-byte preservation.
   A failed check stops the formatter with an `internal formatter error`.

Normal formatting and `--check` share this pipeline.
The command adapter owns file access, configuration, warnings, and exit status.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/formatter/source/mod.rs` | Original bytes, token ranges, syntax views, and final preservation checks. |
| `src/parser/formatting.rs` | The kinds of syntax whose token ranges the compiler's parser records for the formatter. |
| `src/formatter/source/recognize.rs` | Coverage, declaration contexts, exclusions, macros, and grammatical gaps. |
| `src/formatter/source/dispatch.rs` | Syntax ownership, root discovery, and child dispatch. |
| `src/formatter/source/formatters.rs` | Built-in node formatters. |
| `src/formatter/source/document.rs` | Composable layouts, width budgets, and renderer state. |
| `src/formatter/source/ownership.rs` | Source ownership and legacy edit guards. |
| `src/formatter/source/engine.rs` | Full-file scheduling and final validation. |
| `src/formatter/legacy.rs` | Existing block, spacing, and indentation behavior within permitted intervals. |

## Adding an owner

1. Use an existing syntax view, or add one for a new construct.
   Add a kind to `RecordedKind` in `src/parser/formatting.rs`.
   Record it where the parser reads that syntax: save `self.pos` before it, and call `self.record` after it.
   Build the view from those ranges in `recognize.rs`.
1. Implement `NodeFormatter::kind` and `NodeFormatter::format` for that view.
1. Implement `roots` when the formatter owns independent source ranges.
1. Return child nodes from the parent's layout through `context.child` when the parent owns their range.
1. Register the formatter in `builtin_registry()`.
1. Implement `rule` when projects can switch the formatter off.
   Give the rule a name for `[format.rules]` and a default for each style edition.
   `--print-config default` and the check for unknown rule names then include it.
   Add the rule to the table in [the formatter guide](vx-format.md).
   A formatter for part of a larger node has no rule of its own.
1. Read widths from `Shape`, which starts at `max_width`.
   Add a narrower width setting, like `fn_signature_width`, only when a construct needs one.
1. Add other settings only for behavior the formatter implements.
   A new default belongs to a new style edition, so that existing projects keep their output.
1. Test output, preservation, nesting, exclusions, unknown syntax, and repeated-formatting stability.
1. Verify each important assertion with a deliberate implementation regression.

The engine discovers roots through the registry.
Adding an owner does not require a syntax-specific branch in the engine or CLI.
A registration-only test exercises this contract through `format_file_with_registry`.
New language constructs can require shared parser and syntax-view work.
The registry does not automatically understand new grammar.

## Layout ownership

Return a `Document` containing original tokens, spaces, line breaks, child documents, or exact source copies.
Use `Shape` for the current column, indentation, and remaining width.
Use flat and expanded alternatives when layout depends on width.
Rules do not generate replacement token spellings or mutate rendered strings.

The registry rejects duplicate owners for one syntax kind.
The scheduler rejects identical or partially overlapping roots.
An enclosing root owns its complete range and delegates children explicitly.
A root can also hand part of its range back with `Document::Region`, such as the value of a struct field.
The engine formats a region like any other code: with the roots inside it, and with the base formatting between them.
A root that wants the base formatting for its whole range returns a region of that range.
Protected source ranges take precedence over registered roots.
The engine does not schedule the roots of a switched-off rule.
Their code gets the formatting it would get if the rule did not exist.

The renderer requires each original content token exactly once, in source order.
The final validator also checks token spelling and grammatical adjacency.
A macro call retains its original bytes, but its line takes the indentation of the code around it.
A layout must put a whole macro call in one `Preserve` or `Region` document.
Excluded text retains its original bytes, including directive lines.
Unknown syntax produces preservation warnings instead of speculative formatting.
A rule reports the code that it cannot format yet through `NodeFormatter::warnings`.
The engine asks for these warnings only when the rule is on.

See [the source architecture guide](vx-format-source-architecture.md) for state handoffs and coverage boundaries.
