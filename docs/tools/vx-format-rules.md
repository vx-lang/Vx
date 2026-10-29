# Adding a Vx formatting rule

The formatter has an internal rule interface and an explicit registry.
Function declaration layout is the first registered rule.
Existing block normalization, spacing, and fallback indentation remain in the original formatting pipeline.
External plugin loading and a stable third-party API are outside this change.

A [syntax-aware architecture proposal](../discussions/vx_format_syntax_architecture_proposal.md) is pending review.
It describes a possible migration to structure-based dispatch and nested layout composition.
The implementation below remains the current architecture.

## Pipeline

1. Tokenize the source and apply the existing normalization and spacing passes.
1. Build shared delimiter and formatting-exclusion information.
1. Visit tokens in source order and consult the rule registry at eligible positions.
1. Let enabled rules propose a layout using immutable source and configuration context.
1. Render the accepted layout through the shared renderer, or use the existing fallback formatting.
1. Verify lexical token preservation before the existing final whitespace cleanup when a rule changes layout.

Normal formatting and `--check` use this same pipeline.
The command adapter handles configuration and file writes.
Rules perform neither operation.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/formatter/context.rs` | Tokens, matched delimiters, exclusions, configuration, and current renderer position. |
| `src/formatter/registry.rs` | Rule interface, registration, selection, and deterministic execution order. |
| `src/formatter/layout.rs` | Token and whitespace instructions, replacement ownership, and structural validation. |
| `src/formatter/renderer.rs` | Rendering and lexical token-preservation checks. |
| `src/formatter/rules/` | Individual rule implementations. |
| `src/formatter/framework_tests.rs` | Shared extension and protection contracts. |

## Rule interface

Implement `FormatRule` in a new module under `src/formatter/rules/`.
Use `function_signatures.rs` as an example.

The interface provides these methods:

- `name()` returns a unique, stable internal name.
- `starts_at(token)` cheaply identifies positions worth inspecting.
- `enabled(config)` can select behavior from typed configuration. Its default enables the rule.
- `apply(context, layout)` examines the source and proposes a `LayoutEdit` through `LayoutPlan::propose`.

The context provides the current token index, output column, enclosing indentation, and resolved settings.
Shared delimiter matching covers parentheses, square brackets, and braces.
Rules can inspect original token kinds without parsing printed output.

## Layout ownership

A layout replaces a token range beginning at the current position.
Its instructions can reference original tokens, insert a space, or start a line at an indentation level.
The shared renderer converts indentation levels using the configured indentation width.
Rules cannot supply arbitrary source strings.

Every non-whitespace token in the range must appear exactly once, in its original order.
The range must include complete delimiter pairs and must not include the end-of-file token.
Formatting-disabled tokens prevent the proposal from being accepted.
The renderer also checks lexical tokens across replacement boundaries to detect accidental token merging.

Registration order is explicit and deterministic.
Duplicate names and competing proposals at the same position cause assertions with clear messages.
These conditions indicate an implementation error in built-in rules.
They do not silently select one rule's output.

An accepted replacement owns its complete range.
The renderer resumes at the first token after that range.
Rules therefore do not run separately inside a range owned by another rule.
When rules need to cooperate inside one structure, compose their layout decisions within its owning rule.

## Contributor checklist

1. Add the rule module and implement `FormatRule`.
1. Declare the module in `src/formatter/rules/mod.rs`.
1. Add the rule to `builtin_rules()` in `src/formatter/registry.rs` at its intended execution position.
1. Add typed settings to `vx-config` only for behavior the rule implements.
1. Update default configuration output and user documentation when settings change.
1. Add focused cases for output, boundaries, comments, exclusions, nesting, and non-ASCII text where relevant.
1. Verify source-token preservation and repeated-formatting stability.
1. Add CLI coverage when the rule adds configuration or command behavior.
1. Run the focused tests and required mutation checks when test execution is authorized.

New rules do not require branches in the main renderer or changes to the CLI driver.
The existing signature behavior remains covered by formatter and CLI assertions.
