<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Proposal: enforcing the standard library's documentation and life-cycle rules

**Status:** draft for review, 2026-11-17.
**Scope:** life-cycle metadata, compiler gates, executable documentation examples, and CI checks.
**Builds on:** [The standard library RFC](standard_library_tenets_and_layout.md).

The RFC sets the rules: the life-cycle stages, the requirements for stability, and the documentation rules.
This document proposes how to enforce those rules.
Where the two documents overlap, the RFC decides.

This document proposes requirements. It does not implement attributes, compiler gates, a documentation runner, or new tests.
It does not declare any current API stable.

## 1. Proposed decisions

- Keep API contracts and examples beside their declarations in `///` comments.
- Generate reference pages from the compiler's resolved declarations and structured metadata.
- Compile and run ordinary documentation examples by default.
- Require explicit modes and reported reasons for examples that cannot run.
- Give new public APIs lifecycle metadata and an introduction version.
- Record first availability separately from stabilization and deprecation.
- Establish a baseline for existing APIs without automatically declaring them stable.

## 2. Current support and gaps

| Area | Current support | Gap |
| --- | --- | --- |
| API lifecycle | No lifecycle annotations or caller opt-in gates exist today. | Add structured metadata and compiler enforcement. |
| API comparison | [`vx-api diff`](../../src/bin/vx-api.rs) compares two `.vxlib` interfaces and prints the smallest safe version bump. | CI does not run it. It does not know lifecycle states. |
| Source documentation | The parser retains `///` comments on declarations. | No required documentation or example policy for every public API. |
| API reference | [The reference generator](../../scripts/tools/gen_stdlib_reference.py) reads library source signatures and comments. | It uses a separate source scanner, rather than the compiler's resolved declaration model. |
| Documentation examples | [The example checker](../../scripts/tools/check_doc_examples.py) scans Markdown and HTML in CI. | Ordinary examples receive `--parse-only`. Only explicit `run` blocks compile and execute. |
| Source comment examples | Comments can appear in the generated reference. | No CI step compiles them (#3640). |
| Example discovery | The checker recognizes declarations and excludes apparent Rust syntax or incomplete fragments. | Valid examples can disappear through heuristic exclusions. Counts do not establish complete API coverage. |
| Version metadata | API history appears in prose and release notes. | No structured per-item introduction or stabilization metadata exists. |

**Prior art.** Rustdoc runs documentation examples as tests, with hidden setup lines and explicit modes ([the rustdoc book](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html)).
Rust's standard library marks each item with compiler-checked stability metadata ([Rust stability attributes](https://rustc-dev-guide.rust-lang.org/stability.html)).
Its `#[deprecated]` attribute names a replacement, and rustdoc shows it ([the Rust Reference](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute)).

## 3. API lifecycle

### Enforcement of each stage

[RFC section 6](standard_library_tenets_and_layout.md#6-the-life-cycle-of-an-api) defines the stages, the deprecation notice, and the requirements for stability.
This table gives only the proposed enforcement.

| Stage or notice | Proposed enforcement |
| --- | --- |
| Internal | The name starts with `_` until the committee decides on visibility (#489). Report other reachable items as a known gap. |
| Incubating | Require an explicit named feature opt-in across the supported library boundary. |
| Stable | Reject unreviewed incompatible API changes in the release comparison. Build that comparison on `vx-api diff`. |
| Deprecated notice | Warn on use and show deprecation metadata in the reference. |
| Removed | Preserve a migration record and explain obsolete feature opt-ins where possible. |

Passing tests alone does not approve an API design.
A release that changes an incubating API must still document the migration.

Internal and removed states describe publication policy, rather than permission to dereference memory or bypass type checking.
A raw or unsafe API can have any supported lifecycle state.
An incubation gate never makes an unsafe operation safe.

### Items and use paths

Lifecycle metadata must cover functions, methods, types, traits, constants, modules, public fields, variants, and public macros where supported.
Review new trait implementations and blanket implementations as API changes too.
A module's original version must not hide a newer method's version.

Check resolved definitions and every exposed path.
An import alias, re-export, generic instance, or function reference must not bypass a gate.
Reject stable signatures that expose gated types without an approved policy.

Vx has no item visibility today (#489).
Until the committee decides on visibility, an item whose name starts with `_` is internal, as the RFC says.
Every other item is public, and the API inventory treats it as public.
The compiler does not enforce the `_` convention, so the inventory reports this limitation.

### Compiler implementation

Vx has no lifecycle annotations today.
The proposed mechanism must cover the supported public API kinds, including types and traits.
RFC question Q10 decides the source syntax, the attribute schema, and the spelling of the opt-in.

The implementation needs the following work:

1. Parse and check structured attributes beside documentation comments.
1. Collect lifecycle metadata before function-body checking.
1. Carry it through resolution, the frozen registry, specialization, and imports.
1. Preserve it in `.vxlib` interfaces and reject incompatible interface formats.
1. Check calls, function references, type uses, trait uses, fields, variants, and alternate paths.
1. Emit source-located diagnostics for missing opt-ins and deprecated uses.
1. Export the same metadata to documentation and API comparison tools.

Keep this metadata out of symbol identity and type layout.
Changing a lifecycle state must not change the ABI by itself.
The design must follow [the parallel compiler's declaration rules](../language_rules_for_a_parallel_compiler.md).

Use resolved provider ownership for internal-use exemptions.
The implementation must not infer ownership from a matching module name or a filesystem substring.
A defining-module exemption could support the first narrow rollout.
A broader library exemption needs an explicit ownership model before adoption.
A dependency's feature opt-ins must never silently enable those features for its consumer.

Test malformed attributes, unknown features, conflicting states, missing opt-ins, and successful opt-ins.
Repeat the relevant cases through source imports, compiled interfaces, generic calls, methods, and parallel compilation.
Keep library API feature gates distinct from gates for new language syntax.

## 4. Lifecycle metadata

`introduced` means first inclusion in a released standard library.
It does not mean stabilization.
An incubating API can first ship in one release and stabilize in a later one.
Record both versions without rewriting its introduction history.

| Field | Meaning |
| --- | --- |
| `introduced` | First released library version containing this supported API path. |
| `stability` | Base lifecycle state, independent of the deprecation notice. |
| `feature` | Named opt-in for an incubating API. |
| `tracking_issue` | Review and stabilization discussion for that feature. |
| `stable_since` | Release that first committed to the stable contract, if any. |
| `deprecated_since` | Release that first deprecated this API, if any. |
| `replacement` | Preferred replacement or migration instructions, when applicable. |
| `platforms` | Supported hosts or targets, including documented restrictions. |

The following record illustrates the proposed metadata model. It is not implemented compiler syntax.

```json
{
  "item": "std::io::Error::from_raw_os_error",
  "introduced": "<the release that first ships it>",
  "stability": "incubating",
  "feature": "file_io_errors",
  "tracking_issue": "https://github.com/vx-lang/Vx/issues/1341",
  "stable_since": null,
  "deprecated_since": null
}
```

A new method on an existing type gets its own introduction version, and the type keeps its history.
A new name for an old API also needs its own introduction version.
Record the old path and replacement in the migration notes.
Investigate unknown history instead of assigning every existing API the same version.

During implementation, reject malformed versions and impossible lifecycle ordering.
Compare new public APIs against the previous release's API inventory with `vx-api diff`.
Reject missing metadata rather than relying on a reviewer to remember a prose label.
Do not manufacture past stabilization dates when establishing the inventory of older APIs.

Before annotation support exists, use consistent documentation fields as a temporary record.
Show that these fields do not enforce feature opt-in.
Do not render a compiler-enforced incubation badge from prose alone.

## 5. The example runner

### Where examples live

| Form | Location | Purpose |
| --- | --- | --- |
| API example | Declaration's `///` documentation | Demonstrate one API or a small related group. |
| Workflow example | A tested `.vx` fixture included by the book or guide | Demonstrate complete tasks, including error handling and cleanup. |
| Behavioral test | The library test suite | Cover boundaries and failures that would distract from the teaching example. |

Build a searchable example index by module and API.
When one example covers a group of items, the index records each item in the group.
Reuse a workflow fixture through an include instead of copying it into several documents.
Resolve includes and anchors before testing the published example.
Keep source locations so failures identify the original fixture or comment.

### One declaration and documentation model

A future `vx-doc` should consume the compiler's declaration and documentation model.
The name describes proposed tooling, not an existing command.
Emit a structured API inventory for the reference, example index, lifecycle checks, and release comparison.
Do not maintain separate interpretations of signatures in each tool.

Preserve the required information for source modules and compiled interfaces.
Decide which documentation fields belong in `.vxlib` and which can travel in an associated documentation artifact.
Missing metadata must not silently turn an imported incubating API into a stable API.

### Example discovery and preparation

Extract explicitly marked Vx examples from source comments, the book, and supported included fixtures.
Use the language label instead of guessing whether syntax looks like Rust.
Keep Rust comparison blocks as Rust, rather than attempting to compile them as Vx.
Relabel existing Vx examples fenced as Rust during migration. Included `.vx` fixtures must retain their language identity.

Allow examples to declare complete-program or statement-snippet form.
Wrap statement snippets in a generated entry function when needed.
Preserve imports, declaration placement, feature opt-ins, hidden setup, and source line mappings.
An extraction failure is a failed documentation check, not an implicit skip.

### Test modes

The names below describe proposed behavior. They are not current fence syntax or implemented flags.

| Mode | Required result |
| --- | --- |
| `run` | Default for executable examples. Compile, link, run, and check assertions or declared output. |
| `no_run` | Compile and link for the declared target, but do not execute. Require a reason. |
| `compile_fail` | Require the intended compiler diagnostic at the relevant use site. Successful compilation fails the test. |
| `should_panic` | Require the intended panic marker. An unrelated crash, signal, or timeout does not pass. |
| `ignore` | Do not compile or run. Require a reason, tracking reference, and review condition. |
| `text` | Explicit design sketch or non-code content. Exclude it from executable API coverage. |

A compile-failure example must not pass because of a misspelled import or unrelated syntax error.
Check a diagnostic code and the relevant location instead of matching every word of its message.

A default run needs meaningful assertions or checked output for the behavior it describes.
Keep visible example lines identical between the renderer and test input.
Add only declared hidden setup or the generated wrapper.

### Execution and isolation

Use the compiler and runtime built for the test job.
Run examples as ordinary library consumers, without privileged access that application code lacks.
Never enable every incubating feature globally for documentation tests.
Each incubating example must declare the feature it actually needs.

Use isolated temporary directories, bounded execution times, and deterministic inputs.
Avoid external services or personal files in default examples.
Limit parallel jobs according to the repository's core-use rule.

Run each example on both code generators (RFC D3).
Report an unexpected fallback to the other code generator as a failure.
A missing tool, missing hardware device, or missing compiler is not an executed pass.

### Reporting and published output

Report discovered, run, compile-only, expected-failure, expected-panic, ignored, and failed examples separately.
Track example identities and coverage of public items, rather than relying only on totals.
Fail an unexpected empty suite or an unexplained loss of examples.
Require review when an executable example becomes ignored or changes to a weaker mode.

Check that generated pages include the correct examples, lifecycle labels, versions, signatures, and links.
A rendering bug must not publish code different from the tested example.
Test that the reference and the runner consume the same inventory.
Publish versioned reference pages from the matching release artifacts.

## 6. CI gates

| When | Proposed checks |
| --- | --- |
| Pull request | Changed API inventory, lifecycle metadata, documentation examples, missing opt-ins, and `vx-api diff` against the previous release. |
| Release candidate | Versioned reference, example inventory, and compatibility report, using the packaged compiler and library. |

A release report should identify new, changed, deprecated, removed, and still-incubating APIs.
It must name approved skips and lifecycle exceptions.

## 7. Delivery stages

| Stage | Deliverable | Exit evidence |
| --- | --- | --- |
| 1. Establish the baseline | Classify existing APIs, examples, skips, and metadata gaps. | A reviewed inventory distinguishes unknown history from newly introduced APIs. CI reports parsing and execution honestly. |
| 2. Add lifecycle support | Declaration metadata, diagnostics, consumer opt-in, deprecation, interface serialization, and documentation labels. | A small API family passes gate tests across relevant item kinds and every supported import path. |
| 3. Add executable source documentation | Shared declaration model, example extraction, execution modes, includes, and source mappings. | Invalid imports, wrong results, intended panics, diagnostic mismatches, and empty suites all produce the correct failures. |
| 4. Require metadata and examples for new APIs | Version checks, metadata checks, and example checks in CI. | CI rejects a new API without required metadata or documentation. |

Stages can progress independently where their dependencies allow it.
Do not mark a feature as compiler-gated before the compiler enforces its actual public surface.

## 8. Not in this document

This document does not set a testing policy for the standard library.
It also does not enforce the device rules of [RFC section 5](standard_library_tenets_and_layout.md#5-where-library-code-runs), such as an example inside a `spawn on` region (RFC D3 and R6).
Both come later, in one proposal, because most of the testing work that is specific to Vx is about devices.

## 9. Review decisions still needed

RFC question Q10 holds the decisions on the opt-in spelling, the attribute schema, and the temporary representation.

1. Define provider ownership and the exact boundary for internal-use exemptions.
1. Choose how compiled libraries carry documentation and lifecycle metadata.
1. Assign maintainers for API reviews and documentation exceptions.
