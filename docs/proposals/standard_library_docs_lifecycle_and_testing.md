<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Proposal: standard-library documentation, API lifecycle, and testing

**Status:** draft for review, 2026-11-17.
**Release context:** new APIs from the current work target their first release in Vx 0.0.5.
**Scope:** library API policy, documentation tools, executable examples, release metadata, and tests.
**Builds on:** [The standard library RFC](standard_library_tenets_and_layout.md).

The RFC sets the rules: the life-cycle stages, the requirements for stability, and the documentation rules.
This document proposes how to enforce and test those rules.
Where the two documents overlap, the RFC decides.

This document proposes requirements. It does not implement attributes, compiler gates, a documentation runner, or new tests.
It does not declare the current I/O APIs stable.

## 1. Proposed decisions

- Keep API contracts and examples beside their declarations in `///` comments.
- Generate reference pages from the compiler's resolved declarations and structured metadata.
- Compile and run ordinary documentation examples by default.
- Require explicit modes and reported reasons for examples that cannot run.
- Give new public APIs lifecycle metadata and an introduction version.
- Use **Incubating** for APIs that require feature opt-in and may change incompatibly.
- Record first availability separately from stabilization and deprecation.
- Test public behavior separately from compiler internals and runtime bridges.
- Add property tests, injected failures, and benchmarks where the API needs them.
- Establish a baseline for existing APIs without automatically declaring them stable.

Rust supplies useful practices, but matching its API names does not establish equivalent safety, coverage, or maintenance discipline.
Vx should make each published guarantee depend on a test or an explicit review decision.

## 2. Current support and gaps

These observations describe the inspected working tree, including unreleased file I/O changes.
They do not establish what an earlier Vx release shipped.

| Area | Current support | Gap |
| --- | --- | --- |
| API lifecycle | No lifecycle annotations or caller opt-in gates exist today. | Add structured metadata and compiler enforcement. |
| API comparison | [`vx-api diff`](../../src/bin/vx-api.rs) compares two `.vxlib` interfaces and prints the smallest safe version bump. | CI does not run it. It does not know lifecycle states. |
| Source documentation | The parser retains `///` comments on declarations. | No required documentation or example policy for every public API. |
| API reference | [The reference generator](../../scripts/tools/gen_stdlib_reference.py) reads library source signatures and comments. | It uses a separate source scanner, rather than the compiler's resolved declaration model. |
| Documentation examples | [The example checker](../../scripts/tools/check_doc_examples.py) scans Markdown and HTML in CI. | Ordinary examples receive `--parse-only`. Only explicit `run` blocks compile and execute. |
| Source comment examples | Comments can appear in the generated reference. | The checker does not extract examples directly from `.vx` comments. |
| Example discovery | The checker recognizes declarations and excludes apparent Rust syntax or incomplete fragments. | Valid examples can disappear through heuristic exclusions. Counts do not establish complete API coverage. |
| Workflow examples | `book_*` fixtures support book chapters, including [file I/O](../../tests/backend/pass/book_io_write_and_read_a_file.vx). | There is no uniform example index for the full public library. |
| Behavioral tests | Backend fixtures, Rust bridge tests, and checks for both code generators exist. | Tests do not form a complete inventory of public API contracts. |
| Generated tests | [Differential fuzzing](../../scripts/tools/differential_fuzz/README.md) compares generated Vx programs and Rust counterparts. | Compiler-focused generators do not replace properties for each library module. |
| Concurrency | [Library model checks](../../tests/integration_test/library_model_check.rs) and seeded task tests exist. | A handwritten model does not execute the actual Vx implementation. |
| Version metadata | API history appears in prose and release notes. | No structured per-item introduction or stabilization metadata exists. |

The [CI workflow](../../.github/workflows/ci.yml) currently checks documentation examples before generating the stdlib reference.
That order cannot establish that examples in newly generated reference content work.
The source-comment runner proposed below removes that dependency.
Published summaries must distinguish parsing, compilation, execution, and omission.

One existing test weakness needs separate work during rollout.
[The parser property tests](../../tests/integration_test/fuzz.rs) discard the result of `catch_unwind`.
They therefore cannot establish their claimed absence of parser panics.
This proposal records the gap without changing those tests.

## 3. Practices to borrow from Rust

Rustdoc generates API documentation and executes documentation examples as tests.
It supports hidden setup and explicit execution modes.
Assertions make example results testable. See [the rustdoc book](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html).

Rust's standard library uses compiler-supported stability metadata.
Unstable library features require consumer opt-in, while stabilization records a version.
These standard-library attributes are internal Rust infrastructure, not general stable attributes for arbitrary crates.
See [Rust stability attributes](https://rustc-dev-guide.rust-lang.org/stability.html).

Rust deprecation diagnostics identify discouraged APIs and can describe a replacement.
Rustdoc also shows deprecation metadata. See [the Rust Reference](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute).

Rust combines package tests, documentation tests, and compiler tests.
Its standard library relies heavily on documentation tests.
See [Rust's test overview](https://rustc-dev-guide.rust-lang.org/tests/intro.html).
[Core tests](https://github.com/rust-lang/rust/blob/main/library/coretests/Cargo.toml) also define dedicated benchmark targets.

[Miri](https://github.com/rust-lang/miri) detects undefined behavior in supported Rust programs and found bugs in Rust's standard library.
It does not execute every native operating-system interaction.
Vx should retain native platform tests even if it later gains a comparable interpreter.

These practices describe a direction for Vx. This proposal makes no claim that Vx's current tests equal Rust's coverage.
The [core-library parity matrix](../implementation_plans/core_library.md#appendix-a-parity-matrix) tracks API coverage, not testing maturity.

## 4. API lifecycle

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
Types and traits need working gates before the new I/O family can claim complete incubation enforcement.

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

## 5. Availability since Vx 0.0.5

For new APIs in this work, `introduced = 0.0.5` means first inclusion in a released standard library.
It does not mean stabilization.
An incubating API can first ship in 0.0.5 and stabilize in a later release.
Record both dates without rewriting its introduction history.

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
The tracking issue would need to adopt the incubation work before serving as its lifecycle record.

```json
{
  "item": "std::io::Error::from_raw_os_error",
  "introduced": "0.0.5",
  "stability": "incubating",
  "feature": "file_io_errors",
  "tracking_issue": "https://github.com/vx-lang/Vx/issues/1341",
  "stable_since": null,
  "deprecated_since": null
}
```

Existing `File` history remains unchanged when a new method first ships in 0.0.5.
A new migration name also needs its own introduction version.
Record the old path and replacement in the migration notes.
Investigate unknown history instead of assigning every existing API version 0.0.5.

During implementation, reject malformed versions and impossible lifecycle ordering.
Compare new public APIs against the previous release's API inventory with `vx-api diff`.
Reject missing metadata rather than relying on a reviewer to remember a prose label.
Do not manufacture past stabilization dates when establishing the inventory of older APIs.

Before annotation support exists, use consistent documentation fields as a temporary record.
Show that these fields do not enforce feature opt-in.
Do not render a compiler-enforced incubation badge from prose alone.

The 0.0.5 release review must decide what to do if enforcement is incomplete.
The preferred outcome is working gates before publishing APIs under the Incubating contract.
Otherwise, defer exposure or approve an explicit release exception that describes the lack of enforcement.
This proposal does not promise that every implementation stage fits the 0.0.5 release.

## 6. Documentation and example policy

### Documentation beside the API

[RFC section 1.8](standard_library_tenets_and_layout.md#18-documentation) sets the rules for doc comments, examples, and their exceptions.
This section adds the following points.

Document ownership, borrowing, invalidation, initialization, and cleanup where they matter.
When one example covers a group of items, the inventory records each item in the group.
Do not add repeated examples that merely call an API without checking its behavior.

Show the relevant unsafe boundary in rendered examples.
Hidden setup may prepare a temporary resource, but must not conceal the caller's safety obligation.
Examples must not teach unchecked pointer bookkeeping as the normal safe application interface.
Document unavailable safe abstractions as prerequisites, rather than replacing them with misleading wrappers.

### Three forms of examples

| Form | Location | Purpose |
| --- | --- | --- |
| API example | Declaration's `///` documentation | Demonstrate one API or a small related group. |
| Workflow example | A tested `.vx` fixture included by the book or guide | Demonstrate complete tasks, including error handling and cleanup. |
| Behavioral test | The library test suite | Cover boundaries and failures that would distract from the teaching example. |

Build a searchable example index by module and API.
Reuse a workflow fixture through an include instead of copying it into several documents.
Resolve includes and anchors before testing the published example.
Keep source locations so failures identify the original fixture or comment.

The current file-I/O book fixture demonstrates this arrangement.
The proposal does not require moving every useful backend fixture into a new directory immediately.
Classify existing fixtures first, then change their location only when it improves discovery or execution.

## 7. Executable documentation tooling

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
Do not mark unsupported syntax as a successful negative test for borrowing or API safety.

A default run needs meaningful assertions or checked output for the behavior it describes.
A test about compilation alone should say so explicitly.
Keep visible example lines identical between the renderer and test input.
Add only declared hidden setup or the generated wrapper.

### Execution and isolation

Use the compiler and runtime built for the test job.
Reject stale runtime artifacts that could conceal a newly changed bridge.
Run examples as ordinary library consumers, without privileged access that application code lacks.
Never enable every incubating feature globally for documentation tests.
Each incubating example must declare the feature it actually needs.

Use isolated temporary directories, bounded execution times, and deterministic inputs.
Clean resources after success, failure, panic, and timeout.
Avoid external services or personal files in default examples.
Provide controlled fixtures for network, time, filesystem, and hardware requirements.
Limit parallel jobs according to the repository's core-use rule.

Exercise both code generators for supported examples.
Check the intended code generator and report unexpected fallback as a failure.
Run optimized variants where behavior or safety depends on optimization.
At release time, run the changed library examples at `-O0` and `-O3` on supported paths.

A platform skip needs a declared requirement and a job that exercises the supported platform.
A target that can only compile examples must report compilation separately from native execution.
A missing tool, missing hardware device, or missing compiler is not an executed pass.

### Reporting and published output

Report discovered, run, compile-only, expected-failure, expected-panic, ignored, and failed examples separately.
Track example identities and coverage of public items, rather than relying only on totals.
Fail an unexpected empty suite or an unexplained loss of examples.
Require review when an executable example becomes ignored or changes to a weaker mode.

Check that generated pages include the correct examples, lifecycle labels, versions, signatures, and links.
Test the hidden-line transformation and include expansion themselves.
A rendering bug must not publish code different from the tested example.

Generate the reference before checking generated pages in the transitional toolchain.
Once direct source extraction exists, test that the reference and runner consume the same inventory.
Publish versioned reference pages from the matching release artifacts.
Examples for a released API must not depend accidentally on a newer checkout.

## 8. Standard-library testing policy

### Layers of evidence

| Layer | What it establishes | Example checks |
| --- | --- | --- |
| Public contract tests | Observable behavior through supported imports. | Counts, values, error kinds, side effects, and empty inputs. |
| Negative compiler tests | Rejection of invalid programs for the intended reason. | Invalid borrowing, moves, types, and missing feature opt-ins. |
| Runtime bridge tests | The foreign-function interface agrees on types and safety obligations. | Output initialization, native errors, integer widths, and ownership transfer. |
| Property and model-based tests | Invariants across generated values or operation sequences. | Collection contents, iterator consumption, arithmetic boundaries, and I/O progress. |
| Platform tests | Behavior on each supported operating system and target. | Non-UTF-8 Unix paths, file modes, positioning, and host error classification. |
| Concurrency tests | Allowed behavior under different schedules. | Cancellation, wakeups, cleanup, ordering, and deadlock reporting. |
| Safety-tool checks | Memory and synchronization properties within the tool's actual coverage. | Invalid access, leaks, races, and repeated resource release. |
| Benchmarks | Measured performance and resource use. | Allocations, copies, syscall counts, throughput, and latency. |

An internal bridge test does not substitute for a test through `File`.
A compiler representation check does not establish that the program returns the right bytes.
Two code generators can share a frontend defect, so agreement alone is insufficient.
Handwritten models need checks against the actual library implementation.

### Minimum evidence for new APIs

Every new public API needs a contract test or an explicit association with a meaningful shared test.
Cover normal behavior, boundaries, empty cases, and relevant failures.
Test generics, imports, and compiled interfaces where they affect the contract.
Test ownership and cleanup through success and error branches for APIs that own resources.

For new or materially changed assertions, follow the repository's mutation requirement.
Temporarily break the relevant behavior and identify the assertion that fails.
Compilation failure, zero selected tests, and unexpected code-generator fallback do not establish a runtime assertion's coverage.
Restore the implementation and rerun the affected tests.

Compiler-rejection tests need a separate proof that they check the intended diagnostic.
Change the rejected condition or expected diagnostic in a controlled way and check that the test detects the difference.
A test's prose must describe only what its assertions establish.

### Property tests and independent models

Use deterministic seeds and retain failing inputs as regression tests.
Reduce a generated failure to the smallest useful input where the tooling supports it.
Run a bounded seed set on pull requests and a wider set in scheduled jobs.
Record the generator version, seed, compiler options, platform, and code generator in failure reports.

Start with these properties:

- Collections preserve a simple reference model across insert, remove, clear, and lookup sequences.
- Iterators preserve values and consumption counts across adapters and both traversal directions.
- Checked arithmetic agrees with a wider reference calculation at type boundaries.
- I/O helpers preserve byte order and progress across generated short-transfer and error sequences.
- Owning values release each acquired resource exactly once across success and error paths.

Compare with Rust only where Vx and Rust promise the same behavior.
Do not compare hash-table iteration order when neither API guarantees it.
Floating-point comparisons need explicit rules for rounding, tolerances, NaNs, infinities, and signed zero.
Use an independent oracle where two implementations could share the same defect.

### Controlled failures and platform behavior

Use fake readers, writers, clocks, and allocators to force uncommon conditions deterministically.
For I/O, cover short transfers, interruption, EOF, zero-progress writes, empty requests, and errors after partial progress.
Check cleanup when construction fails and when errors propagate through `Result`.
Keep native integration tests alongside these controlled tests.

Only exercise invalid raw-pointer states that the API explicitly accepts, such as a documented null argument.
Do not dereference arbitrary dangling or misaligned pointers in an ordinary test and call the resulting crash a safety check.
Specialized invalid-memory tests need an appropriate tool and a precise expected failure.

Portable tests should assert portable categories and preserved native-code presence.
Host-specific tests can assert exact native classifications under an explicit platform condition.
Keep cross-compilation checks distinct from execution on that host.

### Safety, concurrency, and performance

Use sanitizer checks where instrumentation actually covers the code under test.
Instrumenting the Rust bridge alone does not establish memory safety for code generated from Vx.
A future undefined-behavior interpreter would complement these checks, rather than replace host tests.
Do not claim a general Miri-equivalent capability for the current Vx toolchain.

Continue seeded schedule tests and library model checks.
Test cancellation and failure while resources or buffers remain borrowed.
Keep the model synchronized with the implementation and test known incorrect models as checks of the search itself.

Benchmark APIs where copying, allocation, buffering, or syscall behavior matters.
State workload, compiler options, host, code generator, and measurement method.
Use deterministic allocation or syscall counters for regression gates where possible.
Treat noisy timing results as measurements requiring review, rather than universal thresholds.
Separate correctness gates from performance budgets.

### Devices

TODO: This proposal does not cover code that runs on a device yet.
[RFC section 5](standard_library_tenets_and_layout.md#5-where-library-code-runs) requires the following, and this proposal must add the enforcement:

- Each function that runs on a device has an example and a test inside a `spawn on` region (RFC D3 and R6).
- CI runs those examples and tests on each device target that it has.
- The metadata records where an item runs, separate from the `platforms` field for operating systems.
- A report shows an example that only compiled for a device separately from one that ran on it.

## 9. CI and release checks

| Cadence | Proposed checks |
| --- | --- |
| Pull request | Changed API inventory, metadata, documentation examples, focused contracts, negative tests, bounded property tests, and relevant bridge tests. |
| Pull request affecting compiler or imported libraries | Required repository preflight, both supported code generators, and checks against unexpected fallback. |
| Scheduled | Wider seeds, longer schedule exploration, supported platform jobs, safety tools, and benchmark trends. |
| Release candidate | Full supported matrix, versioned reference, example inventory, compatibility report, and tests using the packaged compiler and library. |

Use the repository's build, test, formatting, and code-generator checks for affected changes.
The proposed policy adds coverage and reporting requirements beyond those checks.

A release report should identify new, changed, deprecated, removed, and still-incubating APIs.
It must name approved skips and lifecycle exceptions.
Do not count a passing build as evidence that every public API received tests.
Do not count an API-presence row in the parity matrix as evidence of behavioral correctness.

## 10. Delivery stages

| Stage | Deliverable | Exit evidence |
| --- | --- | --- |
| 1. Establish the baseline | Classify existing APIs, examples, skips, and metadata gaps. Record new APIs targeting 0.0.5. | A reviewed inventory distinguishes unknown history from newly introduced APIs. CI reports parsing and execution honestly. |
| 2. Add lifecycle support | Declaration metadata, diagnostics, consumer opt-in, deprecation, interface serialization, and documentation labels. | A small API family passes gate tests across relevant item kinds and every supported import path. |
| 3. Add executable source documentation | Shared declaration model, example extraction, execution modes, includes, and source mappings. | Invalid imports, wrong results, intended panics, diagnostic mismatches, and empty suites all produce the correct failures. |
| 4. Require evidence for new APIs | Public contract associations, meaningful examples, version checks, and controlled failure tests. | CI rejects a new API without required metadata, documentation, or tests. |
| 5. Expand testing depth | Module properties, wider platform jobs, safety tools, and performance records. | Failures reproduce from saved inputs, and reports state the actual scope of each check. |

Stages can progress independently where their dependencies allow it.
Do not wait for a new documentation command to improve existing examples and public tests.
Do not mark a feature as compiler-gated before the compiler enforces its actual public surface.

For 0.0.5, review every newly shipped API for introduction metadata, documented stability, meaningful tests, and migration notes.
Execute the new API examples with the release candidate, even during the documentation-tool transition.
Resolve or explicitly approve the incubation-enforcement decision before publishing the release.

## 11. Applying the policy to file I/O

Use the current file APIs as the first complete example of this policy after the required tooling exists.
The [file I/O plan](../implementation_plans/file_io_usability.md) separates implemented raw APIs from gated safe interfaces.
Keep that distinction in reference pages, examples, and release notes.

The initial inventory should include `OpenOptions`, `SeekFrom`, `Seek`, `Error`, their public methods, and the new raw file methods.
Assign 0.0.5 only to APIs that first ship under those names in that release.
Preserve the earlier history of `File` itself.
Record the renamed legacy methods and explain their migration path.

The first examples should demonstrate named opening options, typed seeking, structured errors, and resource cleanup.
Keep raw-pointer examples visibly unsafe.
Do not publish unavailable safe `Read` or `Write` examples as executable examples.
Mark design sketches explicitly and exclude them from completed API coverage.

Public and controlled tests should establish the following behavior:

- Opening a missing file reports an error distinct from reading an empty file.
- Write access alone preserves contents, while explicit creation and truncation have their documented effects.
- Exclusive creation rejects an existing path, and append respects its documented positioning behavior.
- Relative seeks accept signed offsets, while absolute positions retain their unsigned meaning.
- Errors retain their category, operation context, and available native code.
- Short transfers and empty requests have correct counts and memory effects.
- Handles close exactly once through success, failure, and error propagation.

Associate each listed contract with a public API test and its assertions.
A fixture's existence or name does not establish coverage.

## 12. Review decisions still needed

RFC question Q10 holds the decisions on the opt-in spelling, the attribute schema, and the temporary representation.

1. Define provider ownership and the exact boundary for internal-use exemptions.
1. Approve the compatibility and exception policy before 0.0.5 publication.
1. Choose how compiled libraries carry documentation and lifecycle metadata.
1. Assign maintainers for API reviews, documentation exceptions, and scheduled test failures.

Acceptance of this document authorizes a policy direction, not a claim that any implementation stage is complete.
