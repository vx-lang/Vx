# Vx documentation

A map of what is here. The directory holds three different kinds of writing and they are worth
telling apart: the **specification** says what the language is, the **guides** teach it, and the
**design notes** record why it was built the way it was — including decisions that were rejected
and work that is not finished.

If you are just trying to use Vx, start at <https://vxlang.org/docs/>, which is the book: install,
a first program, a language tour, and the standard-library and diagnostic references. This
directory is the deeper material behind it.

## Specification

What the language *is*. These are normative.

| Document | Covers |
| --- | --- |
| [`lang/grammar.md`](lang/grammar.md) | The full EBNF grammar |
| [`lang/syntax.md`](lang/syntax.md) | Syntax, with worked examples of each construct |
| [`lang/types.md`](lang/types.md) | The type system: placement, linearity, typestates |
| [`lang/semantics.md`](lang/semantics.md) | Evaluation rules |
| [`lang/abi.md`](lang/abi.md) | Calling convention and C interoperability |
| [`lang/formal_semantics.md`](lang/formal_semantics.md) | The formal model |
| [`lang/seam_obligations.md`](lang/seam_obligations.md) | What a seam obliges a program to prove |
| [`lang/hosts_and_machines.md`](lang/hosts_and_machines.md) | Machine and host declaration files |
| [`lang/hardware_monad.md`](lang/hardware_monad.md) | The topology algebra behind placement |
| [`semantics/`](semantics/) | Operational, denotational and axiomatic treatments |

## Guides

| Document | Covers |
| --- | --- |
| [`INSTALL.md`](INSTALL.md) | Building the compiler from source, on macOS and Linux |
| [`tutorial.md`](tutorial.md) | A short tour of the language |
| [`tutorial/`](tutorial/) | Generics, and logical operators |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Working on the compiler itself |
| [`adding_a_topology.md`](adding_a_topology.md) | Adding support for new hardware |
| [`git_cheatsheet.md`](git_cheatsheet.md) | Repository conventions |

## Architecture and design

Why the compiler is shaped the way it is. Descriptive rather than normative — where one of these
disagrees with the specification, the specification wins.

| Document | Covers |
| --- | --- |
| [`architecture_executive_summary.md`](architecture_executive_summary.md) | The parallel compiler design, in brief |
| [`parallel_compiler_architecture.md`](parallel_compiler_architecture.md) | The same at length: GIDs, the epoch model, the phase pipeline |
| [`memory_algebra.md`](memory_algebra.md) | Containment, capacity and derived transfer cost |
| [`custom_transfer_contract.md`](custom_transfer_contract.md) | What a hand-written transfer lowering must guarantee |
| [`topology_representation.md`](topology_representation.md) | How topologies are represented internally |
| [`scalable_plugin_system.md`](scalable_plugin_system.md) | How vendors extend the compiler |
| [`npu_hardware_dispatch.md`](npu_hardware_dispatch.md) | Accelerator dispatch |
| [`std_library_design.md`](std_library_design.md) | Why the standard library is layered over a Rust core |
| [`generics_design.md`](generics_design.md) | Monomorphization |
| [`ast_reference.md`](ast_reference.md) | The AST, for people working on the frontend |
| [`spawn_on.md`](spawn_on.md) | The intended asynchronous design — **not implemented** |

## Plans

Documents named `*_plan.md` or `*_implementation_plan.md` are working plans for a piece of work.
They describe an intended end state and are the most likely of anything here to have drifted from
the code. Read them for intent, not as a description of what exists.

## A note on accuracy

Documentation that is wrong is worse than documentation that is missing, so two things are worth
knowing about how this directory is kept honest:

- **Every Vx code block in every document is compiled by CI.** If an example stops compiling, the
  build fails. That check exists because a prior audit found the README's own headline example did
  not parse, and a tutorial teaching a syntax that had never been implemented.
- **Anything not yet built says so**, in the document that describes it, rather than being written
  in the present tense. If you find something here described as working that is not, that is a bug
  worth reporting.
