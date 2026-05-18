## Language
Write code in Rust by default.

## Coding guidelines
- Try to commit as often whenever you think that there is some logical completion of the task.
- Always write detailed commit messages.
- Always run formatters: 'cargo clippy', 'vx-format', 'cargo fmt' and fix all the warnings/errors before commiting.
- Make sure there is a github issue ID attached to each commit unless the commit is cleanup (formatting, minor fixes)
- If you have finished a walkthrough then save the Walkthrough, Task.md, and Implementation Plan in the docs/discussions/ directory.

## Testing
Write unit tests and integration tests for all the code that you write. For testing, prefer using crates like `proptest` for property based testing and `rstest` for test fixtures.

## Usage of AI Tools
- Use AI tools for writing code when you think it is appropriate.
But always make sure that you understand the code that you are writing. And write appropriate comments for the code that you write.

If you are unsure about anything, then ask me.


## Adding Vx language features
- When adding a language feature, make sure that:
  - You also add a test for it in the `tests` directory.
  - You also update all the other files that are related to the new feature.
  - You also update the formal semantics in the `docs/semantics` directory.
  - You also update the tutorial in the `docs/tutorial` directory.
