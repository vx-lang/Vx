## Language

Write code in Rust by default.

## Coding guidelines

- Try to commit as often whenever you think that there is some logical completion of the task.
- Always write detailed commit messages.
- Always run formatters: 'cargo clippy', 'vx-format', 'cargo fmt' and fix all the warnings/errors before commiting.
- Make sure there is a github issue ID attached to each commit unless the commit is cleanup (formatting, minor fixes)
- When planning to add TODO to make progress: better add a `panic!` to prevent accidental usage of the feature and also so that you are forced to implement it when you come back to it. (prefer adding a github issue ID to it as well).
- Dont write issue numbers, or codewords (like S1, C1.2 etc) in the code itself.

## Testing

Write unit tests and integration tests for all the code that you write. For testing, prefer using crates like `proptest` for property based testing and `rstest` for test fixtures.

- A test is not evidence until you have watched it fail. Break the code it covers, see it go red, then put the code back.
- If the break does not turn it red, suspect the break before the test: check it landed where you aimed it (a blind search-and-replace hits the first match, not necessarily yours), and that it removed the behaviour rather than renaming it.

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
