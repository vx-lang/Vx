# GitHub Open Source Release Cleanup Plan

To make the `Vx` repository production-ready for a GitHub release, I've audited the repository and identified several cleanup opportunities.

## Proposed Changes

### 1. `Cargo.toml` Metadata Enhancements
Currently, `Cargo.toml` only contains the package name and version. For a proper open-source release, we need to add standard metadata.
#### [MODIFY] [Cargo.toml](file://<WORKSPACE_ROOT>/Cargo.toml)
- Add `authors`, `description`, `repository`, and `license` fields.
- Add `keywords` (e.g., `compiler`, `heterogeneous`, `mlir`) and `categories`.

### 2. `.gitignore` Hardening
Prevent compilation artifacts from accidentally slipping into the repo.
#### [MODIFY] [.gitignore](file://<WORKSPACE_ROOT>/.gitignore)
- Add rules for `.dylib`, `.o`, `*.log`, and `*.so`.
- Add explicit ignores for old binary tests like `test_tok` and `test_tok.c`.

### 3. Cleanup Legacy & Internal Artifacts
We have several temporary refactoring scripts and AI-generated prompt files floating in the root directory that should not be part of the final GitHub release.
#### [DELETE] [GEMINI.md](file://<WORKSPACE_ROOT>/GEMINI.md)
- This contains the original AI prompt and instructions. Not for public release.
#### [DELETE] [run_ref.c](file://<WORKSPACE_ROOT>/run_ref.c)
- An empty 0-byte scratch file.
#### [MODIFY] Scripts Directory
- Move ad-hoc root Python scripts (`fix_tests.py`, `fix_tests_ast.py`, `refactor_ast.py`, `build_ane_matmul.py`) into `scripts/legacy/` to clean up the root directory without losing historical reference.

### 4. Organize Existing Documentation
#### [MODIFY] [Proposal.md](file://<WORKSPACE_ROOT>/Proposal.md)
- Move this to `docs/Proposal.md` since it contains historical context but shouldn't clutter the root directory (which should only have `README.md` and `ROADMAP.md`).

## User Review Required
> [!IMPORTANT]
> Please review the plan above.
> - Are there any specific details you want me to add to the `Cargo.toml` (e.g., specific Author name/email, License choice like MIT or Apache-2.0)?
> - Should I completely `git rm` the Python scripts instead of moving them to `scripts/legacy`?

Let me know if this plan looks good to execute!
