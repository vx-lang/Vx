# Git Tags & Annotations Cheatsheet

If you are unfamiliar with `git tag` and `git show`, here is a quick reference guide on how annotated tags work in Git.

## What is an Annotated Tag?
In Git, a tag is a pointer to a specific commit. While lightweight tags just point to a commit (like a branch that doesn't change), **annotated tags** are stored as full objects in the Git database. They contain:
- The tagger's name and email
- The date the tag was created
- A tagging message (like a release note)
- A GPG signature (optional)

## Command: `git show <tag-name>`

When you run `git show v1.0` on an annotated tag, Git will output two things:
1. **The Tag Object Metadata**: This includes who tagged it, when it was tagged, and the custom tag message (in our case, the full v1.0 release notes).
2. **The Commit Details**: Immediately following the tag message, it will show the underlying commit hash, the commit author, the commit message, and the `diff` (the code changes) that were part of that specific commit.

### Example Output
```bash
$ git show v1.0
tag v1.0
Tagger: Aditya <aditya@example.com>
Date:   Mon May 11 21:18:00 2026 -0700

Release v1.0: Akar Language Bootstrap Milestone

This marks the v1.0 milestone for the Akar Systems Programming Language...
(Rest of the release notes)

commit ed34e59b2...
Author: Aditya <aditya@example.com>
Date:   Mon May 11 21:10:00 2026 -0700

    Test Suite Reorganization: Segregate passing and failing tests...

diff --git a/tests/...
```

## Useful Tag Commands

- **Create an annotated tag**: `git tag -a v1.0 -m "Release message"`
- **List all tags**: `git tag`
- **View a tag's release notes**: `git show v1.0`
- **Push tags to a remote server (e.g. GitHub)**: `git push origin --tags`
- **Update an existing tag**: `git tag -a -f v1.0 -m "New message"` (Requires force-pushing if already on a remote).
