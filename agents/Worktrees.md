## Worktrees and their build directories

Give each git worktree its own `target/` directory. Do not set `CARGO_TARGET_DIR` to one
directory for all worktrees.

- Cargo names the files of a crate in this repository by its path in the repository, not by
  its absolute path. So `libvxc-<hash>.rlib` has the same name in every worktree. In one shared
  directory, the worktrees overwrite each other's files, and a build can give you a `vxc` that
  holds the code of another branch.
- The pre-commit hook and many scripts run `./target/debug/...` or `./target/release/...`.
- Cargo locks its target directory, so the builds of two worktrees wait for each other.

To avoid a full build in a new worktree, start its `target/` as a copy of the one in the main
checkout. `cp -c` on macOS makes a copy-on-write clone: it is fast, and it takes almost no
disk until a file changes. On Linux, `cp --reflink=auto` does the same on Btrfs and XFS, and
makes a plain copy elsewhere. Run these from the main checkout:

```bash
git worktree add --no-track -b <branch> ../Vx-<name> origin/main
cp -Rc target ../Vx-<name>/target
cp config.local ../Vx-<name>/
ln -s "$PWD/venv" ../Vx-<name>/venv
touch ../Vx-<name>/build.rs
cd ../Vx-<name> && source config.local && cargo build -j <cores minus 2>
```

The first build then compiles the crates of this repository, but not their dependencies.

- `--no-track` keeps git from writing an upstream for the branch into `.git/config`.
- The build script checks once for a Python with coremltools, and keeps the answer.
  `touch build.rs` makes it check again now that `venv` is there. Without it, the build skips
  the Neural Engine models in `ane-primitives/`, and a test in the pre-commit hook fails.
- Run a full `cargo build`. `cargo build --bin vxc` and `cargo test` do not build
  `stdlib/rust_core`. Run it again after you merge a newer `main` into the worktree.
- `git worktree remove ../Vx-<name>` deletes the worktree with its `target/`.
