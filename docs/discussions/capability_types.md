# Capability types

Status: proposed, for review. Issue: #3750. Gaps: #3746, #3747, #3748 and #3751. Where an `impl`
can be: #3754.

A capability is a value that gives the right to do one thing, such as deleting the data or reading
the files of one folder. Code that does not hold the value cannot do the thing. This note answers
two questions:

1. Can a Vx type be a capability that other code cannot make or copy?
1. What else does Vx need, so that a bug cannot become remote code execution?

The short answers: yes, Vx has the rules for it today, with five gaps that small changes
close. And no, it is not enough yet, because any module can open files and sockets with no
capability (section 5).

## 1. What a capability type needs

A capability type keeps three rules:

- **No forging.** Only the module of the type makes a value.
- **No copying.** Code uses a value once, unless the module of the type allows more.
- **No keeping a lent value.** A function that gets `&Cap` uses it during the call, and cannot
  store it.

Vx has the three rules for a type with an invariant:

```vx
struct AdminCap {
  invariant true,
}

pub fn login(password : i32) -> AdminCap {
  assert(password == 1234, "wrong password");
  return AdminCap {};
}

pub fn delete_everything(cap : AdminCap) -> i32 {
  return 1;
}
```

| Rule | Why it holds | Error |
|---|---|---|
| No forging | Only the module of a type with an invariant builds a value of it ([type_invariants.md](type_invariants.md)) | E8014 |
| No copying | A struct moves, also inside an `Option` and into a task | E4001 |
| No keeping a lent value | A result cannot point into the frame of the function | E4005 |

`invariant true` states no fact. It makes `AdminCap` a type that only `auth.vx` can build.

**A capability for one request is lent.** The three rules stop copying, not keeping. A caller that
gets an `AdminCap` by value from `login` can keep it in the state of a server, and use it for each
later request. So `auth.vx` lends a right that must last for one request:

```vx
pub fn with_admin(password : i32, f : fn(&AdminCap) -> i32) -> i32 {
  if password != 1234 {
    return -1;
  }
  let cap = AdminCap {};
  return f(&cap);
}
```

The `AdminCap` stays in the frame of `with_admin`, and `f` gets only a reference. A `fn` captures
nothing, and Vx has no mutable global, so `f` has no place to keep the reference. A lender that
returns the reference gets E4005. Only a right that lasts as long as the program, such as `Files`
or `Root`, goes by value. A closure cannot take a `&AdminCap` yet (#3970).

## 2. The test

On main at 3a4b2c5d5, a second module imported `auth.vx` above. It tried to get an `AdminCap`
without `login`, or to use one value twice:

| Attempt | Result |
|---|---|
| `AdminCap {}` | E8014 |
| `impl Copy for AdminCap {}`, then one value used twice | **Compiles** (#3746) |
| `impl AdminCap { .. }` with new methods | **Compiles** (#3754) |
| `impl Clone` or `impl Default` that builds one | E8014. A `clone` that returns `*self` gets E4008 |
| A closure that passes a captured `AdminCap` on, called twice | **Compiles** (#3751). With a `Vec`, the second call frees it again |
| `mlir!` with `returns : AdminCap`, outside `unsafe` | **Compiles** (#3747) |
| `extern "C" { safe fn mint() -> AdminCap; }`, called outside `unsafe` | **Compiles** (#3748) |
| In `unsafe`, `read` of a `*const i8` cast to `*const AdminCap` | Refused: the cast is not supported |
| In `unsafe`, `read(null<AdminCap>())` | Compiles. The program trusts `unsafe` code (section 4) |
| One `Option<AdminCap>` moved twice | E4001 |
| A `&AdminCap` parameter kept in a struct that the function returns | E4005 |
| An `AdminCap` moved into a task | Compiles, and the value moves. A second use, or a move into two tasks, gets E4001 |
| A generic function that uses its `T` parameter twice | E4001 |

## 3. Close the gaps

- **#3746 and #3754.** Only the file that declares a struct or an enum can write a plain `impl` of
  it, with or without a trait. An impl that must be in another file, because of an import cycle or
  to keep a module apart, is `extern impl`. So `impl Copy for AdminCap` in another file is an
  error, and `extern impl Copy for AdminCap` there is trusted code (section 4). `extern impl` is
  new syntax, and `unsafe impl` keeps its meaning: only `Send`, `Sync` and `Owned` take it. The
  module of the type can still make a capability that copies, such as a read-only right that is
  safe to share.
- **#3747.** Outside the module of a type with an invariant, an `mlir!` block cannot give a value
  of the type. This is the rule for a struct literal, and `unsafe` does not lift it either.
- **#3748.** An `extern` function that returns a type with an invariant, or takes a `&mut` to one,
  cannot be `safe`. Each call then needs `unsafe`.
- **#3751.** A closure whose body moves a captured value runs once. A second call is E4001. This
  is a memory bug as well as a capability bug.
- **#3848.** Each type of `core` and `std` that holds a raw pointer, such as `File`, `TcpStream`,
  `String` or `Location`, has `invariant true`. Before, safe code could build a `TcpStream` from
  the pointer of a `File`. The compiler still writes the `Location` of a `Caller` argument.

## 4. Trusted code

`unsafe` code can always make a value, because it can read one from any address. Rust has the
same rule. So the guarantee is this: code with no `unsafe`, no `extern impl`, no `mlir!` and no
`extern` cannot forge a capability. The modules that use them are the trusted code of the program,
and a review reads each of them.

Two tools keep the trusted code small and easy to find:

- `vxc --no-trusted-code NAME` refuses `unsafe`, `extern impl`, `unsafe impl`, `mlir!`, `extern`
  and a `requires` of an `extern fn` in the package `NAME`, as `#![forbid(unsafe_code)]`
  does in Rust. The program that uses the
  package gives the flag, so a new version of the package cannot add trusted code without an
  error.
- `vxc --audit` lists each use of `unsafe`, `extern impl`, `mlir!` and `extern` in the program,
  with its file and line. For `core` and `std`, it gives only the counts, and
  `vxc --audit-library` lists them too.

## 5. No ambient authority

The rules above stop forged capabilities. They do not stop code that needs no capability. Before
the `Network` capability, any module could `import std::net` and open any socket with no
`unsafe`. A file needed `unsafe` to open, which is trusted code (section 4). Now a safe
`File::open` takes a `&Files`. A capability protects something only when each way to the
operating system needs one:

- `std::fs`, `std::net`, `std::mmap`, `std::env`, the start of a process and the loading of
  native code each take a capability parameter. Environment variables are data, as the command
  line is. But they often hold secrets, such as the token of a service, so reading them is a right
  (`Environment`), as in Deno's `--allow-env`. The command line is a parameter of `main`.
- Standard input, output and error are rights too (`Stdin`, `Stdout`, `Stderr`). `print_s!` and
  `println_s!` take the right as their first argument. `print!` and `println!` take none, so code
  with no right can still print with them.
- Only `main` gets the root capabilities. A library gets only the capabilities that its caller
  passes to it.
- A module narrows a capability, for example from the right to open any file to the right to read
  one folder. The narrow type keeps its open folder in a field, and its invariant keeps the field
  fixed. Each open checks at run time that the name stays in the folder: no `..`, no absolute path,
  and no symbolic link. Each part of the name opens with `openat` and `O_NOFOLLOW`, so the check is
  the same on Linux and macOS. It refuses a link that stays inside the folder too.

<!-- vx-doctest: skip (a sketch: `serve` and the path are not declared) -->

```vx
// `Files` is the right to open any file. `Folder` is the right to read the files of one folder.
// Only `std::fs` can build either one.
fn Folder::open(files : &Files, path : &String) -> Result<Folder, FileError> { .. }
fn Folder::open_file(self : &Folder, name : &String) -> Result<File, FileError> { .. }

fn main(root : Root) -> i32 {
  let adapters = Folder::open(&root.files, &adapters_path)?;
  serve(&adapters);
  return 0;
}
```

A type cannot take back a capability: a value that a function holds stays valid while it holds it.
To revoke a right, the program needs a run-time handle that its owner can clear.

`Processes::run` in `std::process` starts another program, and takes a `&Processes`. The compiler
does not check that program, so `Processes` gives every other right in practice, as Deno's
`--allow-run` does. Native code runs only through `extern`, which is trusted code (section 4). `std::collective::join_job` takes `&Files` and `&Network`, because rank 0
writes the NCCL id to a file and NCCL connects the ranks over the network.

## 6. Effect on a model server

In a Python model server, remote code execution usually comes from loading: a weight file read with
`pickle`, or `trust_remote_code`. In Vx, loading weights reads bytes, for example with `mmap`, and
runs no code. Native code runs only through `extern`. With sections 3 to 5:

- An endpoint that loads a LoRA adapter gets a `Folder` for the adapter folder. A bug in it can
  read the files of that folder, and nothing else, unless the bug is in trusted code.
- The parser of a request gets no capability. A bug in it cannot open a file or start a process.
- The trusted code is the list that `vxc --audit` gives.

These limits hold for a bug in safe code. A memory bug in trusted code, such as an `unsafe` block,
an `extern` function or the GPU runtime, can still do anything. A `Vec` checks each index at run
time (`vec_check_index`). So a bug in a parser that indexes a `Vec` stops the program, and does not
write past the buffer.

## 7. The rules of a parallel compiler

[language_rules_for_a_parallel_compiler.md](../language_rules_for_a_parallel_compiler.md) asks that
no global fact depend on a function body. Each new check reads one declaration:

- #3754 reads the `impl` and the file that declares the type. It agrees with the rule "Copy and
  drop belong to the type".
- #3747 reads one `mlir!` block and the type that it returns.
- #3748 reads one `extern` declaration.
- #3751 reads the body of the closure, in the function that makes it.
- `--no-trusted-code` reads each file of the package on its own, after macro expansion.
- A capability parameter is an ordinary parameter. A signature is the whole contract, so a caller
  sees from it which capabilities a function needs.

## 8. Open questions

1. How does `main` get the root capabilities: as a parameter, as in Austral, or from a call that
   only the module of `main` can make? **Decided (2026-11-23): as a parameter,
   `fn main(root : Root) -> i32`.** A program that needs no capability keeps `fn main() -> i32`.
   A program cannot call `main` or use it as a value (E5004), so no code gets a second `Root`.
1. The package setting is a flag of `vxc`, because Vx has no manifest yet. Does it move into a
   manifest when Vx gets one?
1. What is the type of a parameter that takes a closure that runs once (#3751)?
1. Does a test get root capabilities, and which ones? **Decided (2026-11-23): a test is a program,
   so it gets `Root` in the same way.**

## 9. Steps

1. #3751, because it is a memory bug too.
1. #3754, which also closes #3746. Then #3747 and #3748.
1. `vxc --audit` lists each use of `unsafe`, `mlir!` and `extern`.
1. The package setting.
1. Capability types in `std`. Safe code could open sockets with no `unsafe`, but not files:
   `File::open` was an `unsafe fn` already. So the network came first: `TcpStream::connect`,
   `TcpListener::bind` and `UdpSocket::bind` take a `&Network`, which a program gets as
   `root.network`. Then `File::open` became safe, and takes a `&Files`, which a program gets as
   `root.files`. The raw open, `File::open_c_str`, stays `unsafe`. Then `Folder`, which narrows
   `Files` to one folder. Then `Environment`, the right to read the environment variables, which
   a program gets as `root.env`. Then `Processes`, the right to start another program, which a
   program gets as `root.processes`. Next comes native code.
1. Private functions (`private_functions.md`, #3890). A helper of `std` that reads through a raw
   pointer is sound only when other files cannot call it (#3884).
