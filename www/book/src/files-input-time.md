# Files, input and time

A program often needs text, files, the clock or the network. This chapter shows the parts of the
standard library for these jobs: `std::string`, `std::env`, `std::fs`, `std::io`, `std::time` and
`std::net`.

Many of these functions take a raw pointer, such as a string literal or a buffer. The compiler
cannot check a raw pointer, so you call these functions inside an `unsafe { .. }` block. Files and
the network types also have safe methods, which take a `String` and a `Vec<u8>`.
[Unsafe and FFI](unsafe-and-ffi.md) explains raw pointers.

Each complete program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Strings

`String` is a text that can grow. `String::new()` makes an empty string. `push_c_str` adds a
string literal to the end. `len()` gives the length in bytes, not in characters.

`to_string()` on an integer gives its decimal digits as a `String`. `String::from_c_str` copies a
string literal. `==` on two strings compares their bytes. `parse_int` reads the digits at the
start of a text as an `i32` (`book_io_build_a_string.vx`):

```rust
{{#include ../../../tests/backend/pass/book_io_build_a_string.vx:program}}
```

It prints:

```text
hello, world has 12 bytes
equal: true
parsed 124
```

`println!` moves a `String` argument ([Vx#3235](https://github.com/vx-lang/Vx/issues/3235)). So the
program gets the length before it prints `greeting`.

A string literal is its bytes and a NUL at the end, and the compiler knows how many bytes it has.
`"hello" as *const u8` gives those bytes as a `*const u8`, the type that `String::from_bytes`
below takes. So `String::from_bytes("hello" as *const u8, 5)` makes a `String` with no `unsafe`. A length of
6 would also copy the NUL, and the compiler refuses it (E8006).

`String::from_utf8(&v)` copies the bytes of a `Vec<u8>`, and `String::from_bytes(p, n)` copies
the `n` bytes at the raw pointer `p`. Neither one needs `unsafe`. But a call of `from_bytes` in
safe code must prove that `p` points at `n` bytes, as `v.as_mut_ptr()` and `v.len()` do. Both
give a `Result`. Its `Err` holds a `Utf8Error` when the bytes are not UTF-8, and `e.valid_up_to`
is the number of bytes before the first bad one. `s.to_bytes()` goes the other way: it copies the
bytes of a `String` into a new `Vec<u8>`.

`parse_int` stops at the first byte that is not a digit. So `"12abc"` gives 12, and `"abc"`
gives 0. A `-` is not a sign. `parse_int` does not check for overflow.

`i64::parse(&v, from, to)` reads the decimal integer in bytes `from` to `to` of a `Vec<u8>`, with
no `unsafe`. It allows one `+` or `-` before the digits, and nothing else. `u64::parse` allows only
a `+`. `f64::parse` and `f32::parse` read a float, such as `1.5`, `-2e10`, `inf` or `NaN`. Each
gives a `Result`. Its `Err` holds a `ParseNumberError`: `Empty`, `Invalid` or `OutOfRange`. A
range that is not inside `v` stops the program. `to_string()` on an `f32` or an `f64` gives the
fewest digits that read back as the same value, as in `0.1`.

## The command line

A `main` written `fn main(root : Root, args : Vec<String>) -> i32` gets the words of its command
line in `args`. The program owns them, so it can move them and change them. Each argument comes
after the first word. Whoever starts the program chooses every word, the first one too.
Usually the first word is the name or the path of the program, but it can be empty. A `String`
holds only UTF-8 text, so an argument that is not UTF-8 stops the program before `main` runs.

```vx
import std::root;
import std::string;
import std::vec;

fn main(_root : Root, args : Vec<String>) -> i32 {
  let mut i = 1;
  while i < args.len() {
    println!("{}", args.get_ref(i));
    i = i + 1;
  }
  return 0;
}
```

`vxc args.vx -- one two` prints `one` and `two`. The words are data and not a right, so they are
not part of `Root`. The test is `main_gets_the_words_of_the_command_line.vx`.

## Environment variables

The program that starts a process also chooses its environment variables. They often hold
secrets, such as the token of a service. So reading them is a right, as opening a file is. `main`
gets it in `root.env`, an `Environment`, and passes `&root.env` to the code that reads a variable.
A function that gets no `&Environment` cannot read one.

`env.var(&name)` gives a `Result`:

- `Ok` holds the value, as a `String`.
- `Err(EnvError::NotPresent)` means that the environment has no variable with the name. An empty
  name, and a name that holds `=`, name no variable.
- `Err(EnvError::NotUnicode)` means that the value is not UTF-8 text, so a `String` cannot hold it.

```vx
import core::result;
import std::env;
import std::root;
import std::string;

// Reads `HOME`. A function needs a `&Environment` to read a variable.
fn home(env : &Environment) -> String {
  let name = unsafe { String::from_c_str("HOME") };
  match env.var(&name) {
    Result<String, EnvError>::Ok(value) => {
      return value;
    }
    Result<String, EnvError>::Err(_e) => {
      return String::new();
    }
  }
}

fn main(root : Root) -> i32 {
  println!("{}", home(&root.env));
  return 0;
}
```

The `unsafe` block only makes a `String` from the literal, as [Networks](#networks) explains. The
test is `a_program_reads_its_environment_variables.vx`.

## Files

To open a file, a call needs the right to open files: a `&Files`. Only `main` gets it. A `main`
written `fn main(root : Root) -> i32` gets the rights of the program in `root`. It passes
`&root.files` to `File::open`, or to a function that calls it. Only `std::fs` can make a `Files`,
so a function that gets no `&Files` cannot open a file.

`File::open(files, path, mode)` opens the file at the `String` `path`. The mode is one of the
three values of `OpenMode`:

- `OpenMode::Read()` opens a file that exists, for reading.
- `OpenMode::Write()` makes the file, or makes an existing file empty, for writing.
- `OpenMode::ReadWriteCreate()` opens the file for both, and makes it if it does not exist. It
  keeps the bytes that the file has.

`write(bytes)` writes the bytes of a `Vec<u8>` and gives how many it wrote. `read(buffer)` reads
up to `buffer.len()` bytes into a `Vec<u8>` and gives how many it read. 0 means the end of the
file. `seek(offset, whence)` moves the position of the next read or write, and gives the new
position. `whence` is 0 for the start of the file, 1 for the current position and 2 for the end.

`open`, `read` and `write` give a `Result`. When a call does not work, its `Err` holds a
`FileError`:

- `NotFound`: no file has the path, or a folder of the path is missing.
- `PermissionDenied`: the system does not let the program open the file in the mode.
- `BadPath`: the path holds a zero byte.
- `NotInFolder`: the name leads out of a `Folder`. The end of this section explains `Folder`.
- `Os(code)`: another failure, with the error number of the system.

The `?` operator gives the error to the caller
([Errors, Option and Result](errors.md)).

`File` implements `Drop`, so a file closes at the end of its scope. You do not close it by hand.
[Ownership and borrowing](ownership.md) explains `Drop`.

This program writes three lines to a file, and reads the file back one byte at a time. The test
gives the path of the file in the environment variable `NOTES` (`book_io_write_and_read_a_file.vx`):

```rust
{{#include ../../../tests/backend/pass/book_io_write_and_read_a_file.vx:program}}
```

It prints:

```text
wrote 14 bytes
read 3 lines
the file ends at byte 14
byte 4 is 116, the letter t
```

`write_text` copies the literal into a `Vec<u8>`, because `write` takes bytes. `out` closes at the
end of `write_text`. So `write_and_read` reads all 14 bytes.

`File` also has methods that take raw pointers, for trusted code that gets a path from C:
`open_c_str`, `read_raw` and `write_raw`. They need `unsafe`, and they need no `&Files`. A failed
`open_c_str` gives a `File` that does nothing: each later `read_raw` and `write_raw` gives 0.

A `Folder` is a narrower right: the right to read the files of one folder.
`Folder::open(files, path)` opens the folder at `path`. `folder.open_file(name)` opens the file
`name` inside it, for reading, and `folder.open_folder(name)` gives a `Folder` for a folder inside
it. A name that leads out of the folder gives `FileError::NotInFolder`: an absolute path, a `..`
part, or a symbolic link. So a function that gets only a `&Folder` can read the files of that
folder, and no others. For example, a model server can give the code that loads its weights a
`Folder`, and give the code that reads requests no right at all
(`a_folder_right_reads_only_the_files_inside_its_folder.vx`).

`file.map()` gives the whole file in memory, read-only, as a `Mapping`. It copies nothing: the
system reads each part of the file when the program first reads it. So a program can map a file
of many gigabytes and read only the parts that it needs.

- `mapping.len()` is the size of the file in bytes.
- `mapping.byte(i)` is byte `i`. It stops the program when `i` is outside the file.
- `mapping.as_ptr()` is the address of the first byte, for code that reads through raw
  pointers. An example is a tensor over the weights of a model.

A dropped `Mapping` unmaps the file. A `Mapping` stays valid after its `File` closes. A file open
only for writing gives `FileError::PermissionDenied` (`a_file_mapped_into_memory.vx`). When
another program shortens the file while it is mapped, a read past the new end stops the program
with the signal SIGBUS.

## Standard input and output

Reading standard input and writing to standard output and error are rights, as files are. `main`
gets them in `root.stdin`, a `Stdin`, `root.stdout`, a `Stdout`, and `root.stderr`, a `Stderr`.
Only `std::io` can make them, so a library reads or writes them only when its caller passes them.

- `input.read(&mut buffer)` reads up to `buffer.len()` bytes and gives the count in a `Result`.
  0 means the end of the input.
- `input.read_exact(&mut buffer)` fills all of `buffer`. It gives `StdioError::EndOfInput` when
  the input ends first.
- `out.write_all(&bytes)` writes all of the bytes. They keep their order with `print!`.
- A `select` can wait on a `Stdin`, as it waits on a socket.

`println!` and `print!` do not need the right yet. The older `unsafe` functions `stdin_read`,
`stdout_write` and `stderr_write` take a raw pointer and a length. This loop counts the lines of
the input:

```vx
import core::result;
import std::io;
import std::root;
import std::vec;

fn main(root : Root) -> i32 {
  let mut input = root.stdin;
  let mut byte = Vec<u8>::new();
  byte.push(0u8);
  let mut lines = 0;
  loop {
    match input.read(&mut byte) {
      Result::Ok(n) => {
        if n == 0 {
          break;
        }
        if byte.get(0) == 10u8 {
          lines = lines + 1;
        }
      }
      Result::Err(_e) => {
        break;
      }
    }
  }
  println!("{} lines", lines);
  return 0;
}
```

When `vxc program.vx` runs a program, the program reads the standard input of vxc. So
`printf 'a\nb\n' | vxc program.vx` prints `2 lines`. In a terminal, type the lines, then press
Ctrl-D to end the input. vxc shows what the program prints only when the program ends
([Vx#3458](https://github.com/vx-lang/Vx/issues/3458)).

## Time

`std::time` measures time. `Instant::now()` reads a clock that never goes back. `elapsed()` gives
the time since a reading, as a `Duration`. The difference `later - earlier` of two readings is a
`Duration` too. `sleep(d)` pauses the task for at least the `Duration` `d`. It is a wait, as a
`recv` is: in the shared runtime, other tasks run on the thread of the task while it sleeps. A
cancel of the `tasks` block, or the limit of `within`, ends the sleep early
([Cancellation](tasks.md#cancellation)). `sleep` gives `true` when the whole time passed, and
`false` when it ended early. `sleep_until(at)` sleeps until the `Instant` `at`.

For periodic work, use an `Interval`. `Interval::new(period)` makes one, and each `tick()` waits
for the next tick. Tick `k` is due `k` periods after `Interval::new`. So a late tick does not move
the ticks after it, and the ticks do not drift. `tick()` gives the time when the tick was due, or
`None` after a cancel:

```rust
let mut every_second = Interval::new(Duration::from_secs(1));
loop {
  match every_second.tick() {
    Option<Instant>::Some(_due) => {
      // The periodic work.
    }
    Option<Instant>::None => {
      break;
    }
  }
}
```

When the work takes longer than a period, the program misses ticks. The next `tick()` then comes
at once, and `Interval` skips the missed ticks: the ticks after it are due on time again. `Interval` does not
give the missed ticks in a burst, because a burst gives more work to a program that is already
late. Go's `time.Ticker` also skips missed ticks.

`Duration` comes from `core::time`. `Duration::from_secs`, `from_millis`, `from_micros` and
`from_nanos` make one. `as_secs`, `as_millis` and `as_nanos` give its length, and `subsec_millis`
gives the part that is less than a second. `+` adds two durations
(`book_io_sleep_and_measure.vx`):

```rust
{{#include ../../../tests/backend/pass/book_io_sleep_and_measure.vx:program}}
```

It prints:

```text
slept at least 50 ms: true
1 s and 500 ms
total 3500 ms
```

The program does not print the time that it slept, because that time is different in each run.

## Networks

`std::net` has three types. `TcpListener` waits for TCP connections. `TcpStream` is one TCP
connection. `UdpSocket` sends and receives UDP datagrams, which are single messages. Each address
is a text of the form `host:port`, for example `"127.0.0.1:7878"`.

To open a connection or to bind an address, a call needs the right to use the network: a
`&Network`. As for files, only `main` gets it, in `root`. `main` passes `&root.network` to
`TcpStream::connect`, `TcpListener::bind` and `UdpSocket::bind`, or to a function that calls them. Only `std::net` can make a `Network`, so a
function that gets no `&Network` cannot open a connection. A connection that `accept` gives needs
no other right.

The methods of these types take a `String` address and a `Vec<u8>` buffer, and need no `unsafe`.
The length of the vector is the size of the buffer. Each type also has methods that take raw
pointers. Their names end in `_c_str` or `_raw`.

A string literal is a raw pointer to its bytes. `String::from_c_str("..")` makes a `String` from
it. That call is `unsafe`, because nothing checks that the pointer ends in a NUL byte. A literal
always does, but Vx has no safe way yet to make a `String` or a `Vec<u8>` from a literal. So the
examples below use `unsafe` only for that.

Each call that can fail gives a `Result`. Its `Err` holds a `NetError`, which says why the call
failed:

- `BadAddress`: the address is not `host:port`, holds a NUL byte, or names a host that has no
  address.
- `Refused`: nothing listens at the address.
- `Reset`: the other end reset the connection, or closed it while a write waited.
- `Closed`: the other end closed the connection before `read_exact` got all its bytes.
- `InUse`: another socket uses the address.
- `Cancelled`: the program cancelled the `tasks` block of the task while the call waited.
- `Os(code)`: another failure, with the error number of the system.

The address of the first call in this program has no port. Port 0 in the second asks the system
for any free port (`book_io_network_errors.vx`):

```rust
{{#include ../../../tests/backend/pass/book_io_network_errors.vx:program}}
```

It prints:

```text
bad address
bound a UDP socket
```

The sockets close at the end of their scope, as a `File` does.

A server calls `accept()` on a `TcpListener`. `accept` waits until a client connects, and gives a
`TcpStream`. `read(&mut buffer)` reads up to `buffer.len()` bytes and gives the count. 0 means that
the other end closed the connection. `write(&buffer)` writes bytes and gives the count. This
function sends one byte back to the client. `?` gives the `Err` to the caller:

```vx
fn echo_one_byte(listener : TcpListener) -> Result<u64, NetError> {
  let mut server = listener;
  let mut client = server.accept()?;
  let mut buffer = Vec<u8>::new();
  buffer.push(0u8);
  let _count = client.read(&mut buffer)?;
  return client.write(&buffer);
}
```

TCP is a stream of bytes. A message can arrive in parts, so a `read` can give fewer bytes than the
other end wrote, and a `write` can take fewer bytes than you gave. `read_exact` reads until the
buffer is full, and `write_all` writes all bytes of the buffer. When the connection closes before
the buffer is full, `read_exact` gives `Closed`.

`TcpBufReader` and `TcpBufWriter` hold a `TcpStream` and a buffer of 4096 bytes. `read_line(&mut line)` on a `TcpBufReader` adds the next line, with its newline, to the end of `line`, and gives the
count of bytes. 0 means that the connection closed. A line has no length limit, so a client that
never sends a newline makes `line` grow until the memory runs out. After `read_line`, the reader can
hold the start of the next data, for example the body after a header. So read it with `read` or
`read_exact` on the reader, not on its `TcpStream`.

`write` on a `TcpBufWriter` adds bytes to the buffer, and `flush()` sends them in one write. A
`TcpBufWriter` flushes at the end of its scope too, but then it cannot report an error. In a
cancelled `tasks` block, that flush gives `Cancelled`, and the bytes are lost. `get_mut()` on either
type gives the `TcpStream`, for example to write a reply.

In this program, a client task sends two lines, and the server reads them
(`book_io_network_lines.vx`):

```rust
{{#include ../../../tests/backend/pass/book_io_network_lines.vx:program}}
```

It prints:

```text
line 1 has 11 bytes
line 2 has 12 bytes
the client closed the connection
```

Each wait in `accept`, `read_line` or `flush` is a normal call. On the shared runtime, it parks
only the task that waits, so the other task runs on the same OS thread
([Tasks](tasks.md#why-vx-has-no-async)).

A `UdpSocket` has `recv(&mut buffer)`, which waits for one datagram and gives its length, and
`send_to(&buffer, &addr)`, which sends one datagram.
