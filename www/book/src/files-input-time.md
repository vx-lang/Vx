# Files, input and time

A program often needs text, files, the clock or the network. This chapter shows the parts of the
standard library for these jobs: `std::string`, `std::fs`, `std::io`, `std::time` and `std::net`.

Many of these functions take a raw pointer, such as a string literal or a buffer. The compiler
cannot check a raw pointer, so you call these functions inside an `unsafe { .. }` block. The
network types also have safe methods, which take a `String` and a `Vec<u8>`.
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

`parse_int` stops at the first byte that is not a digit. So `"12abc"` gives 12, and `"abc"`
gives 0. A `-` is not a sign. `parse_int` does not check for overflow.

## Files

`File::open(path, mode)` opens a file. `path` is a text that ends with a zero byte, such as a
string literal. The mode is one of the three values of `OpenMode`:

- `OpenMode::Read()` opens a file that exists, for reading.
- `OpenMode::Write()` makes the file, or makes an existing file empty, for writing.
- `OpenMode::ReadWriteCreate()` opens the file for both, and makes it if it does not exist. It
  keeps the bytes that the file has.

`write(buffer, len)` writes `len` bytes from a buffer and gives how many it wrote. `read(buffer, len)` reads up to `len` bytes into a buffer and gives how many it read. 0 means the end of the
file. `seek(offset, whence)` moves the position of the next read or write, and gives the new
position. `whence` is 0 for the start of the file, 1 for the current position and 2 for the end.

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

`write_text` copies the literal into a `Vec<u8>`, because `write` takes a pointer to `u8` bytes.
`out` closes at the end of `write_text`. So `main` reads all 14 bytes.

A failed `open` gives a `File` that does nothing: each later `read` and `write` gives 0. A missing
file and an empty file look the same. So check the count that `read` and `write` give.

## Standard input and output

`std::io` has three functions. `stdin_read(buffer, len)` reads up to `len` bytes of standard input
and gives the count. 0 means the end of the input. `stdout_write` and `stderr_write` write bytes to
standard output and standard error. This loop counts the lines of the input:

```vx
import std::io;

fn main() -> i32 {
  let mut byte : u8 = 0;
  let mut lines = 0;
  loop {
    let n = unsafe { stdin_read((&mut byte) as *mut u8, 1) };
    if n == 0 {
      break;
    }
    if byte == 10 {
      lines = lines + 1;
    }
  }
  println!("{} lines", lines);
  return 0;
}
```

When `vxc program.vx` runs a program, the program gets no standard input
([Vx#3236](https://github.com/vx-lang/Vx/issues/3236)). So this loop stops at once and prints
`0 lines`. Compile the program with `-c` and run the executable to give it input
([Your first program](first-program.md#compiling-ahead-of-time)).

## Time

`std::time` measures time. `Instant::now()` reads a clock that never goes back. `elapsed()` gives
the time since a reading, as a `Duration`. The difference `later - earlier` of two readings is a
`Duration` too. `sleep(d)` stops the thread for at least the `Duration` `d`.

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

The methods of these types take a `String` address and a `Vec<u8>` buffer, and need no `unsafe`.
The length of the vector is the size of the buffer. Each type also has methods that take raw
pointers. Their names end in `_c_str` or `_raw`, for example `TcpStream::connect_c_str("..")` for a
string literal.

Each call that can fail gives a `Result`. Its `Err` holds a `NetError`, which says why the call
failed:

- `BadAddress`: the address is not `host:port`, or the host has no address.
- `Refused`: nothing listens at the address.
- `Reset`: the other end closed the connection.
- `InUse`: another socket uses the address.
- `Cancelled`: the program cancelled the `tasks` block of the task while the call waited.
- `Os(code)`: another failure, with the error number of the system.

This program uses the raw methods. The address of the first call has no port. Port 0 in the second asks the
system for any free port (`book_io_network_errors.vx`):

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
the buffer is full, `read_exact` gives `Reset`.

`BufReader` and `BufWriter` hold a `TcpStream` and a buffer of 4096 bytes. `read_line(&mut line)`
on a `BufReader` adds the next line, with its newline, to the end of `line`, and gives the count of
bytes. 0 means that the connection closed. `write` on a `BufWriter` adds bytes to the buffer, and
`flush()` sends them in one write. A `BufWriter` flushes at the end of its scope too, but then it
cannot report an error. `get_mut()` on either one gives the `TcpStream`, for example to write a
reply.

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
