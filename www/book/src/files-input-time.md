# Files, input and time

A program often needs text, files, the clock or the network. This chapter shows the parts of the
standard library for these jobs: `std::string`, `std::fs`, `std::io`, `std::time` and `std::net`.

Many of these functions take a raw pointer, such as a string literal or a buffer. The compiler
cannot check a raw pointer, so you call these functions inside an `unsafe { .. }` block.
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

`File::open_raw(path)` opens a file that exists, for reading. `File::create_raw(path)` makes a
file for writing. If the file exists, it makes the file empty. `path` is a text that ends with a
zero byte, such as a string literal. Both functions give a `Result`: `Ok` holds the `File`, and
`Err` holds an `Error`. [Errors](errors.md) explains `Result`.

For other ways to open a file, use `OpenOptions`. Start with `OpenOptions::new()`, and turn on
each choice that you want: `read`, `write`, `append`, `create`, `truncate` or `create_new`.

`write_raw(buffer, len)` writes up to `len` bytes from a buffer, and gives how many it wrote.
`read_raw(buffer, len)` reads up to `len` bytes into a buffer, and gives how many it read. A count
of 0 from `read_raw` means the end of the file. Both functions give a `Result`, so an error is an
`Err` and not a count. The buffer is a raw pointer, and Vx cannot check its size. So each call
goes in an `unsafe` block.

`seek_to` moves the position of the next read or write, and gives the new position.
`SeekFrom::Start(n)` counts from the start of the file, `SeekFrom::Current(n)` from the current
position, and `SeekFrom::End(n)` from the end. `stream_position()` gives the position and does
not move it.

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

`write_text` copies the literal into a `Vec<u8>`, because `write_raw` takes a pointer to `u8`
bytes. `out` closes at the end of `write_text`. So `main` reads all 14 bytes.

The program calls `unwrap`, which stops the program on an `Err`. A real program checks the error.
For example, `error.kind()` gives `ErrorKind::NotFound()` when the file does not exist.

The older `File::open(path, mode)`, `read`, `write` and `seek` are deprecated. A failed `open`
gives a `File` that does nothing, and the program cannot tell a missing file from an empty one.
Use the functions above in new code.

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

Each call that can fail gives a `Result`. Its `Err` holds a `NetError`, which says why the call
failed:

- `BadAddress`: the address is not `host:port`, or the host has no address.
- `Refused`: nothing listens at the address.
- `Reset`: the other end closed the connection.
- `InUse`: another socket uses the address.
- `Cancelled`: the program cancelled the `tasks` block of the task while the call waited.
- `Os(code)`: another failure, with the error number of the system.

This program makes two calls. The address of the first has no port. Port 0 in the second asks the
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
`TcpStream`. `read` and `write` on the stream work as they do on a `File`. This function sends one
byte back to the client. `?` gives the `Err` to the caller:

```vx
fn echo_one_byte(listener : TcpListener) -> Result<u64, NetError> {
  let mut server = listener;
  let mut client = server.accept()?;
  let mut byte : u8 = 0;
  let got = unsafe { client.read((&mut byte) as *mut u8, 1) };
  let _count = got?;
  return unsafe { client.write((&byte) as *const u8, 1) };
}
```

TCP is a stream of bytes. A message can arrive in parts, so a `read` can give fewer bytes than the
other end wrote. To get a whole message, read in a loop until you have all of its bytes.
