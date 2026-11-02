//===- io.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A task of the shared runtime that waits for a socket parks, so that its worker can run other
// tasks (W4 in docs/discussions/waiting_tasks_channels_and_task_errors.md).
//
// Each socket of `std::net` is non-blocking. When a call would block, the caller waits here
// until the socket is ready, and then tries the call again. A fiber puts its socket on the list
// of one poller thread, and parks on a word of its stack. The poller calls poll(2) on the
// sockets of the list. When one is ready, it sets the word and wakes the fiber (wait.rs). A
// thread that runs no fiber calls poll(2) on its socket itself.
//
// poll(2) works the same on Linux and on macOS. epoll and kqueue can replace it when a program
// waits on many sockets.
//
//===----------------------------------------------------------------------===//

use crate::ffi::{sched, wait};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Runs `call` until it does not give `WouldBlock` or `Interrupted`. Before each new try, it
/// waits until `fd` is ready to read, or to write when `write` is set.
pub fn when_ready<T>(
    fd: RawFd,
    write: bool,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    loop {
        match call() {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => wait_ready(fd, write),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            r => return r,
        }
    }
}

/// Waits until `fd` is ready. It can return early, so the caller tries its call again.
fn wait_ready(fd: RawFd, write: bool) {
    let events = if write { libc::POLLOUT } else { libc::POLLIN };
    if sched::fiber_that_can_park().is_null() {
        let mut p = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        unsafe { libc::poll(&mut p, 1, -1) };
        return;
    }
    // The poller sets the word, and then reads it no more. So the word can live on this stack.
    let ready = AtomicI64::new(0);
    poller().add(fd, events, &ready);
    while ready.load(Ordering::SeqCst) == 0 {
        unsafe { wait::vx_wait_while(ready.as_ptr(), 0) };
    }
}

struct Poller {
    /// The sockets that fibers wait for: the fd, its events, and the address of the word to set.
    waiting: Mutex<Vec<(RawFd, i16, usize)>>,
    /// A byte on this pipe makes the poller read the list again.
    wake_write: RawFd,
}

static POLLER: OnceLock<&'static Poller> = OnceLock::new();

impl Poller {
    fn waiting(&self) -> MutexGuard<'_, Vec<(RawFd, i16, usize)>> {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn add(&self, fd: RawFd, events: i16, word: &AtomicI64) {
        self.waiting().push((fd, events, word.as_ptr() as usize));
        // A full pipe already wakes the poller, so a write that fails loses nothing.
        let byte = 1u8;
        unsafe { libc::write(self.wake_write, (&raw const byte).cast(), 1) };
    }
}

/// Whether a fiber waits for a socket. Data from outside the program can wake it.
pub(crate) fn pending() -> bool {
    POLLER.get().is_some_and(|p| !p.waiting().is_empty())
}

fn poller() -> &'static Poller {
    POLLER.get_or_init(|| {
        let mut pipe = [0; 2];
        assert_eq!(
            unsafe { libc::pipe(pipe.as_mut_ptr()) },
            0,
            "the system could not make the pipe of the I/O poller"
        );
        for fd in pipe {
            unsafe { libc::fcntl(fd, libc::F_SETFL, libc::O_NONBLOCK) };
        }
        let p: &'static Poller = Box::leak(Box::new(Poller {
            waiting: Mutex::new(Vec::new()),
            wake_write: pipe[1],
        }));
        std::thread::Builder::new()
            .name("vx-io".into())
            .spawn(move || run(p, pipe[0]))
            .expect("the system could not start the I/O poller thread");
        p
    })
}

fn run(p: &'static Poller, wake_read: RawFd) {
    let mut fds = Vec::new();
    loop {
        fds.clear();
        fds.push(libc::pollfd {
            fd: wake_read,
            events: libc::POLLIN,
            revents: 0,
        });
        fds.extend(p.waiting().iter().map(|&(fd, events, _)| libc::pollfd {
            fd,
            events,
            revents: 0,
        }));
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
            continue;
        }
        if fds[0].revents != 0 {
            let mut bytes = [0u8; 64];
            while unsafe { libc::read(wake_read, bytes.as_mut_ptr().cast(), bytes.len()) } > 0 {}
        }
        // Only `add` changes the list here, and it adds at the end. So entry `i` of the list
        // is entry `i + 1` of `fds`, and an entry added after the poll has no entry in `fds`.
        let mut ready = Vec::new();
        let mut i = 0;
        p.waiting().retain(|&(_, _, word)| {
            i += 1;
            let fired = fds.get(i).is_some_and(|f| f.revents != 0);
            if fired {
                ready.push(word);
            }
            !fired
        });
        for word in ready {
            unsafe {
                (*(word as *const AtomicI64)).store(1, Ordering::SeqCst);
                wait::vx_wake_all(word as *const i64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ffi::sched::Scheduler;
    use crate::ffi::tasks::{tasks_begin, vx_task_start, vx_tasks_end};
    use crate::ffi::wait::{vx_wait_while, vx_wake_all};
    use crate::net::{
        vx_tcp_listener_accept, vx_tcp_listener_bind, vx_tcp_listener_drop, vx_tcp_stream_connect,
        vx_tcp_stream_drop, vx_tcp_stream_read, vx_tcp_stream_write,
    };
    use std::ffi::{c_void, CString};
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    struct Talk {
        listener: *mut c_void,
        address: CString,
        /// 1 after the server accepted.
        accepted: AtomicI64,
        got: Mutex<Vec<u8>>,
    }

    fn talk(env: *mut u8) -> &'static Talk {
        unsafe { &**(env as *const *const Talk) }
    }

    /// Accepts, and then reads before the client wrote. Both wait for the client task, which
    /// runs on the same worker.
    extern "C" fn server(env: *mut u8) {
        let t = talk(env);
        let stream = vx_tcp_listener_accept(t.listener);
        t.accepted.store(1, Ordering::SeqCst);
        unsafe { vx_wake_all(t.accepted.as_ptr()) };
        let mut byte = 0u8;
        let n = vx_tcp_stream_read(stream, &mut byte, 1);
        t.got.lock().unwrap().extend(&[byte][..n as usize]);
        vx_tcp_stream_drop(stream);
    }

    /// Connects, and writes only after the server accepted, so that the read of the server
    /// waits.
    extern "C" fn client(env: *mut u8) {
        let t = talk(env);
        let stream = vx_tcp_stream_connect(t.address.as_ptr());
        while t.accepted.load(Ordering::SeqCst) == 0 {
            unsafe { vx_wait_while(t.accepted.as_ptr(), 0) };
        }
        let byte = b'V';
        assert_eq!(vx_tcp_stream_write(stream, &byte, 1), 1);
        vx_tcp_stream_drop(stream);
    }

    fn env_with<T>(value: T) -> *mut u8 {
        unsafe {
            let e = libc::malloc(std::mem::size_of::<T>()) as *mut T;
            e.write(value);
            e as *mut u8
        }
    }

    #[test]
    fn a_task_that_waits_for_a_socket_lets_another_task_run_on_its_worker() {
        let done = std::sync::Arc::new(AtomicI64::new(0));
        let seen = done.clone();
        std::thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(10);
            while seen.load(Ordering::SeqCst) == 0 {
                if Instant::now() > end {
                    // Not `eprintln!`: the test captures it, and the abort loses it.
                    let _ = std::io::Write::write_all(
                        &mut std::io::stderr(),
                        b"the socket test hangs: a task that waited for a socket did not go on\n",
                    );
                    std::process::abort();
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        // One worker, and the monitor cannot add one. A wait that blocked the worker would
        // leave the client task with no thread to run on.
        let sched = Scheduler::start(1, 1);
        let any_port = CString::new("127.0.0.1:0").unwrap();
        let listener = vx_tcp_listener_bind(any_port.as_ptr());
        assert!(!listener.is_null(), "the test could not listen");
        let bound = unsafe { &*(listener as *const std::net::TcpListener) };
        let t = Talk {
            listener,
            address: CString::new(bound.local_addr().unwrap().to_string()).unwrap(),
            accepted: AtomicI64::new(0),
            got: Mutex::new(Vec::new()),
        };
        let group = tasks_begin(Some(sched));
        unsafe {
            vx_task_start(group, server, env_with(&t as *const Talk));
            vx_task_start(group, client, env_with(&t as *const Talk));
            vx_tasks_end(group);
        }
        done.store(1, Ordering::SeqCst);
        vx_tcp_listener_drop(t.listener);
        assert_eq!(*t.got.lock().unwrap(), b"V");
    }

    /// A thread that runs no fiber waits in poll(2). The client connects and writes only
    /// later, so the accept and the read each find nothing at first.
    #[test]
    fn a_thread_that_waits_for_a_socket_gets_the_data_that_comes_later() {
        let any_port = CString::new("127.0.0.1:0").unwrap();
        let listener = vx_tcp_listener_bind(any_port.as_ptr());
        assert!(!listener.is_null(), "the test could not listen");
        let bound = unsafe { &*(listener as *const std::net::TcpListener) };
        let address = bound.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let mut s = std::net::TcpStream::connect(address).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            std::io::Write::write_all(&mut s, b"V").unwrap();
        });
        let stream = vx_tcp_listener_accept(listener);
        assert!(
            !stream.is_null(),
            "the accept gave up before the client connected"
        );
        let mut byte = 0u8;
        let n = vx_tcp_stream_read(stream, &mut byte, 1);
        assert_eq!(
            (n, byte),
            (1, b'V'),
            "the read gave up before the client wrote"
        );
        client.join().unwrap();
        vx_tcp_stream_drop(stream);
        vx_tcp_listener_drop(listener);
    }
}
