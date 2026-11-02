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
// until the socket is ready, and then tries the call again. The caller puts its socket on the
// list of one poller thread, and waits on a word of its stack (wait.rs): a fiber parks, and a
// thread sleeps. The poller calls poll(2) on the sockets of the list. When one is ready, it
// sets the word and wakes the caller.
//
// A cancel stops the wait (F4). The caller takes its socket off the list before it returns,
// and the poller sets a word only under the lock of the list. So the poller cannot write to a
// word after its wait returned. The call then gives an error. No kernel call still holds the
// buffer of the caller at that point, so the cancel needs no more (F5).
//
// poll(2) works the same on Linux and on macOS. epoll and kqueue can replace it when a program
// waits on many sockets.
//
//===----------------------------------------------------------------------===//

use crate::ffi::wait;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Runs `call` until it does not give `WouldBlock` or `Interrupted`. Before each new try, it
/// waits until `fd` is ready to read, or to write when `write` is set. A cancel of the block of
/// the caller stops the wait, and gives an error.
pub fn when_ready<T>(
    fd: RawFd,
    write: bool,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    loop {
        match call() {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if !wait_ready(fd, write) {
                    return Err(std::io::Error::other("the wait was cancelled"));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            r => return r,
        }
    }
}

/// Waits until `fd` is ready, and gives false when a cancel stopped the wait. It can return
/// early, so the caller tries its call again.
fn wait_ready(fd: RawFd, write: bool) -> bool {
    let events = if write { libc::POLLOUT } else { libc::POLLIN };
    let ready = AtomicI64::new(0);
    let p = poller();
    p.add(fd, events, &ready);
    while ready.load(Ordering::SeqCst) == 0 {
        if crate::ffi::tasks::vx_cancelled() != 0 && p.remove(&ready) {
            return false;
        }
        unsafe { wait::vx_wait_while_or_cancel(ready.as_ptr(), 0) };
    }
    true
}

struct Poller {
    /// The sockets that callers wait for: the fd, its events, and the address of the word to set.
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

    /// Takes the entry of `word` off the list. False when the poller took it first, and so
    /// already set the word.
    fn remove(&self, word: &AtomicI64) -> bool {
        let mut waiting = self.waiting();
        let before = waiting.len();
        waiting.retain(|&(_, _, w)| w != word.as_ptr() as usize);
        waiting.len() < before
    }
}

/// Whether a task or thread waits for a socket. Data from outside the program can wake it.
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
    let mut words = Vec::new();
    loop {
        fds.clear();
        words.clear();
        fds.push(libc::pollfd {
            fd: wake_read,
            events: libc::POLLIN,
            revents: 0,
        });
        for &(fd, events, word) in p.waiting().iter() {
            fds.push(libc::pollfd {
                fd,
                events,
                revents: 0,
            });
            words.push(word);
        }
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
            continue;
        }
        if fds[0].revents != 0 {
            let mut bytes = [0u8; 64];
            while unsafe { libc::read(wake_read, bytes.as_mut_ptr().cast(), bytes.len()) } > 0 {}
        }
        let fired: Vec<usize> = (0..words.len())
            .filter(|&i| fds[i + 1].revents != 0)
            .map(|i| words[i])
            .collect();
        if fired.is_empty() {
            continue;
        }
        // A cancelled caller can take its entry off the list during the poll, and then its
        // word is gone. So only an entry that is still on the list gets its word set, under
        // the lock. A new entry at the same address gets a spurious wake, and tries again.
        let mut woken = Vec::new();
        p.waiting().retain(|&(_, _, word)| {
            if !fired.contains(&word) {
                return true;
            }
            unsafe { (*(word as *const AtomicI64)).store(1, Ordering::SeqCst) };
            woken.push(word);
            false
        });
        for word in woken {
            // The wake reads only the address, so the word can be gone by now.
            unsafe { wait::vx_wake_all(word as *const i64) };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ffi::sched::Scheduler;
    use crate::ffi::tasks::{tasks_begin, vx_task_start, vx_tasks_end, vx_tasks_wait};
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

    /// Stops the test program when a test runs longer than 10 seconds, and names the test.
    struct Watchdog(std::sync::Arc<AtomicI64>);

    impl Watchdog {
        fn new(test: &'static str) -> Watchdog {
            let done = std::sync::Arc::new(AtomicI64::new(0));
            let seen = done.clone();
            std::thread::spawn(move || {
                let end = Instant::now() + Duration::from_secs(10);
                while seen.load(Ordering::SeqCst) == 0 {
                    if Instant::now() > end {
                        // Not `eprintln!`: the test captures it, and the abort loses it.
                        let line = format!("{test} hangs: a wait for a socket did not end\n");
                        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
                        std::process::abort();
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            Watchdog(done)
        }
    }

    impl Drop for Watchdog {
        fn drop(&mut self) {
            self.0.store(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_task_that_waits_for_a_socket_lets_another_task_run_on_its_worker() {
        let _watchdog =
            Watchdog::new("a_task_that_waits_for_a_socket_lets_another_task_run_on_its_worker");
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
        vx_tcp_listener_drop(t.listener);
        assert_eq!(*t.got.lock().unwrap(), b"V");
    }

    /// A thread that runs no fiber sleeps until the socket is ready. The client connects and writes only
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

    /// Whether a caller waits for `fd`.
    fn waits_for(fd: i32) -> bool {
        super::POLLER
            .get()
            .is_some_and(|p| p.waiting().iter().any(|&(f, _, _)| f == fd))
    }

    struct Quiet {
        stream: *mut c_void,
        got: AtomicI64,
    }

    /// Reads from a socket that gets no data.
    extern "C" fn read_until_cancelled(env: *mut u8) {
        let q = unsafe { &**(env as *const *const Quiet) };
        let mut byte = 0u8;
        let n = vx_tcp_stream_read(q.stream, &mut byte, 1);
        q.got.store(n as i64, Ordering::SeqCst);
    }

    /// Leaves a block early while its task waits for a socket that gets no data. The cancel
    /// must stop the wait, and the poller must forget the socket.
    fn a_cancel_stops_a_read(test: &'static str, sched: Option<&'static Scheduler>) {
        let _watchdog = Watchdog::new(test);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let server = listener.accept().unwrap().0;
        server.set_nonblocking(true).unwrap();
        let fd = std::os::fd::AsRawFd::as_raw_fd(&server);
        let q = Quiet {
            stream: Box::into_raw(Box::new(server)).cast(),
            got: AtomicI64::new(-1),
        };
        let group = tasks_begin(sched);
        unsafe {
            vx_task_start(group, read_until_cancelled, env_with(&q as *const Quiet));
            let end = Instant::now() + Duration::from_secs(5);
            while !waits_for(fd) {
                assert!(Instant::now() < end, "the task did not wait for the socket");
                std::thread::sleep(Duration::from_millis(1));
            }
            vx_tasks_wait(group);
            vx_tasks_end(group);
        }
        assert_eq!(
            q.got.load(Ordering::SeqCst),
            0,
            "the cancelled read did not give 0"
        );
        assert!(!waits_for(fd), "the poller still waits for the socket");
        vx_tcp_stream_drop(q.stream);
    }

    #[test]
    fn a_cancel_stops_a_task_that_waits_for_a_socket() {
        a_cancel_stops_a_read(
            "a_cancel_stops_a_task_that_waits_for_a_socket",
            Some(Scheduler::start(1, 1)),
        );
    }

    #[test]
    fn a_cancel_stops_a_task_thread_that_waits_for_a_socket() {
        a_cancel_stops_a_read("a_cancel_stops_a_task_thread_that_waits_for_a_socket", None);
    }
}
