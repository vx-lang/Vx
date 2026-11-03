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
// sets the word and wakes the caller. A connect starts on a non-blocking socket, and waits
// until the socket is ready to write.
//
// A cancel stops the wait (F4). The caller takes its socket off the list before it returns,
// and the poller sets a word only under the lock of the list. So the poller cannot write to a
// word after its wait returned. The call then gives an error. No kernel call still holds the
// buffer of the caller at that point, so the cancel needs no more (F5).
//
// A file is not a socket: poll(2) says that a regular file is always ready, and then a read
// still waits for the disk. So a fiber runs a file call on a helper thread, and parks until the
// call ends. A cancel does not stop that wait, because the call still uses the buffer of the
// caller (F5).
//
// poll(2) works the same on Linux and on macOS. epoll and kqueue can replace it when a program
// waits on many sockets.
//
//===----------------------------------------------------------------------===//

use crate::ffi::{sched, wait};
use std::collections::VecDeque;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

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

/// Connects to `addr`, written as `host:port`, and tries each address of the host in turn. The
/// wait for each connection is as for `when_ready`. The lookup of a host name still blocks the
/// thread.
pub fn connect(addr: &str) -> std::io::Result<std::net::TcpStream> {
    let mut last = std::io::Error::other("the host has no address");
    for a in std::net::ToSocketAddrs::to_socket_addrs(addr)? {
        match connect_to(a) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn connect_to(addr: std::net::SocketAddr) -> std::io::Result<std::net::TcpStream> {
    let (family, storage, len) = socket_address(addr);
    let fd = unsafe { libc::socket(family, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // The stream closes the socket on each error below.
    let stream = unsafe { <std::net::TcpStream as std::os::fd::FromRawFd>::from_raw_fd(fd) };
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    stream.set_nonblocking(true)?;
    let started = unsafe { libc::connect(fd, (&raw const storage).cast(), len) };
    if started == 0 {
        return Ok(stream);
    }
    let e = std::io::Error::last_os_error();
    // After EINTR, the connection goes on as after EINPROGRESS.
    if !matches!(
        e.raw_os_error(),
        Some(libc::EINPROGRESS) | Some(libc::EINTR)
    ) {
        return Err(e);
    }
    loop {
        if !wait_ready(fd, true) {
            return Err(std::io::Error::other("the wait was cancelled"));
        }
        if let Some(e) = stream.take_error()? {
            return Err(e);
        }
        // A wait can return early. The connection is made once the socket has a peer.
        if stream.peer_addr().is_ok() {
            return Ok(stream);
        }
    }
}

/// The C form of `addr`: its family, a `sockaddr_storage` that holds it, and its length.
fn socket_address(addr: std::net::SocketAddr) -> (i32, libc::sockaddr_storage, libc::socklen_t) {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match addr {
        std::net::SocketAddr::V4(a) => {
            let mut sin: libc::sockaddr_in = unsafe { std::mem::zeroed() };
            sin.sin_family = libc::AF_INET as libc::sa_family_t;
            sin.sin_port = a.port().to_be();
            sin.sin_addr.s_addr = u32::from_ne_bytes(a.ip().octets());
            #[cfg(target_os = "macos")]
            {
                sin.sin_len = std::mem::size_of::<libc::sockaddr_in>() as u8;
            }
            unsafe { (&raw mut storage).cast::<libc::sockaddr_in>().write(sin) };
            std::mem::size_of::<libc::sockaddr_in>()
        }
        std::net::SocketAddr::V6(a) => {
            let mut sin6: libc::sockaddr_in6 = unsafe { std::mem::zeroed() };
            sin6.sin6_family = libc::AF_INET6 as libc::sa_family_t;
            sin6.sin6_port = a.port().to_be();
            sin6.sin6_flowinfo = a.flowinfo();
            sin6.sin6_addr.s6_addr = a.ip().octets();
            sin6.sin6_scope_id = a.scope_id();
            #[cfg(target_os = "macos")]
            {
                sin6.sin6_len = std::mem::size_of::<libc::sockaddr_in6>() as u8;
            }
            unsafe { (&raw mut storage).cast::<libc::sockaddr_in6>().write(sin6) };
            std::mem::size_of::<libc::sockaddr_in6>()
        }
    };
    let family = match addr {
        std::net::SocketAddr::V4(_) => libc::AF_INET,
        std::net::SocketAddr::V6(_) => libc::AF_INET6,
    };
    (family, storage, len as libc::socklen_t)
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

/// Runs `call`. A fiber runs it on a helper thread, and parks until it ends, so that its worker
/// can run other tasks. A thread that runs no fiber runs it itself.
pub fn off_the_worker<T>(call: impl FnOnce() -> T) -> T {
    if sched::fiber_that_can_park().is_null() {
        return call();
    }
    let mut out = None;
    let done = AtomicI64::new(0);
    let job: Box<dyn FnOnce() + '_> = Box::new(|| out = Some(call()));
    // The fiber waits below until the helper set `done`, so `call`, `out` and `done` live
    // until the helper no longer uses them. The fiber does not run while the helper runs the
    // call, so the call is the only code that uses what it captured.
    let job: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(job) };
    let word = done.as_ptr() as usize;
    BUSY.fetch_add(1, Ordering::SeqCst);
    helpers().run(Box::new(move || {
        // A panic would leave the fiber parked forever.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
            eprintln!("vx: a file call panicked on a helper thread. The program stops.");
            std::process::abort();
        }
        BUSY.fetch_sub(1, Ordering::SeqCst);
        unsafe {
            (*(word as *const AtomicI64)).store(1, Ordering::SeqCst);
            // The wake reads only the address, so the word can be gone by now.
            wait::vx_wake_all(word as *const i64);
        }
    }));
    while done.load(Ordering::SeqCst) == 0 {
        unsafe { wait::vx_wait_while(done.as_ptr(), 0) };
    }
    out.expect("the helper ran the call")
}

/// The calls that helper threads run now.
static BUSY: AtomicUsize = AtomicUsize::new(0);

type Job = Box<dyn FnOnce() + Send>;

/// Threads that run file calls for fibers. A thread stays when its call ends, and waits for
/// the next one. There is one more thread when each one is busy.
struct Helpers {
    /// The calls that no thread took yet, and the count of threads that wait for one.
    state: Mutex<(VecDeque<Job>, usize)>,
    cond: Condvar,
}

static HELPERS: OnceLock<Helpers> = OnceLock::new();

fn helpers() -> &'static Helpers {
    HELPERS.get_or_init(|| Helpers {
        state: Mutex::new((VecDeque::new(), 0)),
        cond: Condvar::new(),
    })
}

impl Helpers {
    fn state(&self) -> MutexGuard<'_, (VecDeque<Job>, usize)> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn run(&'static self, job: Job) {
        let mut s = self.state();
        s.0.push_back(job);
        // A thread that waits counts until it takes the lock again. So more calls than such
        // threads means that one call has no thread.
        if s.0.len() > s.1 {
            std::thread::Builder::new()
                .name("vx-file".into())
                .spawn(move || self.serve())
                .expect("the system could not start a thread for a file call");
        }
        drop(s);
        self.cond.notify_one();
    }

    fn serve(&self) {
        let mut s = self.state();
        loop {
            if let Some(job) = s.0.pop_front() {
                drop(s);
                job();
                s = self.state();
                continue;
            }
            s.1 += 1;
            s = self.cond.wait(s).unwrap_or_else(|e| e.into_inner());
            s.1 -= 1;
        }
    }
}

/// Whether a task or thread waits for a socket, or a helper runs a file call. The end of
/// either can wake a task.
pub(crate) fn pending() -> bool {
    BUSY.load(Ordering::SeqCst) > 0 || POLLER.get().is_some_and(|p| !p.waiting().is_empty())
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
    /// It also makes the tests that hold one run one at a time, because a test looks at the
    /// whole list of the poller.
    struct Watchdog {
        done: std::sync::Arc<AtomicI64>,
        _turn: std::sync::MutexGuard<'static, ()>,
    }

    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    impl Watchdog {
        fn new(test: &'static str) -> Watchdog {
            let turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let done = std::sync::Arc::new(AtomicI64::new(0));
            let seen = done.clone();
            std::thread::spawn(move || {
                let end = Instant::now() + Duration::from_secs(10);
                while seen.load(Ordering::SeqCst) == 0 {
                    if Instant::now() > end {
                        // Not `eprintln!`: the test captures it, and the abort loses it.
                        let line = format!("{test} hangs: a wait for I/O did not end\n");
                        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
                        std::process::abort();
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            Watchdog { done, _turn: turn }
        }
    }

    impl Drop for Watchdog {
        fn drop(&mut self) {
            self.done.store(1, Ordering::SeqCst);
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
        let _watchdog =
            Watchdog::new("a_thread_that_waits_for_a_socket_gets_the_data_that_comes_later");
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

    #[cfg(target_os = "linux")]
    struct Dial {
        address: CString,
        /// 1 when the connect gave a stream, 0 when it gave null.
        connected: AtomicI64,
        other_ran: AtomicI64,
    }

    #[cfg(target_os = "linux")]
    extern "C" fn connect_to_a_full_listener(env: *mut u8) {
        let d = unsafe { &**(env as *const *const Dial) };
        let stream = vx_tcp_stream_connect(d.address.as_ptr());
        d.connected
            .store(!stream.is_null() as i64, Ordering::SeqCst);
        vx_tcp_stream_drop(stream);
    }

    #[cfg(target_os = "linux")]
    extern "C" fn note_that_it_ran(env: *mut u8) {
        let d = unsafe { &**(env as *const *const Dial) };
        d.other_ran.store(1, Ordering::SeqCst);
    }

    /// Whether a caller waits to write, as a connect does.
    #[cfg(target_os = "linux")]
    fn waits_to_write() -> bool {
        super::POLLER
            .get()
            .is_some_and(|p| p.waiting().iter().any(|&(_, e, _)| e == libc::POLLOUT))
    }

    /// Linux drops a new connection to a listener whose queue of connections is full. So a
    /// connect to it waits. On one worker, the other task must still run, and leaving the
    /// block early must stop the connect.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_task_that_connects_parks_and_a_cancel_stops_it() {
        let _watchdog = Watchdog::new("a_task_that_connects_parks_and_a_cancel_stops_it");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let fd = std::os::fd::AsRawFd::as_raw_fd(&listener);
        // A queue of one connection, and one connection that fills it.
        assert_eq!(unsafe { libc::listen(fd, 0) }, 0);
        let _first = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let d = Dial {
            address: CString::new(listener.local_addr().unwrap().to_string()).unwrap(),
            connected: AtomicI64::new(-1),
            other_ran: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(Scheduler::start(1, 1)));
        unsafe {
            vx_task_start(
                group,
                connect_to_a_full_listener,
                env_with(&d as *const Dial),
            );
            vx_task_start(group, note_that_it_ran, env_with(&d as *const Dial));
            let end = Instant::now() + Duration::from_secs(5);
            while !(waits_to_write() && d.other_ran.load(Ordering::SeqCst) == 1) {
                assert!(
                    Instant::now() < end,
                    "the connect did not wait, or the other task did not run"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            vx_tasks_wait(group);
            vx_tasks_end(group);
        }
        assert_eq!(
            d.connected.load(Ordering::SeqCst),
            0,
            "the cancelled connect did not give null"
        );
        assert!(!waits_to_write(), "the poller still waits for the connect");
    }

    /// A connect to a port where nothing listens gives null, after the wait for the socket.
    #[test]
    fn a_refused_connect_gives_null() {
        let _watchdog = Watchdog::new("a_refused_connect_gives_null");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = CString::new(listener.local_addr().unwrap().to_string()).unwrap();
        drop(listener);
        let stream = vx_tcp_stream_connect(address.as_ptr());
        assert!(stream.is_null(), "a refused connect gave a stream");
    }

    struct Fifo {
        path: CString,
        /// 1 after the reader opened the FIFO.
        opened: AtomicI64,
        got: AtomicI64,
    }

    /// An open of a FIFO to read waits until a writer opens it, and the read waits for data.
    extern "C" fn read_the_fifo(env: *mut u8) {
        let f = unsafe { &**(env as *const *const Fifo) };
        let file = crate::io::vx_file_open(f.path.as_ptr(), 0);
        f.opened.store(1, Ordering::SeqCst);
        unsafe { vx_wake_all(f.opened.as_ptr()) };
        let mut byte = 0u8;
        if crate::io::vx_file_read(file, &mut byte, 1) == 1 {
            f.got.store(byte as i64, Ordering::SeqCst);
        }
        crate::io::vx_file_drop(file);
    }

    extern "C" fn write_the_fifo(env: *mut u8) {
        let f = unsafe { &**(env as *const *const Fifo) };
        let file = crate::io::vx_file_open(f.path.as_ptr(), 1);
        // The write comes after the reader opened, so the read of the reader waits.
        while f.opened.load(Ordering::SeqCst) == 0 {
            unsafe { vx_wait_while(f.opened.as_ptr(), 0) };
        }
        let byte = b'V';
        assert_eq!(crate::io::vx_file_write(file, &byte, 1), 1);
        crate::io::vx_file_drop(file);
    }

    /// On one worker, a task opens and reads a FIFO before the other task opens it to write,
    /// and before it writes. The file calls of the reader must not block the worker.
    #[test]
    fn a_task_that_waits_for_a_file_lets_another_task_run_on_its_worker() {
        let _watchdog =
            Watchdog::new("a_task_that_waits_for_a_file_lets_another_task_run_on_its_worker");
        let path = std::env::temp_dir().join(format!("vx-io-test-fifo-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let path = CString::new(path.to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        let f = Fifo {
            path,
            opened: AtomicI64::new(0),
            got: AtomicI64::new(0),
        };
        let group = tasks_begin(Some(Scheduler::start(1, 1)));
        unsafe {
            vx_task_start(group, read_the_fifo, env_with(&f as *const Fifo));
            vx_task_start(group, write_the_fifo, env_with(&f as *const Fifo));
            vx_tasks_end(group);
        }
        unsafe { libc::unlink(f.path.as_ptr()) };
        assert_eq!(
            f.got.load(Ordering::SeqCst),
            b'V' as i64,
            "the reader did not get the byte"
        );
    }
}
