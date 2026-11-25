//===- crash_report.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The report of a SIGSEGV or SIGBUS: a message and a backtrace with the function, file and line
// of each frame.
//
// The code that faulted can hold any lock, also the lock of the allocator. So the handler must
// not allocate, and it cannot read DWARF itself. `prepare` runs when the program starts. It
// finds `llvm-symbolizer` and the loaded images. The handler then starts the symbolizer with
// `fork` and `execve`, which are safe in a signal handler, and numbers its lines. A frame that
// the symbolizer does not name gets the line of `backtrace_symbols_fd`. Before that line, the
// handler writes the function name from the symbol table of the file. It reads the table with
// `open`, `lseek` and `read` into buffers on the stack.
//
//===----------------------------------------------------------------------===//

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// The frames that the report prints at most.
const CRASH_FRAMES: usize = 64;

/// The seconds that the report may take before SIGALRM ends the program.
const CRASH_REPORT_SECONDS: libc::c_uint = 10;

/// A file that was loaded when the program started: the program or a shared library.
struct Image {
    /// The path in double quotes, as the symbolizer reads it. Empty when the file cannot be read.
    quoted_path: Vec<u8>,
    /// The path that the handler opens to read the symbol table. None when the file cannot be read.
    path: Option<CString>,
    /// The loader put each address of the file this many bytes higher.
    bias: usize,
    /// The image is in memory from `start` to before `end`.
    start: usize,
    end: usize,
}

struct Prepared {
    symbolizer: Option<CString>,
    images: Vec<Image>,
}

static PREPARED: OnceLock<Prepared> = OnceLock::new();

/// Set by the first thread that reports a crash.
static REPORTING: AtomicBool = AtomicBool::new(false);

/// Finds the symbolizer and the loaded images, and installs the handler. Runs before the program.
/// On macOS the report loses the function that crashed, so a crash there gets no handler and no
/// report for now.
pub(crate) fn prepare() {
    if cfg!(target_os = "macos") {
        return;
    }
    PREPARED.get_or_init(|| Prepared {
        symbolizer: find_symbolizer(),
        images: loaded_images(),
    });
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = vx_sigsegv_handler as *const () as libc::sighandler_t;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(libc::SIGSEGV, &action, ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &action, ptr::null_mut());
    }
}

/// `LLVM_SYMBOLIZER_PATH` when it is set, and `llvm-symbolizer` on the `PATH` when it is not.
/// An empty `LLVM_SYMBOLIZER_PATH` turns the symbolizer off.
fn find_symbolizer() -> Option<CString> {
    let executable = |path: &std::path::Path| {
        CString::new(path.as_os_str().as_bytes())
            .ok()
            .filter(|c| unsafe { libc::access(c.as_ptr(), libc::X_OK) } == 0)
    };
    if let Some(path) = std::env::var_os("LLVM_SYMBOLIZER_PATH") {
        return executable(std::path::Path::new(&path));
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .find_map(|dir| executable(&dir.join("llvm-symbolizer")))
}

/// The vDSO of Linux and the libraries in the shared cache of macOS have no file to read.
fn push_image(images: &mut Vec<Image>, path: &[u8], bias: usize, start: usize, end: usize) {
    if start >= end {
        return;
    }
    let readable = CString::new(path)
        .ok()
        .filter(|c| unsafe { libc::access(c.as_ptr(), libc::R_OK) } == 0);
    let mut quoted_path = Vec::new();
    if readable.is_some() && !path.contains(&b'"') {
        quoted_path.push(b'"');
        quoted_path.extend_from_slice(path);
        quoted_path.push(b'"');
    }
    images.push(Image {
        quoted_path,
        path: readable,
        bias,
        start: bias + start,
        end: bias + end,
    });
}

#[cfg(target_os = "linux")]
fn loaded_images() -> Vec<Image> {
    unsafe extern "C" fn add(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        data: *mut c_void,
    ) -> c_int {
        let images = &mut *(data as *mut Vec<Image>);
        let info = &*info;
        let headers = std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize);
        let loads = headers.iter().filter(|h| h.p_type == libc::PT_LOAD);
        let start = loads.clone().map(|h| h.p_vaddr as usize).min();
        let end = loads.map(|h| (h.p_vaddr + h.p_memsz) as usize).max();
        let (Some(start), Some(end)) = (start, end) else {
            return 0;
        };
        // The loader gives the program itself an empty name.
        let name = if info.dlpi_name.is_null() || *info.dlpi_name == 0 {
            match std::env::current_exe() {
                Ok(path) => path.as_os_str().as_bytes().to_vec(),
                Err(_) => return 0,
            }
        } else {
            CStr::from_ptr(info.dlpi_name).to_bytes().to_vec()
        };
        push_image(images, &name, info.dlpi_addr as usize, start, end);
        // The handler reads the symbol table of the program through `/proc/self/exe`. This file
        // is the running program, also when a new build replaced the file at its path.
        if let Some(image) = images.last_mut().filter(|image| image.path.is_some()) {
            if info.dlpi_name.is_null() || *info.dlpi_name == 0 {
                image.path = Some(c"/proc/self/exe".to_owned());
            }
        }
        0
    }
    let mut images = Vec::new();
    unsafe { libc::dl_iterate_phdr(Some(add), &mut images as *mut Vec<Image> as *mut c_void) };
    images
}

#[cfg(target_os = "macos")]
// `libc` asks for the `mach2` crate for the calls of dyld, which are the same calls.
#[allow(deprecated)]
fn loaded_images() -> Vec<Image> {
    let mut images = Vec::new();
    unsafe {
        for i in 0..libc::_dyld_image_count() {
            let header = libc::_dyld_get_image_header(i) as *const libc::mach_header_64;
            let name = libc::_dyld_get_image_name(i);
            if header.is_null() || name.is_null() {
                continue;
            }
            let (mut start, mut end) = (usize::MAX, 0);
            let mut command = header.add(1) as *const u8;
            for _ in 0..(*header).ncmds {
                let load = &*(command as *const libc::load_command);
                if load.cmd == libc::LC_SEGMENT_64 {
                    let segment = &*(command as *const libc::segment_command_64);
                    // `__PAGEZERO` maps no memory.
                    if segment.initprot != 0 {
                        start = start.min(segment.vmaddr as usize);
                        end = end.max((segment.vmaddr + segment.vmsize) as usize);
                    }
                }
                command = command.add(load.cmdsize as usize);
            }
            let bias = libc::_dyld_get_image_vmaddr_slide(i) as usize;
            push_image(
                &mut images,
                CStr::from_ptr(name).to_bytes(),
                bias,
                start,
                end,
            );
        }
    }
    images
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn loaded_images() -> Vec<Image> {
    Vec::new()
}

/// The program counter of the code that faulted, or 0 when it is not known.
unsafe fn faulting_pc(context: *mut c_void) -> usize {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let context = &*(context as *const libc::ucontext_t);
        context.uc_mcontext.gregs[libc::REG_RIP as usize] as usize
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let context = &*(context as *const libc::ucontext_t);
        context.uc_mcontext.pc as usize
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // The start of `ucontext_t`, `mcontext64` and `arm_thread_state64` in the headers of
        // macOS. The `libc` crate does not declare them.
        #[repr(C)]
        struct UContext {
            onstack: c_int,
            sigmask: u32,
            stack: libc::stack_t,
            link: *mut c_void,
            mcsize: usize,
            mcontext: *const MContext,
        }
        #[repr(C)]
        struct MContext {
            exception_state: [u64; 2],
            x: [u64; 29],
            fp: u64,
            lr: u64,
            sp: u64,
            pc: u64,
        }
        let context = &*(context as *const UContext);
        if context.mcontext.is_null() {
            0
        } else {
            (*context.mcontext).pc as usize
        }
    }
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(target_os = "macos", target_arch = "aarch64")
    )))]
    {
        let _ = context;
        0
    }
}

/// Writes to a file descriptor through a buffer on the stack.
struct Output {
    fd: c_int,
    buffer: [u8; 512],
    len: usize,
}

impl Output {
    fn new(fd: c_int) -> Self {
        Output {
            fd,
            buffer: [0; 512],
            len: 0,
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.len == self.buffer.len() {
                self.flush();
            }
            self.buffer[self.len] = b;
            self.len += 1;
        }
    }

    fn number(&mut self, mut n: usize, base: usize, min_digits: usize) {
        let mut digits = [0u8; 20];
        let mut count = 0;
        while n > 0 || count < min_digits.max(1) {
            digits[count] = b"0123456789abcdef"[n % base];
            n /= base;
            count += 1;
        }
        digits[..count].reverse();
        self.bytes(&digits[..count]);
    }

    fn flush(&mut self) {
        let mut done = 0;
        while done < self.len {
            let n = unsafe {
                libc::write(
                    self.fd,
                    self.buffer[done..].as_ptr() as *const c_void,
                    self.len - done,
                )
            };
            if n <= 0 {
                break;
            }
            done += n as usize;
        }
        self.len = 0;
    }
}

/// Reads the output of the symbolizer one frame at a time.
struct Input {
    fd: c_int,
    buffer: [u8; 512],
    pos: usize,
    len: usize,
}

impl Input {
    fn next(&mut self) -> Option<u8> {
        if self.pos == self.len {
            let n = unsafe {
                libc::read(
                    self.fd,
                    self.buffer.as_mut_ptr() as *mut c_void,
                    self.buffer.len(),
                )
            };
            if n <= 0 {
                return None;
            }
            self.pos = 0;
            self.len = n as usize;
        }
        self.pos += 1;
        Some(self.buffer[self.pos - 1])
    }

    /// Copies the lines of one frame to `out`. With `--pretty-print`, the symbolizer ends each
    /// frame with an empty line. Gives false when the symbolizer wrote nothing more.
    fn copy_frame(&mut self, out: &mut Output) -> bool {
        let Some(mut b) = self.next() else {
            return false;
        };
        loop {
            out.bytes(&[b]);
            if b == b'\n' {
                match self.next() {
                    None | Some(b'\n') => return true,
                    // A function that was inlined into the next one.
                    Some(next) => {
                        out.bytes(b"   ");
                        b = next;
                        continue;
                    }
                }
            }
            match self.next() {
                Some(next) => b = next,
                None => {
                    out.bytes(b"\n");
                    return true;
                }
            }
        }
    }
}

/// Starts the symbolizer and writes one request for each frame in `requests`. Gives the process
/// and the file descriptor of its output.
unsafe fn start_symbolizer(
    symbolizer: &CStr,
    images: &[Image],
    frames: &[usize],
    requests: &[Option<usize>],
) -> Option<(libc::pid_t, c_int)> {
    let (mut input, mut output) = ([0; 2], [0; 2]);
    if libc::pipe(input.as_mut_ptr()) != 0 {
        return None;
    }
    if libc::pipe(output.as_mut_ptr()) != 0 {
        libc::close(input[0]);
        libc::close(input[1]);
        return None;
    }
    let argv = [symbolizer.as_ptr(), c"--pretty-print".as_ptr(), ptr::null()];
    let envp: [*const c_char; 1] = [ptr::null()];
    let pid = libc::fork();
    if pid == 0 {
        libc::dup2(input[0], 0);
        libc::dup2(output[1], 1);
        for fd in input.iter().chain(output.iter()) {
            libc::close(*fd);
        }
        libc::execve(symbolizer.as_ptr(), argv.as_ptr(), envp.as_ptr());
        libc::_exit(127);
    }
    libc::close(input[0]);
    libc::close(output[1]);
    if pid < 0 {
        libc::close(input[1]);
        libc::close(output[0]);
        return None;
    }
    let mut out = Output::new(input[1]);
    for (i, request) in requests.iter().enumerate() {
        if let Some(image) = request {
            let image = &images[*image];
            out.bytes(&image.quoted_path);
            out.bytes(b" 0x");
            out.number(frames[i] - image.bias, 16, 1);
            out.bytes(b"\n");
        }
    }
    out.flush();
    libc::close(input[1]);
    Some((pid, output[0]))
}

/// Whether `address` is in an image that was loaded when the program started. On Linux, also
/// whether it is in an image that `dladdr` knows, such as a library loaded later. On macOS, the
/// `__PAGEZERO` segment of the program covers the first 4 GiB, so `dladdr` is not used there.
unsafe fn in_an_image(address: usize) -> bool {
    let known = PREPARED.get().is_some_and(|prepared| {
        prepared
            .images
            .iter()
            .any(|image| image.start <= address && address < image.end)
    });
    #[cfg(target_os = "linux")]
    let known = known || {
        let mut info: libc::Dl_info = std::mem::zeroed();
        libc::dladdr(address as *const c_void, &mut info) != 0
    };
    known
}

/// Reads `buffer.len()` bytes at `offset` in `fd`. Gives false when the file is shorter.
#[cfg(target_os = "linux")]
unsafe fn read_at(fd: c_int, offset: u64, buffer: &mut [u8]) -> bool {
    let Ok(offset) = libc::off_t::try_from(offset) else {
        return false;
    };
    if libc::lseek(fd, offset, libc::SEEK_SET) != offset {
        return false;
    }
    let mut done = 0;
    while done < buffer.len() {
        let n = libc::read(
            fd,
            buffer[done..].as_mut_ptr() as *mut c_void,
            buffer.len() - done,
        );
        if n <= 0 {
            return false;
        }
        done += n as usize;
    }
    true
}

/// Finds the function of each frame in the symbol table (`.symtab`) of the ELF file at `path`. Vx
/// functions are hidden, so only this table names them, and `backtrace_symbols_fd` does not read
/// it. `targets` are the addresses in the file. `names` gets the offset of each name in the file.
#[cfg(target_os = "linux")]
unsafe fn find_functions(path: &CStr, targets: &[Option<usize>], names: &mut [Option<u64>]) {
    let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
    if fd >= 0 {
        read_symbol_table(fd, targets, names);
        libc::close(fd);
    }
}

#[cfg(target_os = "linux")]
unsafe fn read_symbol_table(fd: c_int, targets: &[Option<usize>], names: &mut [Option<u64>]) {
    use libc::{Elf64_Ehdr, Elf64_Shdr, Elf64_Sym};
    unsafe fn read_header<T: Copy>(fd: c_int, offset: u64) -> Option<T> {
        let mut value: T = std::mem::zeroed();
        let size = std::mem::size_of::<T>();
        let bytes = std::slice::from_raw_parts_mut(&mut value as *mut T as *mut u8, size);
        read_at(fd, offset, bytes).then_some(value)
    }
    const SHT_SYMTAB: u32 = 2;
    const STT_FUNC: u8 = 2;
    const SHN_UNDEF: u16 = 0;
    const SECTION_SIZE: u64 = std::mem::size_of::<Elf64_Shdr>() as u64;
    const SYMBOL_SIZE: usize = std::mem::size_of::<Elf64_Sym>();
    const SYMBOLS_PER_READ: usize = 170;

    let Some(header) = read_header::<Elf64_Ehdr>(fd, 0) else {
        return;
    };
    // A 64-bit ELF file with the byte order of this machine.
    let byte_order = if cfg!(target_endian = "little") { 1 } else { 2 };
    if header.e_ident[..6] != [0x7f, b'E', b'L', b'F', 2, byte_order]
        || u64::from(header.e_shentsize) != SECTION_SIZE
    {
        return;
    }
    let section = |index: u64| {
        let offset = index
            .checked_mul(SECTION_SIZE)
            .and_then(|o| o.checked_add(header.e_shoff))?;
        read_header::<Elf64_Shdr>(fd, offset)
    };
    let symbols = (0..u64::from(header.e_shnum))
        .filter_map(section)
        .find(|s| s.sh_type == SHT_SYMTAB && s.sh_entsize == SYMBOL_SIZE as u64);
    let Some(symbols) = symbols else {
        return;
    };
    const SHT_STRTAB: u32 = 3;
    let Some(strings) = section(u64::from(symbols.sh_link)).filter(|s| s.sh_type == SHT_STRTAB)
    else {
        return;
    };
    let count = symbols.sh_size / SYMBOL_SIZE as u64;
    let mut buffer = [0u8; SYMBOLS_PER_READ * SYMBOL_SIZE];
    let mut done = 0;
    while done < count {
        let n = (count - done).min(SYMBOLS_PER_READ as u64) as usize;
        let bytes = &mut buffer[..n * SYMBOL_SIZE];
        let Some(offset) = symbols.sh_offset.checked_add(done * SYMBOL_SIZE as u64) else {
            return;
        };
        if !read_at(fd, offset, bytes) {
            return;
        }
        for entry in bytes.as_chunks::<SYMBOL_SIZE>().0 {
            let symbol = ptr::read_unaligned(entry.as_ptr() as *const Elf64_Sym);
            if symbol.st_info & 0xf != STT_FUNC
                || symbol.st_shndx == SHN_UNDEF
                || symbol.st_name == 0
                || u64::from(symbol.st_name) >= strings.sh_size
            {
                continue;
            }
            let start = symbol.st_value;
            let end = start.saturating_add(symbol.st_size);
            for (target, name) in targets.iter().zip(names.iter_mut()) {
                if let Some(target) = *target {
                    if start <= target as u64 && (target as u64) < end {
                        *name = Some(strings.sh_offset.saturating_add(symbol.st_name.into()));
                    }
                }
            }
        }
        done += n as u64;
    }
}

/// The handler reads ELF files only on Linux.
#[cfg(not(target_os = "linux"))]
unsafe fn find_functions(_path: &CStr, _targets: &[Option<usize>], _names: &mut [Option<u64>]) {}

/// Writes the name at `offset` in the file at `path`, and a space after it. Writes nothing when it
/// cannot read the name.
unsafe fn write_function_name(out: &mut Output, path: &CStr, offset: u64) {
    let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
    if fd < 0 {
        return;
    }
    // A name ends with a 0 byte. The buffer cuts a longer name.
    let mut buffer = [0u8; 256];
    let mut len = 0;
    let found = libc::off_t::try_from(offset)
        .is_ok_and(|offset| libc::lseek(fd, offset, libc::SEEK_SET) == offset);
    while found && len < buffer.len() {
        let n = libc::read(
            fd,
            buffer[len..].as_mut_ptr() as *mut c_void,
            buffer.len() - len,
        );
        if n <= 0 {
            break;
        }
        len += n as usize;
    }
    libc::close(fd);
    let name = &buffer[..len];
    let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(len)];
    if !name.is_empty() {
        out.bytes(name);
        out.bytes(b" ");
    }
}

#[no_mangle]
pub extern "C" fn vx_sigsegv_handler(
    sig: c_int,
    _info: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    // A wild memory access can surface as either SIGSEGV (unmapped page) or SIGBUS (e.g. a far
    // out-of-range address on arm64). The alarm ends the program if the report hangs.
    let message: &[u8] = if sig == libc::SIGBUS {
        b"Caught SIGBUS: Bus Error!\nBacktrace:\n"
    } else {
        b"Caught SIGSEGV: Segmentation Fault!\nBacktrace:\n"
    };
    unsafe {
        // One report at a time. The first one ends the program.
        if REPORTING.swap(true, Ordering::SeqCst) {
            loop {
                libc::pause();
            }
        }
        libc::signal(libc::SIGALRM, libc::SIG_DFL);
        libc::alarm(CRASH_REPORT_SECONDS);
        // A symbolizer that stopped early must not end the program with SIGPIPE.
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        libc::write(2, message.as_ptr() as *const c_void, message.len());

        // The unwinder reads the code at the address of each frame, so it faults on an address
        // outside every image.
        let pc = faulting_pc(context);
        if pc != 0 && !in_an_image(pc) {
            let mut out = Output::new(2);
            out.bytes(b"#0 0x");
            out.number(pc, 16, 16);
            out.bytes(b" outside every loaded image\n");
            out.flush();
            libc::abort();
        }
        let mut raw = [ptr::null_mut(); CRASH_FRAMES];
        let count = libc::backtrace(raw.as_mut_ptr(), CRASH_FRAMES as c_int).max(0) as usize;
        // The frames of the handler and of the signal trampoline come before the code that
        // faulted. Its address is exact, and the other addresses are return addresses.
        let first = raw[..count].iter().position(|&f| f as usize == pc);
        let raw = &raw[first.unwrap_or(0)..count];
        let mut frames = [0usize; CRASH_FRAMES];
        for (i, &f) in raw.iter().enumerate() {
            frames[i] = f as usize;
        }
        let frames = &frames[..raw.len()];

        // The image of each frame. A return address points after the call, so the frame is at
        // the address before it.
        let images = PREPARED
            .get()
            .map_or(&[][..], |prepared| &prepared.images[..]);
        let mut targets = [0usize; CRASH_FRAMES];
        let mut image_of = [None; CRASH_FRAMES];
        for (i, &address) in frames.iter().enumerate() {
            targets[i] = if i == 0 && first.is_some() {
                address
            } else {
                address.wrapping_sub(1)
            };
            image_of[i] = images
                .iter()
                .position(|image| image.start <= targets[i] && targets[i] < image.end);
        }

        // The frames that the symbolizer can name.
        let mut requests = [None; CRASH_FRAMES];
        let mut symbolizer = None;
        if let Some(path) = PREPARED
            .get()
            .and_then(|prepared| prepared.symbolizer.as_ref())
        {
            for (request, image) in requests.iter_mut().zip(image_of) {
                *request = image.filter(|&image| !images[image].quoted_path.is_empty());
            }
            if requests.iter().any(Option::is_some) {
                symbolizer = start_symbolizer(
                    path,
                    images,
                    &targets[..frames.len()],
                    &requests[..frames.len()],
                );
            }
        }

        // The name of each frame in the symbol table of its image. Only a frame that the
        // symbolizer does not name prints it.
        let mut names = [None; CRASH_FRAMES];
        for (index, image) in images.iter().enumerate() {
            let Some(path) = &image.path else {
                continue;
            };
            let mut in_file = [None; CRASH_FRAMES];
            for i in 0..frames.len() {
                if image_of[i] == Some(index) {
                    in_file[i] = Some(targets[i].wrapping_sub(image.bias));
                }
            }
            if in_file.iter().any(Option::is_some) {
                find_functions(path, &in_file[..frames.len()], &mut names[..frames.len()]);
            }
        }

        let mut input = symbolizer.map(|(_, fd)| Input {
            fd,
            buffer: [0; 512],
            pos: 0,
            len: 0,
        });
        let mut out = Output::new(2);
        for (i, &address) in frames.iter().enumerate() {
            out.bytes(b"#");
            out.number(i, 10, 1);
            out.bytes(b" 0x");
            out.number(address, 16, 16);
            out.bytes(b" in ");
            let named = requests[i].is_some()
                && input
                    .as_mut()
                    .is_some_and(|input| input.copy_frame(&mut out));
            if !named {
                // `backtrace_symbols_fd` gives the file and the offset in it, which a symbolizer
                // can read later.
                let path = image_of[i].and_then(|image| images[image].path.as_ref());
                if let (Some(path), Some(offset)) = (path, names[i]) {
                    write_function_name(&mut out, path, offset);
                }
            }
            out.flush();
            if !named {
                libc::backtrace_symbols_fd(&raw[i], 1, 2);
            }
        }
        if let Some((pid, fd)) = symbolizer {
            libc::close(fd);
            libc::waitpid(pid, ptr::null_mut(), 0);
        }
        libc::abort();
    }
}
