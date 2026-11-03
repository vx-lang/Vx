# Testing shadow stacks with QEMU

## What a shadow stack is

A function call puts its return address on the stack. A bug that writes past the end of a
buffer can change that address, and the `ret` then jumps where the attacker wants. This is
how return-oriented programming (ROP) works.

A shadow stack is a second stack that holds only return addresses. Normal stores cannot write
to it. A call puts the return address on both stacks, and a return checks that the two copies
are the same. If they are not, the CPU stops the program.

- On x86-64 this is Intel CET shadow stacks (Linux 6.6 and later).
- On arm64 this is the Guarded Control Stack, GCS (Linux 6.13 and later).

## Why Vx must care

The shared task runtime (`stdlib/rust_core/src/ffi/stack.rs`) switches stacks by itself. A
task runs on a stack from a pool, and a switch moves the stack pointer to another task. The
CPU does not know about this switch. If shadow stacks are on and the switch does not also move
the shadow-stack pointer, the first `ret` after a switch checks the wrong shadow stack, and the
program stops with a fault that names no cause.

So each task stack needs a shadow stack of its own, and the switch must move both. The design
says that the first stack switch must have this, because it costs the most to add later
(section 7 of `docs/discussions/waiting_tasks_channels_and_task_errors.md`). Until the runtime
does it, `StackPool::new` stops the program when shadow stacks are on.

## Why QEMU

Few machines can run shadow stacks today:

- CET needs an Intel CPU from Tiger Lake on, or an AMD CPU from Zen 3 on. Many cloud VMs do
  not pass the feature to the guest.
- GCS needs an Armv9.4 CPU. Almost no hardware has it yet.

QEMU 10.1 and later emulates GCS in user mode (`qemu-aarch64`). Its default CPU in user mode is
`max`, which has GCS. So an x86-64 machine can run arm64 tests with GCS on. QEMU does not
emulate CET, so the x86-64 version cannot be tested this way.

Ubuntu 24.04 has QEMU 8.2, which has no GCS. So the CI job "Rust core tests on aarch64 (qemu)"
builds QEMU 10.2.1 and caches it.

## Check that your QEMU has GCS

```sh
qemu-aarch64 --version    # 10.1 or later
```

This C program turns GCS on, makes a call, and then returns from `main`:

```c
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#ifndef PR_GET_SHADOW_STACK_STATUS
#define PR_GET_SHADOW_STACK_STATUS 74
#define PR_SET_SHADOW_STACK_STATUS 75
#define PR_SHADOW_STACK_ENABLE (1UL << 0)
#endif

__attribute__((noinline)) static int leaf(int x) { return x + 1; }

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  unsigned long status = 0;
  // A system call from inline assembly: a libc wrapper would return through a frame
  // that is not on the shadow stack.
  register long x8 __asm__("x8") = SYS_prctl;
  register long x0 __asm__("x0") = PR_SET_SHADOW_STACK_STATUS;
  register long x1 __asm__("x1") = PR_SHADOW_STACK_ENABLE;
  // prctl refuses a call whose unused arguments are not 0.
  register long x2 __asm__("x2") = 0, x3 __asm__("x3") = 0, x4 __asm__("x4") = 0;
  __asm__ volatile("svc #0"
                   : "+r"(x0)
                   : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4)
                   : "memory");
  prctl(PR_GET_SHADOW_STACK_STATUS, &status, 0, 0, 0);
  printf("set %ld status %lu leaf %d\n", x0, status, leaf(1));
  return 0;
}
```

```sh
aarch64-linux-gnu-gcc -O1 -static gcs.c -o gcs
qemu-aarch64 ./gcs
```

The program prints `set 0 status 1 leaf 2`, and then stops with a segmentation fault. The
fault is the expected result. `main` started before GCS was on, so its return address is
not on the shadow stack, and the return from `main` fails the check. This shows that QEMU
really checks each return. A QEMU older than 10.1 has no GCS. There the `set` should fail, the
status should stay 0, and the program should exit normally. This doc did not test an older QEMU.

## How the switch moves a shadow stack

- Each stack of the pool gets a shadow stack from `map_shadow_stack`, with a token in its top
  word. A stack that goes back to the pool gives its shadow stack back to the kernel. It still
  holds the return addresses of the last task, so the next task gets a new one.
- The switch saves one more word in the frame of the code that stops: the address of a token on
  its shadow stack. `gcsss1` moves to the shadow stack of the code that continues, and `gcsss2`
  puts a token on the old one.
- `chkfeat` says whether shadow stacks are on. A CPU with no shadow stacks runs it as a no-op,
  so the same switch works on every arm64 CPU.
- A new task did not reach its first function by a call, so its shadow stack holds no return
  address for it. A bit in the saved word marks a new frame, and the switch branches to it with
  `br`, not `ret`.
- x86-64 still stops the program when shadow stacks are on. That switch needs CET hardware to
  test.

## Run the runtime tests with GCS

```sh
rustup target add aarch64-unknown-linux-gnu
sudo apt-get install qemu-user gcc-aarch64-linux-gnu    # QEMU 10.1 or later
cd stdlib/rust_core
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
  cargo test --target aarch64-unknown-linux-gnu --lib shadow -- --nocapture
```

Two tests run:

- `tasks_switch_with_shadow_stacks_on` (`stack.rs`): a task switches with the test 100 times.
  Then its stack goes back to the pool, and a second task does the same on it.
- `tasks_park_on_two_workers_with_shadow_stacks_on` (`sched.rs`): two tasks pass a turn back
  and forth on two worker threads, so each one parks and continues many times.

Each test turns GCS on in a child process from `fork`. A return to a frame from before GCS was
on fails, as the C program showed. So the child turns GCS on with a system call from inline
assembly, and it ends with `_exit`, which does not return. The parent reads the exit status of
the child.

When the CPU has no GCS, each test prints "this CPU or kernel has no shadow stacks, so the test
checked nothing", and passes. So check the output for this line. With
`VX_REQUIRE_SHADOW_STACKS=1`, as in CI, each test fails there.

To see a test fail, make the switch skip its shadow-stack part: in `vx_stack_switch`, change
`"tbnz x16, #0, 1f"` to `"b 1f"`. Each test then reports that the child stopped with signal 11.
