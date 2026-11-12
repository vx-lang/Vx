//===- apple_gpu_functions_called_from_two_c_threads.rs - Vx Compiler -*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two C threads call functions from an object that `vxc --action emit-obj` built for the
// Apple GPU. Thread A calls `fill_two`, which fills a tensor on the GPU and then waits on a pipe.
// Thread B runs a function and then writes to the pipe, so `fill_two` goes on to fill a second
// tensor.
//
// - B calls `fill_one` on other memory. Its return waits for its own kernel, and not for the
//   kernel of A. So both kernels of A run after one wait.
// - B calls `hand_over` with the tensor that A's first kernel writes. B then reads the tensor in
//   C, so the return of `hand_over` waits for that kernel.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

const KERNELS: &str = "
extern \"C\" {
  fn read(fd : i32, buf : *mut u8, n : u64) -> i64;
  fn write(fd : i32, buf : *const u8, n : u64) -> i64;
}

// dst = position * scale
fn fill(dst : &mut Tensor<f32, [?], Memory::HBM>, n : i32, scale : f32) -> void {
  spawn on(Topology::Device) {
    for g in 0..(n + 255) / 256 across blocks {
      for l in 0..256 across lanes(256) {
        let i = g * 256 + l;
        if i < n {
          dst[i] = (i as f32) * scale;
        }
      }
    }
  }
  return;
}

// Thread B waits for a byte on the pipe `go`, and sends one back on the pipe `done`.
pub fn fill_two(a : &mut Tensor<f32, [?], Memory::HBM>, b : &mut Tensor<f32, [?], Memory::HBM>, n : i32, go : i32, done : i32) -> void {
  fill(a, n, 1.0);
  let mut byte : u8 = 1;
  unsafe {
    let _sent = write(go, (&byte) as *const u8, 1);
    let _received = read(done, (&mut byte) as *mut u8, 1);
  }
  fill(b, n, 2.0);
  return;
}

pub fn fill_one(c : &mut Tensor<f32, [?], Memory::HBM>, n : i32) -> void {
  fill(c, n, 3.0);
  return;
}

pub fn hand_over(t : &Tensor<f32, [?], Memory::HBM>) -> void {
  return;
}
";

// A rank-1 tensor is 5 C arguments: the allocated pointer, the aligned pointer, the offset, the
// size and the stride.
const HOST: &str = r#"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define T(p) p, p, 0, N, 1
#define TENSOR float *, float *, int64_t, int64_t, int64_t

void fill_two(TENSOR, TENSOR, int32_t, int32_t, int32_t);
void fill_one(TENSOR, int32_t);
void hand_over(TENSOR);

enum { N = 65536 };

static float *tensor(void) {
  float *t = NULL;
  if (posix_memalign((void **)&t, 16384, N * sizeof(float)))
    abort();
  for (int i = 0; i < N; i++)
    t[i] = -7.0f;
  return t;
}

static int go[2], done[2];
static void (*b_runs)(void);

static void *thread_b(void *unused) {
  (void)unused;
  char byte;
  if (read(go[0], &byte, 1) != 1)
    abort();
  b_runs();
  if (write(done[1], &byte, 1) != 1)
    abort();
  return NULL;
}

static float *a, *b, *c;
static float seen;

static void b_fills_c(void) { fill_one(T(c), N); }

static void b_reads_a(void) {
  hand_over(T(a));
  seen = a[N - 1];
}

static void run(const char *name, void (*part)(void)) {
  fprintf(stderr, "%s\n", name);
  a = tensor();
  b = tensor();
  c = tensor();
  b_runs = part;
  if (pipe(go) || pipe(done))
    abort();
  pthread_t other;
  pthread_create(&other, NULL, thread_b, NULL);
  fill_two(T(a), T(b), N, go[1], done[0]);
  pthread_join(other, NULL);
}

int main(void) {
  run("part 1", b_fills_c);
  printf("part 1: %g %g %g\n", a[N - 1], b[N - 1], c[N - 1]);
  run("part 2", b_reads_a);
  printf("part 2: B saw %g\n", seen);
  return 0;
}
"#;

fn run(cmd: &mut Command) -> (String, String) {
    let out = cmd.output().expect("the command did not start");
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

// The `[Vx Metal]` lines of a part of the log about launches and waits, with each kernel name and
// time left out.
fn metal_lines(log: &str, part: &str) -> Vec<String> {
    let after = log.split(part).nth(1).unwrap_or("");
    let section = after.split("\npart ").next().unwrap_or("");
    section
        .lines()
        .filter_map(|l| {
            let l = l.strip_prefix("[Vx Metal] ")?;
            if l.starts_with("made a buffer") || l.starts_with("freed ") {
                return None;
            }
            Some(if l.contains(": grid ") {
                "grid".to_string()
            } else {
                l.split(',').next().unwrap().to_string()
            })
        })
        .collect()
}

#[test]
fn a_function_waits_for_its_own_kernels_and_for_kernels_that_write_its_tensors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("kernels.vx");
    let object = dir.path().join("kernels.o");
    let host_source = dir.path().join("host.c");
    let host = dir.path().join("host");
    std::fs::write(&source, KERNELS).unwrap();
    std::fs::write(&host_source, HOST).unwrap();
    run(Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(root)
        .args(["--host", "default", "--machine"])
        .arg(root.join("fleet/m4-uma.vx"))
        .arg(&source)
        .args(["--action", "emit-obj", "-o"])
        .arg(&object));
    let dispatch = Path::new(env!("NPU_SHARED_LIB_PATH"));
    run(Command::new("cc")
        .arg(&host_source)
        .arg(&object)
        .arg(dispatch)
        .arg(format!(
            "-Wl,-rpath,{}",
            dispatch.parent().unwrap().display()
        ))
        .arg("-o")
        .arg(&host));
    let (output, log) = run(Command::new(&host).env("VX_DISPATCH_VERBOSE", "1"));

    assert!(output.contains("part 1: 65535 131070 196605"), "{output}");
    // The first kernel of A wrote position * 1 before B read it.
    assert!(output.contains("part 2: B saw 65535"), "{output}");
    // A fills `a`, B fills `c` and waits for it, and then A fills `b` and waits for both of its
    // kernels.
    assert_eq!(
        metal_lines(&log, "part 1\n"),
        [
            "grid",
            "grid",
            "waited for 1 kernel",
            "grid",
            "waited for 2 kernels"
        ],
        "{log}"
    );
    // The return of `hand_over` on B waits for the first kernel of A.
    assert_eq!(
        metal_lines(&log, "part 2\n"),
        ["grid", "waited for 1 kernel", "grid", "waited for 1 kernel"],
        "{log}"
    );
}
