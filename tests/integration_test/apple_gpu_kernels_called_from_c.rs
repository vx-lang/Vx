//===- apple_gpu_kernels_called_from_c.rs - Vx Compiler ---------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A C program calls two functions from an object that `vxc --action emit-obj` built for the
// Apple GPU. It calls each one twice: on the GPU, and on the CPU after
// `vx_metal_set_enabled(0)`. Both calls must give the same bytes. The tensors are in memory
// that C allocated, at addresses that are not on a page boundary, and the memory around each
// tensor must not change.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

const KERNELS: &str = "
pub fn blur_rows(dst : &mut Tensor<f32, [?, ?], Memory::HBM>, src : &Tensor<f32, [?, ?], Memory::HBM>, h : i32, w : i32) -> void {
  spawn on(Topology::Device) {
    for y in 0..h across blocks {
      for x in 0..w across lanes(256) {
        let mut acc : f32 = 0.0;
        for d in 0..9 {
          let mut xx = x + d - 4;
          if xx < 0 {
            xx = 0;
          }
          if xx > w - 1 {
            xx = w - 1;
          }
          acc = acc + src[y][xx];
        }
        dst[y][x] = acc / 9.0;
      }
    }
  }
  return;
}

pub fn brighten(dst : &mut Tensor<u8, [?, ?], Memory::HBM>, src : &Tensor<u8, [?, ?], Memory::HBM>, h : i32, w : i32, gain : f32) -> void {
  spawn on(Topology::Device) {
    for y in 0..h across blocks {
      for x in 0..w across lanes(256) {
        for c in 0..4 {
          let mut v = src[y][x * 4 + c] as f32;
          if c < 3 {
            v = v * gain;
          }
          if v > 255.0 {
            v = 255.0;
          }
          dst[y][x * 4 + c] = v as u8;
        }
      }
    }
  }
  return;
}
";

// A rank-2 tensor is 7 C arguments: the allocated pointer, the aligned pointer, the offset,
// the two sizes and the two strides.
const HOST: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void blur_rows(float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
               float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
               int32_t, int32_t);
void brighten(uint8_t *, uint8_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
              uint8_t *, uint8_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
              int32_t, int32_t, float);
void vx_metal_set_enabled(int on);

enum { H = 37, W = 300, GUARD = 5, PITCH = 4 * W + 48 };

// Blurs a fresh copy of the image. The tensors start GUARD floats into their buffers.
static void blur(float *out, int gpu) {
  static float src[H * W + 2 * GUARD];
  for (int i = 0; i < H * W + 2 * GUARD; i++)
    src[i] = (float)((i * 7) % 256) / 255.0f;
  for (int i = 0; i < H * W + 2 * GUARD; i++)
    out[i] = -7.0f;
  vx_metal_set_enabled(gpu);
  blur_rows(out, out + GUARD, 0, H, W, W, 1, src, src + GUARD, 0, H, W, W, 1, H, W);
}

// Brightens a BGRA frame whose rows are PITCH bytes, in a buffer on a page boundary.
static void frame(uint8_t *out, int gpu) {
  static uint8_t *src;
  if (!src && posix_memalign((void **)&src, 16384, H * PITCH))
    return;
  for (int i = 0; i < H * PITCH; i++)
    src[i] = (uint8_t)((i * 5) % 256);
  memset(out, 9, H * PITCH + 2 * GUARD);
  vx_metal_set_enabled(gpu);
  brighten(out, out + GUARD, 0, H, PITCH, PITCH, 1, src, src, 0, H, PITCH, PITCH, 1, H,
           W, 1.5f);
}

int main(void) {
  static float on_gpu[H * W + 2 * GUARD], on_cpu[H * W + 2 * GUARD];
  blur(on_gpu, 1);
  blur(on_cpu, 0);
  int guards = 1;
  for (int i = 0; i < GUARD; i++)
    guards &= on_gpu[i] == -7.0f && on_gpu[GUARD + H * W + i] == -7.0f;
  printf("blur: %s, guards %s, sample %.9g\n",
         memcmp(on_gpu, on_cpu, sizeof on_gpu) ? "different" : "same bytes",
         guards ? "kept" : "changed", on_gpu[GUARD + 12 * W + 34]);

  static uint8_t bright_gpu[H * PITCH + 2 * GUARD], bright_cpu[H * PITCH + 2 * GUARD];
  frame(bright_gpu, 1);
  frame(bright_cpu, 0);
  int padding = 1;
  for (int y = 0; y < H; y++)
    for (int x = 4 * W; x < PITCH; x++)
      padding &= bright_gpu[GUARD + y * PITCH + x] == 9;
  printf("frame: %s, padding %s, pixel %d %d %d %d\n",
         memcmp(bright_gpu, bright_cpu, sizeof bright_gpu) ? "different" : "same bytes",
         padding ? "kept" : "changed", bright_gpu[GUARD + 3 * PITCH],
         bright_gpu[GUARD + 3 * PITCH + 1], bright_gpu[GUARD + 3 * PITCH + 2],
         bright_gpu[GUARD + 3 * PITCH + 3]);
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

#[test]
fn each_kernel_gives_the_same_bytes_on_the_gpu_and_the_cpu() {
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

    assert!(
        output.contains("blur: same bytes, guards kept, sample 0.505882382"),
        "{output}"
    );
    // The source pixel is 32 37 42 47. The colors grow by half, rounded down, and alpha stays.
    assert!(
        output.contains("frame: same bytes, padding kept, pixel 48 55 63 47"),
        "{output}"
    );
    // One call of each function ran on the GPU, and one ran on the CPU.
    assert_eq!(log.matches("[Vx Metal] vx_npu_kernel_").count(), 2, "{log}");
    assert_eq!(
        log.matches("Falling back to CPU libffi dispatch").count(),
        2,
        "{log}"
    );
}
