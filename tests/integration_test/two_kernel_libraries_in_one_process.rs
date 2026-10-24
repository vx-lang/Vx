//===- two_kernel_libraries_in_one_process.rs - Vx Compiler -----*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A C program loads two libraries that `vxc --action emit-obj` built, and calls a
// function from each. Each function fills a tensor with a different number. Both
// libraries name their kernel `vx_npu_kernel_0`, so each call must run the kernel
// of its own library.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::Command;

const FILL: &str = "
pub fn NAME(out: &mut Tensor<f32, [?], Memory::HBM>, n: i32EXTRA) -> void {
  spawn on(Topology::Device) {
    for b in 0..(n + 255) / 256 across blocks {
      for t in 0..256 across lanes {
        if b * 256 + t < n {
          out[b * 256 + t] = VALUE;
        }
      }
    }
  }
  return;
}
";

const LOADER: &str = r#"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
typedef void (*fill_one_fn)(float *, float *, int64_t, int64_t, int64_t, int32_t);
typedef void (*fill_two_fn)(float *, float *, int64_t, int64_t, int64_t, int32_t,
                            int32_t);
int main(int argc, char **argv) {
  void *one = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void *two = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!one || !two) {
    printf("dlopen: %s\n", dlerror());
    return 1;
  }
  float x[4] = {0}, y[4] = {0};
  ((fill_one_fn)dlsym(one, "fill_one"))(x, x, 0, 4, 1, 4);
  ((fill_two_fn)dlsym(two, "fill_two"))(y, y, 0, 4, 1, 4, 7);
  printf("fill_one: %g %g %g %g\n", x[0], x[1], x[2], x[3]);
  printf("fill_two: %g %g %g %g\n", y[0], y[1], y[2], y[3]);
  return 0;
}
"#;

fn run(cmd: &mut Command) -> String {
    let out = cmd.output().expect("the command did not start");
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Builds `lib<name>.so` from a function that fills its tensor with `value`.
fn library(dir: &Path, name: &str, value: &str, extra: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = dir.join(format!("{name}.vx"));
    let object = dir.join(format!("{name}.o"));
    let library = dir.join(format!("lib{name}.so"));
    let text = FILL
        .replace("NAME", name)
        .replace("VALUE", value)
        .replace("EXTRA", extra);
    std::fs::write(&source, text).unwrap();
    run(Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg("--machine")
        .arg(root.join("fleet/a100-80.vx"))
        .arg(&source)
        .args(["--action", "emit-obj", "-o"])
        .arg(&object));
    let dispatch = Path::new(env!("NPU_SHARED_LIB_PATH"));
    run(Command::new("cc")
        .arg("-shared")
        .arg(&object)
        .arg(dispatch)
        .arg(format!(
            "-Wl,-rpath,{}",
            dispatch.parent().unwrap().display()
        ))
        .arg("-o")
        .arg(&library));
    library.to_string_lossy().into_owned()
}

#[test]
fn each_library_runs_its_own_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let one = library(dir.path(), "fill_one", "1.0", "");
    let two = library(dir.path(), "fill_two", "2.0", ", k: i32");
    let loader_source = dir.path().join("loader.c");
    let loader = dir.path().join("loader");
    std::fs::write(&loader_source, LOADER).unwrap();
    run(Command::new("cc")
        .arg(&loader_source)
        .arg("-o")
        .arg(&loader)
        .arg("-ldl"));
    let output = run(Command::new(&loader).args([&one, &two]));
    assert!(output.contains("fill_one: 1 1 1 1"), "{output}");
    assert!(output.contains("fill_two: 2 2 2 2"), "{output}");
}
