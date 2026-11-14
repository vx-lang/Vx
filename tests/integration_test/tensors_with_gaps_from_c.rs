//===- tensors_with_gaps_from_c.rs - Vx Compiler -----------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A C program passes tensors to `pub fn`s that `vxc --action emit-obj` built. Vx reads
// every tensor as packed, so a tensor with gaps, such as rows with padding, must stop
// the program with a message instead of giving wrong values. A dimension of size 1 and
// an empty tensor can have any strides.
//
//===----------------------------------------------------------------------===//

use std::path::Path;
use std::process::{Command, Output};

const LIBRARY: &str = "
pub fn at(t : &Tensor<i32, [?, ?]>, y : i32, x : i32) -> i32 {
  return t[y][x];
}

pub fn fixed_at(t : &Tensor<i32, [2, 3]>, y : i32, x : i32) -> i32 {
  return t[y][x];
}

pub fn at_1d(t : &Tensor<i32, [?]>, i : i32) -> i32 {
  return t[i];
}

pub fn seven(t : &Tensor<i32, [?, ?]>) -> i32 {
  return 7;
}
";

const PROGRAM: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
// A tensor of rank 2 is 7 arguments: allocated, aligned, offset, 2 sizes, 2 strides.
#define T2(p, r, c, s0, s1) (p), (p), 0, (int64_t)(r), (int64_t)(c), (int64_t)(s0), (int64_t)(s1)
int32_t at(int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t);
int32_t fixed_at(int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t,
                 int32_t);
int32_t at_1d(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t);
int32_t seven(int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t, int64_t);
int main(int argc, char **argv) {
  int32_t packed[6] = {0, 1, 2, 10, 11, 12};
  int32_t padded[8] = {0, 1, 2, -1, 10, 11, 12, -1};
  const char *c = argv[1];
  if (!strcmp(c, "packed"))
    printf("packed: %d\n", at(T2(packed, 2, 3, 3, 1), 1, 0));
  else if (!strcmp(c, "padded_rows"))
    printf("padded_rows: %d\n", at(T2(padded, 2, 3, 4, 1), 1, 0));
  else if (!strcmp(c, "padded_rows_fixed_shape"))
    printf("padded_rows_fixed_shape: %d\n", fixed_at(T2(padded, 2, 3, 4, 1), 1, 0));
  else if (!strcmp(c, "every_other_element"))
    printf("every_other_element: %d\n", at_1d(packed, packed, 0, 3, 2, 1));
  else if (!strcmp(c, "one_row"))
    printf("one_row: %d\n", at(T2(packed, 1, 3, 99, 1), 0, 2));
  else if (!strcmp(c, "no_rows"))
    printf("no_rows: %d\n", seven(T2(packed, 0, 3, 5, 2)));
  return 0;
}
"#;

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("the command did not start");
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Builds the C program with the Vx library in `dir`.
fn program(dir: &Path) -> std::path::PathBuf {
    let source = dir.join("library.vx");
    let object = dir.join("library.o");
    let main = dir.join("main.c");
    let binary = dir.join("main");
    std::fs::write(&source, LIBRARY).unwrap();
    std::fs::write(&main, PROGRAM).unwrap();
    run(Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&source)
        .args(["--action", "emit-obj", "-o"])
        .arg(&object));
    run(Command::new("cc")
        .arg(&main)
        .arg(&object)
        .arg("-o")
        .arg(&binary));
    binary
}

fn call(binary: &Path, case: &str) -> Output {
    Command::new(binary)
        .arg(case)
        .output()
        .expect("the C program did not start")
}

#[test]
fn a_tensor_from_c_is_read_or_refused_by_its_strides() {
    let dir = tempfile::tempdir().unwrap();
    let binary = program(dir.path());

    // Packed, and two layouts whose odd strides are never used.
    for (case, expected) in [
        ("packed", "packed: 10"),
        ("one_row", "one_row: 2"),
        ("no_rows", "no_rows: 7"),
    ] {
        let out = call(&binary, case);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains(expected),
            "{case}: expected `{expected}`, got:\n{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // A gap after each row, also in a tensor whose type fixes its shape, and a gap after
    // each element.
    for (case, function) in [
        ("padded_rows", "at"),
        ("padded_rows_fixed_shape", "fixed_at"),
        ("every_other_element", "at_1d"),
    ] {
        let out = call(&binary, case);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let message = format!("parameter 1 of `{function}` is a tensor with gaps");
        assert!(
            !out.status.success() && stdout.is_empty() && stderr.contains(&message),
            "{case}: expected the program to stop with `{message}`, got:\n{stdout}{stderr}"
        );
    }
}
