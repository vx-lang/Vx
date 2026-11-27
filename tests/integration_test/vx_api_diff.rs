//===- vx_api_diff.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `vx-api diff` compares the interfaces of two versions of a library and reports the smallest
// safe version bump. Each test builds the two versions in two directories, so their GIDs differ
// and the tool must match items by name.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

/// Version 1 of the library. Each test changes one thing in it.
const LIBRARY: &str = "\
struct Point {
  x : i32,
  y : i32,
}

enum Color {
  Red,
  Green,
}

impl Point {
  pub fn sum(self : &Point) -> i32 {
    return self.x + self.y;
  }
}

trait Measure {
  fn size(self : &Self) -> i32;
}

impl Measure for Point {
  fn size(self : &Point) -> i32 {
    return self.x;
  }
}

pub fn area(w : i32, h : i32) -> i32 {
  return w * h;
}

pub fn pick(a : &i32, b : &i32) -> &i32 {
  return a;
}

fn half(n : i32) -> i32 {
  return n / 2;
}
";

/// Version 1 with `from` replaced by `to`.
fn changed(from: &str, to: &str) -> String {
    assert!(LIBRARY.contains(from), "the library has no `{from}`");
    LIBRARY.replacen(from, to, 1)
}

/// Build `source` in `dir` and return the path of its `.vxlib`.
fn interface(dir: &Path, source: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let (lib, vxlib) = (dir.join("shapes.vx"), dir.join("shapes.vxlib"));
    std::fs::write(&lib, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&lib)
        .args(["--emit-interface", "-o"])
        .arg(&vxlib)
        .output()
        .expect("failed to run vxc");
    assert!(
        out.status.success(),
        "the library did not build:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    vxlib
}

/// What `vx-api diff` prints for version `old` followed by version `new`.
fn api_diff(test: &str, old: &str, new: &str) -> String {
    let base = std::env::temp_dir().join(format!("vx_api_diff_{test}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (old, new) = (
        interface(&base.join("old"), old),
        interface(&base.join("new"), new),
    );
    let out = Command::new(env!("CARGO_BIN_EXE_vx-api"))
        .arg("diff")
        .args([&old, &new])
        .output()
        .expect("failed to run vx-api");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "vx-api failed:\n{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// Check the reported bump, and that `line` is one of the reported changes.
fn assert_reports(output: &str, bump: &str, line: &str) {
    assert!(
        output.lines().any(|l| l == format!("bump: {bump}")),
        "expected bump: {bump}\n{output}"
    );
    assert!(
        output.lines().any(|l| l == line),
        "expected the line `{line}`\n{output}"
    );
}

#[test]
fn the_same_library_built_in_two_directories_needs_a_patch() {
    let output = api_diff("same", LIBRARY, LIBRARY);
    assert_reports(&output, "patch", "no change to the interface");
}

#[test]
fn a_change_inside_a_function_body_needs_a_patch() {
    let output = api_diff("body", LIBRARY, &changed("return w * h;", "return h * w;"));
    assert_reports(&output, "patch", "no change to the interface");
}

#[test]
fn an_added_function_needs_a_minor_release() {
    let new = format!("{LIBRARY}\npub fn double(n : i32) -> i32 {{\n  return n + n;\n}}\n");
    let output = api_diff("fn_added", LIBRARY, &new);
    assert_reports(&output, "minor", "minor  function `double` was added");
}

#[test]
fn a_removed_function_needs_a_major_release() {
    let new = changed(
        "pub fn area(w : i32, h : i32) -> i32 {\n  return w * h;\n}\n",
        "",
    );
    let output = api_diff("fn_removed", LIBRARY, &new);
    assert_reports(&output, "major", "major  function `area` was removed");
}

#[test]
fn a_function_that_loses_pub_needs_a_major_release() {
    let new = changed("pub fn area(", "fn area(");
    let output = api_diff("loses_pub", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  function `area` is no longer `pub`",
    );
}

#[test]
fn a_method_that_loses_pub_needs_a_major_release() {
    let new = changed("pub fn sum(", "fn sum(");
    let output = api_diff("method_loses_pub", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  method `Point::sum` is no longer `pub`",
    );
}

#[test]
fn a_function_that_becomes_pub_needs_a_minor_release() {
    let new = changed("fn half(", "pub fn half(");
    let output = api_diff("becomes_pub", LIBRARY, &new);
    assert_reports(&output, "minor", "minor  function `half` became `pub`");
}

// A program cannot call a function that is not `pub` through a `.vxlib`. So a change to a
// private function, a new one, and a removed one are no change to the interface.

#[test]
fn a_private_function_with_new_types_needs_a_patch() {
    let new = changed(
        "fn half(n : i32) -> i32 {\n  return n / 2;",
        "fn half(n : i64) -> i64 {\n  return n / 2;",
    );
    let output = api_diff("private_changed", LIBRARY, &new);
    assert_reports(&output, "patch", "no change to the interface");
}

#[test]
fn an_added_private_function_needs_a_patch() {
    let new = format!("{LIBRARY}\nfn third(n : i32) -> i32 {{\n  return n / 3;\n}}\n");
    let output = api_diff("private_added", LIBRARY, &new);
    assert_reports(&output, "patch", "no change to the interface");
}

#[test]
fn a_removed_private_function_needs_a_patch() {
    let new = changed("fn half(n : i32) -> i32 {\n  return n / 2;\n}\n", "");
    let output = api_diff("private_removed", LIBRARY, &new);
    assert_reports(&output, "patch", "no change to the interface");
}

#[test]
fn new_parameter_types_need_a_major_release() {
    let new = changed(
        "fn area(w : i32, h : i32) -> i32",
        "fn area(w : i64, h : i64) -> i64",
    );
    let output = api_diff("params", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  function `area` changed from `fn(i32, i32) -> i32` to `fn(i64, i64) -> i64`",
    );
}

#[test]
fn a_new_return_type_needs_a_major_release() {
    let new = changed(
        "fn area(w : i32, h : i32) -> i32 {\n  return w * h;",
        "fn area(w : i32, h : i32) -> i64 {\n  return (w * h) as i64;",
    );
    let output = api_diff("ret", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  function `area` changed from `fn(i32, i32) -> i32` to `fn(i32, i32) -> i64`",
    );
}

#[test]
fn a_function_that_becomes_unsafe_needs_a_major_release() {
    let new = changed("fn area(", "unsafe fn area(");
    let output = api_diff("becomes_unsafe", LIBRARY, &new);
    assert_reports(&output, "major", "major  function `area` became unsafe");
}

#[test]
fn a_function_that_stops_being_unsafe_needs_a_minor_release() {
    let old = changed("fn area(", "unsafe fn area(");
    let output = api_diff("stops_unsafe", &old, LIBRARY);
    assert_reports(
        &output,
        "minor",
        "minor  function `area` is no longer unsafe",
    );
}

#[test]
fn a_reference_returned_from_another_parameter_needs_a_major_release() {
    let new = changed("  return a;", "  return b;");
    let output = api_diff("ret_prov", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  the reference that function `pick` returns comes from a different parameter",
    );
}

#[test]
fn a_new_struct_field_needs_a_major_release() {
    let new = changed("  y : i32,\n", "  y : i32,\n  z : i32,\n");
    let output = api_diff("field_added", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  struct `Point` has a new field `z`, which code that builds a `Point` must now give",
    );
}

#[test]
fn a_removed_struct_field_needs_a_major_release() {
    let new = changed("  y : i32,\n", "").replacen("self.x + self.y", "self.x", 1);
    let output = api_diff("field_removed", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  field `y` of struct `Point` was removed",
    );
}

#[test]
fn a_new_field_type_needs_a_major_release() {
    let new = changed("  y : i32,\n", "  y : i64,\n").replacen("self.x + self.y", "self.x", 1);
    let output = api_diff("field_type", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  field `y` of struct `Point` changed from `i32` to `i64`",
    );
}

#[test]
fn a_new_enum_variant_needs_a_major_release() {
    let new = changed("  Green,\n", "  Green,\n  Blue,\n");
    let output = api_diff("variant_added", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  enum `Color` has a new variant `Blue`, which a `match` on `Color` must now cover",
    );
}

#[test]
fn a_removed_enum_variant_needs_a_major_release() {
    let new = changed("  Green,\n", "");
    let output = api_diff("variant_removed", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  variant `Color::Green` was removed",
    );
}

#[test]
fn enum_variants_in_a_new_order_need_a_major_release() {
    let new = changed("  Red,\n  Green,\n", "  Green,\n  Red,\n");
    let output = api_diff("variant_order", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  the variants of enum `Color` changed their order",
    );
}

#[test]
fn an_added_struct_and_enum_need_a_minor_release() {
    let new =
        format!("{LIBRARY}\nstruct Size {{\n  w : i32,\n}}\n\nenum Shape {{\n  Square,\n}}\n");
    let output = api_diff("types_added", LIBRARY, &new);
    assert_reports(&output, "minor", "minor  struct `Size` was added");
    assert_reports(&output, "minor", "minor  enum `Shape` was added");
}

#[test]
fn a_method_with_a_new_return_type_needs_a_major_release() {
    let new = changed(
        "fn sum(self : &Point) -> i32 {\n    return self.x + self.y;",
        "fn sum(self : &Point) -> i64 {\n    return (self.x + self.y) as i64;",
    );
    let output = api_diff("method", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  method `Point::sum` changed from `fn(&Point) -> i32` to `fn(&Point) -> i64`",
    );
}

/// A method of a trait impl has the level of the trait, and each trait is public, so the method
/// is API with no `pub`.
#[test]
fn a_trait_method_with_a_new_return_type_needs_a_major_release() {
    let new = changed(
        "  fn size(self : &Self) -> i32;",
        "  fn size(self : &Self) -> i64;",
    )
    .replacen(
        "fn size(self : &Point) -> i32 {\n    return self.x;",
        "fn size(self : &Point) -> i64 {\n    return self.x as i64;",
        1,
    );
    let output = api_diff("trait_method", LIBRARY, &new);
    assert_reports(
        &output,
        "major",
        "major  method `Point::size` changed from `fn(&Point) -> i32` to `fn(&Point) -> i64`",
    );
}

#[test]
fn a_file_that_is_not_an_interface_is_refused() {
    let file = std::env::temp_dir().join(format!("vx_api_diff_bad_{}.vxlib", std::process::id()));
    std::fs::write(&file, "abc").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vx-api"))
        .arg("diff")
        .args([&file, &file])
        .output()
        .expect("failed to run vx-api");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a three-byte file was read as an interface"
    );
    assert!(
        stderr.contains("is not a .vxlib file"),
        "unexpected error: {stderr}"
    );
}
