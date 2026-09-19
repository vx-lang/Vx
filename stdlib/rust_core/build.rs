//===- build.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

fn main() {
    // A macOS dylib records the path it expects to be found at, and every executable linked
    // against it copies that string verbatim. The default is the absolute path of the machine
    // that built it, so a library built on a release runner sends the user's binary looking
    // under /Users/runner/work/... on their own laptop, where dyld fails.
    //
    // `@rpath` defers the decision to whoever links against the library: they pass `-rpath`
    // naming the directory the library actually sits in. `rustc-cdylib-link-arg` applies to
    // the cdylib alone, so the staticlib and the test binaries are untouched.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libvx_std_core.dylib");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
