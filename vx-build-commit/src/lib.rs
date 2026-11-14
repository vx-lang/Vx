//===- lib.rs - Vx Build Commit --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! `commit_suffix!()` is what `vxc --version` prints after the package version: ` (f1fb0333)`
//! for a build from a git checkout of Vx with `VX_VERSION_WITH_COMMIT=1`, and an empty string
//! otherwise.
//!
//! It is a macro, not a constant, so that `concat!` can join it to the version at compile time.
//! `build.rs` writes it.

include!(concat!(env!("OUT_DIR"), "/commit_suffix.rs"));
