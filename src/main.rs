//===- main.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file is the main executable entry point for the Vx compiler CLI.
// It parses command-line arguments using clap and orchestrates compilation
// via the driver module.
//
//===----------------------------------------------------------------------===//

use clap::Parser;
use std::process;
use vxc::driver::{CompilerDriver, DriverOptions};

fn main() {
    let options = DriverOptions::parse();
    let driver = CompilerDriver::new(options);

    if let Err(e) = driver.execute() {
        eprintln!("{}", e);
        process::exit(1);
    }
}
