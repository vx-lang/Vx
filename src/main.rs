//===- main.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("Vx Compiler Internal Error: {}", panic_info);
        eprintln!("Please report this bug at: https://github.com/vx-lang/Vx/issues");
    }));

    let options = DriverOptions::parse();
    let driver = CompilerDriver::new(options);

    if let Err(e) = driver.execute() {
        eprintln!("{}", e);
        process::exit(1);
    }
}
