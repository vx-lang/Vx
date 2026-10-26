//===- vx-api.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `vx-api diff OLD.vxlib NEW.vxlib` prints what changed between the interfaces of two versions
// of a library, then the smallest safe version bump. It reads only the two files it names.
//
//===----------------------------------------------------------------------===//
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use vxc::api_diff::{diff, read_interface, NOT_COMPARED};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Compare the interfaces of two versions of a Vx library"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print what changed between two `.vxlib` files, and the smallest safe version bump.
    Diff {
        /// The interface of the old version.
        old: PathBuf,
        /// The interface of the new version.
        new: PathBuf,
    },
}

fn main() -> ExitCode {
    let Command::Diff { old, new } = Cli::parse().command;
    let interfaces = read_interface(&old).and_then(|o| Ok((o, read_interface(&new)?)));
    let (old, new) = match interfaces {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let diff = diff(&old, &new);
    if diff.changes.is_empty() {
        println!("no change to the interface");
    }
    for change in &diff.changes {
        println!("{:<5}  {}", change.bump, change.description);
    }
    println!("not compared: {NOT_COMPARED}");
    println!("bump: {}", diff.bump());
    ExitCode::SUCCESS
}
