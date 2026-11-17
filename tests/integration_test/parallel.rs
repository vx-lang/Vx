//===- parallel.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one rayon pool that the corpus tests share. Each test runs vxc once per program, and runs
// the programs on this pool, so the tests together use all cores but two.
//
//===----------------------------------------------------------------------===//

use std::sync::Once;

static INIT_RAYON: Once = Once::new();

/// Makes the global rayon pool, with all cores but two, if no test made it yet.
pub fn init_rayon() {
    INIT_RAYON.call_once(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let threads = if cores > 2 { cores - 2 } else { 1 };
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    });
}
