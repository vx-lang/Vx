//===- parallel.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one rayon pool that the corpus tests share. Each test runs vxc once per program, and runs
// the programs on this pool. On a computer that people use, the tests leave two cores free. A CI
// runner has nothing else to run, so there they use all its cores.
//
//===----------------------------------------------------------------------===//

use std::sync::Once;

static INIT_RAYON: Once = Once::new();

/// Makes the global rayon pool, if no test made it yet: with all cores on a CI runner (GitHub
/// Actions sets `CI`), and with all cores but two elsewhere.
pub fn init_rayon() {
    INIT_RAYON.call_once(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let threads = if std::env::var_os("CI").is_some() {
            cores
        } else if cores > 2 {
            cores - 2
        } else {
            1
        };
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    });
}
