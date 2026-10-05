#!/usr/bin/env python3
"""Differential fuzzing for Vx: run the same program several ways and compare what it prints.

Each program runs once per *configuration* -- a set of `vxc` flags, by default the flat code
generator (`flat`) and `--legacy-codegen` (`ast`) -- and, when its generator writes one, as a
Rust twin compiled with `rustc`. Any difference is a bug in one of them, or in the generator.

    fuzz.py list                                   # the generators
    fuzz.py run shadowing --seeds 1-1000           # generate and compare, 1000 programs
    fuzz.py run tensors --seeds 1-300 --config o0=-O0 --keep /tmp/failures
    fuzz.py show shadowing 52 [--rust]             # print one program
    fuzz.py reduce tensors 1133                    # shrink while the same runs still disagree
    fuzz.py reduce tensors 1133 --message "double free" --repeat 5
    fuzz.py compare tests/backend/pass/*.vx        # existing programs across configurations
    fuzz.py compare bench.vx --ignore '[0-9.]+e?-?[0-9]* s'   # with a timing masked
    fuzz.py run owners --seeds 1-1000 --heap --config drops=VX_DROPS=scope

`--config NAME=FLAGS` adds a configuration and may be repeated; giving any replaces the
defaults, so `--config flat= --config o0=-O0` compares the default build with `-O0`. Words like
`VAR=value` before the flags set the environment. `--heap` counts the heap blocks each compiled
program allocates and frees, and reports a program that leaves more than an empty one does. `--vxc`
picks the compiler (default `$CARGO_TARGET_DIR/debug/vxc`, else `target/debug/vxc`). Run with
the LLVM tools and `rustc` on PATH: `source config.local` first.

A new generator is a module in `generators/` with a docstring and `generate(seed)`
returning a `fuzzlib.Program`; see the README.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import atexit
import collections
import multiprocessing
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fuzzlib  # noqa: E402


def default_vxc():
    target = os.environ.get("CARGO_TARGET_DIR", "target")
    return os.path.join(target, "debug", "vxc")


def seed_range(text):
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def check(job):
    """Run one seed every way. Returns (seed, names that disagree, outputs)."""
    generator, seed, vxc, configs, timeout, use_rust, heap = job
    program = fuzzlib.load_generator(generator).generate(seed)
    rust = program.source("rs") if use_rust else None
    with tempfile.TemporaryDirectory() as work:
        outputs = fuzzlib.run_all(program.source("vx"), rust, vxc, configs, timeout, work,
                                  heap=heap)
    return seed, fuzzlib.disagreement(outputs), outputs


def report(results, keep, generator):
    failures = [(seed, names, outputs) for seed, names, outputs in results if names]
    print(f"{len(results) - len(failures)} of {len(results)} agree")
    groups = collections.defaultdict(list)
    for seed, names, outputs in failures:
        groups[fuzzlib.symptom(outputs, names)].append(seed)
    for label, seeds in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        shown = " ".join(str(s) for s in seeds[:8]) + (" ..." if len(seeds) > 8 else "")
        print(f"  {len(seeds):5}  {label}\n         seeds: {shown}")
    if keep and failures:
        os.makedirs(keep, exist_ok=True)
        for seed, names, outputs in failures:
            program = fuzzlib.load_generator(generator).generate(seed) if generator else None
            base = os.path.join(keep, f"{generator or 'file'}_{seed}")
            if program:
                with open(base + ".vx", "w") as f:
                    f.write(program.source("vx"))
                if program.source("rs") is not None:
                    with open(base + ".rs", "w") as f:
                        f.write(program.source("rs"))
            with open(base + ".txt", "w") as f:
                for name, out in outputs.items():
                    f.write(f"== {name}\n{out}\n")
        print(f"kept in {keep}")
    return 1 if failures else 0


def cmd_run(args, configs):
    jobs = [(args.generator, seed, args.vxc, configs, args.timeout, not args.no_rust, args.heap)
            for seed in seed_range(args.seeds)]
    with multiprocessing.Pool(args.jobs) as pool:
        results = sorted(pool.imap_unordered(check, jobs), key=lambda r: r[0])
    return report(results, args.keep, args.generator)


def cmd_show(args, configs):
    program = fuzzlib.load_generator(args.generator).generate(args.seed)
    print(program.source("rs" if args.rust else "vx"))
    return 0


def cmd_reduce(args, configs):
    program = fuzzlib.load_generator(args.generator).generate(args.seed)
    use_rust = not args.no_rust and program.source("rs") is not None

    def outputs_of(p):
        return fuzzlib.run_all(p.source("vx"), p.source("rs") if use_rust else None,
                               args.vxc, configs, args.timeout, heap=args.heap)

    if args.message:
        def fails_once(p):
            return any(args.message in out for out in outputs_of(p).values())
    else:
        target = fuzzlib.disagreement(outputs_of(program))
        if not target:
            print("seed agrees everywhere; nothing to reduce", file=sys.stderr)
            return 1
        print(f"reducing while {', '.join(target)} still disagree", file=sys.stderr)

        def fails_once(p):
            return fuzzlib.disagreement(outputs_of(p)) == target

    def still_fails(p):
        return any(fails_once(p) for _ in range(args.repeat))

    if not still_fails(program):
        print("the seed does not fail as asked", file=sys.stderr)
        return 1
    fuzzlib.reduce(program, still_fails)
    print(program.source("vx"))
    if use_rust and args.rust:
        print(program.source("rs"))
    return 0


def check_file(job):
    path, vxc, configs, timeout, ignore, heap = job
    with open(path) as f:
        source = f.read()
    with tempfile.TemporaryDirectory() as work:
        outputs = fuzzlib.run_all(source, None, vxc, configs, timeout, work, ignore, heap)
    return path, fuzzlib.disagreement(outputs), outputs


def cmd_compare(args, configs):
    if len(configs) < 2 and not args.heap:
        print("compare needs two configurations or more, or --heap", file=sys.stderr)
        return 2
    jobs = [(path, args.vxc, configs, args.timeout, args.ignore, args.heap)
            for path in args.files]
    with multiprocessing.Pool(args.jobs) as pool:
        results = sorted(pool.imap_unordered(check_file, jobs))
    differ = [(p, names, outputs) for p, names, outputs in results if names]
    print(f"{len(results) - len(differ)} of {len(results)} agree")
    for path, names, outputs in differ:
        first = (outputs[names[0]].splitlines() or [""])[0][:90]
        marked = "  (REQUIRES: flat-codegen)" if "REQUIRES: flat-codegen" in open(path).read() else ""
        print(f"  {path}: {', '.join(names)} differ{marked}\n      {names[0]}: {first}")
    return 1 if differ else 0


def main():
    cpus = os.cpu_count() or 4
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--vxc", default=default_vxc())
    common.add_argument("--config", action="append", metavar="NAME=FLAGS",
                        help="a configuration to run; repeat for more (default: flat and ast)")
    common.add_argument("--timeout", type=int, default=60, help="seconds per run")
    common.add_argument("--jobs", type=int, default=max(1, cpus - 2))
    common.add_argument("--heap", action="store_true",
                        help="also report a program that does not free what it allocates")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="the generators", parents=[common])

    run = sub.add_parser("run", help="generate programs and compare their runs", parents=[common])
    run.add_argument("generator")
    run.add_argument("--seeds", default="1-100", help="a seed or a range, e.g. 1-1000")
    run.add_argument("--no-rust", action="store_true", help="compare configurations only")
    run.add_argument("--keep", metavar="DIR", help="write failing programs and outputs here")

    show = sub.add_parser("show", help="print one generated program", parents=[common])
    show.add_argument("generator")
    show.add_argument("seed", type=int)
    show.add_argument("--rust", action="store_true", help="print the Rust twin instead")

    red = sub.add_parser("reduce", help="shrink a failing program", parents=[common])
    red.add_argument("generator")
    red.add_argument("seed", type=int)
    red.add_argument("--message", help="keep deleting while some run prints this text "
                                       "(default: while the same runs disagree)")
    red.add_argument("--repeat", type=int, default=1,
                     help="runs per candidate, for a failure that does not happen every time")
    red.add_argument("--no-rust", action="store_true")
    red.add_argument("--rust", action="store_true", help="also print the reduced Rust twin")

    cmp_ = sub.add_parser("compare", help="run existing .vx files under every configuration",
                          parents=[common])
    cmp_.add_argument("files", nargs="+")
    cmp_.add_argument("--ignore", action="append", default=[], metavar="REGEX",
                      help="replace what this matches before comparing, e.g. a timing")

    args = parser.parse_args()
    configs = [fuzzlib.parse_config(c) for c in args.config] if args.config else fuzzlib.DEFAULT_CONFIGS
    if args.heap:
        work = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, work, True)
        library = fuzzlib.build_heap_counter(work)
        args.heap = (library, fuzzlib.empty_program_blocks(args.vxc, configs, library,
                                                           args.timeout))
    else:
        args.heap = None
    if args.command == "list":
        for name in fuzzlib.generator_names():
            first = fuzzlib.load_generator(name).__doc__.strip().splitlines()[0]
            print(f"{name:12} {first}")
        return 0
    return {"run": cmd_run, "show": cmd_show, "reduce": cmd_reduce,
            "compare": cmd_compare}[args.command](args, configs)


if __name__ == "__main__":
    sys.exit(main())
