#!/bin/bash
# Run one fuzz program on both code generators and in Rust, and compare the output.
#
#   run_one.sh GENERATOR SEED VXC OUTDIR
#
# GENERATOR is shadowing.py or tensors.py. Prints "SEED ok", "SEED rustc-failed", or
# "SEED MISMATCH rust=.. flat=.. ast=..", keeping the program in OUTDIR when it is not ok.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
here=$(dirname "$0")
gen=$1; seed=$2; vxc=$3; out=$4
python3 "$here/$gen" "$seed" "$out" || exit 0
rustc -O -o "$out/p$seed.bin" "$out/p$seed.rs" 2>"$out/p$seed.rserr" || { echo "$seed rustc-failed"; exit 0; }
r=$("$out/p$seed.bin")
fl=$(timeout 60 "$vxc" "$out/p$seed.vx" 2>&1 | grep -v '^\[\|^Warning' | tr -d '\n')
as=$(timeout 60 "$vxc" --legacy-codegen "$out/p$seed.vx" 2>&1 | grep -v '^\[\|^Warning' | tr -d '\n')
if [ "$r" == "$fl" ] && [ "$r" == "$as" ]; then
  echo "$seed ok"
  rm -f "$out/p$seed".*
else
  echo "$seed MISMATCH rust=$r flat=${fl:0:80} ast=${as:0:80}"
fi
