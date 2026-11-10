#!/usr/bin/env python3
"""Compare the output of the Vx program with the output of reference.py.

    python3 apps/qwen/scripts/compare.py REFERENCE_OUTPUT VX_OUTPUT [--tolerance 0.01]

Both outputs have a `prompt=...;` line and one `step=K token=T logit=L ...;` line per step. The
prompts and the tokens must be equal. The logits must be within the tolerance: the GPU adds in
another order than NumPy, so they are close, not equal. The script exits with 1 when a check
fails.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import re
import sys

STEP = re.compile(r"^step=(\d+) token=(\d+) logit=([-+0-9.eE]+)")


def read(path):
    prompt, steps = None, []
    with open(path) as lines:
        for line in lines:
            if line.startswith("prompt="):
                prompt = line.strip()
            match = STEP.match(line)
            if match:
                steps.append((int(match.group(2)), float(match.group(3))))
    return prompt, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", help="the output of reference.py")
    parser.add_argument("vx", help="the output of the Vx program")
    parser.add_argument("--tolerance", type=float, default=0.01, help="the largest allowed logit difference")
    args = parser.parse_args()
    ref_prompt, ref_steps = read(args.reference)
    vx_prompt, vx_steps = read(args.vx)
    if ref_prompt is None or ref_prompt != vx_prompt:
        print(f"the prompts differ: {ref_prompt} and {vx_prompt}")
        return 1
    if not ref_steps or len(ref_steps) != len(vx_steps):
        print(f"the reference has {len(ref_steps)} steps and the Vx program {len(vx_steps)}")
        return 1
    worst = 0.0
    for step, ((ref_token, ref_logit), (vx_token, vx_logit)) in enumerate(zip(ref_steps, vx_steps)):
        if ref_token != vx_token:
            print(f"step {step}: the reference chose {ref_token}, the Vx program {vx_token}")
            return 1
        worst = max(worst, abs(ref_logit - vx_logit))
    within = "within" if worst <= args.tolerance else "above"
    print(f"{len(ref_steps)} steps, same tokens, logits {within} {args.tolerance}")
    print(f"largest logit difference {worst:.2e}")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
