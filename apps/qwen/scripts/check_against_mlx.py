#!/usr/bin/env python3
"""Check reference.py against MLX, an independent implementation of the same model.

    python3 apps/qwen/scripts/check_against_mlx.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw

MLX reads the Hugging Face folder and runs in float32. reference.py reads the converted weight
file. So agreement also checks the converter. The script prints the largest difference of the
logits and fails when the greedy tokens differ. It needs the `mlx-lm` package.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys

import numpy as np

import reference

PROMPT = "def fibonacci(n):\n    "


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", help="the Hugging Face model folder")
    parser.add_argument("weights", help="the weight file that convert.py wrote from that folder")
    parser.add_argument("--steps", type=int, default=32, help="the number of greedy tokens to compare")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="the largest allowed logit difference")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm import load

    mlx_model, tokenizer = load(args.model_dir)
    mlx_model.update(tree_map(lambda p: p.astype(mx.float32), mlx_model.parameters()))
    prompt = tokenizer.encode(PROMPT)

    def mlx_logits(tokens):
        out = mlx_model(mx.array([tokens]))
        mx.eval(out)
        return np.array(out[0], dtype=np.float32)

    model = reference.Model(args.weights)
    worst = float(np.abs(mlx_logits(prompt) - reference.forward(model, prompt, reference.Cache(model.config))).max())
    print(f"prompt of {len(prompt)} tokens: largest logit difference {worst:.2e}")

    tokens = list(prompt)
    failed = False
    for step, (token, logits) in enumerate(reference.generate(model, prompt, args.steps)):
        expected = mlx_logits(tokens)[-1]
        difference = float(np.abs(expected - logits).max())
        worst = max(worst, difference)
        if int(np.argmax(expected)) != token:
            print(f"step {step}: reference.py chose {token}, MLX chose {int(np.argmax(expected))}")
            failed = True
            break
        tokens.append(token)
    print(f"{args.steps} greedy steps, largest logit difference {worst:.2e}")
    print("text: " + repr(tokenizer.decode(tokens[len(prompt) :])))
    if worst > args.tolerance:
        print(f"the largest difference is above the tolerance {args.tolerance}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
