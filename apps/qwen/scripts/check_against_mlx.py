#!/usr/bin/env python3
"""Check reference.py against MLX, an independent implementation of the same model.

    python3 apps/qwen/scripts/check_against_mlx.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw
    python3 apps/qwen/scripts/check_against_mlx.py --tokens 1,2,3 TINY_MOE_DIR tiny-moe.vxw

MLX reads the Hugging Face folder and runs in float32. reference.py reads the converted weight
file. So agreement also checks the converter. The script prints the largest difference of the
logits and fails when the greedy tokens differ. It needs the `mlx-lm` package. A folder with no
tokenizer, such as the one of `make_tiny_model.py --moe --hugging-face`, needs --tokens.

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
    parser.add_argument("--tokens", help="the prompt as token ids, for a folder with no tokenizer")
    args = parser.parse_args()

    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.utils import load_model

    # Keep the weights in bf16, and widen each one to float32 only inside its product, as
    # reference.py does. Widening every weight first would need 30 GB for the 7B model.
    def linear(self, x):
        y = x.astype(mx.float32) @ self["weight"].astype(mx.float32).T
        return y + self["bias"].astype(mx.float32) if "bias" in self else y

    # The same for the matrices of the experts, which MLX multiplies with gather_mm.
    def switch_linear(self, x, indices, sorted_indices=False):
        assert "bias" not in self, "the experts of Qwen3 have no bias"
        weight = self["weight"].astype(mx.float32).swapaxes(-1, -2)
        return mx.gather_mm(x.astype(mx.float32), weight, rhs_indices=indices, sorted_indices=sorted_indices)

    nn.Linear.__call__ = linear
    SwitchLinear.__call__ = switch_linear
    nn.Embedding.__call__ = lambda self, x: self["weight"][x].astype(mx.float32)
    nn.Embedding.as_linear = lambda self, x: x.astype(mx.float32) @ self["weight"].astype(mx.float32).T

    if args.tokens:
        mlx_model, tokenizer = load_model(Path(args.model_dir))[0], None
        prompt = [int(t) for t in args.tokens.split(",")]
    else:
        mlx_model, tokenizer = load(args.model_dir)
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
    if tokenizer is not None:
        print("text: " + repr(tokenizer.decode(tokens[len(prompt) :])))
    else:
        print("tokens: " + ",".join(map(str, tokens[len(prompt) :])))
    if worst > args.tolerance:
        print(f"the largest difference is above the tolerance {args.tolerance}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
