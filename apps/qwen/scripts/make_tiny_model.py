#!/usr/bin/env python3
"""Write a tiny Qwen2 model with random weights, for tests that cannot download a real model.

    python3 apps/qwen/scripts/make_tiny_model.py tiny.vxw
    python3 apps/qwen/scripts/make_tiny_model.py --fp8 tiny-fp8.vxw

The model has the layers of Qwen2.5-Coder at a small size: hidden size 128, 2 layers, 2 query
heads that share 1 key-value head of 64 values, and a vocabulary of 256 tokens. Its weights come
from a xorshift generator with a fixed seed, so the file has the same bytes on every machine and
with every NumPy version.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys

import numpy as np

import qwen_weights

CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_layers": 2,
    "num_q_heads": 2,
    "num_kv_heads": 1,
    "head_dim": 64,
    "vocab_size": 256,
    "max_positions": 64,
    "tied_embeddings": 1,
    "qkv_bias": 1,
    "eos_token_id": 255,
    "fp8": 0,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
}
SEED = 0x2545F4914F6CDD1D


class Xorshift:
    def __init__(self, seed):
        self.state = seed

    def uniform(self, count, low, high):
        """`count` float32 values, spread evenly between `low` and `high`."""
        values = np.empty(count, dtype=np.float64)
        mask = (1 << 64) - 1
        for i in range(count):
            x = self.state
            x ^= (x << 13) & mask
            x ^= x >> 7
            x ^= (x << 17) & mask
            self.state = x
            values[i] = (x >> 11) / float(1 << 53)
        return (low + (high - low) * values).astype(np.float32)


def tiny_model():
    rng = Xorshift(SEED)
    shapes = qwen_weights.tensor_shapes(CONFIG)
    tensors = {}
    for name in qwen_weights.tensor_names(CONFIG):
        shape = shapes[name]
        count = int(np.prod(shape))
        if name.endswith("norm"):
            values = rng.uniform(count, 0.9, 1.1)
        else:
            # About 1 / sqrt(hidden size), so the values in each layer keep a similar size.
            values = rng.uniform(count, -0.15, 0.15)
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values).reshape(shape))
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", help="the weight file to write")
    parser.add_argument("--fp8", action="store_true", help="store the weight matrices as f8e4m3 codes")
    args = parser.parse_args()
    config, tensors = CONFIG, tiny_model()
    if args.fp8:
        config, tensors = qwen_weights.to_fp8(config, tensors)
    qwen_weights.write_weight_file(args.output, config, tensors)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
