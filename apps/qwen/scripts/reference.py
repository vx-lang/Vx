#!/usr/bin/env python3
"""Compute the logits and the greedy tokens of a Qwen2 model with NumPy, in float32.

    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tokenizer DIR --prompt "def f(n):"
    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tensors

It reads the weight file that `convert.py` writes, so the Vx program and this script use the same
bytes. Tests of the Vx program compare with the lines it prints:

    step=0 token=715 logit=16.3279 margin=0.1044;

Step 0 is the token after the prompt. Each later step feeds the previous token back in.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys

import numpy as np

import qwen_weights

# Below this size, the widened float32 copies of all the weights stay in memory between tokens.
CACHE_LIMIT_BYTES = 4 << 30


class Model:
    def __init__(self, path):
        self.config, self.tensors = qwen_weights.read_weight_file(path)
        f32_bytes = sum(array.size * 4 for _, array in self.tensors.values())
        self.cache = {} if f32_bytes <= CACHE_LIMIT_BYTES else None

    def weight(self, name):
        if self.cache is None:
            return qwen_weights.as_f32(self.tensors[name])
        if name not in self.cache:
            self.cache[name] = qwen_weights.as_f32(self.tensors[name])
        return self.cache[name]


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope(x, positions, theta):
    """Rotate each head of x, shaped [tokens, heads, head_dim], by its position.

    The first half of a head pairs with the second half, as in Hugging Face's `rotate_half`.
    """
    half = x.shape[-1] // 2
    inv_freq = 1.0 / theta ** (np.arange(half, dtype=np.float64) * 2 / x.shape[-1])
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    first, second = x[..., :half], x[..., half:]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)


class Cache:
    """The keys and values of every position so far, per layer."""

    def __init__(self, config):
        self.keys = [None] * config["num_layers"]
        self.values = [None] * config["num_layers"]
        self.length = 0


def forward(model, tokens, cache):
    """Run `tokens` after the `cache.length` tokens already in the cache. Return their logits."""
    c = model.config
    hd, qh, kvh = c["head_dim"], c["num_q_heads"], c["num_kv_heads"]
    eps = c["rms_norm_eps"]
    count = len(tokens)
    positions = np.arange(cache.length, cache.length + count)
    x = model.weight("embed")[np.asarray(tokens)]
    for layer in range(c["num_layers"]):
        p = f"layers.{layer}."
        h = rms_norm(x, model.weight(p + "attn_norm"), eps)
        qkv = h @ model.weight(p + "qkv").T
        if c["qkv_bias"]:
            qkv = qkv + model.weight(p + "qkv_bias")
        q = rope(qkv[:, : qh * hd].reshape(count, qh, hd), positions, c["rope_theta"])
        k = rope(qkv[:, qh * hd : (qh + kvh) * hd].reshape(count, kvh, hd), positions, c["rope_theta"])
        v = qkv[:, (qh + kvh) * hd :].reshape(count, kvh, hd)
        if cache.keys[layer] is None:
            cache.keys[layer], cache.values[layer] = k, v
        else:
            cache.keys[layer] = np.concatenate([cache.keys[layer], k])
            cache.values[layer] = np.concatenate([cache.values[layer], v])
        keys, values = cache.keys[layer], cache.values[layer]
        # Query head i uses key-value head i // group: grouped-query attention.
        group = qh // kvh
        scores = np.einsum("tgd,sgd->gts", q, np.repeat(keys, group, axis=1)) / np.float32(np.sqrt(hd))
        visible = np.arange(keys.shape[0])[None, :] <= positions[:, None]
        scores = np.where(visible[None, :, :], scores, -np.inf)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores = scores / scores.sum(axis=-1, keepdims=True)
        attended = np.einsum("gts,sgd->tgd", scores, np.repeat(values, group, axis=1)).reshape(count, qh * hd)
        x = x + attended @ model.weight(p + "attn_out").T
        h = rms_norm(x, model.weight(p + "mlp_norm"), eps)
        gate_up = h @ model.weight(p + "gate_up").T
        gate, up = gate_up[:, : c["intermediate_size"]], gate_up[:, c["intermediate_size"] :]
        x = x + (gate / (1 + np.exp(-gate)) * up) @ model.weight(p + "down").T
    cache.length += count
    x = rms_norm(x, model.weight("final_norm"), eps)
    output = model.weight("embed" if c["tied_embeddings"] else "lm_head")
    return (x @ output.T).astype(np.float32)


def generate(model, prompt, steps):
    """Greedy decoding. Return a list of (token, logits) for each step."""
    cache = Cache(model.config)
    logits = forward(model, prompt, cache)[-1]
    result = []
    for _ in range(steps):
        token = int(np.argmax(logits))
        result.append((token, logits))
        logits = forward(model, [token], cache)[-1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a weight file from convert.py or make_tiny_model.py")
    parser.add_argument("--tokens", help="the prompt as token ids, separated by commas")
    parser.add_argument("--prompt", help="the prompt as text; needs --tokenizer")
    parser.add_argument("--tokenizer", help="a Hugging Face model folder that has tokenizer.json")
    parser.add_argument("--steps", type=int, default=8, help="the number of tokens to generate")
    parser.add_argument("--tensors", action="store_true", help="print the shape and first values of each tensor")
    args = parser.parse_args()
    model = Model(args.weights)

    if args.tensors:
        for name, entry in model.tensors.items():
            first = ", ".join(f"{v:.6g}" for v in qwen_weights.as_f32(entry).reshape(-1)[:4])
            print(f"{name} {entry[0]} {list(entry[1].shape)} [{first}]")
        return 0

    if args.prompt is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
        prompt = tokenizer.encode(args.prompt).ids
    else:
        assert args.tokens, "give the prompt with --tokens or --prompt"
        prompt = [int(t) for t in args.tokens.split(",")]
    print("prompt=" + ",".join(map(str, prompt)) + ";")
    for step, (token, logits) in enumerate(generate(model, prompt, args.steps)):
        # The margin is how far the chosen logit is above the next one. A small margin means that
        # a small rounding difference in another program can choose another token.
        margin = logits[token] - np.partition(logits, -2)[-2]
        print(f"step={step} token={token} logit={logits[token]:.4f} margin={margin:.4f};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
