#!/usr/bin/env python3
"""Compute the logits and the greedy tokens of a Qwen2 or Qwen3 MoE model with NumPy, in float32.

    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tokenizer DIR --prompt "def f(n):"
    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tensors
    python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tokenizer DIR --perplexity FILE

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
        f32_bytes = sum(array.size * (8 if dtype == "u4" else 4) for dtype, array in self.tensors.values())
        self.cache = {} if f32_bytes <= CACHE_LIMIT_BYTES else None

    def weight(self, name, rows=None):
        """The float32 values of a tensor, or of the rows (start, stop) of a matrix."""
        if self.cache is None:
            return qwen_weights.widen(self.tensors, name, rows)
        if name not in self.cache:
            self.cache[name] = qwen_weights.widen(self.tensors, name)
        return self.cache[name] if rows is None else self.cache[name][rows[0] : rows[1]]


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


def silu(x):
    return x / (1 + np.exp(-x))


def mixture_of_experts(model, prefix, h):
    """The sum of the experts that the router chooses for each token of h, each times its weight.

    The router keeps the experts_per_token largest of the softmax over all experts, and scales
    their weights to sum to 1.
    """
    c = model.config
    hidden, inter = c["hidden_size"], c["moe_intermediate_size"]
    scores = h @ model.weight(prefix + "router").T
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores = scores / scores.sum(axis=-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1, kind="stable")[:, : c["experts_per_token"]]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = np.zeros_like(h)
    for expert in np.unique(chosen):
        tokens, slots = np.nonzero(chosen == expert)
        gate_up = h[tokens] @ model.weight(prefix + "experts_gate_up", (expert * 2 * inter, (expert + 1) * 2 * inter)).T
        down = model.weight(prefix + "experts_down", (expert * hidden, (expert + 1) * hidden))
        out[tokens] += weights[tokens, slots][:, None] * ((silu(gate_up[:, :inter]) * gate_up[:, inter:]) @ down.T)
    return out


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
        q = qkv[:, : qh * hd].reshape(count, qh, hd)
        k = qkv[:, qh * hd : (qh + kvh) * hd].reshape(count, kvh, hd)
        v = qkv[:, (qh + kvh) * hd :].reshape(count, kvh, hd)
        if c["qk_norm"]:
            # One RMSNorm over each head, with one weight vector for all the heads of q or of k.
            q = rms_norm(q, model.weight(p + "q_norm"), eps)
            k = rms_norm(k, model.weight(p + "k_norm"), eps)
        q = rope(q, positions, c["rope_theta"])
        k = rope(k, positions, c["rope_theta"])
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
        if c["num_experts"]:
            x = x + mixture_of_experts(model, p, h)
        else:
            gate_up = h @ model.weight(p + "gate_up").T
            gate, up = gate_up[:, : c["intermediate_size"]], gate_up[:, c["intermediate_size"] :]
            x = x + (silu(gate) * up) @ model.weight(p + "down").T
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


def perplexity(model, tokens):
    """exp of the mean negative log-likelihood of each token after the first, given the ones
    before it."""
    logits = forward(model, tokens, Cache(model.config)).astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    chosen = log_probs[np.arange(len(tokens) - 1), np.asarray(tokens[1:])]
    return float(np.exp(-chosen.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a weight file from convert.py or make_tiny_model.py")
    parser.add_argument("--tokens", help="the prompt as token ids, separated by commas")
    parser.add_argument("--prompt", help="the prompt as text; needs --tokenizer")
    parser.add_argument("--tokenizer", help="a Hugging Face model folder that has tokenizer.json")
    parser.add_argument("--steps", type=int, default=8, help="the number of tokens to generate")
    parser.add_argument("--tensors", action="store_true", help="print the first codes and the sum of the codes of each tensor")
    parser.add_argument("--perplexity", help="a text file whose perplexity to print; needs --tokenizer")
    parser.add_argument("--max-tokens", type=int, default=512, help="the number of tokens of --perplexity to use")
    args = parser.parse_args()
    model = Model(args.weights)

    if args.tensors:
        # Bits, not decimals, so that a Vx program can print the same line exactly. A code is the
        # bits of one value as an unsigned integer: 16 for bf16 and f16, 8 for f8e4m3 and 32 for
        # f32. For u4, it is a byte of two codes. The sum adds the codes of the whole tensor.
        c = model.config
        print(f"config hidden={c['hidden_size']} layers={c['num_layers']} vocab={c['vocab_size']};")
        for name, (dtype, array) in model.tensors.items():
            codes = array.reshape(-1).view(f"u{array.dtype.itemsize}")
            first = ",".join(str(int(v)) for v in codes[:4])
            print(f"{name} count={codes.size} codes={first} sum={int(codes.sum(dtype=np.uint64))};")
        return 0

    if args.perplexity is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
        with open(args.perplexity) as text:
            tokens = tokenizer.encode(text.read()).ids[: args.max_tokens]
        print(f"perplexity={perplexity(model, tokens):.4f} tokens={len(tokens)};")
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
