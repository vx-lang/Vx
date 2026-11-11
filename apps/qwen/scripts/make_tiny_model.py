#!/usr/bin/env python3
"""Write a tiny Qwen model with random weights, for tests that cannot download a real model.

    python3 apps/qwen/scripts/make_tiny_model.py tiny.vxw
    python3 apps/qwen/scripts/make_tiny_model.py --fp8 tiny-fp8.vxw
    python3 apps/qwen/scripts/make_tiny_model.py --moe --q4 tiny-moe-q4.vxw
    python3 apps/qwen/scripts/make_tiny_model.py --moe --hugging-face DIR tiny-moe.vxw

The model has the layers of Qwen2.5-Coder at a small size: hidden size 128, 2 layers, 2 query
heads that share 1 key-value head of 64 values, and a vocabulary of 256 tokens. Its weights come
from a xorshift generator with a fixed seed, so the file has the same bytes on every machine and
with every NumPy version.

With --moe, the model has the layers of Qwen3-Coder-30B-A3B: hidden size 256, 2 layers, 4 query
heads and 2 key-value heads of 64 values, and 8 experts of which each token uses 2. Its weights
come from splitmix64, which NumPy computes for all of them at once. They have Hugging Face names
and go through `convert.py`. --hugging-face also writes them as a Hugging Face folder, for
`check_against_mlx.py`.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

import convert
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
    "weight_format": 0,
    "num_experts": 0,
    "experts_per_token": 0,
    "moe_intermediate_size": 0,
    "qk_norm": 0,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
}

# The config.json of the tiny MoE model.
MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 256,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "eos_token_id": 255,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
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
            values = rng.uniform(count, 0.9, 1.1).reshape(shape)
        else:
            # About 1 / sqrt(hidden size), so the values in each layer keep a similar size.
            values = rng.uniform(count, -0.15, 0.15).reshape(shape)
        if len(shape) == 2:
            values = values * block_sizes(name, shape)
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values))
    return tensors


# Each kind of matrix, and each block of 128 x 128 in it, gets another size. Then the fp8 scales
# of the blocks differ, and a kernel that uses the scales of the wrong block or matrix gives other
# logits. With one size everywhere, all scales are almost equal and such a mistake is not seen.
KIND_SIZES = {"embed": 1.0, "qkv": 1.0, "attn_out": 0.7, "gate_up": 1.2, "down": 0.5}
BLOCK_SIZES = (1.0, 0.3, 1.8, 0.6)


def block_sizes(name, shape):
    """The factor of each weight of the matrix `name`, by its kind and its block of 128 x 128."""
    rows, cols = shape
    row_block = np.arange(rows)[:, None] // 128
    col_block = np.arange(cols)[None, :] // 128
    block = np.asarray(BLOCK_SIZES, dtype=np.float32)[(row_block * 3 + col_block) % len(BLOCK_SIZES)]
    return np.float32(KIND_SIZES[name.rsplit(".", 1)[-1]]) * block


def splitmix64_uniform(start, count, low, high):
    """Values number `start` to `start + count - 1` of the stream, spread evenly between `low`
    and `high`. Value i is splitmix64 of the seed plus i + 1 times its constant."""
    i = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(SEED) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (low + (high - low) * ((z >> np.uint64(11)).astype(np.float64) / float(1 << 53))).astype(np.float32)


# The size of each expert. With one size for all, a kernel that uses the scales of the wrong expert
# gives almost the same logits. The router is larger, so that its choice has a clear margin.
EXPERT_SIZES = (1.0, 0.4, 1.6, 0.7)
ROUTER_SIZE = 4.0


def tiny_moe_model():
    """The tensors of the tiny MoE model, as bf16 bits with their Hugging Face names."""
    c = MOE_CONFIG
    hidden, head_dim, inter = c["hidden_size"], c["head_dim"], c["moe_intermediate_size"]
    shapes = {"model.embed_tokens.weight": (c["vocab_size"], hidden)}
    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        shapes[p + "input_layernorm.weight"] = (hidden,)
        shapes[p + "self_attn.q_proj.weight"] = (c["num_attention_heads"] * head_dim, hidden)
        shapes[p + "self_attn.k_proj.weight"] = (c["num_key_value_heads"] * head_dim, hidden)
        shapes[p + "self_attn.v_proj.weight"] = (c["num_key_value_heads"] * head_dim, hidden)
        shapes[p + "self_attn.o_proj.weight"] = (hidden, c["num_attention_heads"] * head_dim)
        shapes[p + "self_attn.q_norm.weight"] = (head_dim,)
        shapes[p + "self_attn.k_norm.weight"] = (head_dim,)
        shapes[p + "post_attention_layernorm.weight"] = (hidden,)
        shapes[p + "mlp.gate.weight"] = (c["num_experts"], hidden)
        for e in range(c["num_experts"]):
            shapes[f"{p}mlp.experts.{e}.gate_proj.weight"] = (inter, hidden)
            shapes[f"{p}mlp.experts.{e}.up_proj.weight"] = (inter, hidden)
            shapes[f"{p}mlp.experts.{e}.down_proj.weight"] = (hidden, inter)
    shapes["model.norm.weight"] = (hidden,)
    shapes["lm_head.weight"] = (c["vocab_size"], hidden)
    tensors, start = {}, 0
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        if name.endswith("norm.weight"):
            values = splitmix64_uniform(start, count, 0.9, 1.1)
        else:
            values = splitmix64_uniform(start, count, -0.1, 0.1)
            if ".experts." in name:
                values *= np.float32(EXPERT_SIZES[int(name.split(".")[5]) % len(EXPERT_SIZES)])
            elif name.endswith("mlp.gate.weight"):
                values *= np.float32(ROUTER_SIZE)
        start += count
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values).reshape(shape))
    return tensors


def write_hugging_face_folder(directory, config, tensors):
    """Write config.json and model.safetensors, with bf16 tensors given as uint16 bits."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as out:
        json.dump(config, out, indent=2)
    header, offset = {}, 0
    for name, (dtype, array) in tensors.items():
        assert dtype == "bf16", f"{name}: {dtype}"
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(os.path.join(directory, "model.safetensors"), "wb") as out:
        out.write(struct.pack("<Q", len(text)) + text)
        for _, array in tensors.values():
            out.write(np.ascontiguousarray(array).data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", help="the weight file to write")
    weight_format = parser.add_mutually_exclusive_group()
    weight_format.add_argument("--fp8", action="store_true", help="store the weight matrices as f8e4m3 codes")
    weight_format.add_argument("--q4", action="store_true", help="store the weight matrices as Q4_0 codes")
    parser.add_argument("--moe", action="store_true", help="write the tiny mixture-of-experts model")
    parser.add_argument("--hugging-face", metavar="DIR", help="also write the MoE model as a Hugging Face folder")
    args = parser.parse_args()
    if args.moe:
        hf = tiny_moe_model()
        if args.hugging_face:
            write_hugging_face_folder(args.hugging_face, MOE_CONFIG, hf)
        config, tensors = convert.convert_tensors(qwen_weights.config_from_json(MOE_CONFIG), hf)
    else:
        assert not args.hugging_face, "--hugging-face needs --moe"
        config, tensors = CONFIG, tiny_model()
    if args.fp8 or args.q4:
        config, tensors = qwen_weights.quantize(config, tensors, "fp8" if args.fp8 else "q4_0")
    qwen_weights.write_weight_file(args.output, config, tensors)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
