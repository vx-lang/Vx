#!/usr/bin/env python3
"""Convert a Qwen2 or Qwen3 MoE model folder from Hugging Face into the weight file of the Qwen app.

    python3 apps/qwen/scripts/convert.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw
    python3 apps/qwen/scripts/convert.py --fp8 ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b-fp8.vxw
    python3 apps/qwen/scripts/convert.py --q4 ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b-q4.vxw
    python3 apps/qwen/scripts/convert.py --q4 --importance importance.npz ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b-q4.vxw

The file keeps the bf16 values exactly. Q, K and V of a layer become one matrix with one bias
vector, Q rows first, and gate and up become one matrix, gate rows first, so that one kernel
reads each of them. The experts of a layer become one matrix for gate and up, and one for down.
With --fp8, the weight matrices become f8e4m3 codes with a scale for each block of 128 x 128.
With --q4, they become Q4_0 codes with a scale for each 32 weights of a row, and lm_head becomes
f8e4m3. --importance takes the file of importance.py, and rounds each Q4_0 matrix to the smallest
error on the inputs that matter most. `qwen_weights.py` describes the file.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys

import numpy as np

import qwen_weights


def joined(parts):
    """Join (dtype name, array) entries along their first axis. They must have one dtype."""
    dtypes = {dtype for dtype, _ in parts}
    assert len(dtypes) == 1, f"cannot join the dtypes {dtypes}"
    return dtypes.pop(), np.concatenate([array for _, array in parts])


def convert(model_dir):
    return convert_tensors(qwen_weights.config_from_hugging_face(model_dir), qwen_weights.read_safetensors(model_dir))


def convert_tensors(config, hf):
    """The tensors of the weight file, from a dict of the tensors of a Hugging Face model. The
    matrices of the experts are functions, so that the file joins them when it writes them."""
    tensors = {"embed": hf["model.embed_tokens.weight"], "final_norm": hf["model.norm.weight"]}
    if not config["tied_embeddings"]:
        tensors["lm_head"] = hf["lm_head.weight"]
    for layer in range(config["num_layers"]):
        source = f"model.layers.{layer}."
        target = f"layers.{layer}."
        attn = source + "self_attn."
        tensors[target + "attn_norm"] = hf[source + "input_layernorm.weight"]
        tensors[target + "qkv"] = joined([hf[attn + p + "_proj.weight"] for p in "qkv"])
        if config["qkv_bias"]:
            tensors[target + "qkv_bias"] = joined([hf[attn + p + "_proj.bias"] for p in "qkv"])
        if config["qk_norm"]:
            tensors[target + "q_norm"] = hf[attn + "q_norm.weight"]
            tensors[target + "k_norm"] = hf[attn + "k_norm.weight"]
        tensors[target + "attn_out"] = hf[attn + "o_proj.weight"]
        tensors[target + "mlp_norm"] = hf[source + "post_attention_layernorm.weight"]
        if config["num_experts"]:
            experts = [f"{source}mlp.experts.{e}." for e in range(config["num_experts"])]
            tensors[target + "router"] = hf[source + "mlp.gate.weight"]
            gate_up = [e + p + "_proj.weight" for e in experts for p in ("gate", "up")]
            tensors[target + "experts_gate_up"] = lambda names=gate_up: joined([hf[n] for n in names])
            down = [e + "down_proj.weight" for e in experts]
            tensors[target + "experts_down"] = lambda names=down: joined([hf[n] for n in names])
        else:
            tensors[target + "gate_up"] = joined([hf[source + "mlp." + p + "_proj.weight"] for p in ("gate", "up")])
            tensors[target + "down"] = hf[source + "mlp.down_proj.weight"]
    return config, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", help="a Hugging Face model folder with config.json and safetensors files")
    parser.add_argument("output", help="the weight file to write")
    weight_format = parser.add_mutually_exclusive_group()
    weight_format.add_argument("--fp8", action="store_true", help="store the weight matrices as f8e4m3 codes")
    weight_format.add_argument("--q4", action="store_true", help="store the weight matrices as Q4_0 codes")
    parser.add_argument("--importance", help="the .npz file of importance.py, which guides the rounding of --q4")
    args = parser.parse_args()
    assert args.importance is None or args.q4, "--importance needs --q4"
    importance = None if args.importance is None else dict(np.load(args.importance))
    config, tensors = convert(args.model_dir)
    if args.fp8 or args.q4:
        config, tensors = qwen_weights.quantize(config, tensors, "fp8" if args.fp8 else "q4_0", importance)
    qwen_weights.write_weight_file(args.output, config, tensors)
    print(f"wrote {args.output}: {len(tensors)} tensors, {config['num_layers']} layers")


if __name__ == "__main__":
    sys.exit(main())
