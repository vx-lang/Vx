#!/usr/bin/env python3
"""Convert a Qwen2 model folder from Hugging Face into the weight file of the Qwen app.

    python3 apps/qwen/scripts/convert.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw
    python3 apps/qwen/scripts/convert.py --fp8 ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b-fp8.vxw

The file keeps the bf16 values exactly. Q, K and V of a layer become one matrix with one bias
vector, Q rows first, and gate and up become one matrix, gate rows first, so that one kernel
reads each of them. With --fp8, the weight matrices become f8e4m3 codes with a scale for each
block of 128 x 128. `qwen_weights.py` describes the file.

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
    config = qwen_weights.config_from_hugging_face(model_dir)
    hf = qwen_weights.read_safetensors(model_dir)
    tensors = {"embed": hf["model.embed_tokens.weight"], "final_norm": hf["model.norm.weight"]}
    if not config["tied_embeddings"]:
        tensors["lm_head"] = hf["lm_head.weight"]
    for layer in range(config["num_layers"]):
        source = f"model.layers.{layer}."
        target = f"layers.{layer}."
        attn = source + "self_attn."
        tensors[target + "attn_norm"] = hf[source + "input_layernorm.weight"]
        tensors[target + "qkv"] = joined([hf[attn + p + "_proj.weight"] for p in "qkv"])
        tensors[target + "qkv_bias"] = joined([hf[attn + p + "_proj.bias"] for p in "qkv"])
        tensors[target + "attn_out"] = hf[attn + "o_proj.weight"]
        tensors[target + "mlp_norm"] = hf[source + "post_attention_layernorm.weight"]
        tensors[target + "gate_up"] = joined([hf[source + "mlp." + p + "_proj.weight"] for p in ("gate", "up")])
        tensors[target + "down"] = hf[source + "mlp.down_proj.weight"]
    return config, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", help="a Hugging Face model folder with config.json and safetensors files")
    parser.add_argument("output", help="the weight file to write")
    parser.add_argument("--fp8", action="store_true", help="store the weight matrices as f8e4m3 codes")
    args = parser.parse_args()
    config, tensors = convert(args.model_dir)
    if args.fp8:
        config, tensors = qwen_weights.to_fp8(config, tensors)
    qwen_weights.write_weight_file(args.output, config, tensors)
    print(f"wrote {args.output}: {len(tensors)} tensors, {config['num_layers']} layers")


if __name__ == "__main__":
    sys.exit(main())
