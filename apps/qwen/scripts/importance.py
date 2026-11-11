#!/usr/bin/env python3
"""Measure how much each input of each weight matrix matters, for `convert.py --importance`.

    python3 apps/qwen/scripts/importance.py qwen3-30b.vxw --tokenizer DIR --out importance.npz FILE...

A matrix multiplies a vector x. The importance of input j is the mean of x[j] * x[j] over the tokens
of a sample text, in a bf16 model. An error in column j of the matrix adds about that much to the
square of the error of each output, so the converter weights the error of column j by it. llama.cpp
calls this an importance matrix. Each expert gets its own, from the tokens that the router sends to it.

The file has one float32 array for each matrix, of shape [groups, columns]: one group for most
matrices, and one for each expert in `experts_gate_up` and `experts_down`. An expert that no token
chose gets the mean over all the experts of its layer.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import sys
import time

import numpy as np

import qwen_weights
import reference


class Model(reference.Model):
    """A reference.Model that reads the experts of a layer in one read of the file each. A bf16
    file of a large model does not fit in memory, so each chunk reads it again. The memory map reads
    it in small pieces, at a few times less speed."""

    def __init__(self, path):
        super().__init__(path)
        self.file = open(path, "rb")
        self.layer, self.read = None, {}

    def weight(self, name, rows=None):
        dtype, array = self.tensors[name]
        if rows is None or dtype != "bf16":
            return super().weight(name, rows)
        layer = name.rsplit(".", 1)[0]
        if layer != self.layer:
            self.layer, self.read = layer, {}
        if name not in self.read:
            mapped = array
            while not isinstance(mapped, np.memmap):
                mapped = mapped.base
            offset = array.ctypes.data - mapped.ctypes.data
            data = os.pread(self.file.fileno(), array.nbytes, offset)
            self.read[name] = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
        return qwen_weights.as_f32(("bf16", self.read[name][rows[0] : rows[1]]))


class Importance:
    """The sums of x[j] * x[j] and the number of tokens, for each matrix and group."""

    def __init__(self, config):
        self.num_experts = config["num_experts"]
        self.sums, self.counts = {}, {}

    def see_input(self, name, rows, x):
        # A model with tied embeddings uses its embedding table as lm_head.
        name = "lm_head" if name == "embed" else name
        groups, group = 1, 0
        if name.rsplit(".", 1)[-1].startswith("experts_"):
            # The rows of one expert, in a matrix that holds the experts one after another.
            groups, group = self.num_experts, rows[0] // (rows[1] - rows[0])
        if name not in self.sums:
            self.sums[name] = np.zeros((groups, x.shape[1]))
            self.counts[name] = np.zeros(groups)
        self.sums[name][group] += np.square(x, dtype=np.float64).sum(axis=0)
        self.counts[name][group] += len(x)

    def means(self):
        result = {}
        for name, sums in self.sums.items():
            counts = self.counts[name]
            everyone = sums.sum(axis=0) / counts.sum()
            result[name] = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], everyone).astype(np.float32)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a bf16 weight file from convert.py")
    parser.add_argument("texts", nargs="+", help="the text files of the sample, read one after another")
    parser.add_argument("--tokenizer", required=True, help="a Hugging Face model folder that has tokenizer.json")
    parser.add_argument("--out", required=True, help="the .npz file to write")
    parser.add_argument("--context", type=int, default=512, help="the tokens of each chunk, which starts with an empty cache")
    parser.add_argument("--chunks-per-text", type=int, help="use at most this many chunks of each text")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    chunks = []
    for path in args.texts:
        with open(path) as text:
            tokens = tokenizer.encode(text.read()).ids
        starts = range(0, len(tokens) - args.context + 1, args.context)[: args.chunks_per_text]
        chunks += [tokens[start : start + args.context] for start in starts]
    model = Model(args.weights)
    assert model.config["weight_format"] == 0, "measure the importance in a bf16 file"
    importance = Importance(model.config)
    model.see_input = importance.see_input
    for index, chunk in enumerate(chunks):
        start = time.time()
        reference.forward(model, chunk, reference.Cache(model.config))
        print(f"chunk {index + 1} of {len(chunks)}: {time.time() - start:.0f} s", flush=True)
    np.savez(args.out, **importance.means())
    print(f"wrote {args.out}: {len(importance.sums)} matrices, {len(chunks) * args.context} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
