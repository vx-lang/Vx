"""Tests of importance.py: the mean square of each input of each matrix, for each expert.

    python3 -m unittest discover -s apps/qwen/scripts -p 'test_*.py'

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import tempfile
import unittest

import numpy as np

import convert
import importance
import make_tiny_model
import qwen_weights as q
import reference


class Means(unittest.TestCase):
    def test_each_expert_gets_the_mean_of_its_own_tokens(self):
        imp = importance.Importance({"num_experts": 4})
        # Expert 0 is rows 0 to 2 of a matrix with 3 rows for each expert, and expert 2 is rows 6 to 8.
        imp.see_input("layers.0.experts_down", (0, 3), np.float32([[1, 2], [3, 4]]))
        imp.see_input("layers.0.experts_down", (6, 9), np.float32([[5, 6]]))
        means = imp.means()["layers.0.experts_down"]
        np.testing.assert_allclose(means[0], [(1 + 9) / 2, (4 + 16) / 2])
        np.testing.assert_allclose(means[2], [25, 36])
        # No token chose expert 1 or 3. They get the mean over the 3 tokens of all the experts.
        np.testing.assert_allclose(means[1], [(1 + 9 + 25) / 3, (4 + 16 + 36) / 3])
        np.testing.assert_allclose(means[3], means[1])

    def test_a_matrix_of_a_layer_has_one_group(self):
        imp = importance.Importance({"num_experts": 4})
        imp.see_input("layers.3.qkv", None, np.float32([[1, -2]]))
        imp.see_input("layers.3.qkv", None, np.float32([[3, 0]]))
        self.assertEqual(imp.means()["layers.3.qkv"].tolist(), [[5, 2]])

    def test_the_embedding_table_counts_as_lm_head(self):
        imp = importance.Importance({"num_experts": 0})
        imp.see_input("embed", None, np.float32([[2, 1]]))
        self.assertEqual(list(imp.means()), ["lm_head"])


class TinyModel(unittest.TestCase):
    def test_the_inputs_of_the_tiny_moe_model(self):
        config, tensors = convert.convert_tensors(q.config_from_json(make_tiny_model.MOE_CONFIG), make_tiny_model.tiny_moe_model())
        tokens = [17, 200, 3, 95, 41, 7, 128]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny-moe.vxw")
            q.write_weight_file(path, config, tensors)
            model = importance.Model(path)
            imp = importance.Importance(model.config)
            model.see_input = imp.see_input
            logits = reference.forward(model, tokens, reference.Cache(model.config))
            # The model of importance.py reads the experts in another way, but the same values.
            plain = reference.Model(path)
            np.testing.assert_array_equal(logits, reference.forward(plain, tokens, reference.Cache(plain.config)))
            means = imp.means()
            c, eps = model.config, model.config["rms_norm_eps"]
            # The first product of the model multiplies the normed embeddings of the tokens.
            h = reference.rms_norm(model.weight("embed")[tokens], model.weight("layers.0.attn_norm"), eps)
            np.testing.assert_allclose(means["layers.0.qkv"], [np.mean(h * h, axis=0)], rtol=1e-6)
        names = {name for name in q.tensor_names(config) if q.is_matrix(name)}
        self.assertEqual(set(means), names)
        self.assertEqual(means["layers.1.experts_gate_up"].shape, (c["num_experts"], c["hidden_size"]))
        self.assertEqual(means["layers.1.experts_down"].shape, (c["num_experts"], c["moe_intermediate_size"]))
        # Each token goes to experts_per_token experts.
        counts = imp.counts["layers.1.experts_down"]
        self.assertEqual(counts.sum(), len(tokens) * c["experts_per_token"])


if __name__ == "__main__":
    unittest.main()
