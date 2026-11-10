"""Tests of qwen_weights.py: the f8e4m3 codes, the block scales and the fp8 weight file.

    python3 -m unittest discover -s apps/qwen/scripts -p 'test_*.py'

The test that compares the codes with the ml_dtypes package runs only when that package is
installed.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import tempfile
import unittest

import numpy as np

import make_tiny_model
import qwen_weights as q

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


class Codes(unittest.TestCase):
    @unittest.skipIf(ml_dtypes is None, "needs the ml_dtypes package")
    def test_codes_equal_those_of_ml_dtypes(self):
        rng = np.random.default_rng(1)
        values = rng.standard_normal(200_000) * np.exp(rng.uniform(-14, 6, 200_000))
        values = np.clip(values, -448, 448).astype(np.float32)
        # The values halfway between two codes, where ties to even decide.
        positive = q.E4M3[:0x7F].astype(np.float64)
        halfway = ((positive[:-1] + positive[1:]) / 2).astype(np.float32)
        for x in (values, halfway, -halfway):
            expected = x.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
            np.testing.assert_array_equal(q.f32_to_e4m3(x), expected)

    def test_values_of_some_codes(self):
        self.assertEqual(q.E4M3[0x38], 1.0)
        self.assertEqual(q.E4M3[0x7E], 448.0)
        self.assertEqual(q.E4M3[0xFE], -448.0)
        self.assertEqual(q.E4M3[0x08], 2.0**-6)
        self.assertEqual(q.E4M3[0x01], 2.0**-9)
        self.assertTrue(np.isnan(q.E4M3[0x7F]) and np.isnan(q.E4M3[0xFF]))

    def test_values_above_448_become_448(self):
        codes = q.f32_to_e4m3(np.float32([449.0, 1000.0, -500.0]))
        self.assertEqual(codes.tolist(), [0x7E, 0x7E, 0xFE])


class BlockScales(unittest.TestCase):
    def setUp(self):
        # 300 x 260 has blocks that the edges of the matrix cut short.
        rng = np.random.default_rng(2)
        self.matrix = (rng.standard_normal((300, 260)) * rng.uniform(0.01, 2.0, (300, 1))).astype(np.float32)
        self.codes, self.scales = q.quantize_fp8(self.matrix)

    def test_the_largest_weight_of_each_block_gets_code_448(self):
        self.assertEqual(self.scales.shape, (3, 3))
        for i in range(3):
            for j in range(3):
                block = self.codes[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
                self.assertEqual(int((block & 0x7F).max()), 0x7E)
                # The agreed scale: the largest size in the block divided by 448.
                largest = np.float32(np.abs(self.matrix[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]).max())
                self.assertEqual(self.scales[i, j], largest / np.float32(448.0))

    def test_each_weight_is_within_half_a_step(self):
        tensors = {"m": ("f8e4m3", self.codes), "m.scale": ("f32", self.scales)}
        widened = q.widen(tensors, "m")
        scale = np.repeat(np.repeat(self.scales, 128, axis=0), 128, axis=1)[:300, :260]
        # A step is 1/8 of the power of two below a value, and 2^-9 below 2^-6, times the scale.
        size = np.abs(self.matrix) / scale
        step = np.where(size < 2.0**-6, 2.0**-9, 2.0 ** (np.floor(np.log2(np.maximum(size, 2.0**-6))) - 3))
        self.assertTrue(np.all(np.abs(widened - self.matrix) <= step / 2 * scale * (1 + 1e-6)))

    def test_a_block_of_zeros_has_scale_one(self):
        codes, scales = q.quantize_fp8(np.zeros((128, 128), dtype=np.float32))
        self.assertEqual(scales.tolist(), [[1.0]])
        self.assertEqual(int(codes.max()), 0)


class Fp8File(unittest.TestCase):
    def test_the_tiny_model_in_fp8_reads_back(self):
        config, tensors = q.to_fp8(make_tiny_model.CONFIG, make_tiny_model.tiny_model())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny-fp8.vxw")
            q.write_weight_file(path, config, tensors)
            read_config, read = q.read_weight_file(path)
            self.assertEqual(read_config["fp8"], 1)
            self.assertEqual(read_config["tied_embeddings"], 0)
            self.assertEqual(list(read), q.tensor_names(read_config))
            self.assertEqual(read["layers.0.qkv"][0], "f8e4m3")
            self.assertEqual(read["layers.0.qkv.scale"][1].shape, (2, 1))
            np.testing.assert_array_equal(read["layers.0.qkv"][1], tensors["layers.0.qkv"][1])
            # The embedding table stays bf16, and lm_head is its fp8 copy.
            np.testing.assert_array_equal(read["embed"][1], make_tiny_model.tiny_model()["embed"][1])
            codes, scales = q.quantize_fp8(q.as_f32(read["embed"]))
            np.testing.assert_array_equal(read["lm_head"][1], codes)
            np.testing.assert_array_equal(read["lm_head.scale"][1], scales)


if __name__ == "__main__":
    unittest.main()
