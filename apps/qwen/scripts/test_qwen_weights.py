"""Tests of qwen_weights.py: the f8e4m3 and Q4_0 codes, their scales, and the quantized weight files.

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

import convert
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


def q4_0_one_row(row):
    """The Q4_0 codes and scales of one row, with the loop of quantize_row_q4_0_ref of llama.cpp,
    in float32 scalars."""
    codes, scales = [], []
    for start in range(0, len(row), 32):
        block = row[start : start + 32]
        largest, extreme = np.float32(0), np.float32(0)
        for v in block:
            if largest < abs(v):
                largest, extreme = abs(v), v
        d = np.float32(extreme / np.float32(-8))
        inverse = np.float32(1) / d if d else np.float32(0)
        codes += [min(15, int(np.float32(v * inverse) + np.float32(8.5))) for v in block]
        scales.append(np.float16(d))
    return codes, scales


def q4_0_one_row_with_importance(row, importance):
    """The Q4_0 codes and scales of one row, with the loops of quantize_row_q4_0_impl and
    make_qx_quants of llama.cpp, in float32 scalars. A group of zeros gets code 8, as above."""
    f = np.float32
    sum_x2 = f(0)
    for v in row:
        sum_x2 = f(sum_x2 + f(v * v))
    sigma2 = f(sum_x2 / f(len(row)))
    codes, scales = [], []
    for start in range(0, len(row), 32):
        x = row[start : start + 32]
        w = [f(importance[start + j] * np.sqrt(f(sigma2 + f(x[j] * x[j])))) for j in range(32)]
        largest, extreme = f(0), f(0)
        for v in x:
            if largest < abs(v):
                largest, extreme = abs(v), v
        if largest == 0:
            codes += [8] * 32
            scales.append(np.float16(0))
            continue

        def round_all(iscale):
            sumlx, suml2, levels = f(0), f(0), []
            for j in range(32):
                # nearest_int of llama.cpp rounds a value halfway between two integers to the even one.
                level = max(-8, min(7, int(np.rint(f(iscale * x[j])))))
                levels.append(level + 8)
                sumlx = f(sumlx + f(f(w[j] * x[j]) * f(level)))
                suml2 = f(suml2 + f(f(w[j] * f(level)) * f(level)))
            return sumlx, suml2, levels

        sumlx, suml2, chosen = round_all(f(f(-8) / extreme))
        scale = f(sumlx / suml2) if suml2 else f(0)
        best = f(scale * sumlx)
        for i in [i for i in range(-9, 10) if i != 0]:
            sumlx, suml2, levels = round_all(f(-f(f(8) + f(f(0.1) * f(i))) / extreme))
            if suml2 > 0 and f(sumlx * sumlx) > f(best * suml2):
                chosen, scale = levels, f(sumlx / suml2)
                best = f(scale * sumlx)
        codes += chosen
        scales.append(np.float16(scale))
    return codes, scales


class Q4_0(unittest.TestCase):
    def test_codes_and_scales_equal_those_of_the_llama_cpp_loop(self):
        rng = np.random.default_rng(3)
        matrix = (rng.standard_normal((6, 256)) * rng.uniform(0.001, 3, (6, 1))).astype(np.float32)
        matrix[1, :32] = 0  # a group of zeros gets scale 0 and code 8
        matrix[2, 32:64] = 0.25  # a group of one value
        # Scale 1, and weights halfway between two codes: llama.cpp rounds them up, not to even.
        matrix[3, :32] = np.float32([-8] + [k + 0.5 for k in range(-8, 7)] * 2 + [7.5])
        # Two weights of the largest size: the first one sets the sign of the scale.
        matrix[4, 64:96] = np.float32([-4, 4] + [1] * 30)
        # A group of -0.0 gets the scale -0.0, as one of +0.0 does.
        matrix[5, :32] = np.float32(-0.0)
        packed, scales = q.quantize_q4_0(matrix)
        codes = q.unpack_u4(packed)
        for r in range(len(matrix)):
            expected_codes, expected_scales = q4_0_one_row(matrix[r])
            self.assertEqual(codes[r].tolist(), expected_codes)
            # The bits, so that -0.0 and +0.0 differ.
            self.assertEqual(scales[r].view(np.uint16).tolist(), [int(s.view(np.uint16)) for s in expected_scales])
        self.assertEqual(codes[1, :32].tolist(), [8] * 32)

    def test_codes_fill_each_word_from_its_low_bits(self):
        # The weight of largest size is -8, so the scale is 1 and weight x gets code x + 8.
        row = np.tile(np.arange(-8, 8, dtype=np.float32), 2)
        packed, scales = q.quantize_q4_0(row[None, :])
        self.assertEqual(scales.tolist(), [[1.0]])
        words = packed.view("<u4")
        self.assertEqual([hex(w) for w in words[0]], ["0x76543210", "0xfedcba98"] * 2)

    def test_each_weight_is_within_half_a_step(self):
        rng = np.random.default_rng(4)
        matrix = (rng.standard_normal((300, 256)) * rng.uniform(0.01, 2.0, (300, 1))).astype(np.float32)
        packed, scales = q.quantize_q4_0(matrix)
        widened = q.widen({"m": ("u4", packed), "m.scale": ("f16", scales)}, "m")
        step = np.repeat(np.abs(scales.astype(np.float32)), 32, axis=1)
        error = np.abs(widened - matrix)
        # Code 15 is the top code, so a weight above it in size is up to one step away. The f16
        # copy of the scale adds at most 8 times its rounding.
        top = q.unpack_u4(packed) == 15
        self.assertTrue(np.all(error[~top] <= step[~top] * (0.5 + 8 * 2.0**-11)))
        self.assertTrue(np.all(error <= step * (1 + 8 * 2.0**-11)))
        # The weight of largest size in each group is exact up to the rounding of the scale.
        largest = np.abs(matrix).reshape(300, -1, 32).max(axis=-1)
        widened_largest = np.abs(widened).reshape(300, -1, 32).max(axis=-1)
        np.testing.assert_allclose(widened_largest, largest, rtol=2.0**-11)

    def test_with_importance_codes_and_scales_equal_those_of_the_llama_cpp_loops(self):
        rng = np.random.default_rng(5)
        matrix = (rng.standard_normal((6, 256)) * rng.uniform(0.001, 3, (6, 1))).astype(np.float32)
        matrix[1, :32] = 0  # a group of zeros gets scale 0 and code 8
        # There is no group of one value. All the tries of such a group have the same error, and
        # the order of the float sums, which differs from llama.cpp in NumPy, chooses one.
        # Two rows share each row of importance, as the rows of one expert do.
        importance = rng.uniform(0, 4, (3, 256)).astype(np.float32)
        importance[0, 64:96] = 0  # a group whose errors do not count gets scale 0
        packed, scales = q.quantize_q4_0(matrix, importance)
        codes = q.unpack_u4(packed)
        for r in range(len(matrix)):
            expected_codes, expected_scales = q4_0_one_row_with_importance(matrix[r], importance[r // 2])
            self.assertEqual(codes[r].tolist(), expected_codes, f"row {r}")
            self.assertEqual(scales[r].view(np.uint16).tolist(), [int(s.view(np.uint16)) for s in expected_scales], f"row {r}")
        self.assertEqual(float(scales[0, 2]), 0.0)

    def test_importance_lowers_the_error_that_it_weights(self):
        rng = np.random.default_rng(6)
        matrix = rng.standard_normal((64, 512)).astype(np.float32)
        importance = rng.lognormal(0, 1.5, (1, 512)).astype(np.float32)

        def error(packed, scales):
            widened = q.widen({"m": ("u4", packed), "m.scale": ("f16", scales)}, "m")
            return float((importance * (widened - matrix) ** 2).sum())

        weighted = error(*q.quantize_q4_0(matrix, importance))
        self.assertLess(weighted, 0.9 * error(*q.quantize_q4_0(matrix, np.ones_like(importance))))
        self.assertLess(weighted, 0.9 * error(*q.quantize_q4_0(matrix)))


class QuantizedFiles(unittest.TestCase):
    def test_the_tiny_model_in_fp8_reads_back(self):
        config, tensors = q.quantize(make_tiny_model.CONFIG, make_tiny_model.tiny_model(), "fp8")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny-fp8.vxw")
            q.write_weight_file(path, config, tensors)
            read_config, read = q.read_weight_file(path)
            self.assertEqual(read_config["weight_format"], 1)
            self.assertEqual(read_config["tied_embeddings"], 0)
            self.assertEqual(list(read), q.tensor_names(read_config))
            self.assertEqual(read["layers.0.qkv"][0], "f8e4m3")
            self.assertEqual(read["layers.0.qkv.scale"][1].shape, (2, 1))
            np.testing.assert_array_equal(read["layers.0.qkv"][1], q.resolve(tensors["layers.0.qkv"])[1])
            # The embedding table stays bf16, and lm_head is its fp8 copy.
            np.testing.assert_array_equal(read["embed"][1], make_tiny_model.tiny_model()["embed"][1])
            codes, scales = q.quantize_fp8(q.as_f32(read["embed"]))
            np.testing.assert_array_equal(read["lm_head"][1], codes)
            np.testing.assert_array_equal(read["lm_head.scale"][1], scales)

    def test_the_tiny_moe_model_in_q4_0_reads_back(self):
        hf = make_tiny_model.tiny_moe_model()
        config, tensors = convert.convert_tensors(q.config_from_json(make_tiny_model.MOE_CONFIG), hf)
        config, tensors = q.quantize(config, tensors, "q4_0")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny-moe-q4.vxw")
            q.write_weight_file(path, config, tensors)
            read_config, read = q.read_weight_file(path)
        self.assertEqual(list(read), q.tensor_names(read_config))
        self.assertEqual((read_config["num_experts"], read_config["experts_per_token"]), (8, 2))
        self.assertEqual((read_config["qk_norm"], read_config["qkv_bias"]), (1, 0))
        dtypes = {name.split(".", 2)[-1]: dtype for name, (dtype, _) in read.items()}
        self.assertEqual(dtypes["experts_gate_up"], "u4")
        self.assertEqual(dtypes["experts_gate_up.scale"], "f16")
        self.assertEqual((dtypes["router"], dtypes["q_norm"], dtypes["lm_head"]), ("bf16", "bf16", "f8e4m3"))
        # 8 experts of 2 * 256 rows of 256 weights, two codes in each byte, and a scale for each 32.
        self.assertEqual(read["layers.1.experts_gate_up"][1].shape, (8 * 512, 128))
        self.assertEqual(read["layers.1.experts_gate_up.scale"][1].shape, (8 * 512, 8))
        # Expert 3 of layer 1: its gate rows, then its up rows, each widened from its own codes.
        p = "model.layers.1.mlp.experts.3."
        expected = np.concatenate([q.as_f32(hf[p + "gate_proj.weight"]), q.as_f32(hf[p + "up_proj.weight"])])
        codes, scales = q.quantize_q4_0(expected)
        widened = q.widen(read, "layers.1.experts_gate_up", rows=(3 * 512, 4 * 512))
        np.testing.assert_array_equal(widened, q.widen({"m": ("u4", codes), "m.scale": ("f16", scales)}, "m"))

    def test_each_expert_of_the_tiny_moe_model_gets_its_own_importance(self):
        hf = make_tiny_model.tiny_moe_model()
        config, tensors = convert.convert_tensors(q.config_from_json(make_tiny_model.MOE_CONFIG), hf)
        rng = np.random.default_rng(7)
        shapes = q.tensor_shapes(dict(config, weight_format=q.WEIGHT_FORMATS["q4_0"], tied_embeddings=0))
        importance = {}
        for name in [n for n in shapes if q.is_matrix(n) and n != "lm_head"]:
            groups = config["num_experts"] if "experts_" in name else 1
            importance[name] = rng.uniform(0.1, 4, (groups, shapes[name][1])).astype(np.float32)
        config, tensors = q.quantize(config, tensors, "q4_0", importance)
        # Expert 3 of layer 1, with row 3 of the importance of its matrix.
        p = "model.layers.1.mlp.experts.3."
        expected = np.concatenate([q.as_f32(hf[p + "gate_proj.weight"]), q.as_f32(hf[p + "up_proj.weight"])])
        codes, scales = q.quantize_q4_0(expected, importance["layers.1.experts_gate_up"][3:4])
        np.testing.assert_array_equal(q.resolve(tensors["layers.1.experts_gate_up"])[1][3 * 512 : 4 * 512], codes)
        np.testing.assert_array_equal(q.resolve(tensors["layers.1.experts_gate_up.scale"])[1][3 * 512 : 4 * 512], scales)


if __name__ == "__main__":
    unittest.main()
