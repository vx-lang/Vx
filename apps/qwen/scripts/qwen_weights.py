"""Read Qwen weights from safetensors files, and read and write the weight file of the Qwen app.

The weight file holds 8-byte little-endian words, so a Vx program can read it with plain `i64`
and `f64` loads and needs no JSON parser:

    word 0       magic, the bytes b"VXQWEN01"
    word 1       format version, 1
    word 2       number of tensors
    word 3       alignment of each tensor, in bytes
    word 4       byte offset of the tensor table
    words 5-20   the integers of CONFIG_INTS, in that order
    words 21-22  rope_theta and rms_norm_eps, as f64

The tensor table has one entry of 16 words (128 bytes) per tensor: the name in words 0-7 (ASCII,
padded with zero bytes), the dtype code, the rank, four dimensions (unused ones are 0), the byte
offset of the data and its size in bytes. The tensors come in the order of `tensor_names`.

`weight_format` says how the file stores the weight matrices of MATRICES. Each other tensor is
bf16 in every format.

    0, bf16    each weight matrix is bf16.
    1, fp8     each weight matrix is f8e4m3 codes, one byte for each weight. The tensor after it,
               `<name>.scale`, has one f32 scale for each block of 128 x 128 weights. A weight is
               the value of its code times the scale of its block.
    2, q4_0    each weight matrix except lm_head is u4 codes, two in each byte, the first weight
               in the low 4 bits. Its `<name>.scale` has one f16 scale for each 32 weights of a
               row. A weight is (code - 8) times its scale, as in the Q4_0 of llama.cpp. lm_head
               is f8e4m3, as in an fp8 file.

A file in fp8 or q4_0 always has `lm_head`, also for a model whose output projection is its
embedding table.

A mixture-of-experts (MoE) model has num_experts > 0. Each of its layers has a `router` with one
row for each expert, and the matrices `experts_gate_up` and `experts_down` in place of `gate_up`
and `down`. These hold expert e after expert e - 1. An expert has 2 * moe_intermediate_size rows
in `experts_gate_up`, its gate rows and then its up rows, and hidden_size rows in `experts_down`.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"VXQWEN01"
VERSION = 1
TABLE_OFFSET = 4096
ENTRY_BYTES = 128
# A Metal buffer over mapped file memory must start on a page, and an Apple Silicon page is 16 KiB.
ALIGNMENT = 16384

CONFIG_INTS = [
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "max_positions",
    "tied_embeddings",
    "qkv_bias",
    "eos_token_id",
    "weight_format",
    "num_experts",
    "experts_per_token",
    "moe_intermediate_size",
    "qk_norm",
]
# The header has room for 16 integers.
CONFIG_INT_SLOTS = 16
assert len(CONFIG_INTS) <= CONFIG_INT_SLOTS
WEIGHT_FORMATS = {"bf16": 0, "fp8": 1, "q4_0": 2}

DTYPE_CODES = {"f32": 0, "bf16": 1, "f8e4m3": 2, "f16": 3, "u4": 4}
DTYPE_NAMES = {code: name for name, code in DTYPE_CODES.items()}
DTYPE_BITS = {"f32": 32, "bf16": 16, "f8e4m3": 8, "f16": 16, "u4": 4}
# The NumPy array of a u4 tensor holds two codes in each uint8, so its last axis is half as long.
STORAGE = {"f32": np.float32, "bf16": np.uint16, "f8e4m3": np.uint8, "f16": np.float16, "u4": np.uint8}

# The weight matrices that an fp8 or q4_0 file quantizes. The kinds of a layer end the name.
MATRICES = ("qkv", "attn_out", "gate_up", "down", "experts_gate_up", "experts_down", "lm_head")
BLOCK = 128
Q4_GROUP = 32


def bf16_to_f32(bits):
    """Widen bf16 values, given as uint16 bit patterns, to float32. The result is exact."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    """Round float32 values to bf16 bit patterns, to nearest with ties to even."""
    values = np.asarray(values, dtype=np.float32)
    assert np.all(np.isfinite(values)), "only finite values round to bf16 here"
    bits = values.view(np.uint32)
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    return ((bits + rounding) >> 16).astype(np.uint16)


def is_matrix(name):
    """Whether an fp8 or q4_0 file quantizes the tensor `name`, and has a `<name>.scale`."""
    return name.rsplit(".", 1)[-1] in MATRICES


def matrix_dtype(config, name):
    """The dtype of the weight matrix `name` in a file with this config."""
    weight_format = config["weight_format"]
    if weight_format == WEIGHT_FORMATS["q4_0"] and name != "lm_head":
        return "u4"
    return "f8e4m3" if weight_format else "bf16"


def tensor_names(config):
    """The names of the tensors in the weight file, in file order."""
    assert config["weight_format"] in WEIGHT_FORMATS.values(), f"weight format {config['weight_format']}"
    assert not (config["weight_format"] and config["tied_embeddings"]), "a quantized file has its own lm_head"
    names = ["embed"]
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        names += [prefix + "attn_norm", prefix + "qkv"]
        if config["qkv_bias"]:
            names.append(prefix + "qkv_bias")
        if config["qk_norm"]:
            names += [prefix + "q_norm", prefix + "k_norm"]
        names += [prefix + "attn_out", prefix + "mlp_norm"]
        if config["num_experts"]:
            names += [prefix + "router", prefix + "experts_gate_up", prefix + "experts_down"]
        else:
            names += [prefix + "gate_up", prefix + "down"]
    names.append("final_norm")
    if not config["tied_embeddings"]:
        names.append("lm_head")
    if config["weight_format"]:
        names = [n for name in names for n in ([name, name + ".scale"] if is_matrix(name) else [name])]
    return names


def tensor_shapes(config):
    """The shape of each tensor of the weight file, by name."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    q_rows = config["num_q_heads"] * head_dim
    qkv_rows = q_rows + 2 * config["num_kv_heads"] * head_dim
    inter = config["intermediate_size"]
    experts, moe_inter = config["num_experts"], config["moe_intermediate_size"]
    shapes = {"embed": (config["vocab_size"], hidden), "final_norm": (hidden,)}
    if not config["tied_embeddings"]:
        shapes["lm_head"] = (config["vocab_size"], hidden)
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        shapes[prefix + "attn_norm"] = (hidden,)
        shapes[prefix + "qkv"] = (qkv_rows, hidden)
        if config["qkv_bias"]:
            shapes[prefix + "qkv_bias"] = (qkv_rows,)
        if config["qk_norm"]:
            shapes[prefix + "q_norm"] = (head_dim,)
            shapes[prefix + "k_norm"] = (head_dim,)
        shapes[prefix + "attn_out"] = (hidden, q_rows)
        shapes[prefix + "mlp_norm"] = (hidden,)
        if experts:
            shapes[prefix + "router"] = (experts, hidden)
            shapes[prefix + "experts_gate_up"] = (experts * 2 * moe_inter, hidden)
            shapes[prefix + "experts_down"] = (experts * hidden, moe_inter)
        else:
            shapes[prefix + "gate_up"] = (2 * inter, hidden)
            shapes[prefix + "down"] = (hidden, inter)
    if config["weight_format"]:
        for name in [n for n in shapes if is_matrix(n)]:
            rows, cols = shapes[name]
            if matrix_dtype(config, name) == "u4":
                assert cols % Q4_GROUP == 0, f"{name}: {cols} columns are not groups of {Q4_GROUP}"
                shapes[name + ".scale"] = (rows, cols // Q4_GROUP)
            else:
                shapes[name + ".scale"] = (-(-rows // BLOCK), -(-cols // BLOCK))
    return shapes


def storage_shape(dtype, shape):
    """The shape of the NumPy array that holds a tensor of this dtype and shape."""
    if dtype != "u4":
        return tuple(shape)
    assert shape[-1] % 2 == 0, f"u4 rows of {shape[-1]} codes do not fill whole bytes"
    return (*shape[:-1], shape[-1] // 2)


def resolve(entry):
    """A (dtype name, array) pair of a tensors dict, whose value can also be a function that
    returns the pair."""
    return entry() if callable(entry) else entry


def write_weight_file(path, config, tensors):
    """Write `tensors`, a dict from name to (dtype name, numpy array), in file order. A value can
    also be a function that returns the pair. The file then gets each tensor when it is written,
    so that only one is in memory at a time."""
    names = tensor_names(config)
    shapes = tensor_shapes(config)
    assert sorted(names) == sorted(tensors), "the tensors do not match the names of the config"
    # The data starts at the first aligned offset after the tensor table.
    data_start = -(-(TABLE_OFFSET + ENTRY_BYTES * len(names)) // ALIGNMENT) * ALIGNMENT
    header = bytearray(data_start)
    words = [VERSION, len(names), ALIGNMENT, TABLE_OFFSET]
    words += [int(config[field]) for field in CONFIG_INTS]
    words += [0] * (CONFIG_INT_SLOTS - len(CONFIG_INTS))
    header[0:8] = MAGIC
    struct.pack_into(f"<{len(words)}q", header, 8, *words)
    struct.pack_into("<2d", header, 8 * (5 + CONFIG_INT_SLOTS), config["rope_theta"], config["rms_norm_eps"])

    offset = data_start
    with open(path, "wb") as out:
        for index, name in enumerate(names):
            dtype, array = resolve(tensors[name])
            shape = shapes[name]
            assert tuple(array.shape) == storage_shape(dtype, shape), f"{name}: shape {array.shape} for {shape}"
            assert array.dtype == STORAGE[dtype], f"{name}: storage {array.dtype} for {dtype}"
            nbytes = int(np.prod(shape)) * DTYPE_BITS[dtype] // 8
            dims = list(shape) + [0] * (4 - len(shape))
            entry = name.encode("ascii").ljust(64, b"\0")
            assert len(entry) == 64, f"{name}: the name is longer than 64 bytes"
            entry += struct.pack("<8q", DTYPE_CODES[dtype], len(shape), *dims, offset, nbytes)
            header[TABLE_OFFSET + index * ENTRY_BYTES : TABLE_OFFSET + (index + 1) * ENTRY_BYTES] = entry
            out.seek(offset)
            out.write(np.ascontiguousarray(array).data)
            offset += -(-nbytes // ALIGNMENT) * ALIGNMENT
        out.truncate(offset)
        out.seek(0)
        out.write(header)


def read_weight_file(path):
    """Return the config and a dict from name to a float32 view of each tensor.

    The views are read from a memory map, and widened from bf16 when they are used.
    """
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert bytes(data[0:8]) == MAGIC, f"{path}: not a weight file of the Qwen app"
    words = struct.unpack_from(f"<{4 + CONFIG_INT_SLOTS}q", data, 8)
    version, count, alignment, table = words[:4]
    assert version == VERSION, f"{path}: format version {version}, expected {VERSION}"
    config = dict(zip(CONFIG_INTS, words[4 : 4 + len(CONFIG_INTS)]))
    config["rope_theta"], config["rms_norm_eps"] = struct.unpack_from("<2d", data, 8 * (5 + CONFIG_INT_SLOTS))
    tensors = {}
    for index in range(count):
        start = table + index * ENTRY_BYTES
        name = bytes(data[start : start + 64]).rstrip(b"\0").decode("ascii")
        dtype, rank, *rest = struct.unpack_from("<8q", data, start + 64)
        dims, offset, nbytes = rest[:4], rest[4], rest[5]
        assert offset % alignment == 0, f"{name}: offset {offset} is not aligned"
        dtype = DTYPE_NAMES[dtype]
        storage = STORAGE[dtype]
        raw = np.frombuffer(data, dtype=storage, count=nbytes // np.dtype(storage).itemsize, offset=offset)
        tensors[name] = (dtype, raw.reshape(storage_shape(dtype, dims[:rank])))
    assert list(tensors) == tensor_names(config), f"{path}: the tensors are not in file order"
    return config, tensors


def as_f32(entry):
    """The float32 values of a bf16 or f32 (dtype name, array) entry of `read_weight_file`."""
    dtype, array = entry
    assert dtype in ("bf16", "f32"), f"use widen() for {dtype}"
    return bf16_to_f32(array) if dtype == "bf16" else np.asarray(array, dtype=np.float32)


def _e4m3_values():
    """The value of each of the 256 f8e4m3 codes: OCP e4m3, bias 7, no infinity, NaN at 0x7F."""
    codes = np.arange(256)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    values = np.where(exponent == 0, mantissa / 8 * 2.0**-6, (1 + mantissa / 8) * 2.0 ** (exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = _e4m3_values()
E4M3_MAX = 448.0


def f32_to_e4m3(values):
    """Round float32 values to f8e4m3 codes, to nearest with ties to even. Values above 448 in
    size become 448."""
    values = np.asarray(values, dtype=np.float32)
    assert np.all(np.isfinite(values)), "only finite values round to f8e4m3 here"
    positive = E4M3[:0x7F]  # codes 0 to 126, in increasing order
    size = np.minimum(np.abs(values), E4M3_MAX)
    high = np.minimum(np.searchsorted(positive, size), 0x7E)
    low = np.maximum(high - 1, 0)
    up, down = positive[high] - size, size - positive[low]
    code = np.where((up < down) | ((up == down) & (high % 2 == 0)), high, low).astype(np.uint8)
    return code | np.where(np.signbit(values), 0x80, 0).astype(np.uint8)


def quantize_fp8(matrix):
    """f8e4m3 codes and f32 block scales for a float32 matrix. The scale of a block of 128 x 128
    is the largest size in it divided by 448, so that block's largest weight becomes 448."""
    rows, cols = matrix.shape
    scales = np.ones((-(-rows // BLOCK), -(-cols // BLOCK)), dtype=np.float32)
    codes = np.empty((rows, cols), dtype=np.uint8)
    for i in range(scales.shape[0]):
        band = np.asarray(matrix[i * BLOCK : (i + 1) * BLOCK], dtype=np.float32)
        for j in range(scales.shape[1]):
            block = band[:, j * BLOCK : (j + 1) * BLOCK]
            largest = np.float32(np.abs(block).max())
            if largest > 0:
                scales[i, j] = largest / np.float32(E4M3_MAX)
            codes[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = f32_to_e4m3(block / scales[i, j])
    return codes, scales


def quantize_q4_0(matrix):
    """u4 codes, two in each byte, and f16 scales for a float32 matrix, rounded as the Q4_0 of
    llama.cpp rounds them (`quantize_row_q4_0_ref`). In each group of 32 weights of a row, the
    weight of the largest size gets code 0, so the scale is that weight divided by -8."""
    rows, cols = matrix.shape
    assert cols % Q4_GROUP == 0, f"{cols} columns are not groups of {Q4_GROUP}"
    codes = np.empty((rows, cols // 2), dtype=np.uint8)
    scales = np.empty((rows, cols // Q4_GROUP), dtype=np.float16)
    # About 16M weights at a time, so that a large matrix needs little memory.
    step = max(1, (1 << 24) // cols)
    for start in range(0, rows, step):
        chunk = np.asarray(matrix[start : start + step], dtype=np.float32)
        assert np.all(np.isfinite(chunk)), "only finite values round to Q4_0 here"
        groups = chunk.reshape(len(chunk), -1, Q4_GROUP)
        largest = np.take_along_axis(groups, np.abs(groups).argmax(axis=-1)[..., None], axis=-1)
        # As in llama.cpp, a group of zeros keeps +0.0 here, also when its weights are -0.0.
        largest = np.where(largest == 0, np.float32(0), largest)
        d = largest / np.float32(-8)
        inverse = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
        # All float32, as in llama.cpp. The sum is at least 0.5, so the floor is its cast to int.
        q = np.minimum(np.float32(15), np.floor(groups * inverse + np.float32(8.5))).astype(np.uint8)
        q = q.reshape(len(chunk), cols)
        codes[start : start + step] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[start : start + step] = d.reshape(len(chunk), -1).astype(np.float16)
    return codes, scales


def unpack_u4(packed):
    """The codes of a u4 array, one in each uint8: the low 4 bits of a byte, then the high 4."""
    return np.stack([packed & 15, packed >> 4], axis=-1).reshape(*packed.shape[:-1], -1)


def widen(tensors, name, rows=None):
    """The float32 values of the tensor `name` of `read_weight_file`, also of a quantized matrix.
    `rows`, a (start, stop) pair, widens only those rows of a matrix."""
    dtype, array = tensors[name]
    start, stop = rows if rows is not None else (0, len(array))
    array = array[start:stop]
    if dtype in ("bf16", "f32"):
        return as_f32((dtype, array))
    scales = tensors[name + ".scale"][1]
    if dtype == "u4":
        group_scales = np.repeat(scales[start:stop].astype(np.float32), Q4_GROUP, axis=1)
        return (unpack_u4(array).astype(np.float32) - 8) * group_scales
    assert dtype == "f8e4m3", f"{name}: cannot widen {dtype}"
    blocks = np.repeat(scales[np.arange(start, stop) // BLOCK], BLOCK, axis=1)[:, : array.shape[1]]
    return E4M3[array] * blocks


def _lazy_pair(compute):
    """Two functions that return the two parts of `compute()`. It runs once for both, and its
    result is freed when both functions have returned their part."""
    result, taken = [], set()

    def part(index):
        def get():
            if not result:
                result.append(compute())
            value = result[0][index]
            taken.add(index)
            if len(taken) == 2:
                result.clear()
                taken.clear()
            return value

        return get

    return part(0), part(1)


def quantize(config, tensors, weight_format):
    """The config and tensors of the fp8 or q4_0 file for a bf16 model. A model with tied
    embeddings gets a copy of its embedding table as `lm_head`. Each quantized matrix and its
    scales are functions, which quantize it when `write_weight_file` gets to it."""
    assert config["weight_format"] == WEIGHT_FORMATS["bf16"], "the model is quantized already"
    tensors = dict(tensors)
    if config["tied_embeddings"]:
        tensors["lm_head"] = tensors["embed"]
    config = dict(config, weight_format=WEIGHT_FORMATS[weight_format], tied_embeddings=0)
    for name in [n for n in tensors if is_matrix(n)]:
        dtype = matrix_dtype(config, name)
        quantizer, scale_dtype = (quantize_q4_0, "f16") if dtype == "u4" else (quantize_fp8, "f32")
        codes, scales = _lazy_pair(lambda entry=tensors[name], quantizer=quantizer: quantizer(as_f32(resolve(entry))))
        tensors[name] = lambda codes=codes, dtype=dtype: (dtype, codes())
        tensors[name + ".scale"] = lambda scales=scales, scale_dtype=scale_dtype: (scale_dtype, scales())
    return config, tensors


def read_safetensors(model_dir):
    """Return a dict from Hugging Face tensor name to (dtype name, numpy array) for a model folder.

    Handles one `model.safetensors` file and a set of shards listed in
    `model.safetensors.index.json`. The arrays are memory maps of the files.
    """
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    storage = {"BF16": ("bf16", np.uint16), "F32": ("f32", np.float32)}
    tensors = {}
    for file in files:
        data = np.memmap(model_dir / file, dtype=np.uint8, mode="r")
        (header_bytes,) = struct.unpack_from("<Q", data, 0)
        header = json.loads(bytes(data[8 : 8 + header_bytes]))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            assert info["dtype"] in storage, f"{name}: dtype {info['dtype']} is not supported"
            dtype, numpy_type = storage[info["dtype"]]
            begin, end = info["data_offsets"]
            raw = np.frombuffer(data, dtype=numpy_type, count=(end - begin) // np.dtype(numpy_type).itemsize,
                                offset=8 + header_bytes + begin)
            tensors[name] = (dtype, raw.reshape(info["shape"]))
    return tensors


def config_from_hugging_face(model_dir):
    """The config of the weight file, from the `config.json` of a model folder."""
    return config_from_json(json.loads((Path(model_dir) / "config.json").read_text()))


def config_from_json(hf):
    """The config of the weight file, from the `config.json` of a Qwen2 or a Qwen3 MoE model."""
    assert hf["model_type"] in ("qwen2", "qwen3_moe"), f"model_type {hf['model_type']} is not supported yet"
    moe = hf["model_type"] == "qwen3_moe"
    if moe:
        assert hf["decoder_sparse_step"] == 1 and not hf["mlp_only_layers"], "each layer must have experts"
        assert hf["norm_topk_prob"], "the weights of the chosen experts must be scaled to sum to 1"
        assert not hf.get("attention_bias", False), "Q, K and V of a Qwen3 MoE model have no bias"
    assert hf.get("rope_scaling") is None, "RoPE scaling is not supported yet"
    eos = hf.get("eos_token_id")
    return {
        "hidden_size": hf["hidden_size"],
        "intermediate_size": hf["intermediate_size"],
        "num_layers": hf["num_hidden_layers"],
        "num_q_heads": hf["num_attention_heads"],
        "num_kv_heads": hf["num_key_value_heads"],
        "head_dim": hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
        "vocab_size": hf["vocab_size"],
        "max_positions": hf["max_position_embeddings"],
        "tied_embeddings": int(hf.get("tie_word_embeddings", False)),
        "qkv_bias": int(not moe),
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "weight_format": WEIGHT_FORMATS["bf16"],
        "num_experts": hf["num_experts"] if moe else 0,
        "experts_per_token": hf["num_experts_per_tok"] if moe else 0,
        "moe_intermediate_size": hf["moe_intermediate_size"] if moe else 0,
        "qk_norm": int(moe),
        "rope_theta": float(hf["rope_theta"]),
        "rms_norm_eps": float(hf["rms_norm_eps"]),
    }
