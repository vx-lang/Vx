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

In a file whose config has fp8 = 1, each weight matrix of MATRICES is f8e4m3 codes, one byte for
each weight. The tensor after it, `<name>.scale`, has one f32 scale for each block of 128 x 128
weights. A weight is the value of its code times the scale of its block. Such a file always has
`lm_head`, also for a model whose output projection is its embedding table.

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
    "fp8",
]
# The header has room for 16 integers. The unused ones are 0, for fields that later models add.
CONFIG_INT_SLOTS = 16

DTYPE_CODES = {"f32": 0, "bf16": 1, "f8e4m3": 2}
DTYPE_NAMES = {code: name for name, code in DTYPE_CODES.items()}
DTYPE_BYTES = {"f32": 4, "bf16": 2, "f8e4m3": 1}
STORAGE = {"f32": np.float32, "bf16": np.uint16, "f8e4m3": np.uint8}

# The weight matrices that an fp8 file stores as f8e4m3 codes. The kinds of a layer end the name.
MATRICES = ("qkv", "attn_out", "gate_up", "down", "lm_head")
BLOCK = 128


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
    """Whether an fp8 file stores the tensor `name` as f8e4m3 codes with block scales."""
    return name.rsplit(".", 1)[-1] in MATRICES


def tensor_names(config):
    """The names of the tensors in the weight file, in file order."""
    assert not (config["fp8"] and config["tied_embeddings"]), "an fp8 file has its own lm_head"
    names = ["embed"]
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        names += [prefix + "attn_norm", prefix + "qkv"]
        if config["qkv_bias"]:
            names.append(prefix + "qkv_bias")
        names += [prefix + "attn_out", prefix + "mlp_norm", prefix + "gate_up", prefix + "down"]
    names.append("final_norm")
    if not config["tied_embeddings"]:
        names.append("lm_head")
    if config["fp8"]:
        names = [n for name in names for n in ([name, name + ".scale"] if is_matrix(name) else [name])]
    return names


def tensor_shapes(config):
    """The shape of each tensor of the weight file, by name."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    q_rows = config["num_q_heads"] * head_dim
    qkv_rows = q_rows + 2 * config["num_kv_heads"] * head_dim
    inter = config["intermediate_size"]
    shapes = {"embed": (config["vocab_size"], hidden), "final_norm": (hidden,)}
    if not config["tied_embeddings"]:
        shapes["lm_head"] = (config["vocab_size"], hidden)
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        shapes[prefix + "attn_norm"] = (hidden,)
        shapes[prefix + "qkv"] = (qkv_rows, hidden)
        if config["qkv_bias"]:
            shapes[prefix + "qkv_bias"] = (qkv_rows,)
        shapes[prefix + "attn_out"] = (hidden, q_rows)
        shapes[prefix + "mlp_norm"] = (hidden,)
        shapes[prefix + "gate_up"] = (2 * inter, hidden)
        shapes[prefix + "down"] = (hidden, inter)
    if config["fp8"]:
        for name in [n for n in shapes if is_matrix(n)]:
            rows, cols = shapes[name]
            shapes[name + ".scale"] = (-(-rows // BLOCK), -(-cols // BLOCK))
    return shapes


def write_weight_file(path, config, tensors):
    """Write `tensors`, a dict from name to (dtype name, numpy array), in file order."""
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
    entries = []
    for index, name in enumerate(names):
        dtype, array = tensors[name]
        assert tuple(array.shape) == shapes[name], f"{name}: shape {array.shape}, expected {shapes[name]}"
        assert array.dtype == STORAGE[dtype], f"{name}: storage {array.dtype} for {dtype}"
        nbytes = array.size * DTYPE_BYTES[dtype]
        dims = list(array.shape) + [0] * (4 - array.ndim)
        entry = name.encode("ascii").ljust(64, b"\0")
        assert len(entry) == 64, f"{name}: the name is longer than 64 bytes"
        entry += struct.pack("<8q", DTYPE_CODES[dtype], array.ndim, *dims, offset, nbytes)
        header[TABLE_OFFSET + index * ENTRY_BYTES : TABLE_OFFSET + (index + 1) * ENTRY_BYTES] = entry
        entries.append((offset, array))
        offset += -(-nbytes // ALIGNMENT) * ALIGNMENT

    with open(path, "wb") as out:
        out.write(header)
        for start, array in entries:
            out.seek(start)
            out.write(np.ascontiguousarray(array).tobytes())
        out.truncate(offset)


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
        shape = tuple(dims[:rank])
        storage = STORAGE[DTYPE_NAMES[dtype]]
        raw = np.frombuffer(data, dtype=storage, count=nbytes // np.dtype(storage).itemsize, offset=offset)
        tensors[name] = (DTYPE_NAMES[dtype], raw.reshape(shape))
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


def widen(tensors, name):
    """The float32 values of the tensor `name` of `read_weight_file`, also an fp8 matrix."""
    dtype, array = tensors[name]
    if dtype != "f8e4m3":
        return as_f32((dtype, array))
    scales = tensors[name + ".scale"][1]
    rows, cols = array.shape
    blocks = np.repeat(np.repeat(scales, BLOCK, axis=0), BLOCK, axis=1)[:rows, :cols]
    return E4M3[array] * blocks


def to_fp8(config, tensors):
    """The config and tensors of the fp8 file for a bf16 model. A model with tied embeddings gets
    an fp8 copy of its embedding table as `lm_head`."""
    assert not config["fp8"], "the model is fp8 already"
    tensors = dict(tensors)
    if config["tied_embeddings"]:
        tensors["lm_head"] = tensors["embed"]
    config = dict(config, fp8=1, tied_embeddings=0)
    for name in [n for n in tensors if is_matrix(n)]:
        codes, scales = quantize_fp8(as_f32(tensors[name]))
        tensors[name] = ("f8e4m3", codes)
        tensors[name + ".scale"] = ("f32", scales)
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
    """The config of the weight file, from the `config.json` of a Qwen2 model folder."""
    hf = json.loads((Path(model_dir) / "config.json").read_text())
    assert hf["model_type"] == "qwen2", f"model_type {hf['model_type']} is not supported yet"
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
        "qkv_bias": 1,
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "fp8": 0,
        "rope_theta": float(hf["rope_theta"]),
        "rms_norm_eps": float(hf["rms_norm_eps"]),
    }
