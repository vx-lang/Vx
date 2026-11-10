# Qwen coder models in Vx

This app runs Qwen coder models with a program written in Vx, on the GPU of an Apple Silicon
Mac. The plan and its milestones are in [#3094](https://github.com/vx-lang/Vx/issues/3094).

The Vx program does not generate text yet. This folder now has the Python tools that prepare the
weights and compute the values that the Vx program must match, and the Vx module that loads the
weights into GPU memory.

## Prepare a model

The scripts need Python 3 with NumPy. `--prompt` also needs the `tokenizers` package, and
`check_against_mlx.py` needs `mlx-lm`.

1. Download a model from Hugging Face, for example
   [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), into a
   folder such as `~/models/Qwen2.5-Coder-0.5B-Instruct`.

1. Convert it into the weight file that the Vx program reads:

   ```sh
   python3 apps/qwen/scripts/convert.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw
   ```

1. Generate a few tokens with the NumPy reference:

   ```sh
   python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw \
       --tokenizer ~/models/Qwen2.5-Coder-0.5B-Instruct --prompt "def fibonacci(n):" --steps 8
   ```

`make_tiny_model.py` writes a model with the same layers and random weights. It is small enough
for tests that cannot download a real model.

## The scripts

| Script | What it does |
| --- | --- |
| `scripts/convert.py` | Converts a Hugging Face Qwen2 folder into a weight file. |
| `scripts/reference.py` | Computes the logits and the greedy tokens with NumPy, in float32. |
| `scripts/check_against_mlx.py` | Compares `reference.py` with MLX on the same model. |
| `scripts/make_tiny_model.py` | Writes a tiny model with random weights. |
| `scripts/qwen_weights.py` | Reads and writes the weight file. Its first comment describes the format. |

## The Vx code

`src/weights.vx` loads a weight file into GPU memory, for the kernels of `std::nn`. A program
imports it as `qwen::weights`, with `vxc --package qwen=apps/qwen/src`.

The tests are in `tests/`. Run them from the root of the Vx repository, with `vxc`, `FileCheck`
and a Python 3 that has NumPy on `PATH`:

```sh
apps/qwen/tests/run.sh
```

`tests/print_weights.vx` also checks a real model. Its lines must equal those of
`reference.py --tensors`:

```sh
vxc apps/qwen/tests/print_weights.vx --package qwen=apps/qwen/src --host default \
    --machine fleet/m4-uma.vx -- qwen-0.5b.vxw
python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw --tensors
```

## The check of the reference

`check_against_mlx.py` compared the reference with MLX in float32 over a 5-token prompt and 32
greedy steps. Both chose the same tokens.

| Model | Largest logit difference | Time on an M4 | Peak memory |
| --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 8.7e-05 | 7 s | 5 GB |
| Qwen2.5-Coder-7B-Instruct | 6.3e-05 | 19 min | 15 GB |

The check fails, as it should, when RoPE uses another theta, when the causal mask lets a token
see the next one, when the program does not add the Q, K and V bias, and when the converter swaps
K and V.
