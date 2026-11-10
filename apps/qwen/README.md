# Qwen coder models in Vx

This app runs Qwen coder models with a program written in Vx, on the GPU of an Apple Silicon
Mac. The plan and its milestones are in [#3094](https://github.com/vx-lang/Vx/issues/3094). The
mixture-of-experts (MoE) model Qwen3-Coder-30B-A3B is the next milestone,
[#3151](https://github.com/vx-lang/Vx/issues/3151). The Vx program runs MoE files in bf16 and in
Q4_0, so far checked on the tiny MoE model.

The program generates tokens with greedy decoding, from a prompt given as token ids. A tokenizer
in Vx comes later, so `reference.py` turns text into token ids for now.

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

1. Generate a few tokens with the NumPy reference. It prints the prompt as token ids:

   ```sh
   python3 apps/qwen/scripts/reference.py qwen-0.5b.vxw \
       --tokenizer ~/models/Qwen2.5-Coder-0.5B-Instruct --prompt "def fibonacci(n):" --steps 8
   ```

1. Generate the same tokens with the Vx program, on the GPU:

   ```sh
   vxc apps/qwen/src/main.vx --package qwen=apps/qwen/src --host default \
       --machine fleet/m4-uma.vx -- qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
   ```

`make_tiny_model.py` writes a model with the same layers and random weights. It is small enough
for tests that cannot download a real model. With `--moe`, it writes a tiny model with the layers
of Qwen3-Coder-30B-A3B.

## The scripts

| Script | What it does |
| --- | --- |
| `scripts/convert.py` | Converts a Hugging Face Qwen2 or Qwen3 MoE folder into a weight file. |
| `scripts/reference.py` | Computes the logits and the greedy tokens with NumPy, in float32. |
| `scripts/check_against_mlx.py` | Compares `reference.py` with MLX on the same model. |
| `scripts/compare.py` | Compares the output of the Vx program with the output of `reference.py`. |
| `scripts/make_tiny_model.py` | Writes a tiny model with random weights. |
| `scripts/qwen_weights.py` | Reads and writes the weight file. Its first comment describes the format. |
| `scripts/test_qwen_weights.py` | Tests the f8e4m3 and Q4_0 codes, their scales and the quantized weight files. |

## fp8 weights

`convert.py --fp8` stores each weight matrix as f8e4m3 codes, one byte for each weight, with one f32
scale for each block of 128 x 128. A weight is the value of its code times the scale of its block.
The norms, the biases and the embedding table stay bf16, so a token reads about half the bytes of
bf16. The Vx program reads an fp8 file like a bf16 file, and runs the fp8 kernels of `std::nn` for
it.

`reference.py` widens the codes and scales exactly to float32. With `--perplexity`, it measures
how well a model predicts a text. On the first 512 tokens of `scripts/tools/gen_error_index.py`:

| Model | bf16 perplexity | fp8 perplexity | Greedy tokens of fp8 |
| --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 20.054 | 20.030 | the same 64 as bf16 |
| Qwen2.5-Coder-7B-Instruct | 10.072 | 10.085 | the same 64 as bf16 |

On the fp8 file, the Vx program chose the same 64 greedy tokens as `reference.py`, with each logit
within 6e-05. One token of the 7B model took 95 ms in fp8 and 156 ms in bf16 on a quiet M4, which
is about 10.5 and 6.4 tokens per second.

## Q4_0 weights

`convert.py --q4` stores each weight matrix except `lm_head` in the Q4_0 format of llama.cpp. A
row has one f16 scale for each 32 weights, and each weight is a 4-bit code from 0 to 15. The
weight is (code - 8) times its scale. The codes round as llama.cpp rounds them, and a test checks
them against the loop of llama.cpp. For Qwen2.5-Coder-7B-Instruct, the codes and scales of all
197 Q4_0 tensors were the same bits as in the Q4_0 GGUF file that Qwen publishes. `lm_head` is fp8, as llama.cpp also keeps its output matrix
above 4 bits.

The Vx program reads a Q4_0 file like the others, and runs the Q4_0 kernels of `std::nn` for it.
Of four 4-bit formats, Q4_0 was the closest to bf16 on the 7B model. The measurement is in
[#3151](https://github.com/vx-lang/Vx/issues/3151#issuecomment-6763017034).

## The Vx code

| File | What it does |
| --- | --- |
| `src/weights.vx` | Loads a weight file into GPU memory, for the kernels of `std::nn`. |
| `src/model.vx` | Makes the tensors and the KV cache once, and runs one token through all layers. |
| `src/main.vx` | Generates tokens with greedy decoding, and prints the lines of `reference.py`. |

A program imports the modules as `qwen::weights` and `qwen::model`, with
`vxc --package qwen=apps/qwen/src`.

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

## The check of the Vx program

`compare.py` compared the Vx program with `reference.py` over 64 greedy steps, after the prompt
`def fibonacci(n):` and a new line. Both chose the same tokens.

| Model | Largest logit difference |
| --- | --- |
| tiny model | 4.7e-05, over 16 steps |
| tiny model, Q4_0 | 4.8e-05, over 16 steps |
| tiny MoE model | 4.7e-05, over 16 steps |
| tiny MoE model, Q4_0 | 5.0e-05, over 16 steps |
| Qwen2.5-Coder-0.5B-Instruct | 5.3e-05 |
| Qwen2.5-Coder-7B-Instruct | 6.0e-05 |

The tiny model runs in CI with `tests/generate.vx`. That test fails when RoPE uses another theta,
when a token runs at the wrong position, and when the two norms of a layer swap their weights.
For the tiny MoE model, it fails when the router reads the hidden state before its norm, when a
token uses one expert in place of two, and when the Q and K norms are missing or swapped.

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

For the tiny MoE model, `make_tiny_model.py --moe --hugging-face DIR` also writes the model as a
Hugging Face folder, which MLX reads:

```sh
python3 apps/qwen/scripts/make_tiny_model.py --moe --hugging-face tiny-moe tiny-moe.vxw
python3 apps/qwen/scripts/check_against_mlx.py --tokens 17,200,3,95,41 tiny-moe tiny-moe.vxw
```

Over 32 greedy steps, both chose the same tokens, and the largest logit difference was 8.7e-06.
The check fails when the router chooses the smallest experts or one expert too few, or when it
does not scale the weights of the chosen experts to sum to 1. It also fails when an expert uses
the down matrix of the next expert, when the converter puts the up rows before the gate rows, or
when the Q and K norms are missing, swapped or after RoPE.
