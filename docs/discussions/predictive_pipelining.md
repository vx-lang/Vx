# Predictive pipelining: several tokens for one read of the weights

Status: design, November 2026. §11 gives the results. Issues: vx-lang/qwen-vx#59 (this design),
vx-lang/Vx#3625 (the time between kernels), vx-lang/Vx#3626 (the wait for the GPU).

## 1. The goal

A generated token of the 30B Qwen app (vx-lang/qwen-vx, Qwen3-Coder-30B-A3B) takes 22.0 ms on the
M4. It reads 1.84 GB, and the GPU reads at most 104 GB/s, so no token can take less than 17.7 ms.
llama.cpp takes 22.1 ms. Kernel work (#3625) can bring a token to about 18.5 ms, but not below
17.7 ms.

The only way below that floor is more than one token for each read of the weights. The CPU
predicts the next tokens (a draft). The GPU then runs the model once over all of them and keeps
the ones that the model itself would have chosen. This is speculative decoding. In code, many
tokens are easy to predict: lines copied from the prompt, closing brackets, indentation. So a
draft can often run 5 or more tokens ahead.

The target: 1.25 times the tokens per second of plain decoding on code tasks, with the same
output as greedy decoding.

## 2. What the measurements on the M4 allow

These numbers are from the M4 (32 GB) on 2026-11-17 (qwen-vx#34, Vx#3626, and the probes for
this design):

| Fact | Number |
| --- | --- |
| GPU read speed alone | 104 GB/s |
| 8 CPU threads alone | 114.7 GB/s |
| CPU and GPU together | 122.6 GB/s (GPU 55.8 + CPU 66.8) |
| GPU speed while 2 CPU threads read 48 GB/s | 68 GB/s |
| A round trip CPU -> GPU -> CPU, plain Metal | 78 to 90 µs |
| A launch and a host read, Vx today | 160 to 250 µs |
| llama.cpp on the CPU alone | 38 tokens/s, against 43.4 on its GPU |

Two rules follow:

1. **The draft must read almost no memory while the GPU works.** The CPU and the GPU share one
   memory bus. A draft model on the CPU that reads 48 GB/s takes a third of the GPU's speed. A
   draft from the tokens already in the context reads almost nothing.
1. **Hand off once for each round, not once for each layer.** A CPU and GPU split of each layer
   costs 48 handoffs a token, at least 3.8 ms. That is more than the 2.6 ms that the extra
   bandwidth saves. One handoff for each round of several tokens costs less than 1%.

## 3. A round

1. The CPU makes a draft of k tokens d1 ... dk from the context.
1. The GPU runs the model over n = k + 1 positions: the last accepted token, then d1 ... dk. It
   writes the K and V of each position into the cache, and the greedy choice at each position.
1. The host reads the n choices (one handoff). The host accepts the draft token di while it equals the
   choice at the position before it. The first choice that differs, or the choice after dk, is
   one more token that the model made itself. So a round gives from 1 to n tokens.
1. The cache keeps the positions of the accepted tokens. The next round writes over the rest, so
   there is nothing to undo.

"Pipelining" is the CPU and the GPU working at the same time: the CPU keeps the drafts and the
indexes of the context up to date while the GPU runs a round.

## 4. The cost of a round

The 2026-11-17 profile (qwen-vx#34) gives the bytes of a token:

| Part | Bytes | For n positions |
| --- | --- | --- |
| QKV, attention output, lm_head, router | 788 MB | read once |
| One expert of one layer | 2.65 MB | each expert that any of the n positions uses, once |
| The KV cache | 98 KB for each position of context (an estimate from the shapes, f16) | once, if attention reads each block once for all n queries |

So a round reads `788 MB + 2.65 MB x U(n) + KV`, where `U(n)` is the sum over the 48 layers of the
number of different experts that the n positions use. One position uses 8 experts in each layer.

The experts make speculation harder on this model. Each position picks its own 8 of 128 experts
in each layer. If the picks were independent, the bytes of a round would be:

| n | Experts in a layer | Bytes of a round | Times one token, short context | Times one token, 8,000 context |
| --- | --- | --- | --- | --- |
| 1 | 8.0 | 1.81 GB | 1.00 | 1.00 |
| 2 | 15.5 | 2.76 GB | 1.53 | 1.37 |
| 4 | 29.1 | 4.49 GB | 2.49 | 2.04 |
| 6 | 41.1 | 6.02 GB | 3.33 | 2.63 |
| 8 | 51.6 | 7.35 GB | 4.07 | 3.14 |

So a round of 4 positions must give 2.5 tokens to break even at a short context, and 2.0 tokens
at 8,000. Nearby tokens of real text pick more of the same experts, so the real numbers are
smaller. How much smaller decides the design, so we measure `U(n)` first (§8, step 0.1).

A long context helps. All n queries read the same KV cache, and the cache is 786 MB at 8,000
positions. So the share of bytes that the n positions use together grows with the context.

## 5. The output stays the same

Greedy decoding comes first. The output of the app with speculation must be the same tokens as
without it. This needs one property of the kernels that run a round: the logits of a position do
not depend on n (batch invariance). The sums of a row must run in the same order for every n.

- The app gets a test that generates with and without speculation and compares the tokens.
- Each new kernel for n positions gets a fixture that compares its rows with the kernel for 1
  position, bit for bit.
- Sampling at a temperature above 0 comes later, with the standard rule that accepts a draft
  token with probability min(1, p / q).

## 6. The draft

The draft runs on the CPU as Vx host code. It reads no weights.

1. **Copy from the context.** An index of the context (the prompt and the output so far) finds the
   longest earlier match of the last tokens. The tokens after that match are the draft. A longer
   match gives a longer draft, as in SuffixDecoding. Edits of a file and code that repeats names
   and signatures give long matches.
1. **Rules of code syntax.** A stack of open brackets and the indentation of the current block give
   the closing brackets and the indentation after a newline. These come from the text of the
   tokens, so the rules do not depend on the tokenizer.
1. **A length that pays.** Each extra draft token adds experts to the round (§4). The draft stops
   when the chance that the model accepts the next token is below its share of the cost. The chance
   comes from how often the model accepted each kind of draft in this session. The cost comes from the
   measured `U(n + 1) - U(n)`.
1. **A small model, later.** If the measurements show tokens that are easy to predict but not
   covered by 1 and 2, a small model can draft them. It must read less than about 2% of the bytes
   of a round, so that the GPU keeps its speed (§2).

## 7. The round on the GPU

- **First version:** the app's prefill path (`pf_chunk` in `src/prefill.vx`) already runs n
  positions with one read of the weights, and it groups the positions by expert. It needs the
  greedy choice at every position, not only the last one.
- **Then, if the first version is slow at small n:** kernels in `std::nn` for n of 2 to 8. A Q4 or
  Q6 product of n vectors loads each weight once and keeps n sums for each row. The experts read
  each expert of the union once. Attention reads each block of the KV cache once for all n
  queries. We tuned the prefill kernels for 2,000 positions, so at n = 4 they may waste most of
  each tile.
- At n = 8, a product does 32 floating point operations for each byte of Q4 weights. At 104 GB/s
  that is about 3.3 TFLOPS, which may be above what the M4's GPU can do. So n may stop at 4 to 6.

## 8. Plan

Each step ends with a number, and the gate after step 0 decides whether the rest happens.

**Step 0: measure, with no change to the app.**

- 0.1 Data: for 12 code prompts of 4 kinds (edit a file, write new code, complete a file, explain
  code), the greedy output of the model, 512 tokens each. Also the 8 experts of every position in
  every layer. A small tool on llama.cpp's library records both, with the same Q4_0 model file.
- 0.2 A simulator: the tokens that each kind of draft would get accepted, for each length. With
  `U(n)` from the router data and the bytes of §4, the time of a token for each prompt.
- 0.3 The time of `pf_chunk` for n = 1 to 8 at 0, 2,000 and 8,000 positions of context. This is
  the cost of a round with today's kernels.
- **Gate:** 1.25 times plain decoding or better, predicted for the edit and complete prompts, and no
  loss on the others. A draft length that adapts can always fall back to n = 1.

**Step 1:** the loop of §3 in the app, with the copy draft and `pf_chunk` for the round. The test
of §5. Times against plain decoding and against llama.cpp, with and without llama.cpp's own
lookup decoding (`llama-lookup`) on the same prompts.

**Step 2:** the kernels for small n in `std::nn` if step 0.3 shows a need for them. The syntax
rules and the length that pays.

**Step 3:** a small draft model if step 0.2 shows a need, and sampling.

## 9. Risks

- Nearby tokens may pick very different experts. Then speculation pays only on prompts that copy
  a lot, such as edits of a file.
- Batch invariance may cost speed in the kernels. If it does, the output is still the greedy
  output of the kernels for n positions, but it may differ from plain decoding in rare ties.
- A fair comparison needs care. `llama-bench` measures plain decoding. llama.cpp also has lookup
  decoding, so step 1 reports both.

## 10. Related work

- Speculative decoding: Leviathan et al. (2023), and Chen et al. (2023).
- Drafts from the context with no model: prompt lookup decoding (Saxena, 2023), SuffixDecoding
  (Oliaro et al., NeurIPS 2025, arXiv 2411.04975), and Cursor's speculative edits (2024).
- Tokens that a grammar forces: SGLang's jump-forward decoding (LMSYS, February 2024).
- Heads that predict several tokens: multi-token prediction (Gloeckle et al., ICML 2024).
- Speculation for mixtures of experts: MoE-Spec (arXiv 2602.16052) and EcoSpec (arXiv 2607.12696).
  Both report that the union of experts can make a round cost more than it saves.

## 11. Results

The app runs the rounds of §3 with the copy draft, at most 7 draft tokens (`--draft 7`). The
kernels of a round are the `_rows` kernels of `std::nn` (step 2). They give the same bits as the
kernels for one position, so both modes gave the same text on all 12 prompts. Plain decoding runs 8
tokens before each read of the result. A round must read its result before the next draft.

These numbers are from the M4 on 2026-11-19: 12 prompts, 512 tokens each, 2 runs, in ms for each
token (geometric mean). llama.cpp runs with `--ignore-eos`.

| Prompts | Vx plain | Vx `--draft 7` | Vx speedup | llama.cpp plain | `llama-lookup` |
| --- | --- | --- | --- | --- | --- |
| Edit a file | 24.09 | 13.09 | 1.84 | 25.50 | 12.55 |
| Complete a file | 23.61 | 23.77 | 0.99 | 25.57 | 26.24 |
| Write new code | 21.91 | 22.32 | 0.98 | 23.93 | 29.27 |
| Explain code | 23.72 | 21.95 | 1.08 | 25.29 | 26.53 |
| All 12 | 23.32 | 19.76 | 1.18 | 25.06 | 22.49 |

- The edits passed the gate of 1.25 times. The completions did not.
- On new code and completions, the drafts give few extra tokens, and a round costs a little more
  than a token. There, the drafts lose 1% to 2%.
- Over all 12 prompts, Vx with drafts is 1.14 times as fast as `llama-lookup`. On edits,
  `llama-lookup` is 4% faster.
