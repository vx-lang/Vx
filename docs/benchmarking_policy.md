# Benchmarking policy

This policy applies to every speed number that Vx or its apps (qwen-vx, glm-vx) publish: release
notes, blog posts, READMEs, PR descriptions and social posts. A new number that does not follow it
does not go out. Numbers that were published before 2026-11-20 stay as they are.

The reader of a claim must be able to answer three questions from the claim and its data alone:
how fast, compared with what, and what else changed.

## 1. The claim

Write each headline claim in this form:

> **A** tok/s against **B** tok/s (**A/B**×), MODEL in PRECISION, at CONTEXT, on HARDWARE, batch N,
> against BASELINE VERSION.

For example: "46.8 against 45.0 generated tokens a second (1.04×), Qwen3-Coder-30B-A3B in Q4_0, after a
60-token prompt, on an M4 with 32 GB, batch 1, against llama.cpp 0.7.0."

- Give the absolute numbers of both sides. A ratio alone does not go out.
- Give the context length. Decode speed falls as the context grows. So for a model, give a table over
  context lengths, not one number.
- Keep prompt processing (prefill), time to the first token and decode speed apart. Do not merge them
  into one number.
- Say where Vx is slower. If Vx loses at some context length, prompt kind or batch size, that row goes
  in the same table.

## 2. A fair comparison

- **Same precision.** Both sides use the same weight format, KV-cache type and arithmetic. List the
  format of each kind of tensor for both sides. If the precision differs, it is a different claim.
- **Same work.** For memory-bound decode, give the bytes that each side reads for each token, and the
  time that this takes at the speed of memory. That is the floor that no program can beat.
- **Same quality.** Show that both sides compute the same model. Use identical tokens, or a
  perplexity or accuracy score measured with one program on both weight files.
- **The best baseline.** Run the other program as its own documentation says, at a recent release.
  When Vx uses a technique, compare with the same technique in the other program. For example, if
  Vx drafts tokens, compare with `llama-lookup`, not with plain `llama-completion`.
- **No tuning on the test set.** Tune thresholds and settings on one set of inputs, and measure on
  another. If that is not possible, say so next to the number.
- **The same inputs.** Both programs read the same prompt files, byte for byte, and generate the same
  number of tokens. Neither stops early at end-of-text.

## 3. Measurement

- **Release builds.** Use the release binary of each program. If that is not possible, record the
  commit and the build type.
- **Resolve every "auto".** Record the value that each `auto` or default setting chose, for example
  flash attention or the thread count.
- **At least 3 runs** of each case. Alternate the order of the programs between rounds, and wait for
  an idle machine before each run.
- **Keep every run**, outliers too. Report the median, the fastest and the slowest run. If a run is
  bad for a known reason, keep it in the data and say why it is out of the summary.
- **Record at the time of the run:** hardware model, OS version, driver, the versions and commits of
  both programs, every command line, the prompt token counts and the date. Missing facts cannot be
  added later.

## 4. Reporting

- **Two tables of setup.** One table lists what both sides share. The other lists what differs.
- **Summaries.** To summarize ratios over several inputs, use the geometric mean. Also give the
  number of inputs where Vx is faster, and the smallest and largest ratio.
- **Every input.** Give a row for each kind of input, not only the total.
- **Where the gain comes from.** List each change with its effect in ms, measured with and without it.
- **Limits.** Give one short list of what the measurement does not cover.
- **Commands.** Give the exact commands, so that anyone can run them again.

## 5. Speculative decoding

- Give the speedup against plain decoding of the same program on the same machine.
- Give the acceptance: accepted and drafted tokens, and accepted tokens for each check.
- Give the largest draft length and the temperature.
- Show that the output is the same as without drafts. Check every pair of runs.
- Name the source of the drafts: a draft model, the context, or extra heads.

## 6. Serving

For a server with many users, give the throughput for each GPU against the speed for each user
(tok/s/user), over a sweep of concurrency, as InferenceMAX does. Give the input and output lengths,
and the P50 and P99 of the time to the first token and of the time for each output token.

## 7. Where the data goes

Each published measurement gets a folder in the repository of the app that it measures, for example
`docs/benchmarks/2026-11-19-m4-decode/`:

- `README.md`: the report, as described in sections 1–6
- `runs.csv`: one row for each run
- `logs/`: the output of each run
- `prompts/` and `run.sh`: the inputs and the script
- `env.txt`: hardware, OS, driver, versions and commits

The release notes or the post links to this folder.

## Checklist

Before a number goes out:

- [ ] absolute numbers of both sides, and a table over context lengths
- [ ] the same precision, work and quality on both sides
- [ ] the best baseline, with the same technique
- [ ] tuning inputs apart from measured inputs, or a note
- [ ] 3 or more runs, all kept, with the median and range
- [ ] release builds, or commits and build types
- [ ] every `auto` setting resolved
- [ ] the losses in the same table
- [ ] the data folder, linked from the claim

## Sources

This policy follows these practices:

- Torsten Hoefler and Roberto Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve
  Ways to Tell the Masses when Reporting Performance Results", SC15
- Gernot Heiser, "Systems Benchmarking Crimes"
- SemiAnalysis InferenceMAX: throughput against interactivity, fixed input and output lengths,
  precision on each chart, public configurations
- MLPerf Inference: a quality target for quantized models, and P99 limits on TTFT and TPOT
- llama.cpp `llama-bench` and its Apple Silicon table: a fixed commit, model and command, pp512 and
  tg128 apart, the mean and standard deviation of 5 runs
- EAGLE and Spec-Bench: speedup and acceptance length, at a stated temperature, on one machine
