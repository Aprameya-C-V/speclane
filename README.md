# SpecLane: Serving Small Hybrid-Attention Models Under Agent Traffic

**A reproducible vLLM configuration study of Qwen3.8-27B-FP8 and Qwen3.5-9B.**
It defines a vanilla baseline, changes one engine setting at a time, and measures throughput,
tail latency and answer quality under traffic shaped like agent loops.

This repository holds the profiles, the workloads, the Modal sweep runner, the deployable
endpoint, the quality check, the raw per-run results and the paper source. Every number below is
generated from the result files in `infer/results/` by `paper/make_results.py`, except the
GSM8K quality check, which is produced by `scripts/gsm8k_endpoint.py`.

Technical report: [`docs/index.html`](docs/index.html), served at
<https://aprameya-c-v.github.io/speclane/>. Paper: `paper/main.pdf` (LaTeX source in `paper/`).
Detailed measurements, bugs and withdrawn claims: [`FINDINGS.md`](FINDINGS.md). Agent-harness
pilot (dropped, inconclusive, code not kept): [`DEVIATIONS.md`](DEVIATIONS.md).

> The techniques studied (speculative decoding, FP8 KV cache, scheduler limits) are existing
> engine features. This is a measurement study, not a new method.

## Results

`agent_sessions` workload, output tokens per second, single run per cell. vLLM 0.33.0,
Qwen3.8-27B-FP8 on one H100 80 GB, Qwen3.5-9B on one L40S 48 GB. Change from baseline in
parentheses; c is concurrency.

| model | profile | c = 1 | c = 16 | p99 TTFT, c = 16 (s) |
|---|---|---|---|---|
| Qwen3.8-27B-FP8 | baseline | 78 | 800 | 1.09 |
| | **MTP-3** | **175 (+124%)** | **1199 (+50%)** | **0.94** |
| | MTP-3 + FP8 KV cache | 163 (+110%) | 1007 (+26%) | 1.20 |
| | FP8 KV cache | 76 (−2%) | 684 (−14%) | 1.46 |
| Qwen3.5-9B | baseline | 43 | 475 | 0.95 |
| | MTP-1 | 61 (+44%) | 617 (+30%) | 0.95 |
| | **MTP-2** | **75 (+77%)** | **753 (+59%)** | **0.74** |
| | FP8 KV cache | 43 (+0%) | 467 (−2%) | 1.42 |

Answer quality is preserved. Same 200 GSM8K test questions (fixed seed), temperature 0,
thinking off, exact match on the final number:

| model | baseline | MTP (recommended profile) |
|---|---|---|
| Qwen3.8-27B-FP8 | 0.940 | 0.935 (MTP-3) |
| Qwen3.5-9B | 0.900 | 0.910 (MTP-2) |

The differences are one and two questions out of 200, within sampling noise. Baselines
reproduced within about 2% across separate boots, so differences below about 5% are not claimed.

## What is new here

1. **A baseline definition and a one-change-at-a-time protocol.** Vanilla means engine defaults
   plus only the flags needed to start and a fixed 16,384-token context, so engine comparisons
   are reproducible.
2. **A startup failure of vanilla vLLM on a hybrid-attention model.** Qwen3.8-27B-FP8 does not
   start on an H100 because the default `max_num_seqs` (1024) exceeds the 745 recurrent-state
   cache blocks that fit. `--max-num-seqs 512` resolves it and is part of the baseline.
3. **MTP speculation is the main lever, the FP8 KV cache is not.** The models' built-in
   multi-token-prediction head raised throughput 50 to 124% (27B) and up to 77% (9B) at no
   measured quality cost; the FP8 KV cache lowered throughput and raised p99 time to first
   token at this scale.
4. **An honest negative result.** A prompt-level agent harness on Terminal-Bench 2.1 and
   τ²-bench showed no reliable gain at the sample sizes affordable; it is reported in
   `DEVIATIONS.md` and the paper claims nothing from it.

## Repo layout

```
infer/models.py        model registry: Hugging Face repo, GPU, price, serve command
infer/profiles.py      baseline and one-change profiles for vLLM and SGLang, flag helpers
infer/workloads.py     shared_prefix and agent_sessions synthetic workloads
infer/bench.py         closed-loop load generator: TTFT, TPOT, throughput
infer/quality.py       GSM8K exact-match guard (sweep-side)
infer/sweep.py         Modal sweep runner: one GPU container per model, results to JSON
infer/endpoint.py      deployable key-protected vLLM endpoint (PROFILE=baseline|tuned)
infer/gateway.py       OpenAI-compatible gateway with routing and failover (unit-tested)
infer/results/         raw per-run results; round2/ holds the partial second round
scripts/gsm8k_endpoint.py  GSM8K check against a deployed, key-protected endpoint
paper/                 LaTeX source, make_results.py (tables, figure), main.pdf
docs/                  technical report (index.html) and the illustration prompt
tests/                 45 tests against a fake backend
FINDINGS.md            every table, the bugs found and the claims withdrawn
DEVIATIONS.md          the dropped agent-harness pilot and its deviations
```

## Reproduce

```bash
# one-time: authenticate Modal, then pick a model
export MODEL_KEY=qwen3.8-27b-fp8            # or qwen3.5-9b

# stage weights (CPU only) and check that every candidate flag exists in this vLLM
modal run infer/modal_app.py::download
modal run infer/sweep.py::validate --engine vllm

# sweep profiles and workloads, with the GSM8K guard
modal run infer/sweep.py::run --engine vllm --profiles baseline,mtp-3,kv-fp8 \
    --workloads shared_prefix,agent_sessions --concurrency 1,16 --quality 100

# serve a key-protected endpoint: baseline or tuned (MTP)
VLLM_API_KEY=... MODEL_KEY=qwen3.8-27b-fp8 PROFILE=tuned modal deploy infer/endpoint.py
uv run --with pandas --with pyarrow --with huggingface_hub --with aiohttp \
    python scripts/gsm8k_endpoint.py <endpoint_url> qwen3.8-27b-fp8 200

# rebuild the paper tables, figure and PDF from the result files
cd paper && uv run --with matplotlib python make_results.py && tectonic main.tex

# tests
.venv/bin/python -m pytest
```

Deployed endpoints bill continuously (`min_containers=1`). Stop them with `modal app stop`.

## Notes on comparability

- **Single runs.** Each cell is one run of 24 requests (96 for 16 concurrent sessions). The
  nearest-rank p99 on 24 requests is the maximum. Only large effects are claimed.
- **Synthetic prompts.** Random-word prompts control caching but have unnatural acceptance rates
  for speculative decoding; real agent traces may differ.
- **One engine version.** Results are for vLLM 0.33.0. SGLang profiles are defined in
  `infer/profiles.py` but were never run.
- **Narrow quality check.** One 200-question arithmetic sample at temperature 0 with thinking
  disabled. It does not cover long agent trajectories, and the FP8 KV cache was not checked.
- **No production claim.** Long-running stability, failure rates and cost under real traffic
  were not measured.

## Status

Complete as a small study. All Modal apps are stopped and the model-weights volume has been
deleted (weights re-download from Hugging Face on the next run). The paper still has
placeholders: authors and date, repository URL, driver and model revision hashes, and a
related-work pass that checks each citation.
