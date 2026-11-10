# Findings

Every throughput and latency number here is generated from the result files in
`infer/results/` by `paper/make_results.py`. The GSM8K numbers come from
`scripts/gsm8k_endpoint.py` run on live endpoints. Agent-study numbers come from
`scripts/score_main.py` and `scripts/score_final.py` over `jobs/` (not committed; see
`.gitignore`). Single runs unless stated.

## 0. Problems found along the way, and what was withdrawn

**Vanilla vLLM cannot start the 27B on an H100.** With defaults the engine stops during
CUDA-graph capture: `max_num_seqs (1024) exceeds available Mamba cache blocks (745)`. Each
decode sequence needs one recurrent-state block, and only 745 fit beside the weights and the
KV cache. `--max-num-seqs 512` resolves it and is part of the baseline, so "baseline" means
"defaults plus what is needed to start".

**An endpoint without tool-calling flags silently zeroes an agent run.** The first
Terminal-Bench pilot gave every task a reward of 0 with 0 tokens: the agent sent tool
definitions with `tool_choice: auto` and vLLM answered HTTP 400. Both profiles now carry
`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, verified
with a real tool-call request before any agent run. That pilot is withdrawn.

**The τ²-bench judge defaults to OpenAI.** Tasks that use an LLM-judged natural-language
assertion called `gpt-5.4-mini` on `api.openai.com` with no key, ended as
`infrastructure_error` and scored 0, in both arms. It made the engineered agent look worse on
dev retail (0.71 versus 0.83) until the cause was found. The judge now points at the vanilla
27B endpoint in every cell (`harness/tau2_judge.py`). Earlier τ² runs that hit it were rerun
or excluded.

**Modal's legacy image builder fails on plain Ubuntu task images.** It runs
`python -m pip` and the image has no `python` (`modal image logs` shows `python: not found`).
Five Terminal-Bench images failed identically in both arms until the run was started with
`MODAL_IMAGE_BUILDER_VERSION=2025.06`. Later `ConflictError`s came from running 30 sandboxes
at once; those trials are excluded, not counted as failures.

Lesson: an agent score is only as trustworthy as the plumbing under it. Every zero was
checked for an infrastructure cause before being read as a model result.

## 1. Throughput

Output tokens per second, change from baseline in parentheses. c is concurrency.

**Qwen3.8-27B-FP8, one H100**

| profile | shared_prefix c=1 | shared_prefix c=16 | agent_sessions c=1 | agent_sessions c=16 |
|---|---|---|---|---|
| baseline | 79 | 743 | 78 | 800 |
| kv-fp8 | 76 (−3%) | 612 (−18%) | 76 (−2%) | 684 (−14%) |
| **mtp-3** | **176 (+123%)** | **1297 (+75%)** | **175 (+124%)** | **1199 (+50%)** |
| mtp-3 + kv-fp8 | 167 (+111%) | 1018 (+37%) | 163 (+110%) | 1007 (+26%) |

**Qwen3.5-9B, one L40S**

| profile | shared_prefix c=1 | shared_prefix c=16 | agent_sessions c=1 | agent_sessions c=16 |
|---|---|---|---|---|
| baseline | 43 | 420 | 43 | 475 |
| kv-fp8 | 43 (−0%) | 399 (−5%) | 43 (+0%) | 467 (−2%) |
| mtp-1 | 61 (+41%) | 499 (+19%) | 61 (+44%) | 617 (+30%) |
| **mtp-2** | **71 (+65%)** | **644 (+53%)** | **75 (+77%)** | **753 (+59%)** |

## 2. Tail latency

p99 time to first token in seconds (nearest rank; on 24 requests this is the maximum).

**Qwen3.8-27B-FP8**

| profile | shared_prefix c=1 | shared_prefix c=16 | agent_sessions c=1 | agent_sessions c=16 |
|---|---|---|---|---|
| baseline | 0.35 | 0.53 | 0.27 | 1.09 |
| kv-fp8 | 0.36 | 1.47 | 0.27 | 1.46 |
| mtp-3 | 0.36 | 0.79 | 0.28 | 0.94 |
| mtp-3 + kv-fp8 | 0.37 | 1.51 | 0.29 | 1.20 |

**Qwen3.5-9B**

| profile | shared_prefix c=1 | shared_prefix c=16 | agent_sessions c=1 | agent_sessions c=16 |
|---|---|---|---|---|
| baseline | 0.53 | 0.69 | 0.41 | 0.95 |
| kv-fp8 | 0.58 | 1.39 | 0.43 | 1.42 |
| mtp-1 | 0.54 | 1.54 | 0.46 | 0.95 |
| mtp-2 | 0.57 | 0.84 | 0.45 | 0.74 |

Read together with section 1: a throughput gain can come with a larger tail. MTP-3 raised
27B throughput 75% at c=16 on `shared_prefix` but also moved p99 from 0.53 s to 0.79 s; the FP8
KV cache moved it to 1.47 s while lowering throughput, which is why it is not recommended.

## 3. Answer quality

Same 200 GSM8K test questions (seed 0), temperature 0, thinking disabled, exact match on the
final number, asked of the live vanilla and tuned endpoints.

| model | baseline | MTP | difference |
|---|---|---|---|
| Qwen3.8-27B-FP8 | 0.940 | 0.935 (MTP-3) | −1 question |
| Qwen3.5-9B | 0.900 | 0.910 (MTP-2) | +2 questions |

No sign that speculation changed answer quality on this sample. The check was not run for the
FP8 KV cache.

## 4. Baseline reproducibility

The same baseline profile, booted separately, reproduced within about 2% in throughput.
Differences below roughly 5% are therefore treated as noise. GSM8K on the baselines was
0.90 to 0.91 across boots.

## 5. A dropped agent-harness study

I also tested whether an engineered prompt harness plus the tuned engine improves agent
accuracy. It did not, at a sample size that could show it. Numbers are paired comparisons of
vanilla (default agent, vanilla engine) against engineered (engineered prompt, tuned engine),
excluding trials that never produced a valid outcome.

| set | n | vanilla | engineered | note |
|---|---|---|---|---|
| τ²-bench airline test (9B) | 19 | 0.84 | 0.84 | tie; interval about ±0.21 |
| τ²-bench retail test (9B) | 5 | 1.00 | 0.80 | one task worse |
| Terminal-Bench 2.1, first 10 test tasks (27B) | 10 | 0.30 | 0.40 | one task better, none worse |
| Terminal-Bench 2.1, final round | 8 | 2 solved | 2 solved | tie; the rest hit sandbox errors |
| Terminal-Bench 2.1, dev tasks (not tuned against) | 5 | 0.20 | 0.80 | 3 better, 0 worse |

The test-split result is no reliable difference. The dev signal is real but small-sample and
was not confirmed on test. Two attempted prompt fixes for a runaway-reasoning failure (output
cap hit on `adaptive-rejection-sampler`) did not help and were dropped. The paper makes no
agent-accuracy claim. Scale, cells and every deviation from the pre-registered plan are in
[`DEVIATIONS.md`](DEVIATIONS.md).

## 6. Not done

- SGLang was never benchmarked (profiles are defined, nothing was run).
- Profiles other than the baselines were not repeated, so run-to-run noise for them is unmeasured.
- FP8 KV cache quality was not measured.
- A second tuning round (adaptive verification, larger batched-token limits, async scheduling
  on top of MTP) was defined in `infer/profiles.py` but its sweep did not finish.
- No real agent traces, no long-running stability or cost measurement.
