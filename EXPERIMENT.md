# Experiment plan (pre-registered 2026-11-09, before any model run)

Question: how much does a co-designed harness and inference engine improve accuracy, time and cost on agent tasks for
small open models, compared with a vanilla harness on a vanilla engine?

Nothing below has been run. Numbers marked "estimate" are guesses to be replaced by measurements.

## Pairs

| | Pair A | Pair B |
|---|---|---|
| Model | Qwen3.8-27B-FP8 on 1x H100 | Qwen3.5-9B on 1x L40S |
| Benchmark | Terminal-Bench 2.1 (89 tasks, Harbor, Modal sandboxes) | tau2-bench (airline, retail, telecom) |
| Vanilla harness | Harbor's built-in `pi` agent, default prompt and settings | tau2's default `llm_agent` (4-line instruction plus policy) |
| Engineered harness | pi with an engineered prompt template and settings (`prompt_template_path`, effort, turn limit) | a custom agent registered through tau2's `Registry.register_agent_factory` |
| Test set (locked) | 30 tasks, `tb21/test_tasks.txt` | the official `test` splits: airline 20, retail 40, telecom 40 |
| Dev set (tuning only) | 10 tasks, `tb21/dev_tasks.txt` | the official `train` splits: airline 30, retail 74, telecom 74 |
| Attempts per task, test | 3 | 4 (needed for pass^4) |

Task selection for Pair A is `tb21/select_tasks.py` (seeds 0 and 1, stratified by category, fingerprint
`49c8e993afea3525`). An earlier Terminal-Bench 4.0 selection (`tb4/`) is kept as an optional hard-mode check only.

Why Terminal-Bench 2.1 for the 27B: Qwen reports 73.0 for it (vendor-run, Terminus harness), so there is headroom in both
directions; it has no GPU tasks; most agent timeouts are 15 to 60 minutes with 1-CPU, 2 GB sandboxes, so attempts are cheap.
Terminal-Bench 3.0 and 4.0 tasks have much longer timeouts and heavier sandboxes, and open models score near zero there.

## Systems compared (per pair)

- S0 vanilla: stock harness + vanilla engine.
- S3 full system: engineered harness + tuned engine. **The primary comparison is S3 against S0.**
- S1 harness only and S2 engine only: ablations, run only if budget remains, in the order S1 then S2.

Vanilla engine = vLLM defaults plus the flags required to serve the model and a context length (131072 for Pair A, 32768
for Pair B; the same in every cell). Tuned engine = the profile chosen by the sweeps (`infer/sweep.py`) on workloads that
resemble these runs, subject to the quality guard (`infer/quality.py`), then frozen before any test run.

## Metrics

Primary:
- Pair A: resolve rate = mean over test tasks of the fraction of attempts that pass the task verifier.
- Pair B: pass^1 and pass^4 over the 100 test tasks, and per domain.

Secondary (reported, not used to declare success): pass@3 (Pair A); wall-clock and tokens per attempt; GPU and sandbox
cost per attempt; cost per resolved task; harness turns.

## Analysis

- Paired by task. Difference S3 minus S0 with a 95% interval from a bootstrap that resamples tasks (10,000 resamples).
- No claim of an effect unless the interval excludes zero. Intervals are reported even when they include zero.
- Per-domain and per-category breakdowns are exploratory.
- Infrastructure failures (sandbox crash, endpoint outage) are rerun and counted separately. Agent failures, timeouts and
  malformed-output failures count as failures.

## Integrity rules

1. Harness and prompt changes use the dev/train sets only. Test sets are not run until the harness is frozen.
2. Freeze = a git tag and a hash of the harness files, the engine profile, model revision, engine version and all
   sampling settings, recorded in `RESULTS.md` before the first test run.
3. Any change after the freeze, and any deviation from this plan, is written to `DEVIATIONS.md` with the reason, and the
   affected results are labelled.
4. All runs are kept. Negative and null results are reported.
5. Sampling follows the model card (thinking mode: temperature 1.0, top_p 0.95, top_k 20) in both vanilla and engineered
   cells unless the engineered harness changes it as a documented part of the harness.
6. Pair B uses the same user-simulator model and settings in every cell. It is a self-hosted Qwen model, so scores are not
   comparable with the public tau2 leaderboard or with Qwen's reported 79.1.

## Phases, rough budget (estimates; each phase needs a go-ahead before spending)

| Phase | What | Estimate |
|---|---|---|
| 0 | Selection, plan, scripts (done, free) | $0 |
| 1 | CPU weight downloads, flag validation, tiny-model smoke test on an A10G | under $2 |
| 2 | Serving sweeps for 27B (H100) and 9B (L40S) on agent-like workloads, quality guard | $40-55 |
| 3 | Vanilla pilots: 27B on 5 dev tasks, 9B on a slice of train; decide go or no-go | $10-20 |
| 4 | Harness development on dev/train, with failure analysis | $20-40 |
| 5 | Final test runs, core S0 and S3 for both pairs | $60-130 |
| 6 | Ablations S1, S2 if budget remains; analysis and write-up | rest |

Total estimate $130-250 against $280 of credit. The error bars are about 2x. Stop and reassess if spending reaches $200
before Phase 5 starts.

## Go / no-go checks

- After the Pair A pilot: if vanilla pi solves under 10% of dev tasks, find out why (setup error, endpoint, prompt format)
  before continuing; if it is genuine, fall back to a smaller task subset of the easiest category or report the floor.
- After the Pair B pilot: if vanilla pass^1 is above 90% on train, there is little headroom; weight the analysis toward
  pass^4 and the harder domains.
- If the engine sweeps show no tuned profile that beats the baseline by a margin larger than run-to-run noise, say so and run
  the full-system cell with the baseline engine, reporting that the engine contributed nothing measurable.

## Known limits

- 30 tasks and 3 attempts give an interval of roughly +/- 5 points on a rate near 30%; effects smaller than about 10-12
  points will not be distinguishable from noise.
- Lossless engine settings (speculation, scheduling) should not change accuracy; their effect is expected in time and
  cost. Accuracy gains are expected mainly from the harness.
- Qwen's 73.0 on Terminal-Bench 2.1 and 79.1 on tau2 were measured with different harnesses and settings.
