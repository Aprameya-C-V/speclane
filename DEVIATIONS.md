# Deviations from EXPERIMENT.md

The main study was run at a much smaller scale than pre-registered, because of a hard time limit. Results are a small-sample demonstration, not a powered test.

1. **Scale.** Terminal-Bench 2.1: the first 10 of the 30 locked test tasks, 1 trial each (pre-registered: 30 tasks, k>=3). tau2-bench: airline `test` split, 20 tasks, 1 trial each (pre-registered: several domains, k=4).
2. **Cells.** Only S0 (vanilla harness, vanilla engine) and S3 (engineered harness, tuned engine) were run. S1/S2 ablations were not, so harness and engine effects cannot be separated in the agent results (engine effects are measured separately in the throughput sweeps).
3. **No dev-split tuning.** The engineered prompt (`harness/tb_prompt.j2`) and agent (`harness/tau2_engineered.py`) were written once and not iterated on any dev data; they were frozen before the test runs.
4. **Image builds.** Modal's legacy image builder failed on 5 Terminal-Bench task images (`python: not found`). Both arms were rerun on those tasks with `MODAL_IMAGE_BUILDER_VERSION=2025.06`; other tasks used the default builder. A task whose build failed in both attempts is reported as an infrastructure failure, not a model failure.
5. **User simulator.** The tau2 user simulator is the vanilla Qwen3.8-27B-FP8 endpoint in both cells (temperature 0, thinking off). Scores are comparable across the two cells only, not with the public leaderboard.
6. **Concurrency.** Terminal-Bench cells ran at the same time on different endpoints; the vanilla 27B endpoint also served the tau2 user simulator, so it carried extra load during the S0 runs.
7. **Quality of the tuned engine.** MTP speculative decoding is lossless in principle, but no GSM8K quality check on the MTP engines was completed.
8. **Dropped pilot.** The tau2 retail baseline pilot (9B, train, 10 tasks) failed with a script error while the runner was being edited and is not reported.
