# Deviations from the pre-registered plan

The agent-harness pilot ran at a much smaller scale than planned, because of a hard time limit.
It is a small-sample demonstration, not a powered test. The runnable code for it is not part of
this repository; the numbered list below and the paired numbers in `FINDINGS.md` section 5 are
the complete record, and the paper claims nothing from it.

1. **Scale.** Terminal-Bench 2.1: the first 10 of the 30 locked test tasks, one trial each
   (planned: 30 tasks, k >= 3). tau2-bench: airline `test` split, 20 tasks, one trial each
   (planned: several domains, k = 4).
2. **Cells.** Only S0 (vanilla prompt, vanilla engine) and S3 (engineered prompt, tuned engine)
   ran. The S1 and S2 ablations did not, so the prompt and engine effects cannot be separated in
   the agent numbers; engine effects are measured separately in the throughput sweeps.
3. **No dev-split tuning.** The engineered prompt and agent were written once and not iterated on
   any dev data; they were frozen before the test runs. The dev-split result reported in
   `FINDINGS.md` was measured after freezing, not used to tune.
4. **Image builds.** The legacy image builder failed on 5 Terminal-Bench task images
   (`python: not found`). Both arms were rerun on those tasks with a newer builder version; a
   task whose build failed in both attempts is reported as an infrastructure failure, not a
   model failure.
5. **User simulator.** The tau2 user simulator was the vanilla Qwen3.8-27B-FP8 endpoint in both
   cells (temperature 0, thinking off). Scores are comparable across the two cells only, not
   with the public leaderboard.
6. **Concurrency.** Terminal-Bench cells ran at the same time on different endpoints, and the
   vanilla 27B endpoint also served the tau2 user simulator, so it carried extra load during the
   S0 runs.
7. **Dropped pilot.** The tau2 retail baseline pilot (9B, train, 10 tasks) failed with a script
   error while the runner was being edited and is not reported.
