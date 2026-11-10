"""Deterministic, pre-registered task selection for the Terminal-Bench 2.1 experiment (Qwen3.8-27B).

Same rule as tb4/select_tasks.py, applied to metadata.json (fetched from the harborframework/terminal-bench-2.1 mirror):
  1. Start from the 89 tasks.
  2. Drop tasks that need a GPU (none in 2.1, but the rule is kept for reproducibility).
  3. Test set: 30 tasks, stratified by category in proportion to category size, Random(0) inside each category.
  4. Dev set: 10 further tasks, same stratification, Random(1). Harness tuning may look only at the dev set.

    python tb21/select_tasks.py        # prints the selection and writes selection.json, test_tasks.txt, dev_tasks.txt
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "tb4"))
from select_tasks import DEV_SEED, DEV_SIZE, TEST_SEED, TEST_SIZE, stratified_sample  # noqa: E402


def main() -> None:
    rows = json.loads((HERE / "metadata.json").read_text())
    tasks = [{"name": r["name"], "category": r["category"] or "unknown", "gpus": r["gpus"], "agent_s": r["agent_s"]} for r in rows]
    eligible = [t for t in tasks if t["gpus"] == 0]
    test = stratified_sample(eligible, TEST_SIZE, TEST_SEED)
    test_names = {t["name"] for t in test}
    dev = stratified_sample([t for t in eligible if t["name"] not in test_names], DEV_SIZE, DEV_SEED)
    selection = {
        "dataset": "harborframework/terminal-bench-2.1",
        "total_tasks": len(tasks),
        "excluded_gpu": sorted(t["name"] for t in tasks if t["gpus"] > 0),
        "test": [t["name"] for t in test],
        "dev": [t["name"] for t in dev],
        "test_categories": {c: sum(1 for t in test if t["category"] == c) for c in sorted({t["category"] for t in test})},
        "test_timeouts_s": {t["name"]: t["agent_s"] for t in test},
    }
    selection["fingerprint"] = hashlib.sha256(json.dumps({k: selection[k] for k in ("test", "dev")}, sort_keys=True).encode()).hexdigest()[:16]
    (HERE / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    (HERE / "test_tasks.txt").write_text("\n".join(selection["test"]) + "\n")
    (HERE / "dev_tasks.txt").write_text("\n".join(selection["dev"]) + "\n")
    print(json.dumps({k: v for k, v in selection.items() if k != "test_timeouts_s"}, indent=1))
    long_tasks = {n: s for n, s in selection["test_timeouts_s"].items() if s and s > 3600}
    print("test tasks with timeouts over 1 hour:", long_tasks or "none")


if __name__ == "__main__":
    main()
