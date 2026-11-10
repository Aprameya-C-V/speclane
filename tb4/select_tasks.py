"""Deterministic, pre-registered task selection for the Terminal-Bench 4.0 experiment.

Rule (fixed before any model was run):
  1. Start from the 66 tasks of terminal-bench@4.0.0.
  2. Drop tasks that need a GPU (3 tasks); the experiment sandboxes are CPU-only.
  3. Test set: 30 tasks, stratified by category in proportion to category size (largest-remainder rounding),
     sampled with Random(TEST_SEED) inside each category.
  4. Dev set: 10 further tasks from what is left, same stratification, Random(DEV_SEED). Prompt and harness tuning may
     look only at the dev set. The test set stays locked until the final runs.

    python tb4/select_tasks.py <directory containing the downloaded task folders> [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import tomllib

TEST_SEED = 0
DEV_SEED = 1
TEST_SIZE = 30
DEV_SIZE = 10


def load_tasks(root: pathlib.Path) -> list[dict]:
    tasks = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.toml").exists()):
        data = tomllib.loads((folder / "task.toml").read_text())
        metadata, environment = data.get("metadata", {}), data.get("environment", {})
        tasks.append({
            "name": folder.name,
            "category": metadata.get("category", "unknown"),
            "expert_hours": metadata.get("expert_time_estimate_hours"),
            "gpus": environment.get("gpus", 0) or 0,
        })
    return tasks


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` slots across categories in proportion to their size."""
    size = sum(counts.values())
    exact = {c: total * n / size for c, n in counts.items()}
    quota = {c: int(v) for c, v in exact.items()}
    leftovers = sorted(counts, key=lambda c: (exact[c] - quota[c], counts[c], c), reverse=True)
    for category in leftovers[: total - sum(quota.values())]:
        quota[category] += 1
    return quota


def stratified_sample(tasks: list[dict], total: int, seed: int) -> list[dict]:
    by_category: dict[str, list[dict]] = {}
    for task in tasks:
        by_category.setdefault(task["category"], []).append(task)
    quota = allocate({c: len(v) for c, v in by_category.items()}, total)
    rng = random.Random(seed)
    chosen = []
    for category in sorted(by_category):
        chosen += rng.sample(sorted(by_category[category], key=lambda t: t["name"]), min(quota[category], len(by_category[category])))
    return sorted(chosen, key=lambda t: t["name"])


def select(tasks: list[dict]) -> dict:
    eligible = [t for t in tasks if t["gpus"] == 0]
    test = stratified_sample(eligible, TEST_SIZE, TEST_SEED)
    test_names = {t["name"] for t in test}
    dev = stratified_sample([t for t in eligible if t["name"] not in test_names], DEV_SIZE, DEV_SEED)
    return {
        "dataset": "terminal-bench/terminal-bench@4.0.0",
        "total_tasks": len(tasks),
        "excluded_gpu": sorted(t["name"] for t in tasks if t["gpus"] > 0),
        "test": [t["name"] for t in test],
        "dev": [t["name"] for t in dev],
        "test_categories": {c: sum(1 for t in test if t["category"] == c) for c in sorted({t["category"] for t in test})},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--write", action="store_true", help="write selection.json and the .txt lists next to this script")
    args = parser.parse_args()
    selection = select(load_tasks(args.root))
    selection["fingerprint"] = hashlib.sha256(json.dumps({k: selection[k] for k in ("test", "dev")}, sort_keys=True).encode()).hexdigest()[:16]
    print(json.dumps(selection, indent=1))
    if args.write:
        here = pathlib.Path(__file__).parent
        (here / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
        (here / "test_tasks.txt").write_text("\n".join(selection["test"]) + "\n")
        (here / "dev_tasks.txt").write_text("\n".join(selection["dev"]) + "\n")


if __name__ == "__main__":
    main()
