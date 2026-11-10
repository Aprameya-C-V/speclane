"""Scores the main study: paired per-task comparison for tau2 (airline test) and Terminal-Bench 2.1 (first 10 test tasks)."""
import glob, json, os, random, sys

def tb(job_dirs):
    out = {}
    for jd in job_dirs:
        for p in glob.glob(f"jobs/{jd}/*/result.json"):
            if os.path.dirname(p) == f"jobs/{jd}": continue
            r = json.load(open(p)); task = os.path.basename(os.path.dirname(p)).rsplit("__", 1)[0]
            rew = (r.get("verifier_result") or {}).get("rewards")
            err = (r.get("exception_info") or {}).get("exception_type")
            if rew is not None: out[task] = ("ok", rew.get("reward", 0.0))
            elif task not in out or out[task][0] != "ok": out[task] = ("infra" if err else "unfinished", None)
    return out

def tau(tag):
    p = glob.glob(f"external/tau2-bench/data/simulations/{tag}/results.json")
    if not p: return {}
    d = json.load(open(p[0])); res = {}
    for s in d["simulations"]:
        res[s["task_id"]] = (s.get("reward_info") or {}).get("reward", 0.0)
    return res

def boot(diffs, n=10000):
    if not diffs: return (float("nan"),) * 3
    random.seed(0); m = sum(diffs) / len(diffs)
    bs = sorted(sum(random.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return m, bs[int(.025 * n)], bs[int(.975 * n)]

for name, a, b in (("tau2 airline test (9B)", tau("test-9b-S0"), tau("test-9b-S3")),):
    common = sorted(set(a) & set(b)); print(name, "tasks", len(common))
    print(" S0 mean %.3f  S3 mean %.3f" % (sum(a[t] for t in common) / len(common), sum(b[t] for t in common) / len(common)))
    print(" paired diff mean %.3f  95%% CI [%.3f, %.3f]" % boot([b[t] - a[t] for t in common]))
s0, s3 = tb(["main-tb-S0", "main-tb-S0-retry"]), tb(["main-tb-S3", "main-tb-S3-retry"])
print("Terminal-Bench 2.1 (27B)")
for t in sorted(set(s0) | set(s3)): print(" ", t, s0.get(t), s3.get(t))
both = [t for t in s0 if t in s3 and s0[t][0] == "ok" and s3[t][0] == "ok"]
print(" scored in both:", len(both), " S0 %.2f S3 %.2f" % (sum(s0[t][1] for t in both), sum(s3[t][1] for t in both)))
