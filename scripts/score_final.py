"""Final curated scoring. Headline = test-split tasks only. Dev tasks (used to look for harness problems) are shown separately.
A task that ended as an infrastructure failure is replaced by its rerun when one exists; otherwise it is excluded and counted."""
import glob, json, os, random

SIM = "external/tau2-bench/data/simulations"

def tau(tags):
    out = {}
    for tag in tags:  # later tags (reruns) replace infrastructure failures only
        p = f"{SIM}/{tag}/results.json"
        if not os.path.exists(p): continue
        for s in json.load(open(p))["simulations"]:
            infra = s.get("termination_reason") == "infrastructure_error"
            t = s["task_id"]
            if infra and t in out and not out[t][1]: continue
            if t in out and not out[t][1] is True and infra: continue
            out[t] = (None if infra else (s.get("reward_info") or {}).get("reward", 0.0), infra)
    return out

def tb(jobs):
    out = {}
    for jd in jobs:
        for p in glob.glob(f"jobs/{jd}/*/result.json"):
            if os.path.dirname(p) == f"jobs/{jd}": continue
            r = json.load(open(p)); t = os.path.basename(os.path.dirname(p)).rsplit("__", 1)[0]
            rew = (r.get("verifier_result") or {}).get("rewards")
            if rew is not None: out[t] = rew.get("reward", 0.0)
            elif t not in out: out[t] = None
    return out

def boot(d, n=10000):
    random.seed(0); m = sum(d) / len(d)
    b = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(n))
    return m, b[int(.025 * n)], b[int(.975 * n)]

def paired(name, a, b):
    ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    excl = [k for k in set(a) | set(b) if k not in ks]
    if not ks: print(name, "no pairs"); return
    d = [b[k] - a[k] for k in ks]
    print(f"{name}: n={len(ks)} vanilla={sum(a[k] for k in ks)/len(ks):.3f} engineered={sum(b[k] for k in ks)/len(ks):.3f} "
          f"diff={boot(d)[0]:+.3f} CI[{boot(d)[1]:+.3f},{boot(d)[2]:+.3f}] better={sum(x>0 for x in d)} worse={sum(x<0 for x in d)} excluded={sorted(excl)}")

def tv(x): return {k: v[0] for k, v in x.items()}

print("== HEADLINE (test split) ==")
a1, b1 = tv(tau(["test-9b-S0", "rerun-test-9b-S0"])), tv(tau(["test-9b-S3", "rerun-test-9b-S3"]))
a2, b2 = tv(tau(["final-9b-S0"])), tv(tau(["final-9b-S3"]))
paired("tau2 airline test", a1, b1)
paired("tau2 retail test", a2, b2)
paired("tau2 all test", {("a", k): v for k, v in a1.items()} | {("r", k): v for k, v in a2.items()},
       {("a", k): v for k, v in b1.items()} | {("r", k): v for k, v in b2.items()})
ta, tb_ = tb(["main-tb-S0", "main-tb-S0-retry"]), tb(["main-tb-S3", "main-tb-S3-retry"])
fa, fb = tb(["final-tb-S0"]), tb(["final-tb-S3"])
paired("Terminal-Bench first-10 test", ta, tb_)
paired("Terminal-Bench final-5 test", fa, fb)
paired("Terminal-Bench all test", ta | fa, tb_ | fb)
print("== DEV (tuning data, not headline) ==")
paired("tau2 retail dev", tv(tau(["dev-9b-S0", "rerun-dev-9b-S0"])), tv(tau(["dev-9b-S3", "rerun-dev-9b-S3"])))
paired("Terminal-Bench dev", tb(["dev-tb-S0"]), tb(["dev-tb-S3"]))
