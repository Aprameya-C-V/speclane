"""Builds the paper's tables, figure and headline numbers from measured results. Nothing is typed in by hand.

    python paper/make_results.py                       # reads infer/results/*.json (finished sweeps)
    python paper/make_results.py --from-log LOG ...    # interim: reads the progress lines printed by a running sweep

Output (paper/generated/): engine_<model>.tex (throughput table), ttft_<model>.tex, fig_throughput.pdf, numbers.tex.
A number that has no data yet is written as \\todo{pending}, so an unfinished paper cannot show a made-up value.
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent
RESULTS = HERE.parent / "infer" / "results"
OUT = HERE / "generated"

MODELS = {"qwen3.8-27b-fp8": ("Qwen3.8-27B-FP8", "H100"), "qwen3.5-9b": ("Qwen3.5-9B", "L40S")}
WORKLOADS = ["shared_prefix", "agent_sessions"]
CONCURRENCIES = [1, 16]


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("+", "+")


def load_finished(paths: list[pathlib.Path]) -> dict:
    """{(model, profile, workload, concurrency): {"tok_s", "ttft_p99", ...}} from finished sweep files; later files win."""
    data: dict = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        for record in json.loads(path.read_text()):
            if "runs" not in record or "error" in record:
                continue
            for run in record["runs"]:
                s = run["summary"]
                data[(record["model"], record["profile"], run["workload"], run["concurrency"])] = {
                    "tok_s": s["output_tokens_per_s"], "ttft_p99": s["ttft_p99_s"], "ok": s["ok"], "requests": s["requests"],
                    "cost": s.get("cost_per_million_output_tokens_usd")}
    return data


def load_interim(paths: list[pathlib.Path], model_of: dict[str, str]) -> dict:
    """Progress lines ({"profile":..., "out_tok_s":...}) from sweep logs. model_of maps a log file name to its model key."""
    data: dict = {}
    for path in paths:
        model = model_of.get(path.name)
        if not model:
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("{") and "out_tok_s" in line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data[(model, d["profile"], d["workload"], d["concurrency"])] = {"tok_s": d["out_tok_s"], "ttft_p99": d["ttft_p99"]}
    return data


def delta(value: float, base: float | None) -> str:
    return "" if not base else f" ({(value / base - 1) * 100:+.0f}\\%)"


def profile_order(data: dict, model: str) -> list[str]:
    names = []
    for key in data:
        if key[0] == model and key[1] not in names:
            names.append(key[1])
    return sorted(names, key=lambda n: (n != "baseline", n))


def write_tables(data: dict) -> None:
    for model, (label, gpu) in MODELS.items():
        profiles = profile_order(data, model)
        for kind, field, fmt in (("engine", "tok_s", "{:.0f}"), ("ttft", "ttft_p99", "{:.2f}")):
            cols = "".join("r" for _ in WORKLOADS for _ in CONCURRENCIES)
            lines = [r"\begin{tabular}{l" + cols + "}", r"\toprule",
                     "Profile & " + " & ".join(f"{tex(w)}, c={c}" for w in WORKLOADS for c in CONCURRENCIES) + r" \\", r"\midrule"]
            if not profiles:
                lines.append(r"\multicolumn{" + str(1 + len(WORKLOADS) * len(CONCURRENCIES)) + r"}{c}{\todo{pending}} \\")
            for profile in profiles:
                cells = []
                for w in WORKLOADS:
                    for c in CONCURRENCIES:
                        row = data.get((model, profile, w, c))
                        base = data.get((model, "baseline", w, c))
                        if not row:
                            cells.append("--")
                        elif kind == "engine":
                            cells.append(fmt.format(row[field]) + (delta(row[field], base[field]) if base and profile != "baseline" else ""))
                        else:
                            cells.append(fmt.format(row[field]))
                lines.append(tex(profile) + " & " + " & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}"]
            (OUT / f"{kind}_{model.replace('.', '').replace('-', '_')}.tex").write_text("\n".join(lines) + "\n")


def write_figure(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 3.2), sharey=False)
    for ax, (model, (label, gpu)) in zip(axes, MODELS.items()):
        profiles = profile_order(data, model)
        width = 0.8 / max(1, len(profiles))
        for i, profile in enumerate(profiles):
            values = [data.get((model, profile, "agent_sessions", c), {}).get("tok_s", 0) for c in CONCURRENCIES]
            ax.bar([j + i * width for j in range(len(CONCURRENCIES))], values, width, label=profile)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(CONCURRENCIES))], [f"concurrency {c}" for c in CONCURRENCIES])
        ax.set_title(f"{label} on {gpu}, agent sessions", fontsize=9)
        ax.set_ylabel("output tokens / s")
        if profiles:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_throughput.pdf")
    fig.savefig(HERE / "figures" / "fig_throughput.png", dpi=150)


def write_numbers(data: dict) -> None:
    """Headline numbers as LaTeX macros, so the prose always matches the tables."""
    macros = []

    def add(name: str, value: str) -> None:
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    specs = [("TwentySeven", "qwen3.8-27b-fp8", "mtp-3"), ("Nine", "qwen3.5-9b", "mtp-1")]
    for label, model, best in specs:
        for workload, wlabel in (("agent_sessions", "Agent"), ("shared_prefix", "Shared")):
            for c in CONCURRENCIES:
                base = data.get((model, "baseline", workload, c))
                top = data.get((model, best, workload, c))
                stem = f"{label}{wlabel}C{ {1: 'One', 16: 'Sixteen'}[c] }"  # LaTeX command names cannot contain digits
                add(f"{stem}Base", f"{base['tok_s']:.0f}" if base else r"\todo{pending}")
                add(f"{stem}Best", f"{top['tok_s']:.0f}" if top else r"\todo{pending}")
                add(f"{stem}Gain", f"{(top['tok_s'] / base['tok_s'] - 1) * 100:.0f}" if base and top else r"\todo{pending}")
    (OUT / "numbers.tex").write_text("\n".join(macros) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-log", nargs="*", help="interim: sweep log files; model is inferred from the file name")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    if args.from_log:
        model_of = {pathlib.Path(p).name: ("qwen3.8-27b-fp8" if "27b" in p else "qwen3.5-9b") for p in args.from_log}
        data = load_interim([pathlib.Path(p) for p in args.from_log], model_of)
        source = "interim logs"
    else:
        data = load_finished(sorted(RESULTS.glob("*.json")))
        source = "finished sweeps"
    write_tables(data)
    write_figure(data)
    write_numbers(data)
    print(f"{len(data)} measurements from {source}; wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
