"""One-GPU configuration sweep on Modal.

For one model, starts each serving profile (vLLM or SGLang) inside a single GPU container, runs the workloads at several
concurrency levels from the same machine (no network noise), records boot time and server metrics, optionally scores
quality, then stops the server and moves to the next profile. Results are written to a volume after every profile, so a
crash loses at most one.

    MODEL_KEY=qwen3.8-27b-fp8 modal run infer/modal_app.py::download           # once; CPU only
    MODEL_KEY=qwen3.8-27b-fp8 modal run infer/sweep.py::validate --engine vllm # CPU only: do the flags exist?
    MODEL_KEY=qwen3.8-27b-fp8 modal run infer/sweep.py::run --engine vllm --profiles baseline,mtp-3 \\
        --workloads chat_short,shared_prefix --concurrency 1,16,64
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request

import modal

from models import MODELS, PORT
from profiles import Profile, candidates, combine, flags_in, serve_command

MODEL_KEY = os.environ.get("MODEL_KEY", "qwen3.8-27b-fp8")
CONFIG = MODELS[MODEL_KEY]
DATA = "/models"
RESULTS = "/results"

app = modal.App(f"sweep-{MODEL_KEY}")
models_volume = modal.Volume.from_name("models-vol", create_if_missing=True)
results_volume = modal.Volume.from_name("results-vol", create_if_missing=True)


def engine_image(engine: str) -> modal.Image:
    """vLLM from PyPI (>= 0.17.0 per the recipes). SGLang's install is unverified: the Qwen3.5 card asks for its main branch."""
    # A CUDA *devel* image: vLLM and FlashInfer compile or locate GPU kernels at startup and need nvcc, which Debian slim
    # lacks (the first sweep died with "FlashInfer kernels are disabled: nvcc missing"). The installed vLLM 0.33.0 uses torch cu130.
    base = modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    base = base.uv_pip_install("vllm>=0.17.0") if engine == "vllm" else base.uv_pip_install("sglang[all]")
    return (
        base.uv_pip_install("aiohttp", "pandas", "pyarrow", "huggingface_hub", "hf_transfer")
        .env({"HF_HOME": f"{DATA}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL_KEY": MODEL_KEY,
              "CUDA_HOME": "/usr/local/cuda"})
        .add_local_python_source("models", "profiles", "bench", "workloads", "quality")
    )


class ServerDied(RuntimeError):
    pass


def diagnose_log(path: str = "/tmp/server.log") -> str:
    """The lines of the server log most likely to hold the root cause, plus its last lines. The outer vLLM error
    ("Engine core initialization failed, see root cause above") names nothing, so the whole log is searched."""
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return "no server log"
    # The engine process logs its own traceback as "ERROR ... ]" lines; its last lines hold the real exception.
    engine_errors = [line[:300] for line in lines if " ERROR " in line and "^^^" not in line]
    exceptions = [line[:300] for line in lines if any(k in line for k in ("Error:", "Exception:", "OutOfMemory", "CUDA error", "RuntimeError"))]
    return ("ENGINE ERROR LINES (last 25):\n" + "\n".join(engine_errors[-25:]) +
            "\nEXCEPTION LINES (last 10):\n" + "\n".join(exceptions[-10:]) +
            "\nLAST LINES:\n" + "\n".join(line[:200] for line in lines[-8:]))


def start_server(profile: Profile) -> subprocess.Popen:
    """Start the engine and copy its output to /tmp/server.log and, minus progress bars, to the container's stdout so
    that `modal run` shows what the server is doing while it starts."""
    env = {**os.environ, **profile.env}
    process = subprocess.Popen(serve_command(MODEL_KEY, profile), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True, text=True, bufsize=1, errors="replace")

    def pump() -> None:
        with open("/tmp/server.log", "w") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "it/s]" not in line and "%|" not in line:
                    print("[srv] " + line.rstrip()[:300], flush=True)

    threading.Thread(target=pump, daemon=True).start()
    return process


def wait_ready(process: subprocess.Popen, timeout_s: float) -> float:
    """Seconds until /health answers. Raises ServerDied, with the log tail, if the server exits first."""
    started = time.time()
    while time.time() - started < timeout_s:
        if process.poll() is not None:
            raise ServerDied(diagnose_log())
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return time.time() - started
        except Exception:
            time.sleep(2)
    raise ServerDied(f"not ready after {timeout_s}s\n" + diagnose_log())


def stop_server(process: subprocess.Popen) -> None:
    """Stop the server and its children. A server that already died (a failed launch) is not an error here."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        try:  # children of a dead launcher can still hold the GPU
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)  # let the GPU memory be released before the next profile


def scrape_metrics() -> list[str]:
    """Speculation and prefix-cache counters from the server's metrics endpoint, for acceptance-rate analysis."""
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5).read().decode()
    except Exception:
        return []
    keep = ("spec_decode", "spec_accept", "prefix_cache", "cache_hit")
    return [line for line in text.splitlines() if line and not line.startswith("#") and any(k in line for k in keep)][:40]


def run_workload(workload_name: str, concurrency: int, requests: int, model: str) -> dict:
    from bench import run_benchmark, run_sessions, summarize
    from workloads import WORKLOADS, build_conversations, build_sessions, is_session_workload

    url = f"http://127.0.0.1:{PORT}"
    workload = WORKLOADS[workload_name]
    hourly = CONFIG.gpu_hourly_usd
    if is_session_workload(workload):
        sessions = max(concurrency, requests // workload.turns)
        asyncio.run(run_sessions(url, model, build_sessions(workload, 2, seed=999), workload.output_tokens, 2))  # warm-up
        results, wall = asyncio.run(run_sessions(url, model, build_sessions(workload, sessions), workload.output_tokens, concurrency))
    else:
        warm = build_conversations(workload, 4, seed=999)
        asyncio.run(run_benchmark(url, model, warm, workload.output_tokens, concurrency=2))
        results, wall = asyncio.run(run_benchmark(url, model, build_conversations(workload, requests), workload.output_tokens, concurrency=concurrency))
    return dataclasses.asdict(summarize(results, wall, hourly))


def split_repeat(name: str) -> tuple[str, int]:
    """'baseline*3' means run the baseline profile three times (to measure run-to-run noise)."""
    base, _, count = name.partition("*")
    return base, int(count) if count else 1


def resolve_profiles(engine: str, names: list[str]) -> list[tuple[Profile, int]]:
    """Named candidates, or 'tuned:a+b+c' to combine several candidates into one profile. A name may end in '*N' to repeat it;
    the result is one (profile, repeat index) entry per run."""
    pool = {p.name: p for p in candidates(MODEL_KEY, engine)}
    chosen: list[tuple[Profile, int]] = []
    for spec in names:
        name, repeats = split_repeat(spec)
        for index in range(repeats):
            chosen.append((_resolve_one(pool, name, engine), index))
    return chosen


def _resolve_one(pool: dict[str, Profile], name: str, engine: str) -> Profile:
    """One profile by name, or 'tuned:a+b+c' combining several candidates."""
    if name.startswith("tuned:"):
        return combine(name, [pool[n] for n in name[len("tuned:"):].split("+")], MODEL_KEY)
    return pool[name]


def run_sweep(engine: str, profiles: list[str], workloads: list[str], concurrency: list[int], requests: int,
              quality_questions: int = 0) -> list[dict]:
    from huggingface_hub import snapshot_download

    snapshot_download(CONFIG.repo, local_files_only=True)  # fail fast if the weights were not staged
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = f"{RESULTS}/{MODEL_KEY}-{engine}-{stamp}.jsonl"
    records = []
    for profile, repeat in resolve_profiles(engine, profiles):
        record: dict = {"model": MODEL_KEY, "engine": engine, "profile": profile.name, "repeat": repeat, "args": list(profile.args),
                        "gpu": CONFIG.gpu, "gpu_hourly_usd": CONFIG.gpu_hourly_usd}
        process = start_server(profile)
        try:
            record["boot_s"] = round(wait_ready(process, CONFIG.startup_timeout_s), 1)
            record["runs"] = []
            for workload in workloads:
                for level in concurrency:
                    summary = run_workload(workload, level, requests, CONFIG.repo)
                    record["runs"].append({"workload": workload, "concurrency": level, "summary": summary})
                    print(json.dumps({"profile": profile.name, "repeat": repeat, "workload": workload, "concurrency": level,
                                      "out_tok_s": summary["output_tokens_per_s"], "ttft_p99": summary["ttft_p99_s"]}), flush=True)
            record["metrics"] = scrape_metrics()
            if quality_questions:
                from quality import evaluate, load_gsm8k
                record["quality"] = asyncio.run(evaluate(f"http://127.0.0.1:{PORT}", CONFIG.repo, load_gsm8k(quality_questions)))
        except ServerDied as error:
            record["error"] = str(error)
            print(json.dumps({"profile": profile.name, "server_failed": str(error)[:5000]}), flush=True)
        finally:
            stop_server(process)
        records.append(record)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        results_volume.commit()
    return records


common = dict(volumes={DATA: models_volume, RESULTS: results_volume}, gpu=CONFIG.gpu, timeout=6 * 3600)


@app.function(image=engine_image("vllm"), **common)
def sweep_vllm(profiles: list[str], workloads: list[str], concurrency: list[int], requests: int, quality_questions: int) -> list[dict]:
    return run_sweep("vllm", profiles, workloads, concurrency, requests, quality_questions)


@app.function(image=engine_image("sglang"), **common)
def sweep_sglang(profiles: list[str], workloads: list[str], concurrency: list[int], requests: int, quality_questions: int) -> list[dict]:
    return run_sweep("sglang", profiles, workloads, concurrency, requests, quality_questions)


def check_flags(engine: str) -> dict:
    """Which of this model's candidate flags the installed engine does not recognise. Runs on CPU."""
    wanted = flags_in(candidates(MODEL_KEY, engine))
    attempts = (
        [["vllm", "serve", "--help=all"], ["vllm", "serve", "--help"], ["python", "-m", "vllm.entrypoints.openai.api_server", "--help"]]
        if engine == "vllm"
        else [["python", "-m", "sglang.launch_server", "--help"]]
    )
    best, tried = "", []
    for command in attempts:
        try:
            text = subprocess.run(command, capture_output=True, text=True, timeout=600)
            output = text.stdout + text.stderr
        except Exception as error:
            output = f"error: {error}"
        tried.append({"command": " ".join(command), "chars": len(output)})
        if len(output) > len(best):
            best = output
    missing = sorted(flag for flag in wanted if flag not in best)
    return {"engine": engine, "checked": len(wanted), "missing": missing, "help_chars": len(best), "tried": tried,
            "snippet": best[:600], "version": subprocess.run(["python", "-c", "import vllm;print(vllm.__version__)"], capture_output=True, text=True).stdout.strip()
            if engine == "vllm" else ""}


@app.function(image=engine_image("vllm"), cpu=2, memory=8192, timeout=1800)
def validate_vllm() -> dict:
    return check_flags("vllm")


@app.function(image=engine_image("sglang"), cpu=2, memory=8192, timeout=1800)
def validate_sglang() -> dict:
    return check_flags("sglang")


@app.function(image=engine_image("vllm"), cpu=2, memory=8192, timeout=900)
def env_info() -> dict:
    """What the vLLM image actually contains, for choosing a CUDA base image."""
    def run(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()[:300]

    return {
        "torch_cuda": run("python -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null"),
        "vllm": run("python -c 'import vllm;print(vllm.__version__)' 2>/dev/null | tail -1"),
        "flashinfer": run("pip list 2>/dev/null | grep -i -E 'flashinfer|deep.?gemm|xformers|^triton'"),
        "tools": run("which nvcc gcc g++ ninja cc make 2>&1"),
        "nvidia_pip": run("pip list 2>/dev/null | grep -i -E 'nvidia-cuda|cuda-toolkit|nvidia-cublas' | head -5"),
    }


@app.local_entrypoint()
def info() -> None:
    print(json.dumps(env_info.remote(), indent=2))


@app.local_entrypoint()
def validate(engine: str = "vllm") -> None:
    """CPU-only check that every candidate flag exists in the engine version that would be installed."""
    result = (validate_vllm if engine == "vllm" else validate_sglang).remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def run(engine: str = "vllm", profiles: str = "baseline", workloads: str = "chat_short", concurrency: str = "1,16,64",
        requests: int = 64, quality: int = 0) -> None:
    """Run the sweep on a GPU and save the records under results/."""
    function = sweep_vllm if engine == "vllm" else sweep_sglang
    records = function.remote(profiles.split(","), workloads.split(","), [int(c) for c in concurrency.split(",")], requests, quality)
    os.makedirs("results", exist_ok=True)
    out = f"results/{MODEL_KEY}-{engine}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as handle:
        json.dump(records, handle, indent=1)
    for record in records:
        print(record["profile"], "boot_s=", record.get("boot_s"), "error=" + record["error"][:120] if "error" in record else "")
    print("saved", out)
