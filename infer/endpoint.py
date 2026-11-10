"""A model endpoint for agent runs: one model, one engine profile, behind an API key.

    export VLLM_API_KEY=...                         # secret; also give it to the client that calls the endpoint
    MODEL_KEY=qwen3.8-27b-fp8 PROFILE=baseline modal deploy infer/endpoint.py   # vanilla engine
    MODEL_KEY=qwen3.8-27b-fp8 PROFILE=tuned    modal deploy infer/endpoint.py   # tuned engine (round-1 winner)

PROFILE=baseline is the vanilla profile of infer/profiles.py (defaults plus what is needed to start). PROFILE=tuned is the
round-1 winner: MTP speculation (3 draft tokens on the 27B, 2 on the 9B), no FP8 KV cache. Both use the same context length,
so a comparison between them changes only the engine settings. The key is read from the environment at deploy time and stored
in a Modal secret; it is never printed.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request

import modal

from models import MODELS, PORT
from profiles import baseline_args, candidates, serve_command, set_flag

MODEL_KEY = os.environ.get("MODEL_KEY", "qwen3.8-27b-fp8")
PROFILE = os.environ.get("PROFILE", "baseline")
CONFIG = MODELS[MODEL_KEY]
# Agent trajectories are long: Terminal-Bench runs need a large window, tau2 conversations are short.
CONTEXT = int(os.environ.get("CONTEXT", "131072" if MODEL_KEY == "qwen3.8-27b-fp8" else "32768"))
DATA = "/models"
SERVED_NAME = MODEL_KEY

app = modal.App(f"ep-{MODEL_KEY}-{PROFILE}{os.environ.get('APP_SUFFIX', '')}")
models_volume = modal.Volume.from_name("models-vol", create_if_missing=True)
SECRET = modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm>=0.17.0", "huggingface_hub")
    .env({"HF_HOME": f"{DATA}/hf", "CUDA_HOME": "/usr/local/cuda", "MODEL_KEY": MODEL_KEY, "PROFILE": PROFILE, "CONTEXT": str(CONTEXT)})
    .add_local_python_source("models", "profiles")
)


def command() -> list[str]:
    pool = {p.name: p for p in candidates(MODEL_KEY, "vllm")}
    if PROFILE == "baseline":
        profile = pool["baseline"]
    elif PROFILE == "tuned":
        profile = pool["mtp-3" if MODEL_KEY == "qwen3.8-27b-fp8" else "mtp-2"]
    else:
        profile = pool[PROFILE]
    args = set_flag(profile.args, "--max-model-len", str(CONTEXT))
    args = set_flag(args, "--served-model-name", SERVED_NAME)
    # An agent sends tool definitions with tool_choice "auto"; without these flags vLLM answers 400 and the agent never reaches the
    # model (found in the first Terminal-Bench pilot). They are part of serving an agent at all, so both profiles carry them.
    # Parser names: the Qwen3.5-9B model card and the vLLM recipe for Qwen3.8-27B-FP8.
    args = set_flag(args, "--enable-auto-tool-choice", "")
    args = set_flag(args, "--tool-call-parser", "qwen3_coder")
    args = set_flag(args, "--reasoning-parser", "qwen3")
    return serve_command(MODEL_KEY, type(profile)(profile.name, profile.engine, args, profile.description))


def phase(name: str, started: float) -> None:
    print(f'{{"phase": "{name}", "model": "{MODEL_KEY}", "profile": "{PROFILE}", "seconds": {time.time() - started:.1f}}}', flush=True)


@app.cls(
    image=image,
    gpu=CONFIG.gpu,
    volumes={DATA: models_volume},
    secrets=[SECRET],
    scaledown_window=1200,
    min_containers=1,  # always warm while deployed, so agent requests never hit a cold start; stop the app to stop billing
    max_containers=1,
    startup_timeout=3600,
    timeout=6 * 3600,
)
@modal.concurrent(max_inputs=256)
class Server:
    @modal.enter()
    def start(self) -> None:
        self.started = time.time()
        self.process = subprocess.Popen(command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        errors="replace", start_new_session=True)

        def pump() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                if "it/s]" not in line and "%|" not in line:
                    print("[srv] " + line.rstrip()[:300], flush=True)

        threading.Thread(target=pump, daemon=True).start()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.process.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        phase("engine_ready", self.started)

    @modal.web_server(PORT, startup_timeout=3600)
    def serve(self) -> None:
        """vLLM is already listening; this only publishes the port."""

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
