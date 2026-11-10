"""Modal deployment of one model behind vLLM's OpenAI-compatible server.

Pick the model with the MODEL_KEY environment variable (see models.py):

    modal run infer/modal_app.py::download            # stage weights on the volume (CPU only, no GPU cost)
    modal deploy infer/modal_app.py                   # deploy the server; it scales to zero when idle

Each phase of startup prints one JSON line ({"phase": ..., "seconds": ...}) so cold-start cost can be measured.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

import modal

from models import MODELS, PORT, serve_command

MODEL_KEY = os.environ.get("MODEL_KEY", "tiny")
CONFIG = MODELS[MODEL_KEY]
DATA = "/models"

app = modal.App(f"infer-{MODEL_KEY}")
volume = modal.Volume.from_name("models-vol", create_if_missing=True)


def build_image() -> modal.Image:
    spec = CONFIG.image
    if spec.kind == "registry":
        image = modal.Image.from_registry(spec.value, add_python="3.12")
    else:
        image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(spec.value, pre=spec.pre)
    return image.uv_pip_install("huggingface_hub", "hf_transfer").env(
        {"HF_HOME": f"{DATA}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL_KEY": MODEL_KEY, **CONFIG.env}
    ).add_local_python_source("models")


image = build_image()


def phase(name: str, started: float) -> None:
    print(json.dumps({"phase": name, "model": MODEL_KEY, "seconds": round(time.time() - started, 1)}), flush=True)


@app.function(image=image, volumes={DATA: volume}, timeout=6 * 3600, cpu=4, memory=16384)
def download() -> None:
    """Copy the model's weights from Hugging Face to the volume. Runs on CPU, so it costs no GPU time."""
    from huggingface_hub import snapshot_download

    started = time.time()
    snapshot_download(repo_id=CONFIG.repo)
    volume.commit()
    phase("download_done", started)


@app.cls(
    image=image,
    gpu=CONFIG.gpu,
    volumes={DATA: volume},
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
    startup_timeout=CONFIG.startup_timeout_s,
    timeout=6 * 3600,
)
@modal.concurrent(max_inputs=256)
class Server:
    @modal.enter()
    def start(self) -> None:
        self.started = time.time()
        self.process = subprocess.Popen(serve_command(CONFIG))
        phase("process_started", self.started)
        # Block until vLLM answers, so the container only counts as ready when it can serve.
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.process.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        phase("engine_ready", self.started)

    @modal.web_server(PORT, startup_timeout=CONFIG.startup_timeout_s)
    def serve(self) -> None:
        """vLLM is already listening; this only publishes the port."""

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        phase("stopped", self.started)
