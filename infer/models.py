"""Model registry: where each model's weights live, how its vLLM server is launched, and what GPUs it needs.

Every flag below was copied from the vLLM recipe for that model (recipes.vllm.ai) unless a comment says it was
adapted. Adapted flags are unverified until a real run proves them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PORT = 8000


@dataclass(frozen=True)
class ImageSpec:
    """How to build the container image that runs vLLM."""

    kind: str  # "pip": install vllm from PyPI; "registry": pull a published image
    value: str  # pip requirement, or registry tag
    pre: bool = False  # allow pre-release wheels (nightly builds)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    repo: str
    gpu: str  # Modal GPU string, e.g. "B200:4"
    gpu_hourly_usd: float  # price of the whole GPU set, from `modal billing rates`
    image: ImageSpec
    env: dict[str, str] = field(default_factory=dict)
    serve_args: tuple[str, ...] = ()
    startup_timeout_s: int = 3600
    weights_gb: int = 0
    notes: str = ""


# Modal rates read on 2026-11-08 (`modal billing rates`): A10G $1.10/h, L40S $1.95/h, H100 $3.95/h, B200 $6.25/h.
A10G_USD_PER_HOUR = 1.10
L40S_USD_PER_HOUR = 1.95
H100_USD_PER_HOUR = 3.95
B200_X4_USD_PER_HOUR = 4 * 6.25

MODELS: dict[str, ModelConfig] = {
    # A tiny model for proving the server, gateway and benchmark code before any expensive GPU is rented.
    "tiny": ModelConfig(
        key="tiny",
        repo="Qwen/Qwen3-0.6B",
        gpu="A10G",
        gpu_hourly_usd=1.10,
        image=ImageSpec("pip", "vllm"),
        serve_args=("--max-model-len", "8192", "--gpu-memory-utilization", "0.85"),
        startup_timeout_s=900,
        weights_gb=2,
        notes="Smoke-test model. Not part of the experiment.",
    ),
    # The Qwen family used for the project. All are Apache-2.0, run on released vLLM (>= 0.17.0) and ship an MTP draft
    # head; flags come from recipes.vllm.ai/Qwen/<model> (TP1 plus MTP speculation). Drop speculation with
    # overrides={"--speculative-config": None}. GPU choice is by size: weights plus KV cache must fit one GPU.
    "qwen3.5-4b": ModelConfig(
        key="qwen3.5-4b",
        repo="Qwen/Qwen3.5-4B",
        gpu="A10G",
        gpu_hourly_usd=A10G_USD_PER_HOUR,
        image=ImageSpec("pip", "vllm>=0.17.0"),
        env={"VLLM_USE_RUST_FRONTEND": "1"},
        serve_args=(
            "--trust-remote-code",
            "--tensor-parallel-size", "1",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}',
        ),
        startup_timeout_s=1200,
        weights_gb=9,
        notes="Dense 4B. Recipe sets --enforce-eager only for Intel Arc hardware, so it is omitted here.",
    ),
    "qwen3.5-9b": ModelConfig(
        key="qwen3.5-9b",
        repo="Qwen/Qwen3.5-9B",
        gpu="L40S",
        gpu_hourly_usd=L40S_USD_PER_HOUR,
        image=ImageSpec("pip", "vllm>=0.17.0"),
        env={"VLLM_USE_RUST_FRONTEND": "1"},
        serve_args=(
            "--trust-remote-code",
            "--tensor-parallel-size", "1",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}',
        ),
        startup_timeout_s=1200,
        weights_gb=19,
        notes="Dense 9B, gated-delta-network hybrid attention.",
    ),
    "qwen3.8-27b-fp8": ModelConfig(
        key="qwen3.8-27b-fp8",
        repo="Qwen/Qwen3.8-27B-FP8",
        gpu="H100",
        gpu_hourly_usd=H100_USD_PER_HOUR,
        image=ImageSpec("pip", "vllm>=0.17.0"),
        env={"VLLM_USE_RUST_FRONTEND": "1"},
        serve_args=(
            "--tensor-parallel-size", "1",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        ),
        startup_timeout_s=1800,
        weights_gb=31,
        notes="Top small dense model on Artificial Analysis. Linear attention on 48 of 64 layers.",
    ),
    "qwen3.6-35b-a3b-fp8": ModelConfig(
        key="qwen3.6-35b-a3b-fp8",
        repo="Qwen/Qwen3.6-35B-A3B-FP8",
        gpu="H100",
        gpu_hourly_usd=H100_USD_PER_HOUR,
        image=ImageSpec("pip", "vllm>=0.17.0"),
        env={"VLLM_USE_RUST_FRONTEND": "1"},
        serve_args=(
            "--trust-remote-code",
            "--tensor-parallel-size", "1",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}',
        ),
        startup_timeout_s=1800,
        weights_gb=37,
        notes="Mixture of experts, 35B total and 3B active. The recipe's own text does not mention speculative decoding.",
    ),
    # Recipe: recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash, B200 tensor-parallel preset, TP4, FP8 checkpoint 1017.
    # Deferred: not part of the current plan (too expensive for the credits); kept for reference.
    "deepseek-v4.1-flash": ModelConfig(
        key="deepseek-v4.1-flash",
        repo="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        gpu="B200:4",
        gpu_hourly_usd=B200_X4_USD_PER_HOUR,
        image=ImageSpec("pip", "vllm>=0.31.0"),
        env={
            "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_RUST_FRONTEND": "1",
        },
        serve_args=(
            "--tokenizer-mode", "deepseek_v41",
            "--tensor-parallel-size", "4",
            "--kv-cache-dtype", "fp8",
            "--attention-config", '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}',
            "--gpu-memory-utilization", "0.93",
            "--engram-config", '{"cpu_offload":false}',
            "--kernel-config", '{"enable_flashinfer_autotune":true,"sparse_indexer_topk_backend":"deep_select"}',
            "--max-num-seqs", "8",
            "--max-num-active-seqs", "1",
            "--prefill-schedule-interval", "16",
            "--max-num-batched-tokens", "4096",
            "--compilation-config", '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092]}',
            "--max-cudagraph-capture-size", "4092",
            "--tool-call-parser", "deepseek_v41",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "deepseek_v41",
            "--speculative-config", '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}',
            "--language-model-only",
        ),
        startup_timeout_s=7200,
        weights_gb=510,
        notes="The recipe preset targets low concurrency (max-num-seqs 8, one active sequence). Throughput runs need overrides.",
    ),
    # Recipe: recipes.vllm.ai/zai-org/GLM-5.3-Flash, the TP4 + MTP variant. Needs a nightly vLLM wheel (>= 0.29.0).
    "glm-5.3-flash": ModelConfig(
        key="glm-5.3-flash",
        repo="zai-org/GLM-5.3-Flash",
        gpu="B200:4",
        gpu_hourly_usd=B200_X4_USD_PER_HOUR,
        image=ImageSpec("registry", "vllm/vllm-openai:nightly"),
        env={"VLLM_ENGINE_READY_TIMEOUT_S": "3600", "VLLM_USE_RUST_FRONTEND": "1"},
        serve_args=(
            "--tokenizer-mode", "glm5_next",
            "--tensor-parallel-size", "4",
            "--kv-cache-dtype", "fp8",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":5}',
            "--tool-call-parser", "glm47",
            "--reasoning-parser", "glm47",
            "--enable-auto-tool-choice",
            "--served-model-name", "zai-org/GLM-5.3-Flash",
        ),
        startup_timeout_s=3600,
        weights_gb=328,
        notes="The recipe's default is TP8 (8x B200); TP4 with MTP is its second listed variant.",
    ),
    # Recipe: recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash. The recipe command is TP8; its text says TP4 works on Blackwell.
    "minimax-m3.1-flash": ModelConfig(
        key="minimax-m3.1-flash",
        repo="MiniMaxAI/MiniMax-M3.1-Flash",
        gpu="B200:4",
        gpu_hourly_usd=B200_X4_USD_PER_HOUR,
        image=ImageSpec("registry", "vllm/vllm-openai:minimax-m31-flash"),
        env={"VLLM_FLOAT32_MATMUL_PRECISION": "high"},
        serve_args=(
            "--block-size", "128",
            "--disable-hybrid-kv-cache-manager",
            "--tensor-parallel-size", "4",  # ADAPTED from the recipe's 8; unverified
            "--kv-cache-dtype", "nvfp4",
            "--attention-config", '{"indexer_kv_dtype": "nvfp4", "use_prefill_query_quantization": true}',
            "--compilation-config", '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
            "--tool-call-parser", "minimax_m3",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "minimax_m3",
            "--speculative-config", '{"method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_TS","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}',
        ),
        startup_timeout_s=3600,
        weights_gb=252,
        notes="License is 'other': read the terms before any commercial use.",
    ),
}


def serve_command(config: ModelConfig, overrides: dict[str, str | None] | None = None, port: int = PORT) -> list[str]:
    """The `vllm serve` command for a model. `overrides` maps a flag to a new value, or to None to drop the flag."""
    args = list(config.serve_args)
    for flag, value in (overrides or {}).items():
        if flag in args:
            index = args.index(flag)
            if value is None:
                # Drop the flag and its value, if it has one.
                end = index + 2 if index + 1 < len(args) and not args[index + 1].startswith("--") else index + 1
                del args[index:end]
            else:
                args[index + 1] = value
        elif value is not None:
            args += [flag, value]
    return ["vllm", "serve", config.repo, "--host", "0.0.0.0", "--port", str(port), *args]


def cost_per_million_tokens(hourly_usd: float, tokens_per_second: float) -> float:
    """Dollars per million tokens for a GPU set that sustains `tokens_per_second`."""
    if tokens_per_second <= 0:
        return float("inf")
    return hourly_usd / (tokens_per_second * 3600) * 1_000_000
