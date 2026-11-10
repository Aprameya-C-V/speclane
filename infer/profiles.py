"""Serving profiles: the vanilla baseline and tuned candidates for vLLM and SGLang.

A baseline is the engine's own defaults plus only what is needed to serve the model and a common context length, so
vLLM and SGLang are compared on equal memory terms. Every candidate changes one thing from the baseline, so its effect
can be read directly; the sweep later combines the winners.

Flags come from the Qwen model cards, the vLLM recipes and the SGLang cookbook for Qwen3.8-27B. Anything not found in
those sources is marked "unverified". Flag spelling is checked against each engine's --help before any GPU is rented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from models import MODELS, PORT

CONTEXT_LENGTH = 16384  # same for every profile and engine

# Flags a model needs to start at all, per engine (from the recipes, model cards and first launches).
# Qwen3.8-27B-FP8 on one H100 with vLLM 0.33.0: the default max_num_seqs (1024) exceeds the 745 Mamba cache blocks that fit,
# so vanilla vLLM fails with "max_num_seqs (1024) exceeds available Mamba cache blocks (745)". 512 is the smallest round value
# below that limit and is part of the baseline, so the baseline is "defaults plus what is needed to start".
REQUIRED: dict[str, dict[str, tuple[str, ...]]] = {
    "qwen3.5-9b": {"vllm": ("--trust-remote-code",), "sglang": ()},
    "qwen3.8-27b-fp8": {"vllm": ("--max-num-seqs", "512"), "sglang": ()},
}


@dataclass(frozen=True)
class Profile:
    name: str
    engine: str  # "vllm" or "sglang"
    args: tuple[str, ...]
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)


def set_flag(args: tuple[str, ...], flag: str, value: str | None) -> tuple[str, ...]:
    """Replace a flag's value, add the flag, or (value None) remove it. Flags with no value are written as flag only."""
    items = list(args)
    if flag in items:
        index = items.index(flag)
        has_value = index + 1 < len(items) and not items[index + 1].startswith("--")
        end = index + 2 if has_value else index + 1
        del items[index:end]
    if value is not None:
        items += [flag] if value == "" else [flag, value]
    return tuple(items)


def flag_pairs(args: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """The (flag, value) pairs of an argument list, so two lists can be compared without regard to order."""
    pairs, i = set(), 0
    while i < len(args):
        has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
        pairs.add((args[i], args[i + 1] if has_value else ""))
        i += 2 if has_value else 1
    return frozenset(pairs)


def baseline_args(model_key: str, engine: str) -> tuple[str, ...]:
    required = REQUIRED[model_key][engine]
    if engine == "vllm":
        return (*required, "--max-model-len", str(CONTEXT_LENGTH))
    return (*required, "--context-length", str(CONTEXT_LENGTH), "--enable-metrics")


def mtp_vllm(model_key: str, tokens: int) -> str:
    """vLLM speculative config. The 9B card uses method qwen3_next_mtp; the recipes use plain mtp. Unverified which is faster."""
    method = "qwen3_next_mtp" if model_key == "qwen3.5-9b" else "mtp"
    return f'{{"method":"{method}","num_speculative_tokens":{tokens}}}'


def mtp_vllm_adaptive(model_key: str, tokens: int) -> str:
    """MTP speculation with vLLM's adaptive verification (decides per step how much of the draft to verify).
    Documented for MTP and DSpark in vLLM's docs; unverified on these two models until a run proves the flag is accepted."""
    method = "qwen3_next_mtp" if model_key == "qwen3.5-9b" else "mtp"
    return f'{{"method":"{method}","num_speculative_tokens":{tokens},"enable_adaptive_verification":true}}'


def best_mtp_tokens(model_key: str) -> int:
    """Draft length of the round-1 winner (27B: 3 from the recipe; 9B: 2 from the model card)."""
    return 3 if model_key == "qwen3.8-27b-fp8" else 2


def candidates(model_key: str, engine: str) -> list[Profile]:
    """The baseline plus one-factor-at-a-time variations for this model and engine."""
    base = baseline_args(model_key, engine)
    profiles = [Profile("baseline", engine, base, "engine defaults, common context length")]

    def variant(name: str, description: str, *changes: tuple[str, str | None]) -> None:
        args = base
        for flag, value in changes:
            args = set_flag(args, flag, value)
        if flag_pairs(args) == flag_pairs(base):  # already part of the baseline (a required flag): not a variation
            return
        profiles.append(Profile(name, engine, args, description))

    if engine == "vllm":
        variant("gpu-util-95", "more memory for KV cache", ("--gpu-memory-utilization", "0.95"))
        variant("kv-fp8", "FP8 KV cache: more concurrent tokens, lossy", ("--kv-cache-dtype", "fp8"))
        variant("seqs-256", "allow 256 concurrent sequences", ("--max-num-seqs", "256"))
        variant("seqs-512", "allow 512 concurrent sequences", ("--max-num-seqs", "512"))
        variant("batched-8k", "8192 tokens per scheduler step", ("--max-num-batched-tokens", "8192"))
        variant("batched-16k", "16384 tokens per scheduler step", ("--max-num-batched-tokens", "16384"))
        variant("async-scheduling", "overlap scheduling with execution", ("--async-scheduling", ""))
        variant("no-prefix-cache", "prefix caching off, to measure what it is worth", ("--no-enable-prefix-caching", ""))
        for tokens in (1, 2, 3, 5):
            variant(f"mtp-{tokens}", f"MTP speculation, {tokens} draft tokens", ("--speculative-config", mtp_vllm(model_key, tokens)))
        variant("mtp-adaptive", "MTP with adaptive verification at the round-1 draft length",
                ("--speculative-config", mtp_vllm_adaptive(model_key, best_mtp_tokens(model_key))))
        # Round-1 finding: with speculation vLLM caps scheduled tokens at 2048 and suggests raising max_num_batched_tokens.
        variant("mtp-batched-8k", "round-1 MTP winner plus 8192 tokens per scheduler step",
                ("--speculative-config", mtp_vllm(model_key, best_mtp_tokens(model_key))), ("--max-num-batched-tokens", "8192"))
        variant("mtp-batched-16k", "round-1 MTP winner plus 16384 tokens per scheduler step",
                ("--speculative-config", mtp_vllm(model_key, best_mtp_tokens(model_key))), ("--max-num-batched-tokens", "16384"))
        variant("mtp-async", "round-1 MTP winner plus async scheduling",
                ("--speculative-config", mtp_vllm(model_key, best_mtp_tokens(model_key))), ("--async-scheduling", ""))
    else:
        variant("mem-0.90", "more memory for KV cache", ("--mem-fraction-static", "0.90"))
        variant("kv-fp8", "FP8 KV cache: more concurrent tokens, lossy", ("--kv-cache-dtype", "fp8_e4m3"))
        variant("chunked-4k", "4096-token prefill chunks", ("--chunked-prefill-size", "4096"))
        variant("chunked-8k", "8192-token prefill chunks", ("--chunked-prefill-size", "8192"))
        variant("running-256", "allow 256 running requests", ("--max-running-requests", "256"))
        for backend in ("fa3", "flashinfer", "triton"):
            variant(f"attn-{backend}", f"{backend} attention backend", ("--attention-backend", backend))
        variant("mtp-3", "EAGLE-style MTP, 3 steps (SGLang cookbook setting)",
                ("--speculative-algorithm", "EAGLE"), ("--speculative-num-steps", "3"),
                ("--speculative-eagle-topk", "1"), ("--speculative-num-draft-tokens", "4"))
        variant("mamba-lazy", "lazy radix-cache strategy for linear attention layers (cookbook)",
                ("--mamba-radix-cache-strategy", "extra_buffer_lazy"))
    return profiles


def combine(name: str, profiles: list[Profile], model_key: str) -> Profile:
    """Merge the changes of several candidate profiles into one tuned profile, later profiles winning on conflicts."""
    engine = profiles[0].engine
    base = baseline_args(model_key, engine)
    args = base
    for profile in profiles:
        items = list(profile.args)
        i = 0
        while i < len(items):
            flag = items[i]
            has_value = i + 1 < len(items) and not items[i + 1].startswith("--")
            value = items[i + 1] if has_value else ""
            if flag not in base or (has_value and base[base.index(flag) + 1] != value):
                args = set_flag(args, flag, value)
            i += 2 if has_value else 1
    return Profile(name, engine, args, "combined: " + ", ".join(p.name for p in profiles))


def serve_command(model_key: str, profile: Profile, port: int = PORT) -> list[str]:
    repo = MODELS[model_key].repo
    if profile.engine == "vllm":
        return ["vllm", "serve", repo, "--host", "0.0.0.0", "--port", str(port), *profile.args]
    return ["python", "-m", "sglang.launch_server", "--model-path", repo, "--host", "0.0.0.0", "--port", str(port), *profile.args]


def flags_in(profiles: list[Profile]) -> set[str]:
    """Every flag used by these profiles, for checking against an engine's --help."""
    return {item for profile in profiles for item in profile.args if item.startswith("--")}
