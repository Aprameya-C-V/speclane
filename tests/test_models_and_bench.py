import random

import pytest

from bench import RequestResult, percentile, run_benchmark, summarize, synthetic_prompt
from fake_backend import FakeBackend
from models import MODELS, cost_per_million_tokens, serve_command


def test_serve_command_uses_recipe_flags_and_drops_overrides():
    config = MODELS["deepseek-v4.1-flash"]
    command = serve_command(config)
    assert command[:3] == ["vllm", "serve", "deepseek-ai/DeepSeek-V4.1-Flash-1017"]
    assert command[command.index("--tensor-parallel-size") + 1] == "4"

    tuned = serve_command(config, {"--max-num-seqs": "256", "--max-num-active-seqs": None, "--max-num-batched-tokens": "8192"})
    assert tuned[tuned.index("--max-num-seqs") + 1] == "256"
    assert "--max-num-active-seqs" not in tuned
    assert tuned[tuned.index("--max-num-batched-tokens") + 1] == "8192"
    # The original is untouched.
    assert "--max-num-active-seqs" in serve_command(config)


def test_serve_command_drops_a_flag_without_a_value():
    command = serve_command(MODELS["deepseek-v4.1-flash"], {"--enable-auto-tool-choice": None})
    assert "--enable-auto-tool-choice" not in command
    assert "--tool-call-parser" in command


def test_every_model_has_a_gpu_price_and_a_port():
    for config in MODELS.values():
        assert config.gpu_hourly_usd > 0
        assert "--port" in serve_command(config)


def test_qwen_models_use_one_gpu_and_can_drop_speculation():
    for key in ("qwen3.5-4b", "qwen3.5-9b", "qwen3.8-27b-fp8", "qwen3.6-35b-a3b-fp8"):
        config = MODELS[key]
        assert config.repo.startswith("Qwen/") and "B200" not in config.gpu and ":" not in config.gpu
        command = serve_command(config)
        assert command[command.index("--tensor-parallel-size") + 1] == "1"
        assert "--speculative-config" in command
        plain = serve_command(config, {"--speculative-config": None})
        assert "--speculative-config" not in plain and "--tensor-parallel-size" in plain
    # More speculative tokens than the recipe uses, for a sweep.
    config = MODELS["qwen3.8-27b-fp8"]
    tuned = serve_command(config, {"--speculative-config": '{"method":"mtp","num_speculative_tokens":5}'})
    assert '"num_speculative_tokens":5' in tuned[tuned.index("--speculative-config") + 1]


def test_cost_per_million_tokens():
    # $25/hour at 1,000 tokens/s: 3.6M tokens/hour, so $6.94 per million.
    assert cost_per_million_tokens(25, 1000) == pytest.approx(6.944, abs=0.001)
    assert cost_per_million_tokens(25, 0) == float("inf")


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == 2
    assert percentile([1, 2, 3, 4], 0.99) == 4
    assert percentile([], 0.5) == 0


def test_summarize_computes_throughput_and_errors():
    results = [
        RequestResult(ok=True, ttft_s=0.1, e2e_s=1.1, output_tokens=11),
        RequestResult(ok=True, ttft_s=0.3, e2e_s=1.3, output_tokens=11),
        RequestResult(ok=False, error="http 500"),
    ]
    summary = summarize(results, wall_s=2.0, gpu_hourly_usd=25)
    assert summary.ok == 2 and summary.requests == 3
    assert summary.output_tokens_per_s == 11.0
    assert summary.tpot_p50_ms == pytest.approx(100.0)  # (1.0 s decode) / 10 gaps
    assert summary.errors == {"http 500": 1}
    assert summary.cost_per_million_output_tokens_usd == pytest.approx(25 / (11 * 3600) * 1e6, abs=0.01)


def test_synthetic_prompts_differ():
    rng = random.Random(1)
    assert synthetic_prompt(100, rng) != synthetic_prompt(100, rng)


async def test_closed_loop_benchmark_against_a_fake_server():
    fake = FakeBackend(tokens=8, token_delay_s=0.005, first_token_delay_s=0.02)
    url = await fake.start()
    try:
        conversations = [[{"role": "user", "content": f"hello {i}"}] for i in range(12)]
        results, wall = await run_benchmark(url, "fake", conversations, output_tokens=8, concurrency=4)
    finally:
        await fake.stop()
    assert len(results) == 12 and all(r.ok for r in results)
    assert all(r.output_tokens == 8 for r in results)
    assert all(r.ttft_s >= 0.02 for r in results)
    assert len(fake.requests) == 12 and fake.requests[0]["stream"] is True


async def test_open_loop_benchmark_arrives_on_schedule():
    fake = FakeBackend(tokens=3)
    url = await fake.start()
    try:
        conversations = [[{"role": "user", "content": "x"}] for _ in range(10)]
        results, wall = await run_benchmark(url, "fake", conversations, output_tokens=3, rate=50.0, seed=3)
    finally:
        await fake.stop()
    assert len(results) == 10 and all(r.ok for r in results)
    starts = sorted(r.start_s for r in results)
    assert starts[-1] > starts[0]  # arrivals were spread out, not fired together


async def test_server_errors_are_recorded_not_raised():
    fake = FakeBackend()
    fake.fail_requests = True
    url = await fake.start()
    try:
        results, _ = await run_benchmark(url, "fake", [[{"role": "user", "content": "x"}]] * 3, 4, concurrency=2)
    finally:
        await fake.stop()
    assert [r.ok for r in results] == [False, False, False]
    assert "http 500" in results[0].error


async def test_run_benchmark_needs_exactly_one_mode():
    with pytest.raises(ValueError):
        await run_benchmark("http://x", "m", [], 1)
    with pytest.raises(ValueError):
        await run_benchmark("http://x", "m", [], 1, concurrency=1, rate=1.0)
