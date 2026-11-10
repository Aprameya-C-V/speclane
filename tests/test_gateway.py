import json

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from fake_backend import FakeBackend
from gateway import Backend, Gateway, LeastLoaded, RoundRobin, make_app, parse_metrics


def test_parse_metrics_reads_vllm_gauges_and_ignores_the_rest():
    text = (
        "# HELP x y\n"
        'vllm:num_requests_running{model_name="m"} 3.0\n'
        'vllm:num_requests_waiting{model_name="m"} 7\n'
        'vllm:kv_cache_usage_perc{model_name="m"} 0.42\n'
        'vllm:other{a="b"} 9\n'
    )
    assert parse_metrics(text) == {"running": 3.0, "waiting": 7.0, "kv_usage": 0.42}
    assert parse_metrics('vllm:gpu_cache_usage_perc{m="x"} 0.5') == {"kv_usage": 0.5}
    assert parse_metrics("") == {}


def test_least_loaded_prefers_the_idle_backend():
    busy = Backend("busy", "u", waiting=5, running=4)
    idle = Backend("idle", "u")
    assert LeastLoaded().choose([busy, idle], None) is idle  # type: ignore[arg-type]
    # Open requests count even before the server reports them.
    idle.inflight = 20
    assert LeastLoaded().choose([busy, idle], None) is busy  # type: ignore[arg-type]


def test_round_robin_alternates():
    a, b = Backend("a", "u"), Backend("b", "u")
    policy = RoundRobin()
    assert [policy.choose([a, b], None).name for _ in range(4)] == ["a", "b", "a", "b"]  # type: ignore[arg-type]


async def start_gateway(backends: list[Backend], policy, tmp_path):
    gateway = Gateway(backends, policy, log_path=str(tmp_path / "g.jsonl"), poll_interval_s=0.05)
    server = TestServer(make_app(gateway))
    await server.start_server()
    return gateway, server


async def chat(server: TestServer, model: str = "auto") -> tuple[int, str, str]:
    async with aiohttp.ClientSession() as session:
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 3}
        async with session.post(str(server.make_url("/v1/chat/completions")), json=body) as response:
            return response.status, response.headers.get("X-Backend", ""), (await response.read()).decode()


async def test_streams_the_backend_answer_and_logs_the_request(tmp_path):
    fake = FakeBackend(tokens=3)
    url = await fake.start()
    gateway, server = await start_gateway([Backend("one", url, models=["fake"])], LeastLoaded(), tmp_path)
    try:
        status, backend, text = await chat(server)
        assert status == 200 and backend == "one"
        assert text.count("data:") >= 4 and "[DONE]" in text
        assert fake.requests[0]["model"] == "fake"  # "auto" was rewritten to the backend's model name
    finally:
        await server.close()
        await fake.stop()
    record = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert record["status"] == 200 and record["backend"] == "one" and record["ttft_s"] is not None


async def test_routes_around_a_loaded_backend_using_polled_metrics(tmp_path):
    busy, idle = FakeBackend(waiting=30, running=8, kv_usage=0.9), FakeBackend()
    busy_url, idle_url = await busy.start(), await idle.start()
    gateway, server = await start_gateway([Backend("busy", busy_url), Backend("idle", idle_url)], LeastLoaded(), tmp_path)
    try:
        await gateway.poll(gateway.backends[0])
        await gateway.poll(gateway.backends[1])
        assert gateway.backends[0].waiting == 30 and gateway.backends[0].kv_usage == 0.9
        chosen = [(await chat(server))[1] for _ in range(4)]
        assert chosen == ["idle"] * 4
    finally:
        await server.close()
        await busy.stop()
        await idle.stop()


async def test_fails_over_when_a_backend_returns_server_errors(tmp_path):
    broken, good = FakeBackend(), FakeBackend()
    broken.fail_requests = True
    broken_url, good_url = await broken.start(), await good.start()
    gateway, server = await start_gateway([Backend("broken", broken_url), Backend("good", good_url)], RoundRobin(), tmp_path)
    try:
        status, backend, _ = await chat(server)
        assert status == 200 and backend == "good"
        assert gateway.backends[0].failures == 1
    finally:
        await server.close()
        await broken.stop()
        await good.stop()


async def test_named_model_only_goes_to_matching_backends(tmp_path):
    a, b = FakeBackend(model="a"), FakeBackend(model="b")
    a_url, b_url = await a.start(), await b.start()
    gateway, server = await start_gateway([Backend("a", a_url, models=["model-a"]), Backend("b", b_url, models=["model-b"])], RoundRobin(), tmp_path)
    try:
        assert [(await chat(server, "model-b"))[1] for _ in range(3)] == ["b", "b", "b"]
    finally:
        await server.close()
        await a.stop()
        await b.stop()


async def test_returns_503_when_nothing_is_healthy(tmp_path):
    fake = FakeBackend()
    url = await fake.start()
    backend = Backend("only", url)
    backend.healthy = False
    gateway, server = await start_gateway([backend], RoundRobin(), tmp_path)
    try:
        # The poller may mark it healthy again; stop it so the test controls state.
        gateway.poller.cancel()
        status, _, _ = await chat(server)
        assert status == 503
    finally:
        await server.close()
        await fake.stop()


async def test_state_endpoint_reports_backends(tmp_path):
    fake = FakeBackend()
    url = await fake.start()
    gateway, server = await start_gateway([Backend("one", url)], RoundRobin(), tmp_path)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(str(server.make_url("/gateway/state"))) as response:
                state = await response.json()
        assert state[0]["name"] == "one" and "kv_usage" in state[0]
    finally:
        await server.close()
        await fake.stop()
