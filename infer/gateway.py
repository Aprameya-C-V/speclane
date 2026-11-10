"""A small inference gateway: one OpenAI-compatible endpoint in front of several model servers.

It polls each server's health and vLLM metrics, picks a server per request with a pluggable policy, streams the
answer through, fails over when a server is down, and logs one JSON line per request.

    python infer/gateway.py --backend deepseek=https://a.modal.run --backend glm=https://b.modal.run --policy least-loaded
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Protocol

import aiohttp
from aiohttp import web

# vLLM metric names; they have changed between releases, so the poller also accepts the older forms.
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


@dataclass
class Backend:
    name: str
    url: str
    models: list[str] = field(default_factory=list)  # served model names; empty means any
    healthy: bool = True
    running: float = 0.0
    waiting: float = 0.0
    kv_usage: float = 0.0
    inflight: int = 0  # requests this gateway has open to the server right now
    served: int = 0
    failures: int = 0


@dataclass
class RequestInfo:
    model: str  # requested model name, or "auto"
    prompt_chars: int
    has_tools: bool


class Policy(Protocol):
    def choose(self, candidates: list[Backend], request: RequestInfo) -> Backend: ...


class RoundRobin:
    def __init__(self) -> None:
        self.counter = 0

    def choose(self, candidates: list[Backend], request: RequestInfo) -> Backend:
        backend = candidates[self.counter % len(candidates)]
        self.counter += 1
        return backend


class LeastLoaded:
    """Pick the backend with the least work queued or running, using live server state plus our own open requests."""

    def __init__(self, kv_weight: float = 4.0) -> None:
        self.kv_weight = kv_weight

    def load(self, backend: Backend) -> float:
        return backend.inflight + backend.waiting + 0.5 * backend.running + self.kv_weight * backend.kv_usage

    def choose(self, candidates: list[Backend], request: RequestInfo) -> Backend:
        return min(candidates, key=self.load)


POLICIES = {"round-robin": RoundRobin, "least-loaded": LeastLoaded}


def parse_metrics(text: str) -> dict[str, float]:
    """Read the vLLM gauges the router needs out of Prometheus text; missing ones are left out."""
    wanted = {"running": RUNNING, "waiting": WAITING, "kv_usage": KV_USAGE}
    found: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        for key, names in wanted.items():
            if name in names and key not in found:
                found[key] = float(line.rsplit(" ", 1)[1])
    return found


class Gateway:
    def __init__(self, backends: list[Backend], policy: Policy, log_path: str | None = None, poll_interval_s: float = 1.0) -> None:
        self.backends = backends
        self.policy = policy
        self.log_path = log_path
        self.poll_interval_s = poll_interval_s
        self.session: aiohttp.ClientSession | None = None
        self.poller: asyncio.Task | None = None

    async def start(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=900))
        self.poller = asyncio.create_task(self.poll_forever())

    async def stop(self, app: web.Application) -> None:
        if self.poller:
            self.poller.cancel()
        if self.session:
            await self.session.close()

    async def poll_forever(self) -> None:
        while True:
            await asyncio.gather(*(self.poll(backend) for backend in self.backends))
            await asyncio.sleep(self.poll_interval_s)

    async def poll(self, backend: Backend) -> None:
        assert self.session
        try:
            async with self.session.get(f"{backend.url}/health", timeout=aiohttp.ClientTimeout(total=3)) as response:
                backend.healthy = response.status == 200
            if backend.healthy:
                async with self.session.get(f"{backend.url}/metrics", timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        state = parse_metrics(await response.text())
                        backend.running = state.get("running", backend.running)
                        backend.waiting = state.get("waiting", backend.waiting)
                        backend.kv_usage = state.get("kv_usage", backend.kv_usage)
        except Exception:
            backend.healthy = False

    def candidates(self, request: RequestInfo, exclude: set[str]) -> list[Backend]:
        pool = [b for b in self.backends if b.healthy and b.name not in exclude]
        if request.model != "auto":
            pool = [b for b in pool if not b.models or request.model in b.models or request.model == b.name]
        return pool

    def log(self, record: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    async def chat(self, request: web.Request) -> web.StreamResponse:
        assert self.session
        body = await request.json()
        info = RequestInfo(
            model=body.get("model", "auto"),
            prompt_chars=sum(len(str(m.get("content", ""))) for m in body.get("messages", [])),
            has_tools=bool(body.get("tools")),
        )
        tried: set[str] = set()
        started = time.perf_counter()
        while True:
            candidates = self.candidates(info, tried)
            if not candidates:
                self.log({"ts": time.time(), "status": 503, "model": info.model, "tried": sorted(tried)})
                return web.json_response({"error": "no healthy backend"}, status=503)
            backend = self.policy.choose(candidates, info)
            tried.add(backend.name)
            payload = {**body, "model": backend.models[0]} if info.model == "auto" and backend.models else body
            backend.inflight += 1
            response: web.StreamResponse | None = None
            ttft = None
            try:
                async with self.session.post(f"{backend.url}/v1/chat/completions", json=payload) as upstream:
                    if upstream.status >= 500:
                        raise ConnectionError(f"upstream {upstream.status}")
                    response = web.StreamResponse(status=upstream.status, headers={
                        "Content-Type": upstream.headers.get("Content-Type", "application/json"),
                        "X-Backend": backend.name,
                    })
                    await response.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        await response.write(chunk)
                    await response.write_eof()
                backend.served += 1
                self.log({"ts": time.time(), "status": 200, "backend": backend.name, "model": info.model,
                          "ttft_s": ttft, "e2e_s": time.perf_counter() - started, "prompt_chars": info.prompt_chars})
                return response
            except Exception as error:
                backend.failures += 1
                if response is not None and response.prepared:
                    # Bytes already reached the client; a retry would duplicate output, so end the stream.
                    self.log({"ts": time.time(), "status": "broken", "backend": backend.name, "error": str(error)[:80]})
                    return response
                backend.healthy = False if isinstance(error, (ConnectionError, aiohttp.ClientConnectionError)) else backend.healthy
            finally:
                backend.inflight -= 1

    async def state(self, request: web.Request) -> web.Response:
        return web.json_response([{k: getattr(b, k) for k in
                                   ("name", "url", "models", "healthy", "running", "waiting", "kv_usage", "inflight", "served", "failures")}
                                  for b in self.backends])


def make_app(gateway: Gateway) -> web.Application:
    app = web.Application()
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.stop)
    app.router.add_post("/v1/chat/completions", gateway.chat)
    app.router.add_get("/gateway/state", gateway.state)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", action="append", required=True, help="name=url, repeatable")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="least-loaded")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log", default="gateway.jsonl")
    args = parser.parse_args()
    backends = []
    for spec in args.backend:
        name, url = spec.split("=", 1)
        backends.append(Backend(name=name, url=url.rstrip("/")))
    web.run_app(make_app(Gateway(backends, POLICIES[args.policy](), args.log)), port=args.port)


if __name__ == "__main__":
    main()
