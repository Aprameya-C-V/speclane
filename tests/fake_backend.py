"""A fake OpenAI-compatible server with controllable speed and load, for testing the gateway and benchmark."""
from __future__ import annotations

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer


class FakeBackend:
    def __init__(self, tokens: int = 5, token_delay_s: float = 0.0, first_token_delay_s: float = 0.0,
                 running: float = 0, waiting: float = 0, kv_usage: float = 0.0, model: str = "fake") -> None:
        self.tokens = tokens
        self.token_delay_s = token_delay_s
        self.first_token_delay_s = first_token_delay_s
        self.running, self.waiting, self.kv_usage = running, waiting, kv_usage
        self.model = model
        self.healthy = True
        self.fail_requests = False
        self.requests: list[dict] = []
        self.server: TestServer | None = None

    async def health(self, request: web.Request) -> web.Response:
        return web.Response(status=200 if self.healthy else 503)

    async def metrics(self, request: web.Request) -> web.Response:
        return web.Response(text=(
            "# HELP vllm:num_requests_running running\n"
            f"vllm:num_requests_running{{model_name=\"{self.model}\"}} {self.running}\n"
            f"vllm:num_requests_waiting{{model_name=\"{self.model}\"}} {self.waiting}\n"
            f"vllm:kv_cache_usage_perc{{model_name=\"{self.model}\"}} {self.kv_usage}\n"
        ))

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        if self.fail_requests:
            return web.Response(status=500, text="boom")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(self.first_token_delay_s)
        count = min(self.tokens, body.get("max_tokens", self.tokens))
        for i in range(count):
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(self.token_delay_s)
        usage = {"choices": [], "usage": {"completion_tokens": count, "prompt_tokens": 7}}
        await response.write(f"data: {json.dumps(usage)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def start(self) -> str:
        app = web.Application()
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/v1/chat/completions", self.chat)
        self.server = TestServer(app)
        await self.server.start_server()
        return str(self.server.make_url("")).rstrip("/")

    async def stop(self) -> None:
        if self.server:
            await self.server.close()
