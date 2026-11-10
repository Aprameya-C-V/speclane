"""Load generator for OpenAI-compatible servers: time to first token, per-token latency, throughput and cost.

Closed loop (--concurrency N): N requests in flight at all times; measures peak throughput.
Open loop (--rate R): Poisson arrivals at R requests per second; measures latency under a given load, the way
real traffic behaves, because a slow server cannot slow the arrivals down.

    python infer/bench.py --url http://localhost:8000 --model m --concurrency 16 --requests 64 --input-tokens 1000 --output-tokens 256
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field

import aiohttp

from models import cost_per_million_tokens


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    output_tokens: int = 0
    error: str = ""
    start_s: float = 0.0  # offset from the start of the run
    prompt_tokens: int = 0  # as counted by the server, when it reports usage
    text: str = ""  # the generated text, kept so callers can score or continue it
    session: int = -1  # multi-turn runs: which conversation this belongs to
    turn: int = 0


@dataclass
class Summary:
    requests: int
    ok: int
    wall_s: float
    output_tokens: int
    output_tokens_per_s: float
    requests_per_s: float
    total_tokens_per_s: float  # prompt plus output tokens, as the server counts them
    prompt_tokens: int
    ttft_p50_s: float
    ttft_p90_s: float
    ttft_p99_s: float
    e2e_p50_s: float
    e2e_p90_s: float
    e2e_p99_s: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    cost_per_million_output_tokens_usd: float | None = None
    errors: dict[str, int] = field(default_factory=dict)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; 0 for no data."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))]


def summarize(results: list[RequestResult], wall_s: float, gpu_hourly_usd: float | None = None) -> Summary:
    good = [r for r in results if r.ok]
    tokens = sum(r.output_tokens for r in good)
    tpot = [(r.e2e_s - r.ttft_s) / (r.output_tokens - 1) * 1000 for r in good if r.output_tokens > 1]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error[:80]] = errors.get(r.error[:80], 0) + 1
    tokens_per_s = tokens / wall_s if wall_s > 0 else 0.0
    return Summary(
        requests=len(results),
        ok=len(good),
        wall_s=round(wall_s, 3),
        output_tokens=tokens,
        output_tokens_per_s=round(tokens_per_s, 2),
        requests_per_s=round(len(good) / wall_s, 3) if wall_s > 0 else 0.0,
        total_tokens_per_s=round((tokens + sum(r.prompt_tokens for r in good)) / wall_s, 2) if wall_s > 0 else 0.0,
        prompt_tokens=sum(r.prompt_tokens for r in good),
        ttft_p50_s=round(percentile([r.ttft_s for r in good], 0.5), 4),
        ttft_p90_s=round(percentile([r.ttft_s for r in good], 0.9), 4),
        ttft_p99_s=round(percentile([r.ttft_s for r in good], 0.99), 4),
        e2e_p50_s=round(percentile([r.e2e_s for r in good], 0.5), 4),
        e2e_p90_s=round(percentile([r.e2e_s for r in good], 0.9), 4),
        e2e_p99_s=round(percentile([r.e2e_s for r in good], 0.99), 4),
        tpot_p50_ms=round(statistics.median(tpot), 2) if tpot else 0.0,
        tpot_p90_ms=round(percentile(tpot, 0.9), 2),
        cost_per_million_output_tokens_usd=(
            round(cost_per_million_tokens(gpu_hourly_usd, tokens_per_s), 3) if gpu_hourly_usd else None
        ),
        errors=errors,
    )


WORDS = ("alpha river stone engine paper window garden signal copper marble forest lantern harbor orbit velvet "
         "thunder ribbon canvas meadow quartz saddle timber violet anchor bridge candle desert ember falcon").split()


def synthetic_prompt(input_tokens: int, rng: random.Random) -> str:
    """Random words, about 1.3 tokens each, so the prompt is not served from a prefix cache by accident."""
    return " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(max(1, int(input_tokens / 2))))


def load_trace(path: str) -> list[list[dict]]:
    """Replay file: one JSON object per line with `messages` (chat format) or `prompt` (plain text)."""
    conversations = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                conversations.append(row["messages"] if "messages" in row else [{"role": "user", "content": row["prompt"]}])
    return conversations


async def send_request(session: aiohttp.ClientSession, url: str, payload: dict, run_start: float) -> RequestResult:
    """One streaming chat completion; time to first token is the first chunk that carries text."""
    started = time.perf_counter()
    result = RequestResult(ok=False, start_s=started - run_start)
    chunks = 0
    usage_tokens = None
    parts: list[str] = []
    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload) as response:
            if response.status != 200:
                result.error = f"http {response.status}: {(await response.text())[:60]}"
                return result
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                event = json.loads(line[5:])
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                    result.prompt_tokens = event["usage"].get("prompt_tokens") or 0
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        if chunks == 0:
                            result.ttft_s = time.perf_counter() - started
                        chunks += 1
                    if delta.get("content"):
                        parts.append(delta["content"])
        result.e2e_s = time.perf_counter() - started
        result.text = "".join(parts)
        result.output_tokens = usage_tokens if usage_tokens is not None else chunks
        result.ok = chunks > 0
        if not result.ok:
            result.error = "no tokens"
    except Exception as error:  # a failed request is a data point, not a crash
        result.error = f"{type(error).__name__}: {error}"[:80]
    return result


def make_payload(model: str, messages: list[dict], output_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def run_benchmark(
    url: str,
    model: str,
    conversations: list[list[dict]],
    output_tokens: int,
    concurrency: int | None = None,
    rate: float | None = None,
    seed: int = 0,
) -> tuple[list[RequestResult], float]:
    """Run all conversations; `concurrency` selects closed loop, `rate` open loop. Returns results and wall time."""
    if (concurrency is None) == (rate is None):
        raise ValueError("give exactly one of concurrency or rate")
    rng = random.Random(seed)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        run_start = time.perf_counter()
        tasks: list[asyncio.Task] = []
        if concurrency is not None:
            queue: asyncio.Queue = asyncio.Queue()
            for messages in conversations:
                queue.put_nowait(messages)
            results: list[RequestResult] = []

            async def worker() -> None:
                while True:
                    try:
                        messages = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    results.append(await send_request(session, url, make_payload(model, messages, output_tokens), run_start))

            await asyncio.gather(*(worker() for _ in range(concurrency)))
        else:
            assert rate is not None
            for messages in conversations:
                tasks.append(asyncio.create_task(send_request(session, url, make_payload(model, messages, output_tokens), run_start)))
                await asyncio.sleep(rng.expovariate(rate))
            results = list(await asyncio.gather(*tasks))
        return results, time.perf_counter() - run_start


async def run_sessions(
    url: str,
    model: str,
    sessions: list[tuple[str, list[str]]],
    output_tokens: int,
    concurrency: int,
) -> tuple[list[RequestResult], float]:
    """Multi-turn conversations: each session is (system prompt, user turns). Turns run in order and every request
    carries the whole history, as an agent loop does, so a shared system prompt and growing prefix can be cached.
    `concurrency` sessions run at once."""
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    queue: asyncio.Queue = asyncio.Queue()
    for index, session in enumerate(sessions):
        queue.put_nowait((index, session))
    results: list[RequestResult] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        run_start = time.perf_counter()

        async def worker() -> None:
            while True:
                try:
                    index, (system, turns) = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                messages = [{"role": "system", "content": system}]
                for turn, user_text in enumerate(turns):
                    messages.append({"role": "user", "content": user_text})
                    result = await send_request(http, url, make_payload(model, messages, output_tokens), run_start)
                    result.session, result.turn = index, turn
                    results.append(result)
                    if not result.ok:
                        break
                    messages.append({"role": "assistant", "content": result.text})

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        return results, time.perf_counter() - run_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--rate", type=float, help="open-loop arrivals per second")
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--trace", help="JSONL conversations to replay instead of synthetic prompts")
    parser.add_argument("--gpu-hourly-usd", type=float, help="price of the GPUs, for cost per million tokens")
    parser.add_argument("--out", help="write the summary and raw results as JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.trace:
        conversations = load_trace(args.trace)[: args.requests]
    else:
        conversations = [[{"role": "user", "content": synthetic_prompt(args.input_tokens, rng)}] for _ in range(args.requests)]
    results, wall = asyncio.run(run_benchmark(args.url, args.model, conversations, args.output_tokens, args.concurrency, args.rate, args.seed))
    summary = summarize(results, wall, args.gpu_hourly_usd)
    print(json.dumps(asdict(summary), indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": asdict(summary), "results": [asdict(r) for r in results], "args": vars(args)}, handle)


if __name__ == "__main__":
    main()
