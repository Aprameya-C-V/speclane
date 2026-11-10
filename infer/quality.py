"""Quality guard: a speed optimization that changes answers is not an optimization.

Scores the model on GSM8K test questions (grade-school math) by exact numeric answer, so a profile's accuracy can be
compared with the baseline's. Lossless features (speculative decoding with exact verification) should leave accuracy
unchanged; lossy ones (FP8 KV cache) can move it, and this measures by how much.
"""
from __future__ import annotations

import asyncio
import re
import random

import aiohttp

from bench import make_payload, send_request

INSTRUCTION = "Solve the problem step by step. End with a line of the form 'Answer: <number>'."


def build_prompt(question: str) -> str:
    return f"{question}\n\n{INSTRUCTION}"


def to_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """The number after the last 'Answer:'; if there is none, the last number in the text."""
    marked = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if marked:
        return to_number(marked[-1])
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return to_number(numbers[-1]) if numbers else None


def reference_answer(solution: str) -> float | None:
    """GSM8K solutions end with '#### <number>'."""
    return to_number(solution.split("####")[-1])


def accuracy(predictions: list[float | None], references: list[float | None]) -> float:
    correct = sum(1 for p, r in zip(predictions, references) if p is not None and r is not None and abs(p - r) < 1e-6)
    return correct / len(references) if references else 0.0


def load_gsm8k(count: int, seed: int = 0) -> list[tuple[str, float | None]]:
    """(question, reference answer) pairs from the GSM8K test split; downloads from Hugging Face."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    frame = pd.read_parquet(path)
    rows = list(zip(frame["question"], frame["answer"]))
    random.Random(seed).shuffle(rows)
    return [(question, reference_answer(answer)) for question, answer in rows[:count]]


async def evaluate(url: str, model: str, problems: list[tuple[str, float | None]], concurrency: int = 32,
                   max_tokens: int = 1024) -> dict:
    """Ask every problem once at temperature 0 and score the answers."""
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    queue: asyncio.Queue = asyncio.Queue()
    for index, problem in enumerate(problems):
        queue.put_nowait((index, problem))
    answers: dict[int, float | None] = {}
    failures = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:

        async def worker() -> None:
            nonlocal failures
            while True:
                try:
                    index, (question, _) = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = make_payload(model, [{"role": "user", "content": build_prompt(question)}], max_tokens)
                # Thinking mode makes answers long; the guard compares engines on the same short-answer setting.
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                result = await send_request(http, url, payload, 0.0)
                if result.ok:
                    answers[index] = extract_answer(result.text)
                else:
                    failures += 1
                    answers[index] = None

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    predictions = [answers.get(i) for i in range(len(problems))]
    references = [reference for _, reference in problems]
    return {"questions": len(problems), "accuracy": round(accuracy(predictions, references), 4), "failures": failures}
