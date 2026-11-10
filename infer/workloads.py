"""Benchmark workloads that resemble real traffic, instead of one random-prompt shape.

Each workload stresses a different part of an engine:
  chat_short       short prompt, short answer: scheduler overhead and per-token latency
  rag_long         long prompt, short answer: prefill speed
  reasoning_long   short prompt, long answer: decode speed and speculation
  shared_prefix    one long shared system prompt, unique questions: prefix caching
  agent_sessions   multi-turn conversations with a shared system prompt and growing history: prefix caching under load
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from bench import synthetic_prompt


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    input_tokens: int
    output_tokens: int
    shared_prefix_tokens: int = 0
    turns: int = 1  # more than 1 makes it a multi-turn session workload


WORKLOADS: dict[str, Workload] = {
    "chat_short": Workload("chat_short", "short prompt, short answer", 300, 300),
    "rag_long": Workload("rag_long", "long prompt, short answer", 6000, 256),
    "reasoning_long": Workload("reasoning_long", "short prompt, long answer", 500, 2048),
    "shared_prefix": Workload("shared_prefix", "long shared system prompt, unique question", 150, 256, shared_prefix_tokens=3000),
    "agent_sessions": Workload("agent_sessions", "6-turn sessions, shared system prompt, growing history", 200, 200, shared_prefix_tokens=2000, turns=6),
}


def is_session_workload(workload: Workload) -> bool:
    return workload.turns > 1


def build_conversations(workload: Workload, count: int, seed: int = 0) -> list[list[dict]]:
    """Independent single-turn requests. A shared prefix is the same system message in every request."""
    rng = random.Random(seed)
    system = synthetic_prompt(workload.shared_prefix_tokens, random.Random(seed + 1)) if workload.shared_prefix_tokens else ""
    conversations = []
    for _ in range(count):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": synthetic_prompt(workload.input_tokens, rng)})
        conversations.append(messages)
    return conversations


def build_sessions(workload: Workload, count: int, seed: int = 0) -> list[tuple[str, list[str]]]:
    """Multi-turn sessions: (system prompt, user turns). All sessions share the same system prompt."""
    rng = random.Random(seed)
    system = synthetic_prompt(workload.shared_prefix_tokens, random.Random(seed + 1))
    return [(system, [synthetic_prompt(workload.input_tokens, rng) for _ in range(workload.turns)]) for _ in range(count)]
