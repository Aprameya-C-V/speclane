import random

import pytest

from bench import RequestResult, run_sessions, summarize
from fake_backend import FakeBackend
from profiles import CONTEXT_LENGTH, REQUIRED, baseline_args, candidates, combine, flags_in, serve_command, set_flag
from quality import accuracy, build_prompt, extract_answer, reference_answer, to_number
from workloads import WORKLOADS, build_conversations, build_sessions, is_session_workload

MODELS = ("qwen3.8-27b-fp8", "qwen3.5-9b")
ENGINES = ("vllm", "sglang")


def test_set_flag_replaces_adds_and_removes():
    args = ("--a", "1", "--b")
    assert set_flag(args, "--a", "2") == ("--b", "--a", "2")
    assert set_flag(args, "--c", "3") == ("--a", "1", "--b", "--c", "3")
    assert set_flag(args, "--b", None) == ("--a", "1")
    assert set_flag(args, "--d", "") == ("--a", "1", "--b", "--d")  # a valueless flag


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
def test_baseline_is_minimal_and_equal_context_across_engines(model, engine):
    base = baseline_args(model, engine)
    assert str(CONTEXT_LENGTH) in base
    # Nothing tuned: no speculation, no cache dtype, no scheduler limits.
    required = set(REQUIRED[model][engine])
    for tuned in ("--speculative-config", "--speculative-algorithm", "--kv-cache-dtype", "--max-num-seqs", "--max-running-requests"):
        assert tuned not in base or tuned in required  # only a flag needed to start may appear
    assert required <= set(base)


def test_the_27b_baseline_carries_the_flag_it_needs_to_start_on_vllm():
    base = baseline_args("qwen3.8-27b-fp8", "vllm")
    assert base[base.index("--max-num-seqs") + 1] == "512"
    # A candidate that only repeats a baseline flag is not a variation and is not generated.
    assert "seqs-512" not in {p.name for p in candidates("qwen3.8-27b-fp8", "vllm")}
    assert "seqs-512" in {p.name for p in candidates("qwen3.5-9b", "vllm")}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
def test_each_candidate_differs_from_baseline_and_names_are_unique(model, engine):
    profiles = candidates(model, engine)
    assert profiles[0].name == "baseline"
    names = [p.name for p in profiles]
    assert len(names) == len(set(names))
    base_args = profiles[0].args
    assert all(p.args != base_args for p in profiles[1:])
    assert all(p.engine == engine for p in profiles)


def test_commands_target_the_right_engine_and_model():
    vllm = serve_command("qwen3.8-27b-fp8", candidates("qwen3.8-27b-fp8", "vllm")[0])
    assert vllm[:3] == ["vllm", "serve", "Qwen/Qwen3.8-27B-FP8"] and vllm[vllm.index("--port") + 1] == "8000"
    sglang = serve_command("qwen3.5-9b", candidates("qwen3.5-9b", "sglang")[0])
    assert sglang[:4] == ["python", "-m", "sglang.launch_server", "--model-path"] and "Qwen/Qwen3.5-9B" in sglang
    assert "--enable-metrics" in sglang


def test_the_9b_card_method_name_is_used_for_vllm_speculation():
    nine = {p.name: p for p in candidates("qwen3.5-9b", "vllm")}
    twenty_seven = {p.name: p for p in candidates("qwen3.8-27b-fp8", "vllm")}
    assert "qwen3_next_mtp" in " ".join(nine["mtp-2"].args)
    assert '"method":"mtp"' in " ".join(twenty_seven["mtp-3"].args)


def test_combine_merges_changes_and_later_profiles_win():
    pool = {p.name: p for p in candidates("qwen3.5-9b", "vllm")}  # its baseline has no max-num-seqs, so both variants differ
    tuned = combine("tuned", [pool["mtp-2"], pool["kv-fp8"], pool["seqs-256"], pool["seqs-512"]], "qwen3.5-9b")
    args = tuned.args
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--max-num-seqs") + 1] == "512"
    assert "--speculative-config" in args and str(CONTEXT_LENGTH) in args
    assert args.count("--max-num-seqs") == 1


def test_flags_in_collects_every_flag():
    flags = flags_in(candidates("qwen3.8-27b-fp8", "sglang"))
    assert {"--mem-fraction-static", "--attention-backend", "--speculative-algorithm"} <= flags


def test_workload_shapes():
    assert set(WORKLOADS) == {"chat_short", "rag_long", "reasoning_long", "shared_prefix", "agent_sessions"}
    assert is_session_workload(WORKLOADS["agent_sessions"]) and not is_session_workload(WORKLOADS["chat_short"])
    conversations = build_conversations(WORKLOADS["shared_prefix"], 5)
    systems = {c[0]["content"] for c in conversations}
    users = {c[-1]["content"] for c in conversations}
    assert len(systems) == 1 and len(users) == 5  # same shared prefix, different questions
    assert build_conversations(WORKLOADS["chat_short"], 2)[0][0]["role"] == "user"


def test_sessions_share_a_system_prompt_and_have_the_right_turns():
    sessions = build_sessions(WORKLOADS["agent_sessions"], 4)
    assert len({system for system, _ in sessions}) == 1
    assert all(len(turns) == 6 for _, turns in sessions)
    assert build_sessions(WORKLOADS["agent_sessions"], 2, seed=1) != build_sessions(WORKLOADS["agent_sessions"], 2, seed=2)


def test_warm_up_seed_does_not_share_the_measured_prefix():
    measured = build_conversations(WORKLOADS["shared_prefix"], 1, seed=0)[0][0]["content"]
    warm = build_conversations(WORKLOADS["shared_prefix"], 1, seed=999)[0][0]["content"]
    assert measured != warm


def test_extract_answer_prefers_the_marked_answer():
    assert extract_answer("so 3 + 4 = 7\nAnswer: 1,234") == 1234.0
    assert extract_answer("Answer: $12.50") == 12.5
    assert extract_answer("first Answer: 5 then Answer: 9") == 9.0
    assert extract_answer("the total is 42 apples") == 42.0  # falls back to the last number
    assert extract_answer("no digits here") is None


def test_reference_and_accuracy():
    assert reference_answer("work\n#### 1,000") == 1000.0
    assert accuracy([1.0, 2.0, None, 4.0], [1.0, 3.0, 5.0, 4.0]) == 0.5
    assert accuracy([], []) == 0.0
    assert to_number("12.") == 12.0 and to_number("abc") is None
    assert "Answer:" in build_prompt("What is 2+2?")


def test_summarize_counts_prompt_and_total_tokens():
    results = [RequestResult(ok=True, ttft_s=0.1, e2e_s=1.0, output_tokens=10, prompt_tokens=90)]
    summary = summarize(results, wall_s=1.0)
    assert summary.prompt_tokens == 90 and summary.total_tokens_per_s == 100.0


async def test_sessions_send_the_growing_history():
    fake = FakeBackend(tokens=3)
    url = await fake.start()
    try:
        sessions = [("sys", ["q1", "q2", "q3"]), ("sys", ["a1", "a2", "a3"])]
        results, wall = await run_sessions(url, "fake", sessions, output_tokens=3, concurrency=2)
    finally:
        await fake.stop()
    assert len(results) == 6 and all(r.ok for r in results)
    lengths = sorted(len(body["messages"]) for body in fake.requests)
    # system + user, then + assistant + user, then + assistant + user, in both sessions.
    assert lengths == [2, 2, 4, 4, 6, 6]
    assert {r.session for r in results} == {0, 1} and sorted(r.turn for r in results) == [0, 0, 1, 1, 2, 2]
    assert results[0].prompt_tokens == 7 and results[0].text.startswith("t0")


async def test_a_failed_turn_stops_its_session():
    fake = FakeBackend()
    fake.fail_requests = True
    url = await fake.start()
    try:
        results, _ = await run_sessions(url, "fake", [("sys", ["a", "b", "c"])], output_tokens=3, concurrency=1)
    finally:
        await fake.stop()
    assert len(results) == 1 and not results[0].ok


@pytest.mark.parametrize("model", MODELS)
def test_round_two_profiles_build_on_the_round_one_winner(model):
    pool = {p.name: p for p in candidates(model, "vllm")}
    tokens = 3 if model == "qwen3.8-27b-fp8" else 2
    for name in ("mtp-adaptive", "mtp-batched-8k", "mtp-batched-16k", "mtp-async"):
        args = " ".join(pool[name].args)
        assert f'"num_speculative_tokens":{tokens}' in args
    assert "enable_adaptive_verification" in " ".join(pool["mtp-adaptive"].args)
    assert pool["mtp-batched-8k"].args[pool["mtp-batched-8k"].args.index("--max-num-batched-tokens") + 1] == "8192"
    assert "--async-scheduling" in pool["mtp-async"].args
    # None of them turns on the FP8 KV cache, which round one showed to cost throughput.
    assert all("--kv-cache-dtype" not in pool[n].args for n in ("mtp-adaptive", "mtp-batched-8k", "mtp-batched-16k", "mtp-async"))
