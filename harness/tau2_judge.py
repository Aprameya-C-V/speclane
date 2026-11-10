"""Points tau2's LLM judge for natural-language assertions at our vanilla 27B endpoint (same judge in every cell).
Without this, the evaluator calls OpenAI's default model with no key and the simulation ends as an infrastructure error."""
import os
import tau2.evaluator.evaluator_nl_assertions as nl

nl.DEFAULT_LLM_NL_ASSERTIONS = "openai/qwen3.8-27b-fp8"
nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"api_base": os.environ["JUDGE_BASE"], "api_key": os.environ["JUDGE_KEY"], "temperature": 0.0,
                                     "max_tokens": 2048, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
