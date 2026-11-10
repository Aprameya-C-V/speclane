"""Runs tau2-bench with an engineered agent in place of the default `llm_agent`.

    python harness/tau2_engineered.py run --domain airline ...    (same arguments as `tau2 run`; use --agent llm_agent)

tau2's command line only accepts its built-in agent names, so this replaces the factory registered under "llm_agent" before the
command runs. The change is only the system prompt: same model, same tools, same policy text, same loop.
"""
import sys

from tau2.agent.llm_agent import LLMAgent, create_llm_agent  # noqa: F401
import tau2_judge  # noqa: F401
from tau2.cli import main
from tau2.registry import registry

ENGINEERED_INSTRUCTION = """
You are a customer service agent. Follow the <policy> below exactly. In each turn you do exactly one thing: either send one message to the user, or make one tool call. Never both. Generate valid JSON arguments only.

Before any action that changes the database (booking, cancelling, modifying, exchanging, returning, refunding, transferring, updating):
1. Verify the user's identity in the way the policy requires.
2. Look up the current state with read-only tools. Copy identifiers and values exactly from tool results; never guess or invent them.
3. Check every policy rule that applies. If a rule forbids the request, refuse and explain; never invent an exception.
4. Tell the user the exact action and its consequences, and get an explicit yes when the policy requires confirmation. Then make the call once.

Keep a list of everything the user asked for. Before you end the conversation, make sure each item is done or properly refused and tell the user the outcome. If information is missing, ask for it instead of assuming.
""".strip()

ENGINEERED_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class EngineeredAgent(LLMAgent):
    @property
    def system_prompt(self) -> str:
        return ENGINEERED_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=ENGINEERED_INSTRUCTION)


def create_engineered_agent(tools, domain_policy, **kwargs):
    return EngineeredAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
    )


registry._agent_factories["llm_agent"] = create_engineered_agent

if __name__ == "__main__":
    sys.argv[0] = "tau2"
    main()
