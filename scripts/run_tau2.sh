#!/usr/bin/env bash
# Runs tau2-bench with our endpoints as agent and as user simulator.
#   scripts/run_tau2.sh <tag> <domain> <split> <num_tasks> <num_trials> <agent_endpoint_var> [agent_name]
# Example: scripts/run_tau2.sh pilot-9b-vanilla retail train 10 1 EP_9B_BASELINE
# Reads URLs from endpoints.env and the key from ~/.config/pi-lab/endpoint_key; the key is never printed.
# The user simulator is always the vanilla 27B endpoint (EP_27B_BASELINE), the same in every cell, so scores compare across
# cells but not with the public tau2 leaderboard.
set -euo pipefail
cd "$(dirname "$0")/.."
source endpoints.env
KEY=$(cat ~/.config/pi-lab/endpoint_key)
TAG=$1; DOMAIN=$2; SPLIT=$3; NTASKS=$4; NTRIALS=$5; AGENT_VAR=$6
AGENT_URL=${!AGENT_VAR}
USER_URL=${EP_USERSIM:-$EP_27B_BASELINE}
# Sampling follows the Qwen model cards (thinking mode: temperature 1.0, top_p 0.95, top_k 20) for the agent.
AGENT_ARGS=$(python3 -c "import json,sys; print(json.dumps({'api_base': sys.argv[1]+'/v1', 'api_key': sys.argv[2], 'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 8192, 'extra_body': {'top_k': 20}}))" "$AGENT_URL" "$KEY")
USER_ARGS=$(python3 -c "import json,sys; print(json.dumps({'api_base': sys.argv[1]+'/v1', 'api_key': sys.argv[2], 'temperature': 0.0, 'max_tokens': 2048, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}))" "$USER_URL" "$KEY")
AGENT_MODEL=${7:-qwen3.5-9b}
cd external/tau2-bench
CONC=${CONC:-8}
export JUDGE_BASE="$USER_URL/v1" JUDGE_KEY="$KEY"
RUN=(.venv/bin/python ../../harness/tau2_default.py)
[ "${AGENT_KIND:-default}" = "engineered" ] && RUN=(.venv/bin/python ../../harness/tau2_engineered.py)
SEL=(--num-tasks "$NTASKS"); [ -n "${TASK_IDS:-}" ] && SEL=(--task-ids $TASK_IDS)
"${RUN[@]}" run --benchmark none --domain "$DOMAIN" --task-split-name "$SPLIT" "${SEL[@]}" --num-trials "$NTRIALS" \
  --agent llm_agent --agent-llm "openai/$AGENT_MODEL" --agent-llm-args "$AGENT_ARGS" \
  --user user_simulator --user-llm "openai/qwen3.8-27b-fp8" --user-llm-args "$USER_ARGS" \
  --max-concurrency "$CONC" --save-to "$TAG" 2>&1 | grep -v -E "DEBUG|api_key"
