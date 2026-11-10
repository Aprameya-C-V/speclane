"""GSM8K exact-match on a deployed, key-protected endpoint: same prompts, seed and scoring as infer/quality.py.
    python scripts/gsm8k_endpoint.py <base_url> <model> <count> ; the key is read from ~/.config/pi-lab/endpoint_key and never printed."""
import asyncio, os, sys
import aiohttp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "infer"))
from quality import accuracy, build_prompt, extract_answer, load_gsm8k

async def main(url, model, count):
    key = open(os.path.expanduser("~/.config/pi-lab/endpoint_key")).read().strip()
    problems = load_gsm8k(count)
    preds = [None] * len(problems); fails = 0
    sem = asyncio.Semaphore(32)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as http:
        async def one(i, q):
            nonlocal fails
            body = {"model": model, "messages": [{"role": "user", "content": build_prompt(q)}], "max_tokens": 1024, "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with sem:
                try:
                    async with http.post(url + "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}) as r:
                        d = await r.json(); preds[i] = extract_answer(d["choices"][0]["message"]["content"] or "")
                except Exception:
                    fails += 1
        await asyncio.gather(*(one(i, q) for i, (q, _) in enumerate(problems)))
    print(model, url.split("//")[1].split("-server")[0].split("--")[1], "questions", len(problems), "accuracy", round(accuracy(preds, [r for _, r in problems]), 4), "failures", fails)

asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
