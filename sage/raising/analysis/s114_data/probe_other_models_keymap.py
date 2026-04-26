#!/usr/bin/env python3
"""S114 — Do other Thor models share the empty-response failure on '1=UP'?

S113 noted gemma4:e4b empty-response. We've confirmed it. Quick check:
do other models also fail, or is gemma4:e4b unique?
"""
import json
import time
import requests

OLLAMA = "http://localhost:11434/api/generate"
MODELS = ["gemma4:e4b", "gemma3:12b", "qwen2.5:3b", "phi4:14b"]
PROMPTS = ["Hello", "1=UP", "1=UP 2=DOWN 3=LEFT 4=RIGHT. Pick a number."]


def probe(model, prompt, options):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    try:
        t0 = time.time()
        r = requests.post(OLLAMA, json=payload, timeout=60)
        r.raise_for_status()
        d = r.json()
        return {
            "response": d.get("response", "").strip(),
            "eval_count": d.get("eval_count"),
            "done_reason": d.get("done_reason"),
            "elapsed": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {"error": str(e)}


print(f"{'model':14s} {'prompt':45s} {'sampler':18s} {'eval':>4s} {'reason':>10s}  response")
print("-" * 130)
for model in MODELS:
    requests.post(OLLAMA, json={"model": model, "keep_alive": 0})
    time.sleep(2)
    for prompt in PROMPTS:
        # Greedy
        r1 = probe(model, prompt, {"temperature": 0.0, "num_predict": 60})
        # Default sampler
        r2 = probe(model, prompt, {"num_predict": 60})
        for label, r in [("greedy", r1), ("default", r2)]:
            if "error" in r:
                resp = f"ERROR: {r['error'][:50]}"
                print(f"{model:14s} {prompt[:45]:45s} {label:18s}  err  err  {resp}")
                continue
            empty = len(r["response"]) == 0
            mark = " ←EMPTY" if empty else ""
            print(f"{model:14s} {prompt[:45]:45s} {label:18s} {r['eval_count']:>4} "
                  f"{r['done_reason']:>10}  {repr(r['response'][:60])}{mark}")
        print()
    requests.post(OLLAMA, json={"model": model, "keep_alive": 0})
    time.sleep(1)
