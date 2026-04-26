#!/usr/bin/env python3
"""S114 — Confirm the sampler-specific fix on gemma4:e4b.

Hypothesis: greedy (temp=0) on gemma4:e4b lands on tokens that decode
to empty (likely special-token IDs filtered by Ollama). Adding min_p
or higher temperature avoids those.
"""
import json
import time
import requests

GEN = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b"

requests.post(GEN, json={"model": MODEL, "keep_alive": 0})
time.sleep(2)

def probe(prompt, options, label=""):
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": options}
    t0 = time.time()
    r = requests.post(GEN, json=payload, timeout=60)
    r.raise_for_status()
    d = r.json()
    resp = d.get("response", "")
    print(f"  [{label:18s}] opts={json.dumps(options):60s}  eval={d.get('eval_count'):>3} "
          f"reason={d.get('done_reason'):>10}  resp={repr(resp[:55])}")
    return resp.strip()

prompt = "Hello"

print(f"=== gemma4:e4b sampler matrix on {repr(prompt)} ===\n")

# Pure greedy
probe(prompt, {"temperature": 0.0, "num_predict": 30}, "greedy")

# Tiny temp
probe(prompt, {"temperature": 0.001, "num_predict": 30}, "temp=0.001")
probe(prompt, {"temperature": 0.01, "num_predict": 30}, "temp=0.01")
probe(prompt, {"temperature": 0.1, "num_predict": 30}, "temp=0.1")
probe(prompt, {"temperature": 0.5, "num_predict": 30}, "temp=0.5")
probe(prompt, {"temperature": 1.0, "num_predict": 30}, "temp=1.0")

# Greedy + min_p
probe(prompt, {"temperature": 0.0, "min_p": 0.01, "num_predict": 30}, "greedy+min_p=0.01")
probe(prompt, {"temperature": 0.0, "min_p": 0.05, "num_predict": 30}, "greedy+min_p=0.05")
probe(prompt, {"temperature": 0.0, "min_p": 0.1, "num_predict": 30}, "greedy+min_p=0.1")

# Temp + min_p (the working combo from T2.1)
probe(prompt, {"temperature": 0.7, "min_p": 0.05, "num_predict": 30}, "T0.7+min_p=0.05")

# Greedy + top_k
probe(prompt, {"temperature": 0.0, "top_k": 1, "num_predict": 30}, "greedy+top_k=1")
probe(prompt, {"temperature": 0.0, "top_k": 5, "num_predict": 30}, "greedy+top_k=5")
probe(prompt, {"temperature": 0.0, "top_k": 50, "num_predict": 30}, "greedy+top_k=50")

# Top_p variations
probe(prompt, {"temperature": 0.0, "top_p": 0.5, "num_predict": 30}, "greedy+top_p=0.5")
probe(prompt, {"temperature": 0.0, "top_p": 0.95, "num_predict": 30}, "greedy+top_p=0.95")

# Try Gemma's own defaults from the modelfile
probe(prompt, {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "num_predict": 30}, "gemma defaults")

# Try without ANY options (truly default)
probe(prompt, {"num_predict": 30}, "no options")

print("\n=== Same matrix on '1=UP' (game prompt) ===\n")
prompt = "1=UP 2=DOWN 3=LEFT 4=RIGHT"

probe(prompt, {"temperature": 0.0, "num_predict": 60}, "greedy")
probe(prompt, {"temperature": 0.0, "min_p": 0.05, "num_predict": 60}, "greedy+min_p=0.05")
probe(prompt, {"temperature": 0.7, "min_p": 0.05, "num_predict": 60}, "T0.7+min_p=0.05")
probe(prompt, {"num_predict": 60}, "no options")

requests.post(GEN, json={"model": MODEL, "keep_alive": 0})
print("\n[unloaded]")
