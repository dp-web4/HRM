"""S115: Test gemma4:26b on Thor with same failing prompts as gemma4:e4b.
If 26b also fails -> gemma4-family-on-Thor issue. If 26b works -> e4b-specific."""
import json, requests, sys

OLLAMA = "http://127.0.0.1:11434"
PROMPTS = [
    "Hello",
    "What color is the sky?",
    "Why is the sky blue?",
    "1=UP",
    "1=UP 2=DOWN 3=LEFT 4=RIGHT",
    "What is 2+2?",
    "What is the capital of France? Reply: 1=Paris 2=London",
]
MODELS = ["gemma4:26b", "gemma4:e4b"]
SAMPLERS = [
    ("greedy", {"temperature": 0.0, "seed": 42, "num_predict": 80}),
    ("default", {"seed": 42, "num_predict": 80}),
]

results = []
for model in MODELS:
    for prompt in PROMPTS:
        for sname, sopts in SAMPLERS:
            try:
                r = requests.post(f"{OLLAMA}/api/generate",
                                  json={"model": model, "prompt": prompt, "stream": False, "options": sopts},
                                  timeout=120)
                d = r.json()
                resp = d.get("response", "")
                eval_count = d.get("eval_count", 0)
                results.append({
                    "model": model, "prompt": prompt, "sampler": sname,
                    "response": resp[:200], "eval_count": eval_count,
                    "empty": len(resp.strip()) == 0,
                })
                print(f"{model:14s} | {sname:8s} | empty={len(resp.strip())==0} | eval={eval_count:3d} | {prompt[:40]:40s} -> {resp[:80]!r}")
            except Exception as e:
                print(f"{model:14s} | {sname:8s} | ERROR: {e}")
                results.append({"model": model, "prompt": prompt, "sampler": sname, "error": str(e)})

with open("gemma4_26b_keymap_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
print("\n--- SUMMARY ---")
for model in MODELS:
    for sname, _ in SAMPLERS:
        n_empty = sum(1 for r in results if r.get("model")==model and r.get("sampler")==sname and r.get("empty"))
        n = sum(1 for r in results if r.get("model")==model and r.get("sampler")==sname)
        print(f"{model:14s} | {sname:8s} | empty: {n_empty}/{n}")
