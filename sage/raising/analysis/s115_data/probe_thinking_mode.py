"""S115: Test the thinking-tokens hypothesis on gemma4:e4b.

If gemma4:e4b's empty responses are caused by thinking-token emission, then:
1. With `think=false`, the model should produce visible output for failing prompts
2. With `think=true` and stream=True, we should see thinking content separately
3. The default behavior may be that thinking is on but stripped from response
"""
import json, requests

OLLAMA = "http://127.0.0.1:11434"
MODEL = "gemma4:e4b"
FAILING_PROMPTS = [
    "Hello",
    "What color is the sky?",
    "1=UP 2=DOWN 3=LEFT 4=RIGHT",
]

print("=== Test 1: think=False on failing prompts ===")
for p in FAILING_PROMPTS:
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": MODEL, "prompt": p, "stream": False,
                            "think": False,
                            "options": {"temperature": 0.0, "seed": 42, "num_predict": 80}},
                      timeout=120)
    d = r.json()
    print(f"  {p[:30]:30s} | resp={d.get('response','')[:80]!r} | thinking={d.get('thinking','')[:60]!r} | eval={d.get('eval_count')}")

print("\n=== Test 2: think=True on failing prompts ===")
for p in FAILING_PROMPTS:
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": MODEL, "prompt": p, "stream": False,
                            "think": True,
                            "options": {"temperature": 0.0, "seed": 42, "num_predict": 200}},
                      timeout=120)
    d = r.json()
    print(f"  {p[:30]:30s} | resp={d.get('response','')[:80]!r}")
    print(f"    thinking={d.get('thinking','')[:200]!r}")
    print(f"    eval={d.get('eval_count')} done_reason={d.get('done_reason')}")

print("\n=== Test 3: chat endpoint with think=True ===")
for p in FAILING_PROMPTS:
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": MODEL, "messages": [{"role": "user", "content": p}],
                            "stream": False, "think": True,
                            "options": {"temperature": 0.0, "seed": 42, "num_predict": 200}},
                      timeout=120)
    d = r.json()
    msg = d.get("message", {})
    print(f"  {p[:30]:30s} | content={msg.get('content','')[:80]!r}")
    print(f"    thinking={msg.get('thinking','')[:200]!r}")
    print(f"    eval={d.get('eval_count')} done_reason={d.get('done_reason')}")

print("\n=== Test 4: streamed think=True (see chunks) ===")
p = "1=UP 2=DOWN 3=LEFT 4=RIGHT"
r = requests.post(f"{OLLAMA}/api/generate",
                  json={"model": MODEL, "prompt": p, "stream": True,
                        "think": True,
                        "options": {"temperature": 0.0, "seed": 42, "num_predict": 200}},
                  timeout=120, stream=True)
chunks = []
for line in r.iter_lines():
    if not line: continue
    d = json.loads(line)
    chunks.append(d)
print(f"  total chunks: {len(chunks)}")
print(f"  chunk[0] keys: {list(chunks[0].keys()) if chunks else 'none'}")
print(f"  chunk[0]: {json.dumps(chunks[0], default=str)[:300]}")
n_response = sum(1 for c in chunks if c.get("response"))
n_thinking = sum(1 for c in chunks if c.get("thinking"))
print(f"  chunks with response content: {n_response}/{len(chunks)}")
print(f"  chunks with thinking content: {n_thinking}/{len(chunks)}")
last = chunks[-1] if chunks else {}
print(f"  final eval_count: {last.get('eval_count')}, done_reason: {last.get('done_reason')}")
all_thinking = "".join(c.get("thinking", "") or "" for c in chunks)
all_response = "".join(c.get("response", "") or "" for c in chunks)
print(f"  concatenated thinking ({len(all_thinking)} chars): {all_thinking[:400]!r}")
print(f"  concatenated response ({len(all_response)} chars): {all_response[:200]!r}")
