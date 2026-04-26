"""S115: Validate think:false fix end-to-end against the kinds of prompts
production uses (game-style with action keymap). Test both gemma4:e4b and gemma4:26b.

Question: with think:false set (as production does since Apr 18), do gemma4 models
produce parseable, non-empty outputs for the prompt shapes that S114 reported as broken?
"""
import json, requests, re

OLLAMA = "http://127.0.0.1:11434"
MODELS = ["gemma4:e4b", "gemma4:26b"]

# Production-shaped lean prompt (approximation of build_lean_prompt output)
PROD_LEAN_PROMPT = """Game: lp85, Level 1, Step 5
Goal: Move the avatar to the green tile.

Actions:
1=UP   2=DOWN   3=LEFT   4=RIGHT   5=SELECT   6=CLICK

Lookahead (next-state pixel-diff if you take each action):
1=UP: 0
2=DOWN: 47
3=LEFT: 0
4=RIGHT: 0
5=SELECT: 0
6=CLICK: 0

Recent: (none)
Hint from NN: action=DOWN, conf=0.62

Pick exactly one action and respond:
ACTION=<1-6>"""

# Also test the simpler S114 prompt
SIMPLE_KEYMAP = "1=UP 2=DOWN 3=LEFT 4=RIGHT"

ACTION_RE = re.compile(r"ACTION\s*=\s*<?(\w+)>?", re.IGNORECASE)

print("=== Production-Shape Prompt Test (think:false) ===")
for model in MODELS:
    for label, prompt in [("simple", SIMPLE_KEYMAP), ("prod_lean", PROD_LEAN_PROMPT)]:
        # Match production payload exactly: think:false, NO options
        r = requests.post(f"{OLLAMA}/api/chat",
                          json={"model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "stream": False, "think": False},
                          timeout=120)
        d = r.json()
        msg = d.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        m = ACTION_RE.search(content)
        captured = m.group(1) if m else None
        print(f"  {model:14s} | {label:9s} | empty={len(content.strip())==0} | "
              f"eval={d.get('eval_count')} | captured_action={captured!r}")
        print(f"    response[:200]: {content[:200]!r}")
        if thinking:
            print(f"    thinking[:80]: {thinking[:80]!r}")
        print()

print("=== Demonstrate the Template-Copy Silent-Fallback ===")
# The S114 finding: when LLM copies the literal `<1-6>` template, regex matches `1`
# because \w+ stops at the hyphen. Confirm this happens on the prod model.
samples = [
    "ACTION=<1-6>",            # literal template copy
    "ACTION=<1-6> X=<0-63> Y=<0-63>",  # full template
    "ACTION=2",                 # a real numeric answer
    "ACTION=<2>",               # bracketed numeric
    "I think ACTION=DOWN",      # named answer
]
for s in samples:
    m = ACTION_RE.search(s)
    print(f"  input={s!r:50s} -> capture={m.group(1) if m else None!r}")
