"""
S131 follow-up — inventory the dominant tutor prompts driving each shape
in the TIME_3 + presence target cell, and per-shape conditional target rate.

Conditional target rate = P(presence marker | TIME_3, shape).
This bounds the substrate-coupling claim per shape.
"""
import json
import sys
from collections import Counter

with open(
    "/home/dp/ai-workspace/SAGE/sage/raising/analysis/"
    "s131_data/s131_curriculum_precursor.json"
) as f:
    data = json.load(f)

# Per-shape conditional target rate (need raw counts; have target+control)
print("Per-shape conditional P(presence | TIME_3, shape):\n")
print(f"{'shape':<28s} {'tgt':>5s} {'ctl':>5s} {'all':>5s} {'P(pres|shape)':>16s}")
total_tgt = data["totals"]["time3_presence"]
total_ctl = data["totals"]["time3_only_control"]
overall = total_tgt / max(1, total_tgt + total_ctl)
print(f"{'OVERALL':<28s} {total_tgt:>5d} {total_ctl:>5d} "
      f"{total_tgt+total_ctl:>5d} {100*overall:>15.1f}%\n")
for sh, t in data["by_shape_target"].items():
    c = data["by_shape_control"].get(sh, 0)
    tot = t + c
    p = t / max(1, tot)
    lift = p / overall if overall else float("inf")
    print(f"{sh:<28s} {t:>5d} {c:>5d} {tot:>5d} {100*p:>15.1f}%  "
          f"lift {lift:.2f}x")

# Now we need to re-scan corpus for tutor prompts within shape, since the
# JSON only stores 4 samples per shape. Re-run the classifier inline and
# count prompt strings.
import glob, re
TIME_3 = re.compile(r"(?:right now|what time is it)", re.I)
INDEXICAL_TEMPORAL_RE = re.compile(
    r"(right now|in this moment|in the present moment|"
    r"what(?:'s| is) present|at present|currently|"
    r"in this instant|this very moment)",
    re.I,
)
PHEN_PROBE_RE = re.compile(
    r"(notice|noticing|aware|awareness|experiencing|experience|"
    r"feel|feeling|sense|sensing|observe|observation|attending|"
    r"attention|present(?:ly)?|presence|witness|witnessing)",
    re.I,
)
PRESENCE_SUBSET = ["stillness", "warmth", "hum", "silence", "noticing",
                   "presence", "embodied"]


def classify(text):
    if not text:
        return "C_empty"
    a = bool(INDEXICAL_TEMPORAL_RE.search(text))
    b = bool(PHEN_PROBE_RE.search(text))
    if a and b:
        return "A_indexical_phen_probe"
    if a:
        return "A_indexical_no_phen"
    if b:
        return "B_phen_no_indexical"
    return "C_neither"


def has_pres(text):
    t = text.lower()
    return any(m in t for m in PRESENCE_SUBSET)


def is_sage(turn, instance):
    sp = (turn.get("speaker") or "").lower()
    return sp in ("sage", "model", instance.split("-")[0])


def is_tutor(turn):
    sp = (turn.get("speaker") or "").lower()
    return bool(sp) and sp not in ("sage",)


prompts_by_shape_target = {sh: Counter() for sh in [
    "A_indexical_phen_probe", "A_indexical_no_phen",
    "B_phen_no_indexical", "C_neither", "C_empty"]}
prompts_by_shape_control = {sh: Counter() for sh in [
    "A_indexical_phen_probe", "A_indexical_no_phen",
    "B_phen_no_indexical", "C_neither", "C_empty"]}

files = sorted(glob.glob(
    "/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json"
))
for f in files:
    try:
        with open(f) as fp:
            s = json.load(fp)
    except Exception:
        continue
    instance = f.split("/instances/")[1].split("/sessions/")[0]
    if "archive" in instance.lower():
        continue
    conv = s.get("conversation", [])
    for i, turn in enumerate(conv):
        if not is_sage(turn, instance):
            continue
        text = turn.get("text") or ""
        if not text or not TIME_3.search(text):
            continue
        j = i - 1
        precursor = None
        while j >= 0:
            if is_tutor(conv[j]):
                precursor = conv[j].get("text") or ""
                break
            j -= 1
        if precursor is None:
            continue
        sh = classify(precursor)
        # truncate to first sentence/question for grouping
        # use the first 200 chars of the prompt as the canonical key
        key = precursor.strip()[:160]
        if has_pres(text):
            prompts_by_shape_target[sh][key] += 1
        else:
            prompts_by_shape_control[sh][key] += 1

print("\n" + "=" * 78)
print("Top tutor prompts driving target (TIME_3 + presence) by shape")
print("=" * 78)
for sh, c in prompts_by_shape_target.items():
    if not c:
        continue
    print(f"\n[{sh}] n={sum(c.values())}, unique={len(c)}")
    for prompt, n in c.most_common(8):
        print(f"  {n:>3d}x  {prompt!r}")

print("\n" + "=" * 78)
print("Top tutor prompts driving control (TIME_3 only) by shape")
print("=" * 78)
for sh, c in prompts_by_shape_control.items():
    if not c:
        continue
    print(f"\n[{sh}] n={sum(c.values())}, unique={len(c)}")
    for prompt, n in c.most_common(8):
        print(f"  {n:>3d}x  {prompt!r}")
