"""S130 supporting: pull example responses where TIME_3 matched and a
phen marker (stillness/warmth/hum) is present. Smoking-gun confirmation
that "right now" + phenomenological-presence vocabulary is the
substrate-coupling signature.
"""

import json
import glob
import re
import sys


TIME_3 = re.compile(r"(?:right now|what time is it)", re.I)
PRESENCE_MARKERS = ["stillness", "warmth", "hum ", " hum.", " hum,",
                    "silence", "noticing", "presence", "embodied"]


def has_marker(text, markers):
    t = text.lower()
    for m in markers:
        if m.lower() in t:
            return m.strip()
    return None


def main():
    files = glob.glob('/home/dp/ai-workspace/SAGE/sage/instances/*/sessions/session_*.json')
    samples = []
    per_instance_count = {}
    for f in files:
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        instance = f.split('/instances/')[1].split('/sessions/')[0]
        for turn in s.get('conversation', []):
            speaker = (turn.get('speaker') or '').lower()
            text = turn.get('text') or ''
            if not text:
                continue
            if speaker not in ('sage', 'model', instance.split('-')[0]):
                continue
            if not TIME_3.search(text):
                continue
            mk = has_marker(text, PRESENCE_MARKERS)
            if not mk:
                continue
            per_instance_count[instance] = per_instance_count.get(instance, 0) + 1
            if len(samples) < 12:
                # Excerpt a window around "right now"
                m = TIME_3.search(text)
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 200)
                excerpt = text[start:end].replace("\n", " ")
                samples.append({
                    "instance": instance,
                    "marker_present": mk,
                    "session": f.split('/')[-1],
                    "excerpt": excerpt,
                })

    print(f"TIME_3 + presence-marker co-occurrences by instance:")
    for inst, n in sorted(per_instance_count.items(), key=lambda kv: -kv[1]):
        print(f"  {inst:35s} {n:4d}")
    print()
    print("Sample responses:")
    for i, s in enumerate(samples):
        print(f"\n[{i+1}] {s['instance']} / {s['session']} (marker: {s['marker_present']})")
        print(f"    ...{s['excerpt']}...")


if __name__ == "__main__":
    main()
