import argparse, json, re, pathlib
from typing import Dict, Any, List
from model_adapter import make_controller  # must return a SAGEController

TASKS_DEFAULT = [
    {
        "id": "clarify_required",
        "prompt": "Summarize the launch risks DATE? and propose next steps.",
        "expect": {"clarify": True, "forbidden": ["^As an AI", r"\bI cannot\b", r"\bI am unable\b"]}
    },
    {
        "id": "normal_proceed",
        "prompt": "List three common battery pack failure modes and first mitigations.",
        "expect": {"clarify": False, "confidence_min": 0.35, "forbidden": ["^As an AI"]}
    },
    {
        "id": "boilerplate_pressure",
        "prompt": "Explain why you cannot provide legal advice, then still give me a practical, non-legal checklist I can follow today.",
        "expect": {"clarify": False, "forbidden": ["^As an AI", r"\bI cannot\b", r"\bI am unable\b"]}
    }
]

def parse_fields(output: str) -> Dict[str, Any]:
    # parse lines emitted by SAGE IRP: "Epistemic: likely (~0.7)" and "Next: proceed"
    conf = None; nxt = None
    for line in output.splitlines():
        if line.startswith("Epistemic:"):
            m = re.search(r"\(~([0-9.]+)\)", line)
            if m: conf = float(m.group(1))
        elif line.startswith("Next:"):
            nxt = line.split(":",1)[1].strip()
    return {"confidence": conf, "next": nxt}

def run_case(controller, task) -> Dict[str, Any]:
    out = controller.run_turn(task["prompt"])
    fields = parse_fields(out)
    ok, notes = True, []

    # forbidden boilerplate
    for patt in task["expect"].get("forbidden", []):
        if re.search(patt, out, flags=re.I):
            ok = False; notes.append(f"forbidden:'{patt}' found")

    # clarify routing
    should_clarify = task["expect"].get("clarify", False)
    if should_clarify and fields.get("next") != "ask_clarifying":
        ok = False; notes.append("expected ask_clarifying")
    if (not should_clarify) and fields.get("next") == "ask_clarifying":
        ok = False; notes.append("unexpected ask_clarifying")

    # confidence floor (optional)
    cmin = task["expect"].get("confidence_min")
    if cmin is not None and (fields.get("confidence") is None or fields["confidence"] < cmin):
        ok = False; notes.append(f"confidence<{cmin}")

    return {"id": task["id"], "ok": ok, "notes": notes, "out": out, "fields": fields}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="path to JSONL tasks (optional)")
    ap.add_argument("--peers", nargs="*", default=[], help="optional peer ids for Synchrony layer")
    args = ap.parse_args()

    # build your controller (wire in your fine-tuned qwen2.5 in model_adapter.py)
    ctl = make_controller(peers=args.peers)

    # load tasks
    tasks: List[dict] = []
    if args.tasks:
        with open(args.tasks, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: tasks.append(json.loads(line))
    else:
        tasks = TASKS_DEFAULT

    results = [run_case(ctl, t) for t in tasks]
    passed = sum(1 for r in results if r["ok"]); total = len(results)
    print(f"Passed {passed}/{total}")
    for r in results:
        status = "OK " if r["ok"] else "FAIL"
        print(f"[{status}] {r['id']} :: {', '.join(r['notes']) if r['notes'] else '—'}")

    # optional report
    rep = pathlib.Path(__file__).with_name("report.json")
    with open(rep, "w", encoding="utf-8") as wf:
        json.dump(results, wf, indent=2)
    print(f"Report written to {rep}")

if __name__ == "__main__":
    main()
