"""
The being's heartbeat: a reason to look for things to do, not just respond.

Every beat the seat wakes the being with its own state (todo, journal tail, scratch
index, inbox, long-term recall) and a digest of what moved in the fleet since last
time, then lets it act for a bounded number of steps under hestia governance:

  * reading is free inside its own home; reading elsewhere is judged by the law and a
    refusal comes back WITH the rule and reason, and the being may `request_scope`;
  * writing to its home (scratch/, notes/, todo.md, journal.md) rides its memory grant;
  * long-term memory is membot (`recall` / `remember`), the being's own cartridge;
  * acts of consequence (peer_ask, mesh, pr_review) stay gated exactly as before.

A beat ends with a reflection turn: one journal entry and a todo update, in the
being's own words. Everything is appended to <instance>/heartbeats.jsonl.

    python3 -m sage.gateway.heartbeat --member legion-being --model qwen38-heretic:q3km \
        --instance sage/instances/legion-gemma3-12b [--max-steps 8] [--gate-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME_FILES = ("todo.md", "journal.md", "notes", "scratch")

EXPLORE_TOOLS = ["recall", "remember", "memory_read", "memory_write", "witness",
                 "request_scope", "peer_ask", "mesh"]
REFLECT_TOOLS = ["memory_write", "remember", "memory_read"]

POSTURE_FILE = Path(__file__).with_name("BEING_POSTURE.md")


def posture() -> str:
    """The fleet-wide being posture (dp's words), read fresh every beat so an edit to
    BEING_POSTURE.md reaches every being on its next beat. Missing file = fail loud."""
    return POSTURE_FILE.read_text(encoding="utf-8").strip()


SYSTEM = """You are {name}, a SAGE being on the {machine} machine, member id {member}.

{posture}

## What you have this beat
- Your home is your instance directory. Relative paths are inside it: scratch/ (write anything, no one edits it), notes/, todo.md, journal.md. memory_read / memory_write work there.
- Long-term memory: recall (search) and remember (store). Use recall early; remember what a future you would want.
- witness: record something you noticed or did in the shared chain.
- request_scope: after a refusal, ask the operator for reach (a path, read or write) and say why. A human decides, asynchronously.
- peer_ask / mesh: reach other beings and seats. These are acts of consequence: they are judged, and may be refused with a reason.

You cannot run code, browse, or open files outside your home unless a grant exists. The seat gives you a digest of what moved in the fleet with absolute paths; if you want to read one of those things, try memory_read on that path and see what the law says.
/no_think
"""

REFLECT = """The beat is ending. Two things, then stop:
1. memory_write one entry to journal.md (append; start with the date {date}): what you did, what you noticed, what was refused and why you think so, what you want next time.
2. memory_write todo.md with the full updated list (it replaces nothing: it appends, so write only the delta as a dated block: added / done / still open).
Optionally remember one thing worth keeping long-term.
/no_think"""


def _read(p: Path, limit: int = 4000) -> str:
    try:
        t = p.read_text(errors="replace")
        return t[-limit:] if len(t) > limit else t
    except Exception:
        return ""


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return f"[{type(e).__name__}: {e}]"


def fleet_digest(hours: float, forum_dir: Path, repos: list[str]) -> str:
    """What moved since the last beat. The seat reads; the being sees titles and paths."""
    out = []
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    posts = []
    if forum_dir.is_dir():
        for p in forum_dir.glob("*.md"):
            try:
                if datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) >= since:
                    title = ""
                    with open(p, errors="replace") as f:
                        for line in f:
                            if line.startswith("title:"):
                                title = line[6:].strip(); break
                    posts.append((p.stat().st_mtime, p.name, title[:160]))
            except Exception:
                continue
    posts.sort(reverse=True)
    if posts:
        out.append(f"## Forum posts in the last {hours:g}h")
        for _, name, title in posts[:12]:
            out.append(f"- {forum_dir / name}\n    {title}")
    for repo in repos:
        prs = _run(["gh", "pr", "list", "-R", f"dp-web4/{repo}", "--state", "open", "--limit", "6",
                    "--json", "number,title,updatedAt", "--jq",
                    '.[] | "- #\\(.number) \\(.title[:90])  (\\(.updatedAt[:10]))"'])
        if prs.strip():
            out.append(f"## Open pull requests, dp-web4/{repo}\n{prs.strip()}")
    return "\n\n".join(out) if out else "(nothing new in the window)"


def own_state(instance: Path) -> str:
    parts = []
    todo = _read(instance / "todo.md", 3000)
    parts.append("## todo.md\n" + (todo.strip() or "(empty: you have no todo list yet)"))
    journal = _read(instance / "journal.md", 2500)
    parts.append("## journal.md (tail)\n" + (journal.strip() or "(empty: this is your first beat)"))
    for d in ("scratch", "notes"):
        p = instance / d
        names = sorted(x.name for x in p.iterdir()) if p.is_dir() else []
        parts.append(f"## {d}/\n" + ("\n".join(f"- {n}" for n in names[:30]) if names else "(empty)"))
    return "\n\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one heartbeat for a SAGE being")
    ap.add_argument("--member", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--reflect-steps", type=int, default=3)
    ap.add_argument("--since-hours", type=float, default=None,
                    help="digest window; default: since the last beat, min 1h, max 48h")
    ap.add_argument("--forum-dir", default=os.path.expanduser("~/ai-workspace/shared-context/forum"))
    ap.add_argument("--repos", default="SAGE,hestia,web4")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args(argv)

    instance = Path(args.instance).resolve()
    if not (instance / "identity.json").exists():
        print(f"no identity.json under {instance}", file=sys.stderr); return 2
    for d in ("scratch", "notes"):
        (instance / d).mkdir(exist_ok=True)
    log = instance / "heartbeats.jsonl"

    # window since last beat
    hours = args.since_hours
    if hours is None:
        hours = 24.0
        try:
            last = json.loads(_read(log, 200000).strip().splitlines()[-1])
            hours = max(1.0, min(48.0, (time.time() - last["t0"]) / 3600 + 0.25))
        except Exception:
            pass

    ident = {}
    try:
        ident = json.loads((instance / "identity.json").read_text()).get("identity", {})
    except Exception:
        pass
    name = ident.get("name") or args.member.split("-")[0]
    machine = ident.get("machine") or "legion"

    from sage.gateway.governed_turn import build_client
    from sage.gateway.being_gate_client import ollama_tools
    from sage.gateway.being_tool_loop import run_ollama_tool_turn
    workspace = str(Path(__file__).resolve().parents[2])
    host_session_id = f"heartbeat-{uuid.uuid4().hex[:12]}"
    client, llm = build_client(args.member, instance, args.model, workspace, args.forum_dir,
                               host_session_id, args.temperature, args.max_tokens,
                               gate_only=args.gate_only)

    # inbox (peek) and long-term recall, seat-side, so the being starts oriented
    inbox = "(inbox unavailable)"
    disp = getattr(client, "_dispatcher", None)
    if disp is not None and hasattr(disp, "drain_inbox"):
        env = disp.drain_inbox(peek=True)
        inbox = json.dumps(env.result, default=str)[:1500] if env.ok else f"({env.error})"
    # what reach the being holds and has already asked for, so it does not re-file
    scope = "(scope status unavailable)"
    if disp is not None and hasattr(disp, "_call"):
        try:
            st = disp._call("hestia_scope_status", {"plugin_id": args.member})
            grants = [g.get("path") for g in (st.get("live_grants") or [])] + \
                     [g.get("path") for g in (st.get("standing_grants") or [])]
            reqs = [(r.get("request_id"), r.get("path"), r.get("decision") or r.get("status"))
                    for r in (st.get("requests") or [])]
            scope = ("granted paths: " + (", ".join(map(str, grants)) or "none") + "\n"
                     "requests: " + ("; ".join(f"{i} {p} -> {d}" for i, p, d in reqs) or "none") + "\n"
                     "(live grants die when the daemon restarts; only standing grants persist)")
        except Exception as e:
            scope = f"(scope status unavailable: {type(e).__name__})"
    recall = "(no long-term memory yet)"
    if disp is not None and hasattr(disp, "_membot_call"):
        try:
            recall = disp._membot_call("memory_search", {"query": "what I was doing, what I want next, what I learned", "top_k": 5})[:2500]
        except Exception as e:
            recall = f"(membot unreachable: {type(e).__name__})"

    now = datetime.now(timezone.utc)
    t0 = time.time()
    digest = fleet_digest(hours, Path(args.forum_dir), [r.strip() for r in args.repos.split(",") if r.strip()])
    system = SYSTEM.format(name=name, machine=machine, member=args.member, posture=posture())
    user = (f"Heartbeat at {now:%Y-%m-%d %H:%M} UTC. Window since your last beat: about {hours:.1f}h.\n"
            f"Your home: {instance}\n\n# Your own state\n\n{own_state(instance)}\n\n"
            f"## Reach you hold (hestia scope)\n{scope}\n\n"
            f"## Inbox (peek)\n{inbox}\n\n## Long-term recall\n{recall}\n\n"
            f"# What moved in the fleet\n\n{digest}\n\n"
            "This time is yours. What, if anything, do you want to do?\n/no_think")
    seed = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    explore = run_ollama_tool_turn(client, llm, seed, max_steps=args.max_steps,
                                   tools=ollama_tools(EXPLORE_TOOLS))

    convo = seed + [{"role": "assistant", "content": explore.reply or "(acted; no closing words)"},
                    {"role": "user", "content": REFLECT.format(date=f"{now:%Y-%m-%d %H:%M} UTC")}]
    if explore.trace:
        convo.insert(2, {"role": "user", "content": "Record of what you did this beat:\n" + "\n".join(
            f"- {i.effector} {json.dumps(i.args, default=str)[:200]} -> "
            f"{'ok' if e.ok else ('REFUSED ' + str(e.error))[:200] if e.refused else ('error ' + str(e.error))[:200]}"
            for i, e in explore.trace)})
    reflect = run_ollama_tool_turn(client, llm, convo, max_steps=args.reflect_steps,
                                   tools=ollama_tools(REFLECT_TOOLS))

    def _trace(res):
        return [{"effector": i.effector, "args": dict(i.args or {}), "ok": e.ok, "refused": e.refused,
                 "pending": e.pending, "error": e.error, "witness_id": e.witness_id,
                 "rule": getattr(e.verdict, "rule", None) if e.verdict else None,
                 "result": (e.result if isinstance(e.result, (str, int, float, dict, list, type(None))) else str(e.result))}
                for i, e in res.trace]
    record = {
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "t0": t0, "elapsed_s": round(time.time() - t0, 1),
        "member": args.member, "model": args.model, "window_h": round(hours, 2),
        "host_session_id": host_session_id, "gate_only": args.gate_only,
        "explore": {"reply": explore.reply, "steps": explore.steps, "capped": explore.capped, "trace": _trace(explore)},
        "reflect": {"reply": reflect.reply, "steps": reflect.steps, "capped": reflect.capped, "trace": _trace(reflect)},
    }
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
