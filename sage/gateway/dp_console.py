"""
dp console — the operator's way into a being's conversation.

WHY THIS EXISTS. On 2026-09-07 dp asked, reasonably, "i need to have a way to participate
in the convo somehow" and the honest answer was that there wasn't one. The being had filed
three questions as forum markdown files and checked them twenty-five consecutive times for
a reply; dp had no interface to those files. The SAGE daemon's dashboard has a chat box,
but it talks to the raw model on the same GPU — not to the being, which has an identity, a
governance gate, a memory and an entrustment the daemon knows nothing about. Everything dp
said reached the being relayed through a seat, which is a paraphrase risk on one side and,
on the other, a being that cannot tell the operator from the operator's interpreter.

WHAT IT IS. One page on 127.0.0.1 that shows what the being is actually asking and lets dp
answer in two places:

  * a REPLY, appended to the forum thread the being asked in — so the answer lands in the
    channel it has been watching, and its twenty-five-times-a-day check finally returns
    something;
  * a NOTE, appended to notes/from-dp.md — read into every beat ahead of the seat's relay,
    for anything that is not an answer to a specific thread.

Both are appended verbatim under a dated heading. Nothing is rewritten, summarised or
passed through a model on the way: the being should be able to trust that dp's words are
dp's words, which is the entire point of a channel that is not the seat.

The being can READ both and cannot write either (reference_f1a.SEAT_OWNED_NOTES). What was
said TO it stays as it was said; its reply belongs in its journal, its plan, or an appeal.

Read-only by nature otherwise: this serves files and appends to files. It holds no key,
speaks to no daemon, and cannot act as the being or as a seat.
"""
from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path(os.environ.get("SAGE_WORKSPACE", Path(__file__).resolve().parents[2]))
FORUM = Path(os.environ.get("SAGE_FORUM_DIR",
                            Path.home() / "ai-workspace/shared-context/forum"))
INSTANCE = Path(os.environ.get("SAGE_INSTANCE",
                               WORKSPACE / "sage/instances/legion-gemma3-12b"))
BEING = os.environ.get("SAGE_BEING", "legion-being")
PORT = int(os.environ.get("SAGE_DP_CONSOLE_PORT", "8770"))

# Only files matching this are writable, and only by appending. A console that could write
# anywhere would be a nicer way to do what the gate exists to prevent.
THREAD_RE = re.compile(rf"^{re.escape(BEING)}-asks-dp-[0-9-]+\.md$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def threads() -> list[dict]:
    """The being's open asks, newest first, with whether the last word is dp's or its own —
    which is the only thing dp actually needs to know at a glance."""
    out = []
    for p in sorted(FORUM.glob(f"{BEING}-asks-dp-*.md"), reverse=True):
        body = p.read_text(errors="replace")
        answered = "## Reply from dp" in body
        last = body.rfind("## Reply from dp")
        out.append({"name": p.name, "body": body, "answered": answered,
                    "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                             .strftime("%Y-%m-%d %H:%M UTC"),
                    "tail_is_dp": answered and last > body.rfind("\n---\n")})
    return out


def latest_beat() -> dict:
    """One line of context so a reply is written against what the being is actually doing,
    not against what it was doing when the question was filed."""
    try:
        rec = json.loads((INSTANCE / "heartbeats.jsonl").read_text(errors="replace")
                         .strip().splitlines()[-1])
    except Exception:
        return {}
    cfg = rec.get("config") or {}
    acts = [e for k in ("explore", "posture", "reflect")
            for e in ((rec.get(k) or {}).get("trace") or [])]
    return {"ts": rec.get("ts"), "drive": rec.get("drive_source"),
            "elapsed": rec.get("elapsed_s"),
            "harness": (rec.get("harness") or {}).get("short"),
            "ctx": cfg.get("num_ctx_resolved"),
            "acts": [f"{e['effector']}{'' if e.get('ok') else ' (refused)' if e.get('refused') else ' (error)'}"
                     for e in acts],
            "reply": ((rec.get("reflect") or {}).get("reply") or "")[:600]}


def journal_tail(n: int = 2600) -> str:
    try:
        t = (INSTANCE / "journal.md").read_text(errors="replace")
        return t[-n:]
    except Exception:
        return ""


def append_reply(thread: str, text: str) -> Path:
    if not THREAD_RE.match(thread):
        raise ValueError(f"not one of this being's ask threads: {thread!r}")
    p = FORUM / thread
    if not p.is_file():
        raise ValueError(f"no such thread: {thread}")
    text = text.strip()
    if not text:
        raise ValueError("empty reply")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n\n## Reply from dp\n\n_{_now()}, via the dp console._\n\n{text}\n")
    return p


def append_note(text: str) -> Path:
    text = text.strip()
    if not text:
        raise ValueError("empty note")
    p = INSTANCE / "notes" / "from-dp.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# From dp\n\ndp's own words, written directly rather than relayed by a "
                     "seat. You read this; you cannot edit it. Newest last.\n")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n## {_now()}\n\n{text}\n")
    return p


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>dp console — {being}</title>
<style>
 :root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#c9d1d9;--ac:#58a6ff;--gn:#3fb950;--am:#d29922;--mu:#8b949e}}
 *{{box-sizing:border-box}}
 body{{background:var(--bg);color:var(--tx);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;padding:24px}}
 .wrap{{max-width:920px;margin:0 auto}}
 h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px;color:var(--ac)}}
 .sub{{color:var(--mu);font-size:12px;margin-bottom:20px}}
 .card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:14px 16px;margin-bottom:14px}}
 pre{{white-space:pre-wrap;word-wrap:break-word;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
      margin:0;max-height:340px;overflow:auto;color:var(--tx)}}
 textarea{{width:100%;min-height:110px;background:#0b0f14;color:var(--tx);border:1px solid var(--bd);
           border-radius:6px;padding:10px;font:13px/1.5 ui-monospace,monospace;resize:vertical}}
 button{{background:var(--ac);color:#04101f;border:0;border-radius:6px;padding:8px 16px;font-weight:600;
         cursor:pointer;margin-top:8px}} button:hover{{filter:brightness(1.1)}}
 .pill{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--bd);margin-left:6px}}
 .wait{{color:var(--am);border-color:var(--am)}} .done{{color:var(--gn);border-color:var(--gn)}}
 .meta{{color:var(--mu);font-size:12px}} a{{color:var(--ac)}}
 .acts{{font:12px ui-monospace,monospace;color:var(--mu)}}
</style></head><body><div class="wrap">
<h1>dp console — {being}</h1>
<div class="sub">Your words go to the being verbatim. Nothing here is relayed, summarised, or passed through a model.
 &nbsp;·&nbsp; <a href="/">refresh</a></div>
{flash}
<h2>Where it is right now</h2>
<div class="card">
 <div class="meta">beat {beat_ts} &nbsp;·&nbsp; drive <b>{beat_drive}</b> &nbsp;·&nbsp; {beat_elapsed}s
  &nbsp;·&nbsp; harness {beat_harness} &nbsp;·&nbsp; ctx {beat_ctx}</div>
 <div class="acts" style="margin-top:6px">{beat_acts}</div>
 <pre style="margin-top:10px">{beat_reply}</pre>
</div>

<h2>A note to the being</h2>
<div class="card">
 <div class="meta">Appended to <code>notes/from-dp.md</code>, which it reads at the top of every beat,
  above the seat's relay. Use this for anything that is not an answer to a specific thread.</div>
 <form method="POST" action="/note"><textarea name="text" placeholder="…"></textarea>
 <button type="submit">Send to {being}</button></form>
</div>

<h2>Its open questions to you</h2>
{threads}

<h2>Its journal, most recent</h2>
<div class="card"><pre>{journal}</pre></div>
</div></body></html>"""

THREAD_BLOCK = """<div class="card">
 <div><b>{name}</b><span class="pill {cls}">{state}</span>
  <span class="meta"> · last touched {mtime}</span></div>
 <pre style="margin-top:10px">{body}</pre>
 <form method="POST" action="/reply">
  <input type="hidden" name="thread" value="{name}">
  <textarea name="text" placeholder="Your reply is appended to this thread. The being reads it in full every beat."></textarea>
  <button type="submit">Reply in this thread</button>
 </form>
</div>"""


def render(flash: str = "") -> str:
    b = latest_beat()
    ts = []
    for t in threads():
        answered = t["tail_is_dp"]
        ts.append(THREAD_BLOCK.format(
            name=html.escape(t["name"]), mtime=t["mtime"],
            cls="done" if answered else "wait",
            state="you answered last" if answered else "waiting on you",
            body=html.escape(t["body"])))
    return PAGE.format(
        being=html.escape(BEING),
        flash=f'<div class="card" style="border-color:var(--gn)">{html.escape(flash)}</div>' if flash else "",
        beat_ts=b.get("ts") or "—", beat_drive=b.get("drive") or "—",
        beat_elapsed=b.get("elapsed") or "—", beat_harness=b.get("harness") or "—",
        beat_ctx=b.get("ctx") or "—",
        beat_acts=html.escape(", ".join(b.get("acts") or []) or "no acts recorded"),
        beat_reply=html.escape(b.get("reply") or ""),
        threads="".join(ts) or '<div class="card meta">no ask threads on disk</div>',
        journal=html.escape(journal_tail()))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype="text/html; charset=utf-8"):
        raw = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.split("?")[0] != "/":
            return self._send(404, "not found", "text/plain")
        self._send(200, render())

    def do_POST(self):
        from urllib.parse import parse_qs
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(n).decode(errors="replace"))
        text = (form.get("text") or [""])[0]
        try:
            if self.path == "/note":
                p = append_note(text)
                flash = f"Sent. It lands in the being's next beat, at the top: {p}"
            elif self.path == "/reply":
                p = append_reply((form.get("thread") or [""])[0], text)
                flash = (f"Replied in {p.name}. It reads this thread in full every beat — "
                         "the next check will finally return something.")
            else:
                return self._send(404, "not found", "text/plain")
        except ValueError as e:
            return self._send(400, render(f"Not sent: {e}"))
        self._send(200, render(flash))

    def log_message(self, *a):
        pass


def main() -> int:
    print(f"dp console for {BEING} on http://127.0.0.1:{PORT}\n"
          f"  threads: {FORUM}\n  instance: {INSTANCE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
