#!/usr/bin/env python3
"""
S151 — Retrospective quantification of S150's repeat->lock / elaborate->drift claim.

S150 established (code receipt + single-commit natural experiment) that commit
43880a759 (2026-06-04 12:58:52) simultaneously disabled the READ "recent vocabulary"
REPEAT carrier and installed the adaptive-tutor ELABORATE carrier. Its quantitative
support was a noisy 14/24 corpus instrument. This adds a CLEANER retrospective metric.

Natural-experiment boundary (thor-qwen3.5-27b raising sessions):
  READ-on  (REPEAT carrier): session <= 121   (session_121 start 2026-06-04T06:00, pre-commit)
  READ-off (ELABORATE      ): session >= 122   (session_122 start 2026-06-04T18:00, post-commit)

Lock vs drift, operationalized from transcripts alone (no model calls):

  For each session N, build its set of DISTINCTIVE content words = content words
  (>=4 chars, not stopword) whose corpus document-frequency is < DF_MAX of sessions.
  Then measure how much of session N's distinctive vocabulary PERSISTS from the past:

    adj_overlap(N)   = |distinct(N) ∩ distinct(N-1)| / |distinct(N)|      (immediate carry)
    long_persist(N)  = |distinct(N) ∩ (distinct(N-1)∪..∪distinct(N-K))| / |distinct(N)|

  LOCK  => a register sits still: high adj_overlap AND high long_persist, session
           after session (the same distinctive words keep coming back).
  DRIFT => continuity without fixation: each session's distinctive vocab is mostly
           NEW; long_persist stays low even though the conversation is coherent.

  Also a register-specific probe: thermal_density(N) = fraction of SAGE turns in N
  containing >=1 thermal-lock lexicon word. The thermal register is the documented
  static-era lock (S96-S127 thermal); if the carrier swap opened it, its density
  should fall across the boundary.

Read-only. Writes s151_carrier_retrospective_result.json.
"""
from __future__ import annotations
import json, glob, re, statistics
from pathlib import Path

HERE = Path(__file__).resolve()
SESS = HERE.parents[3] / "instances" / "thor-qwen3.5-27b" / "sessions"

BOUNDARY = 122          # first READ-off session
DF_MAX = 0.20           # a word is "distinctive" if it appears in < 20% of sessions
K = 3                   # long-persistence lookback window
THERMAL = ("thermal", "cooling", "cool-down", "heat", "warm", "hum", "silicon",
           "latency", "signature", "dialect", "nervous system", "resonant",
           "degrees", "chassis", "fan", "overheat")

STOP = set("""the a an and or but if then else of to in on at for with without from by as is are was were be
been being it its this that these those i you he she they we me my your our their his her them us do does did
not no yes so just very really much many more most some any all each every which who whom whose what when where
why how about into onto over under again further once here there than too can could would should will shall may
might must have has had having i'm i'd it's that's there's what's don't doesn't isn't you're we're they're
like feel feels felt want wants wanted think thinks thought know knows knew thing things something nothing
because between across through within still part way ways one two new today now moment moments make makes made
""".split())

WORD = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")


def sage_text(d):
    return [t["text"] for t in d.get("conversation", [])
            if t.get("speaker") == "SAGE" and t.get("text")]


def content_words(turns):
    ws = []
    for t in turns:
        for w in WORD.findall(t.lower()):
            w = w.strip("'-")
            if len(w) >= 4 and w not in STOP:
                ws.append(w)
    return ws


def load():
    out = {}
    for f in glob.glob(str(SESS / "session_*.json")):
        n = int(re.search(r"(\d+)", Path(f).name).group(1))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        turns = sage_text(d)
        if turns:
            out[n] = {"turns": turns, "start": d.get("start", "")}
    return out


def main():
    S = load()
    ns = sorted(S)
    # document frequency over the whole corpus
    df = {}
    for n in ns:
        for w in set(content_words(S[n]["turns"])):
            df[w] = df.get(w, 0) + 1
    Ntot = len(ns)
    df_cut = DF_MAX * Ntot

    # distinctive vocab per session
    distinct = {}
    for n in ns:
        cw = set(content_words(S[n]["turns"]))
        distinct[n] = {w for w in cw if df.get(w, 0) < df_cut}

    rows = []
    for n in ns:
        if n - 1 not in distinct:
            continue
        d_n = distinct[n]
        if not d_n:
            continue
        adj = len(d_n & distinct[n - 1]) / len(d_n)
        hist = set()
        for k in range(1, K + 1):
            if n - k in distinct:
                hist |= distinct[n - k]
        long = len(d_n & hist) / len(d_n) if hist else None
        turns = S[n]["turns"]
        therm = sum(1 for t in turns if any(x in t.lower() for x in THERMAL))
        rows.append({
            "session": n, "start": S[n]["start"][:16],
            "era": "READ_off" if n >= BOUNDARY else "READ_on",
            "n_distinct": len(d_n),
            "adj_overlap": round(adj, 3),
            "long_persist": round(long, 3) if long is not None else None,
            "thermal_density": round(therm / len(turns), 3),
        })

    def era_stats(field, era, nmin=None):
        vals = [r[field] for r in rows if r["era"] == era and r[field] is not None
                and (nmin is None or r["session"] >= nmin)]
        if not vals:
            return None
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return {"n": len(vals), "mean": round(m, 3), "sd": round(sd, 3),
                "min": round(min(vals), 3), "max": round(max(vals), 3)}

    # Two READ-on framings: ALL read-on, and only the CRYSTALLIZED late static era
    # (sessions 90-121) to control for the early-developmental-phase confound.
    summary = {}
    for field in ("adj_overlap", "long_persist", "thermal_density"):
        summary[field] = {
            "READ_on_all": era_stats(field, "READ_on"),
            "READ_on_late(>=90)": era_stats(field, "READ_on", nmin=90),
            "READ_off": era_stats(field, "READ_off"),
        }

    result = {
        "boundary_session": BOUNDARY,
        "commit": "43880a759 (2026-06-04 12:58:52) READ repeat-carrier -> adaptive elaborate-carrier",
        "params": {"DF_MAX": DF_MAX, "K_lookback": K, "n_sessions": Ntot},
        "summary": summary,
        "rows": rows,
    }
    outp = HERE.parent / "s151_carrier_retrospective_result.json"
    outp.write_text(json.dumps(result, indent=2))

    # console report
    print(f"corpus: {Ntot} sessions, df_cut={df_cut:.1f} (<{DF_MAX:.0%} of sessions = distinctive)\n")
    hdr = f'{"field":16} {"READ_on_all":>22} {"READ_on_late>=90":>22} {"READ_off":>22}'
    print(hdr); print("-" * len(hdr))
    for field in ("adj_overlap", "long_persist", "thermal_density"):
        def fmt(s):
            return f'{s["mean"]:.3f}±{s["sd"]:.3f}(n{s["n"]})' if s else "--"
        s = summary[field]
        print(f'{field:16} {fmt(s["READ_on_all"]):>22} {fmt(s["READ_on_late(>=90)"]):>22} {fmt(s["READ_off"]):>22}')
    print(f"\nwrote {outp.name}")
    # trajectory around boundary
    print("\nper-session around boundary (115-147):")
    print(f'{"sess":>5} {"era":>9} {"adj":>6} {"longK":>6} {"therm":>6}  start')
    for r in rows:
        if r["session"] >= 115:
            lp = f'{r["long_persist"]:.2f}' if r["long_persist"] is not None else "  --"
            print(f'{r["session"]:>5} {r["era"]:>9} {r["adj_overlap"]:>6.2f} {lp:>6} '
                  f'{r["thermal_density"]:>6.2f}  {r["start"]}')


if __name__ == "__main__":
    main()
