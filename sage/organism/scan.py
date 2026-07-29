#!/usr/bin/env python3
"""scan — the instrument scan (INSTRUMENT_SCAN.md).

dp 2026-07-29: "you can't fly a plane IFR by one instrument alone... fixating on just one will put you in
the ground. All instruments matter, and together they enable situational awareness."

Emits the full panel from a run's artifacts and runs the CROSS-CHECKS. The cross-check is the load-bearing
part: a silently-wrong instrument reads plausibly and is caught by DISAGREEMENT with the others, never by
inspecting it alone — which is exactly the shape of this codebase's characteristic failure (the 2026-07-28
census: thirteen failures, all producing well-formed plausible values, none raising).

An instrument with no source reads U/S (unserviceable), never 0. An instrument that is not reporting is not
an instrument reading zero — conflating those is the same error as calling an unexercised organ inert.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

US = "U/S"


def _fmt(v, suffix="", dec=0):
    return US if v is None else (f"{v:.{dec}f}{suffix}")


def read_rows(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def read_liveness(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path))
        return d.get("liveness", d)
    except Exception:
        return {}


def panel(rows, live, history=None, reserves=None):
    """Compute every instrument. None == U/S, never 0."""
    p = {}
    n = len(rows)

    # CAPABILITY (altimeter) — where we are now, against the constant policy
    if n:
        cold = sum(r.get("cold_ok") for r in rows) / n
        warm = sum(r.get("warm_ok") for r in rows) / n
        maj = Counter(r.get("actual") for r in rows).most_common(1)[0]
        p["capability"] = {"n": n, "cold": 100 * cold, "warm": 100 * warm,
                           "baseline": 100 * maj[1] / n, "baseline_action": maj[0],
                           "resolution": 100.0 / n}
    else:
        p["capability"] = None

    # ACTIVITY (airspeed) — is it doing work? below stall = not flying
    if n:
        used = sum(1 for r in rows if r.get("used", 0) > 0)
        retr = sum(1 for r in rows if r.get("retrieved", 0) > 0)
        p["activity"] = {"steps": n, "memory_delivered_pct": 100 * retr / n,
                         "memory_used_pct": 100 * used / n,
                         "levels_attempted": sum(1 for r in rows if r.get("level_attempted")),
                         "levels_cleared": sum(1 for r in rows if r.get("level_cleared"))}
    else:
        p["activity"] = None

    # COORDINATION (turn coordinator) — cooperating or fighting?
    if n:
        harmful = sum(1 for r in rows if r.get("used", 0) > 0
                      and r.get("cold_ok") and not r.get("warm_ok"))
        helped = sum(1 for r in rows if r.get("used", 0) > 0
                     and not r.get("cold_ok") and r.get("warm_ok"))
        usedn = max(sum(1 for r in rows if r.get("used", 0) > 0), 1)
        p["coordination"] = {"harmful_retrieval_pct": 100 * harmful / usedn,
                             "helpful_retrieval_pct": 100 * helped / usedn,
                             "changed_pct": 100 * sum(1 for r in rows if r.get("changed")) / n}
    else:
        p["coordination"] = None

    # COMPOSITION (attitude indicator) — whole, connected, insulated? THE MASTER INSTRUMENT.
    # Contamination is computable from rows ALONE and is the single most valuable cross-check, so it must
    # never depend on an optional liveness file. Burying it there was the same class of error the panel
    # exists to catch: the key reading silently unavailable.
    contam = None
    if n:
        rej = [r for r in rows if r.get("retrieved", 0) > 0 and r.get("used", 0) == 0]
        if rej:
            contam = 100 * sum(1 for r in rej if r.get("changed")) / len(rej)
    comps = (live or {}).get("components", {})
    if comps or contam is not None:
        rungs = {name: c.get("top") for name, c in comps.items()}
        reached_used = [k for k, v in rungs.items() if v in ("used", "affected")]
        errs = sum(len(c.get("errors", [])) for c in comps.values())
        p["composition"] = {"components": len(comps), "reached_used": len(reached_used),
                            "errors": errs, "contamination_pct": contam,
                            "top_rungs": rungs, "ladder_available": bool(comps)}
    else:
        p["composition"] = None

    # LEARNING (VSI) — trend, not level
    if history and len(history) >= 2:
        xs = list(range(len(history)))
        ws = [h["warm_pct"] for h in history]
        cs = [h["cold_pct"] for h in history]
        mx = sum(xs) / len(xs)

        def slope(ys):
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs) or 1
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        p["learning"] = {"warm_slope": slope(ws), "cold_slope": slope(cs), "passes": len(history)}
    else:
        p["learning"] = None

    # COURSE (nav) — transfer radius actually achieved
    p["course"] = {"radius_demonstrated": "none (next-action prediction on replays)",
                   "levels_played": (p["activity"] or {}).get("levels_attempted", 0) if p["activity"] else None}

    p["reserves"] = reserves
    return p


def render(p) -> str:
    L = ["", "=" * 78, "INSTRUMENT SCAN", "=" * 78]
    c = p.get("capability")
    L.append(f"  ALTIMETER   capability : " + (US if not c else
             f"cold {c['cold']:.0f}%  warm {c['warm']:.0f}%  vs baseline {c['baseline']:.0f}% "
             f"('{c['baseline_action']}')  [n={c['n']}, resolution {c['resolution']:.0f}%/step]"))
    l = p.get("learning")
    L.append(f"  VSI         learning   : " + (US if not l else
             f"warm {l['warm_slope']:+.2f}%/pass   cold {l['cold_slope']:+.2f}%/pass  "
             f"[{l['passes']} passes]"))
    co = p.get("composition")
    L.append(f"  ATTITUDE    composition: " + (US if not co else
             (f"{co['reached_used']}/{co['components']} components reached USED, {co['errors']} errors, "
              if co.get("ladder_available") else "ladder U/S, ")
             + f"contamination {_fmt(co['contamination_pct'], '%')}"))
    a = p.get("activity")
    L.append(f"  AIRSPEED    activity   : " + (US if not a else
             f"{a['steps']} steps  memory delivered {a['memory_delivered_pct']:.0f}% / "
             f"used {a['memory_used_pct']:.0f}%  levels played {a['levels_attempted']} "
             f"cleared {a['levels_cleared']}"))
    cd = p.get("coordination")
    L.append(f"  TURN-COORD  coordination: " + (US if not cd else
             f"of retrievals used — helpful {cd['helpful_retrieval_pct']:.0f}% / "
             f"HARMFUL {cd['harmful_retrieval_pct']:.0f}%   decisions changed {cd['changed_pct']:.0f}%"))
    L.append(f"  NAV         course     : {p['course']['radius_demonstrated']}")
    L.append(f"  FUEL        reserves   : {p.get('reserves') or US}")
    return "\n".join(L)


def crosscheck(p) -> list:
    """The load-bearing part: instruments that DISAGREE indicate a failed instrument, not a fact."""
    w = []
    c, co, a, cd, l = (p.get(k) for k in ("capability", "composition", "activity",
                                          "coordination", "learning"))
    if c and co and co.get("contamination_pct") is not None:
        if c["warm"] > c["cold"] and co["contamination_pct"] > 25:
            w.append(f"CAPABILITY says warm>cold (+{c['warm']-c['cold']:.0f}pp) but COMPOSITION says "
                     f"{co['contamination_pct']:.0f}% of decisions changed with NOTHING accepted. "
                     f"Incompatible — the capability reading is not trustworthy.")
    if c and a and a["memory_used_pct"] < 5 and abs(c["warm"] - c["cold"]) > 5:
        w.append(f"CAPABILITY shows a memory effect while ACTIVITY says memory was used on only "
                 f"{a['memory_used_pct']:.0f}% of steps. The effect is not coming from where it is claimed.")
    if c and abs(c["warm"] - c["cold"]) * c["n"] / 100 < 1.0 and abs(c["warm"] - c["cold"]) > 0:
        w.append(f"CAPABILITY delta ({c['warm']-c['cold']:+.0f}pp) is below resolution "
                 f"({c['resolution']:.0f}%/step at n={c['n']}) — unresolved, not small.")
    if l and abs(l["cold_slope"]) > 2.0:
        w.append(f"LEARNING: cold slope {l['cold_slope']:+.2f}%/pass is NOT flat. The model is stateless "
                 f"and prompts identical, so this indicates leakage. Warm slope uninterpretable.")
    if cd and cd["harmful_retrieval_pct"] > cd["helpful_retrieval_pct"]:
        w.append(f"COORDINATION: retrieval is net HARMFUL ({cd['harmful_retrieval_pct']:.0f}% vs "
                 f"{cd['helpful_retrieval_pct']:.0f}% helpful) — organs fighting, not cooperating.")
    if a and a["levels_attempted"] == 0:
        w.append("NAV/AIRSPEED: zero levels played. Everything here is prediction on recorded replays — "
                 "a proxy, not the objective. No transfer radius is being demonstrated.")
    if c and (co is None or not co.get("ladder_available")):
        w.append("Liveness ladder U/S while CAPABILITY is being read — the master instrument is partly "
                 "missing. Do not fly on the altimeter alone.")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows"); ap.add_argument("--liveness"); ap.add_argument("--bundle")
    ap.add_argument("--reserves")
    a = ap.parse_args()
    rows = read_rows(a.rows)
    live = read_liveness(a.liveness or a.bundle)
    history = None
    if a.bundle and os.path.exists(a.bundle):
        try:
            history = json.load(open(a.bundle)).get("history")
        except Exception:
            history = None
    p = panel(rows, live, history=history, reserves=a.reserves)
    print(render(p))
    w = crosscheck(p)
    print("\n  CROSS-CHECK")
    if not w:
        print("    instruments agree — no contradiction detected (agreement is not proof of correctness)")
    for x in w:
        print(f"    ** {x}")
    print("\n  A single instrument carries a READING, never a verdict.\n")


if __name__ == "__main__":
    main()
