#!/usr/bin/env python3
"""liveness — the six-rung ladder (PRD v3 §0).

    enabled → entered → produced → admitted → used → affected

Our old `_produce` convention stopped at rung 3, which is exactly why two separate questions were
unanswerable for months:
  - "vision was on" never told us whether vision MATTERED (the 7-22/23 scored wash);
  - membot's four silent-drop incidents each looked like "a channel with nothing to say".

A channel that returns nothing must be DISTINGUISHABLE from a channel that was never asked. That is the
whole job of this module. It never swallows an exception and never reports success it cannot evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

RUNGS = ["enabled", "entered", "produced", "admitted", "used", "affected"]


@dataclass
class Component:
    name: str
    rungs: dict = field(default_factory=dict)      # rung -> {"t", "detail"}
    counts: dict = field(default_factory=dict)     # e.g. {"hits": 3, "items": 8}
    errors: list = field(default_factory=list)

    def top_rung(self) -> Optional[str]:
        reached = [r for r in RUNGS if r in self.rungs]
        return reached[-1] if reached else None

    def reached(self, rung: str) -> bool:
        return rung in self.rungs


class Ladder:
    """Records how far each component actually got. Fail-closed by contract (`require`)."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.components: dict[str, Component] = {}
        self.events: list[dict] = []

    def _c(self, name: str) -> Component:
        return self.components.setdefault(name, Component(name))

    def mark(self, component: str, rung: str, detail: Any = None, **counts):
        if rung not in RUNGS:
            raise ValueError(f"unknown rung {rung!r}; valid: {RUNGS}")
        c = self._c(component)
        c.rungs[rung] = {"t": time.time(), "detail": detail}
        for k, v in counts.items():
            c.counts[k] = v
        self.events.append({"t": time.time(), "component": component, "rung": rung,
                            "detail": detail, **counts})
        return self

    def error(self, component: str, exc: BaseException, context: str = ""):
        """Record a failure LOUDLY. Never used to make a failure look like an absence."""
        c = self._c(component)
        msg = f"{type(exc).__name__}: {exc}" + (f" [{context}]" if context else "")
        c.errors.append(msg)
        self.events.append({"t": time.time(), "component": component, "rung": "ERROR", "detail": msg})
        print(f"[liveness] ERROR {component}: {msg}", flush=True)
        return self

    def flow(self, component: str, n_in: int, n_out: int, reason: str = ""):
        """RULE 2 — every filter and retrieval reports (in -> out -> dropped).

        Zero-in-zero-out was invisible for months in this stack: a tag filter that matched nothing and a
        provenance filter that excluded nothing both looked exactly like 'nothing to report'. A filter
        that drops EVERYTHING or drops NOTHING across a run is suspicious by construction, so both
        extremes are called out here rather than left for someone to notice.
        """
        dropped = n_in - n_out
        c = self._c(component)
        c.counts.update({"in": n_in, "out": n_out, "dropped": dropped})
        flag = ""
        if n_in > 0 and n_out == 0:
            flag = "  ** DROPPED EVERYTHING — filter may be starved **"
        elif n_in > 0 and dropped == 0 and reason:
            flag = "  ** DROPPED NOTHING — filter may be inert **"
        self.events.append({"t": time.time(), "component": component, "rung": "flow",
                            "in": n_in, "out": n_out, "dropped": dropped, "reason": reason})
        print(f"[flow] {component}: in={n_in} out={n_out} dropped={dropped}"
              + (f" ({reason})" if reason else "") + flag, flush=True)
        return self

    def require(self, component: str, rung: str):
        """Fail-closed gate. A milestone run must not 'succeed' with an inert component."""
        c = self.components.get(component)
        if c is None or not c.reached(rung):
            got = c.top_rung() if c else None
            raise RuntimeError(
                f"LIVENESS FAILURE: '{component}' required at rung '{rung}' but reached '{got}'. "
                f"errors={(c.errors if c else [])}")

    def report(self) -> str:
        lines = [f"=== LIVENESS LADDER ({self.run_id}) ==="]
        w = max([len(n) for n in self.components] + [9])
        lines.append("  " + "component".ljust(w) + "  " + " ".join(r[:4].ljust(4) for r in RUNGS) + "   counts")
        for name, c in self.components.items():
            marks = " ".join(("  ✓ " if c.reached(r) else "  · ") for r in RUNGS)
            cnt = " ".join(f"{k}={v}" for k, v in c.counts.items())
            lines.append(f"  {name.ljust(w)}  {marks}   {cnt}" + ("  ERR:" + "; ".join(c.errors) if c.errors else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id,
                "components": {n: {"rungs": list(c.rungs), "counts": c.counts, "errors": c.errors,
                                   "top": c.top_rung()} for n, c in self.components.items()},
                "events": self.events}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
