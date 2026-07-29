#!/usr/bin/env python3
"""ablation — RULE 1: every channel must have a demonstrated ablation delta.

    "It's wired" is not a claim. "Turning it off changes the result" is.

This is the counter-instrument to silent resolution. In the 2026-07-28 census, thirteen failures all
produced well-formed plausible results: a dead mount returned a valid status, 13,602 lost writes produced
a valid empty cart, a starved filter returned a valid [], a truncated record read as a valid record, an
import resolved to a valid wrong file. Not one raised. Every one of them would have been caught on the day
it started by a single question — **does turning this channel OFF change the output?** — because in each
case the answer was already "no".

So a channel is not "integrated" because it is enabled, entered, or even producing. It is integrated when
disabling it MOVES a downstream metric.

*** SCOPE (dp 2026-07-29, ORGANS_ARE_THE_REFERENCE_DESIGN.md). This harness prices IMPLEMENTATIONS, never
ORGANS. Biology ran the organ-level ablation over hundreds of millions of years with death as the loss
function; anything optional was optimised out. A zero delta means THIS CODE is inert, unconnected or
untrained — it NEVER means the organ is unnecessary. The hippocampus does not owe us a delta on
next-action prediction. The LLM is the frontal lobe: removing it is a LOBOTOMY, not a control, and no
such configuration may be reported as a baseline. At epoch zero, flat is a work item — connected?
exercised? right representation? trained? — not a verdict. ***

This generalizes the memory ablation the PRD already requires (§12.1) to every channel in the stack.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEQ = os.path.abspath(os.path.join(_HERE, "..", "tools", "sequence_corpus"))
if _SEQ in sys.path:
    sys.path.remove(_SEQ)
sys.path.append(_SEQ)
if _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)


@dataclass
class Channels:
    """Which channels are live for a run. Every one must be switchable, or it cannot be ablated —
    and a channel that cannot be ablated cannot be shown to matter."""
    vision: bool = True
    memory: bool = True
    object_dynamics: bool = True
    action_attribution: bool = True
    organ_context: bool = True        # organ blocks admitted into the composed context at all
    selection_feedback: bool = True   # re-rank retrieval by past USED/REJECTED verdicts

    def off(self, name: str) -> "Channels":
        d = asdict(self)
        if name not in d:
            raise ValueError(f"unknown channel {name!r}; known: {sorted(d)}")
        d[name] = False
        return Channels(**d)

    def names(self):
        return [k for k in asdict(self)]


@dataclass
class ChannelResult:
    channel: str
    metric_on: float
    metric_off: float
    n: int = 0
    note: str = ""

    @property
    def delta(self) -> float:
        return self.metric_on - self.metric_off

    @property
    def demonstrated(self) -> bool:
        return abs(self.delta) > 1e-9

    @property
    def verdict(self) -> str:
        if self.demonstrated:
            return f"DEMONSTRATED ({self.delta:+.3f})"
        return "NOT DEMONSTRATED — disabling it changed nothing; treat as inert until shown otherwise"


def run_ablation(run_fn: Callable[[Channels], float],
                 channels: Optional[Channels] = None,
                 only: Optional[list] = None,
                 label: str = "") -> list[ChannelResult]:
    """Run `run_fn` once with everything on, then once per channel with that channel OFF.

    run_fn(channels) -> a single downstream metric (higher = better). It must be the metric you actually
    care about, not the channel's own readout: a channel that converges beautifully on its own output and
    moves nothing downstream has not earned its place.
    """
    ch = channels or Channels()
    targets = only or ch.names()
    print(f"\n=== ABLATION{' — ' + label if label else ''} ===", flush=True)
    base = run_fn(ch)
    print(f"  all channels ON -> metric = {base:.3f}", flush=True)

    results = []
    for name in targets:
        off = run_fn(ch.off(name))
        r = ChannelResult(channel=name, metric_on=base, metric_off=off)
        results.append(r)
        print(f"  {name:20s} OFF -> {off:.3f}   {r.verdict}", flush=True)
    return results


def report(results: list[ChannelResult], out_path: Optional[str] = None) -> str:
    lines = ["", "=" * 74, "ABLATION REPORT — a channel earns its place by changing the outcome",
             "=" * 74,
             f"  {'channel':22s} {'on':>7s} {'off':>7s} {'delta':>8s}   verdict"]
    inert = []
    for r in sorted(results, key=lambda x: -abs(x.delta)):
        lines.append(f"  {r.channel:22s} {r.metric_on:7.3f} {r.metric_off:7.3f} {r.delta:+8.3f}   "
                     + ("demonstrated" if r.demonstrated else "NOT DEMONSTRATED"))
        if not r.demonstrated:
            inert.append(r.channel)
    if inert:
        lines += ["", f"  ** {len(inert)} channel(s) inert on this metric: {', '.join(inert)}",
                  "     Disabling them changed nothing. Either the metric is insensitive to them, or they",
                  "     are not doing work. Both are findings; neither may be reported as 'integrated'."]
    else:
        lines += ["", "  every channel moved the metric — none is silently inert on this measure"]
    txt = "\n".join(lines)
    print(txt, flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump([asdict(r) | {"delta": r.delta, "demonstrated": r.demonstrated}
                       for r in results], f, indent=2)
    return txt


# ---------------------------------------------------------------- self-test
def _selftest():
    """A channel that genuinely contributes, and one that is wired but inert — the harness must tell
    them apart. This is the sham control for the instrument itself."""
    def fake_run(ch: Channels) -> float:
        score = 0.5
        if ch.vision:
            score += 0.2          # really contributes
        if ch.memory:
            score += 0.0          # wired, produces output, changes nothing — the silent-resolution shape
        return score
    res = run_ablation(fake_run, only=["vision", "memory"], label="self-test (sham)")
    report(res)
    ok = ({r.channel for r in res if r.demonstrated} == {"vision"})
    print(f"\n  self-test {'PASS' if ok else 'FAIL'}: harness separates contributing from inert")
    return ok


if __name__ == "__main__":
    _selftest()
