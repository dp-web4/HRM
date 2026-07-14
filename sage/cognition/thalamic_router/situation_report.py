"""Perception-as-integrator V1 — SituationReport organ.

Item #7 from the SAGE-WM math-to-brain build order
(`forum/nomad-sage-wm-math-to-brain-mapping-2026-05-31.md` §5).

The math wants a perception update `μ̇ = −∂F/∂μ` that fuses all evidence
channels into a unified per-model state estimate. The framing has been
canonical since 2026-05-01
(`arc-agi-3/phase2/brain-arch/perception-as-integrator-2026-05-01.md`):

  Perception is not "what the frame says." Perception is "what the
  situation is, given everything I have access to right now."

Five channels, one SituationReport, fed into one cortex turn:

  now    — current frame + animation delta
  then   — past episodes / fleet wisdom / cartridge retrievals
  laws   — cross-game patterns / structural regularities
  self_  — own WM state + recent action history (interoception)
  salient — the integrator's pick of what matters here (learned, future)

This V1 is INSTRUMENTATION-ONLY (per the math-foundation discipline).
It builds the SituationReport from existing channel sources, logs its
shape and per-channel content size, and does NOT change the cortex
prompt or any action selection. Behavior is unchanged from v37; the
data stream is added.

Gated by `SAGE_PERCEPTION_INTEGRATOR=1`. Default OFF.

V2 (later): re-render the cortex prompt from the SituationReport as a
single coherent SITUATION block instead of disjoint sections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SituationReport:
    """Unified per-cycle situational read across all sensory channels.

    Fields follow the perception-as-integrator framing. Each channel
    holds a structured payload (any of: str / dict / list); the integrator
    organ is the authority on the schema for each. None when the channel
    is unavailable this cycle.
    """
    # Channels
    now: Optional[Dict[str, Any]] = None         # current frame + delta
    then: Optional[Dict[str, Any]] = None        # past episodes / cartridge
    laws: Optional[Dict[str, Any]] = None        # cross-game patterns
    self_: Optional[Dict[str, Any]] = None       # WM state + action history
    salient: Optional[Dict[str, Any]] = None     # integrator's pick (V2)

    # Provenance
    game: Optional[str] = None
    level: Optional[int] = None
    cycle: Optional[int] = None
    sources_present: List[str] = field(default_factory=list)
    sources_missing: List[str] = field(default_factory=list)

    def channel_coverage(self) -> float:
        """Fraction of the 5 canonical channels that have non-empty content
        this cycle. A metric for the integrator's situational completeness.
        Range [0, 1]."""
        present = sum(1 for ch in (self.now, self.then, self.laws,
                                   self.self_, self.salient)
                      if ch is not None)
        return present / 5.0

    def summary_one_line(self) -> str:
        """Compact log-friendly summary for the [SITUATION] tag."""
        bits = []
        for name, ch in (("now", self.now), ("then", self.then),
                         ("laws", self.laws), ("self", self.self_),
                         ("salient", self.salient)):
            mark = "✓" if ch is not None else "·"
            sz = _channel_size(ch) if ch is not None else 0
            bits.append(f"{name}{mark}{sz}")
        return " ".join(bits)


def _channel_size(ch: Any) -> int:
    """Rough size proxy for a channel's content — number of items if dict
    or list, character count for strings, 1 for atomic, 0 for None."""
    if ch is None:
        return 0
    if isinstance(ch, dict):
        return len(ch)
    if isinstance(ch, list):
        return len(ch)
    if isinstance(ch, str):
        return len(ch)
    return 1


def _try_get(obj, *attrs, default=None):
    """Walk a chain of attribute names safely. Returns default if any
    intermediate is missing."""
    cur = obj
    for a in attrs:
        cur = getattr(cur, a, None)
        if cur is None:
            return default
    return cur


def _build_now_channel(wm, fd) -> Optional[Dict[str, Any]]:
    """Current frame + animation delta. Pulls from the FrameData and the
    WM's recent observation buffer if available."""
    payload: Dict[str, Any] = {}
    # Frame summary — count of objects visible at the WM level
    obj_count = 0
    objs = _try_get(wm, "objects", default=None)
    if objs:
        obj_count = len(objs) if hasattr(objs, "__len__") else 0
    payload["object_count"] = obj_count
    # Frame dims (used by perception)
    payload["frame_idx"] = _try_get(fd, "frame_idx", default=None)
    # Recent delta — last two frames' rough size diff
    frame = _try_get(fd, "frame", default=None)
    if frame is not None and hasattr(frame, "__len__") and len(frame) >= 2:
        payload["animation_frames_available"] = len(frame)
    # Available actions are part of "now"
    payload["available_actions"] = _try_get(fd, "available_actions", default=[])
    return payload if payload else None


def _build_then_channel(wm) -> Optional[Dict[str, Any]]:
    """Past — cartridge retrievals, recent strategy articulations.

    Reads from `wm._last_cartridge_retrievals` (populated by the
    CARTRIDGE-RETRIEVE V1 instrumentation when SAGE_CARTRIDGE_RETRIEVE=1).
    Empty/missing → channel returns None."""
    retrievals = _try_get(wm, "_last_cartridge_retrievals", default=None)
    if not retrievals:
        return None
    return {
        "n_cartridge_candidates": len(retrievals),
        "preview": [r.get("preview", "")[:80] for r in retrievals[:3]],
    }


def _build_laws_channel(wm) -> Optional[Dict[str, Any]]:
    """Cross-game / structural regularities. Reads from causal_rules
    (per-game laws the WM has accumulated) and the rule-harvester overlay
    if present."""
    rules = _try_get(wm, "causal_rules", default=None)
    if not rules:
        return None
    # Count of rules with positive evidence — the "active laws" pool
    active = [r for r in rules if getattr(r, "evidence_count", 0) > 0]
    if not active:
        return None
    return {
        "n_active_rules": len(active),
        "actions_with_rules": sorted({
            (r.action.split()[0].upper() if isinstance(r.action, str) else "?")
            for r in active
        }),
    }


def _build_self_channel(wm) -> Optional[Dict[str, Any]]:
    """Interoception — WM state + recent action outcomes.

    Includes faith portfolio snapshot (best candidate, trust distribution),
    recent learning log entries, level progression context.
    """
    payload: Dict[str, Any] = {}
    # Faith portfolio snapshot
    portfolio = _try_get(wm, "faith_portfolio", default=None)
    if portfolio is not None and getattr(portfolio, "candidates", None):
        best_score = portfolio.best_score()
        n_cands = len(portfolio.candidates)
        payload["faith_n_candidates"] = n_cands
        payload["faith_best_score"] = round(best_score, 3)
        # Names of candidates with trust above commit threshold
        commit_tau = getattr(portfolio, "commit_tau", 0.45)
        ready = [c.id for c in portfolio.candidates.values()
                 if c.trust >= commit_tau]
        payload["faith_commit_ready_ids"] = ready
    # Level progression context
    payload["level"] = _try_get(wm, "level", default=0)
    payload["levels_completed"] = _try_get(wm, "_levels_completed_at_plan_start",
                                            default=None)
    return payload if payload else None


def _build_salient_channel(wm) -> Optional[Dict[str, Any]]:
    """Integrator's pick of what matters here. V1 placeholder — returns
    None always. V2 will compute this from the cross-channel signal."""
    return None


def build_situation_report(wm, fd, cycle: Optional[int] = None) -> SituationReport:
    """Build a SituationReport from the current game state.

    Pulls each channel from its canonical source organ. Logs the channel
    coverage. Returns the report for downstream V2 use (currently
    informational only).

    Gated only by the calling code's SAGE_PERCEPTION_INTEGRATOR check;
    this function itself is always safe to call (returns an empty-ish
    SituationReport if no channels populate).
    """
    now = _build_now_channel(wm, fd)
    then = _build_then_channel(wm)
    laws = _build_laws_channel(wm)
    self_ = _build_self_channel(wm)
    salient = _build_salient_channel(wm)

    sources_present = []
    sources_missing = []
    for name, ch in (("now", now), ("then", then), ("laws", laws),
                     ("self_", self_), ("salient", salient)):
        if ch is not None:
            sources_present.append(name)
        else:
            sources_missing.append(name)

    return SituationReport(
        now=now, then=then, laws=laws, self_=self_, salient=salient,
        game=_try_get(wm, "game", default=None),
        level=_try_get(wm, "level", default=None),
        cycle=cycle,
        sources_present=sources_present,
        sources_missing=sources_missing,
    )


def log_situation_report(sr: SituationReport) -> None:
    """Emit a one-line [SITUATION] log of the report's coverage.

    Format: `[SITUATION] <game> L<level> cyc<N>: <channels> coverage=<pct>`
    Channels show as `name<mark><size>` where mark is ✓ if present, · if
    missing, and size is the rough payload size. Used by analysis tooling
    to track which channels are reliably populated across the corpus.
    """
    game = sr.game or "?"
    lvl = sr.level if sr.level is not None else "?"
    cyc = sr.cycle if sr.cycle is not None else "?"
    cov_pct = int(sr.channel_coverage() * 100)
    print(f"[SITUATION] {game} L{lvl} cyc{cyc}: "
          f"{sr.summary_one_line()} coverage={cov_pct}%",
          flush=True)
