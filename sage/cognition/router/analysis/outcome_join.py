#!/usr/bin/env python3
"""Cross-stream timestamp join: router records → game outcomes.

Router shadow records capture (state → decision) pairs at the router's
decision boundary. They do NOT capture the game-level outcome that
followed (the PRD explicitly scoped the records to kernel-internal
state per §5).

But the outcome IS captured — separately, in `ARC-SAGE/knowledge/
visual-memory/{game}/run_*/run.json`. Each run.json represents a game
session with a start timestamp (dir name) and an end (file mtime).
Router records with `metadata.source=gameplay` whose timestamps fall
inside a session window can be joined to that session's outcome.

Output: enriched records with `metadata.game_outcome = {...}`, plus a
summary showing how many decisions led to WIN vs LOSS outcomes.

This is the external-validation signal pipeline. When Phase 4 (RPE-
grounded online learning) lands, training consumes outcome-joined
records instead of synthetic reward signals, so the router learns
"decisions that correlated with winning" rather than "decisions that
matched the teacher."

Scope fence (deliberate):
- Game outcomes only. Raising-session outcomes are a separate tool
  (needs raising-track T3-delta extraction). Deferred.
- Approximate boundaries. Sessions have wall-clock start (dir name)
  and end (file mtime). No per-step timestamps in current run.json.
  Fine-grained attribution within a session is out of scope.
- No mutation of source records. Enrichment produces copies.

Spec: shared-context/arc-agi-3/phase2/brain-arch/thalamic-router-prd.md
      §5 (data non-goals), §4 Phase 4 (RPE-grounded learning)
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# ───────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────

@dataclass
class GameSession:
    """One game-play run with a wall-clock window + outcome."""

    game: str                      # e.g. "bp35"
    run_dir: str                   # absolute path to the run_* dir
    start_ts: float                # unix seconds, parsed from dir name if present
    end_ts: float                  # unix seconds, taken from run.json mtime
    total_steps: int
    levels_completed: int          # best finished level (0-indexed + 1, or 0 if none)
    win_levels: int                # PRD/game-specific "win levels proved" count
    final_level_index: Optional[int]
    won: bool                      # heuristic: reached final level OR explicit result

    def contains(self, ts: float) -> bool:
        return self.start_ts <= ts <= self.end_ts

    def outcome_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "run_dir": self.run_dir,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "final_level_index": self.final_level_index,
            "won": self.won,
            "session_start": self.start_ts,
            "session_end": self.end_ts,
            "total_steps": self.total_steps,
        }


@dataclass
class JoinSummary:
    """Aggregate statistics from a join pass."""

    records_seen: int = 0
    records_by_source: Dict[str, int] = field(default_factory=dict)
    gameplay_records: int = 0
    gameplay_matched: int = 0
    gameplay_unmatched: int = 0
    sessions_touched: int = 0

    # Outcome-linked decision stats
    decisions_by_action_and_outcome: Dict[str, Dict[str, int]] = field(default_factory=dict)
    decisions_by_plugin_and_outcome: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten defaultdicts
        return d


# ───────────────────────────────────────────────────────────────────
# Session loading
# ───────────────────────────────────────────────────────────────────

_RUN_DIR_TS_RE = re.compile(r"run_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_"
                            r"(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})")


def _dir_start_ts(run_dir: Path) -> Optional[float]:
    """Parse start timestamp from dir name (run_YYYYMMDD_HHMMSS).

    Returns None when the dir is a semantic name (e.g. run_L5_winning).
    Caller falls back to file mtime.
    """
    match = _RUN_DIR_TS_RE.match(run_dir.name)
    if not match:
        return None
    import datetime as dt
    d = match.groupdict()
    try:
        naive = dt.datetime(
            int(d["y"]), int(d["m"]), int(d["d"]),
            int(d["H"]), int(d["M"]), int(d["S"]),
        )
        # Assume local-time naming. Many machines stamp in local wall
        # clock; we'll convert via timestamp() which treats as local.
        return naive.timestamp()
    except (ValueError, OSError):
        return None


def _classify_outcome(data: Dict[str, Any]) -> Tuple[int, int, Optional[int], bool]:
    """Extract (levels_completed, win_levels, final_level_index, won).

    Run JSONs vary in schema. We support the bp35/lf52/dc22 style
    (win_levels_proved + final_level_index) and the generic
    (levels_completed + result) style.
    """
    win_levels = int(data.get("win_levels_proved") or 0)
    final_idx = data.get("final_level_index")
    levels_completed = int(
        data.get("levels_completed") or
        (final_idx + 1 if isinstance(final_idx, int) else 0) or
        win_levels
    )
    # "won" is True if explicit result says so, or win_levels == expected level count
    result = str(data.get("result") or "").upper()
    won = (result == "WIN") or (
        win_levels > 0
        and isinstance(final_idx, int)
        and win_levels >= final_idx
    )
    return levels_completed, win_levels, (final_idx if isinstance(final_idx, int) else None), won


def load_game_sessions(games_dir: Path) -> List[GameSession]:
    """Walk ARC-SAGE/knowledge/visual-memory/{game}/run_*/run.json."""
    games_dir = Path(games_dir)
    sessions: List[GameSession] = []
    if not games_dir.exists():
        return sessions

    for game_dir in sorted(games_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        for run_dir in sorted(game_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            run_json = run_dir / "run.json"
            if not run_json.exists():
                continue
            try:
                data = json.loads(run_json.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            start_ts = _dir_start_ts(run_dir)
            try:
                end_ts = run_json.stat().st_mtime
            except OSError:
                continue
            if start_ts is None:
                # Semantic-named dir — use file mtime as both ends
                # (best we can do; these sessions have no wall-clock window).
                start_ts = end_ts

            levels_completed, win_levels, final_idx, won = _classify_outcome(data)
            sessions.append(GameSession(
                game=game,
                run_dir=str(run_dir),
                start_ts=start_ts,
                end_ts=end_ts,
                total_steps=int(data.get("total_steps") or len(data.get("steps") or [])),
                levels_completed=levels_completed,
                win_levels=win_levels,
                final_level_index=final_idx,
                won=won,
            ))
    return sessions


# ───────────────────────────────────────────────────────────────────
# Joiner
# ───────────────────────────────────────────────────────────────────

class OutcomeJoiner:
    """Maps router record timestamps to game-session outcomes.

    Uses binary search on session start_ts for O(log n) lookup.
    Handles overlapping sessions (different games, or same game on
    different machines) by returning the most specific match — the
    session whose window contains ts AND whose start_ts is latest.
    """

    def __init__(self, sessions: Iterable[GameSession]):
        self._sessions: List[GameSession] = sorted(sessions, key=lambda s: s.start_ts)
        self._starts = [s.start_ts for s in self._sessions]

    def find_session(self, ts: float) -> Optional[GameSession]:
        """Return the session containing `ts`, or None."""
        if not self._sessions:
            return None
        # Find the latest session whose start_ts ≤ ts
        idx = bisect.bisect_right(self._starts, ts) - 1
        # Walk backward through candidates; any whose window contains ts wins
        while idx >= 0:
            s = self._sessions[idx]
            if s.contains(ts):
                return s
            # Earlier sessions may still contain (if they ended after a later start)
            # but bisect only found the latest start — continue scanning only
            # if prior session's end could cover ts
            if self._sessions[idx].end_ts < ts:
                return None
            idx -= 1
        return None

    def __len__(self) -> int:
        return len(self._sessions)


# ───────────────────────────────────────────────────────────────────
# Record loading + enrichment
# ───────────────────────────────────────────────────────────────────

def _iter_records(records_dir: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for shard in sorted(Path(records_dir).glob("**/*.jsonl*")):
        open_fn = gzip.open if shard.suffix == ".gz" else open
        try:
            with open_fn(shard, "rt") as f:
                for line in f:
                    try:
                        yield shard, json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
        except (EOFError, OSError):
            continue


def enrich_records(
    records_dir: Path,
    joiner: OutcomeJoiner,
    output_path: Optional[Path] = None,
) -> JoinSummary:
    """Walk records, attach game outcomes where applicable, emit summary.

    If output_path is given, enriched records are written to it as
    JSONL (gzipped iff the path ends in .gz). Otherwise only the
    summary is computed.

    Original records are never mutated on disk. The caller keeps the
    captured artifacts untouched.
    """
    summary = JoinSummary()
    by_action_and_outcome: Dict[str, Counter] = defaultdict(Counter)
    by_plugin_and_outcome: Dict[str, Counter] = defaultdict(Counter)
    sessions_touched: set = set()

    out = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = (gzip.open if str(output_path).endswith(".gz") else open)(
            str(output_path), "wt", encoding="utf-8"
        )

    try:
        for _shard, rec in _iter_records(records_dir):
            summary.records_seen += 1
            src = (rec.get("metadata") or {}).get("source", "unknown")
            summary.records_by_source[src] = summary.records_by_source.get(src, 0) + 1
            if src == "gameplay":
                summary.gameplay_records += 1
                ts = float(rec.get("timestamp") or 0.0)
                session = joiner.find_session(ts)
                if session:
                    summary.gameplay_matched += 1
                    sessions_touched.add(session.run_dir)
                    # Don't mutate; shallow-copy + attach
                    enriched = dict(rec)
                    meta = dict(rec.get("metadata") or {})
                    meta["game_outcome"] = session.outcome_dict()
                    enriched["metadata"] = meta
                    rec_for_stats = enriched
                    if out is not None:
                        out.write(json.dumps(enriched) + "\n")
                    # Tally
                    action = (rec.get("router_output") or {}).get("action", "unknown")
                    plugin = (rec.get("router_output") or {}).get("plugin") or "∅"
                    outcome_key = "won" if session.won else "lost"
                    by_action_and_outcome[action][outcome_key] += 1
                    by_plugin_and_outcome[plugin][outcome_key] += 1
                else:
                    summary.gameplay_unmatched += 1
                    if out is not None:
                        out.write(json.dumps(rec) + "\n")
            else:
                if out is not None:
                    out.write(json.dumps(rec) + "\n")
    finally:
        if out is not None:
            out.close()

    summary.sessions_touched = len(sessions_touched)
    summary.decisions_by_action_and_outcome = {
        k: dict(v) for k, v in by_action_and_outcome.items()
    }
    summary.decisions_by_plugin_and_outcome = {
        k: dict(v) for k, v in by_plugin_and_outcome.items()
    }
    return summary


# ───────────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────────

def _print_summary(s: JoinSummary) -> None:
    print("=" * 60)
    print("Router record → game outcome join")
    print("=" * 60)
    print(f"  Records seen                : {s.records_seen}")
    print(f"  By source                   : {s.records_by_source}")
    print(f"  Gameplay records            : {s.gameplay_records}")
    print(f"  Matched to a game session   : {s.gameplay_matched}"
          + (f"  ({s.gameplay_matched / max(s.gameplay_records, 1) * 100:.1f}% of gameplay)"
             if s.gameplay_records else ""))
    print(f"  Unmatched gameplay          : {s.gameplay_unmatched}")
    print(f"  Distinct sessions touched   : {s.sessions_touched}")
    if s.decisions_by_action_and_outcome:
        print()
        print("  Decisions by action × outcome:")
        for action, counts in sorted(s.decisions_by_action_and_outcome.items()):
            won, lost = counts.get("won", 0), counts.get("lost", 0)
            total = won + lost
            win_rate = won / total if total else 0.0
            print(f"    {action:10s}: won={won:6d} lost={lost:6d} "
                  f"win_rate={win_rate:.3f} (n={total})")
    if s.decisions_by_plugin_and_outcome:
        print()
        print("  Decisions by plugin × outcome (top 10 by total):")
        top = sorted(
            s.decisions_by_plugin_and_outcome.items(),
            key=lambda kv: -(kv[1].get("won", 0) + kv[1].get("lost", 0)),
        )[:10]
        for plugin, counts in top:
            won, lost = counts.get("won", 0), counts.get("lost", 0)
            total = won + lost
            win_rate = won / total if total else 0.0
            print(f"    {plugin:20s}: won={won:6d} lost={lost:6d} "
                  f"win_rate={win_rate:.3f} (n={total})")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records-dir", required=True,
                   help="Path to router shadow partitions (per-machine or _aggregate).")
    p.add_argument("--games-dir", required=True,
                   help="Path to ARC-SAGE/knowledge/visual-memory/ (game run trees).")
    p.add_argument("--output", default=None,
                   help="Optional JSONL output path for enriched records "
                        "(gzipped if ends in .gz).")
    p.add_argument("--summary-json", default=None,
                   help="Optional JSON path for the aggregate summary.")
    args = p.parse_args()

    sessions = load_game_sessions(Path(args.games_dir))
    print(f"Loaded {len(sessions)} game sessions from {args.games_dir}")

    joiner = OutcomeJoiner(sessions)
    summary = enrich_records(
        Path(args.records_dir),
        joiner,
        output_path=Path(args.output) if args.output else None,
    )
    _print_summary(summary)

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nWrote summary: {args.summary_json}")

    # Exit 0 always — this is a reporting tool, not a gate.
    return 0


if __name__ == "__main__":
    sys.exit(main())
