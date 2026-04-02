"""
GameKnowledgeBase — Persistent lived experience for ARC-AGI-3 games.
Version: 1.0 (2026-04-01)

This is the cognitive scaffold that turns repeated sessions into understanding.
Each session, the LLM's general game intelligence combines with game-specific
lived experience stored here to produce increasingly competent play.

See arc-agi-3/LIVED_EXPERIENCE.md for the design philosophy.

Storage: arc-agi-3/experiments/cartridges/{game_family}.knowledge.json
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


@dataclass
class ObjectRecord:
    """A known game object — piece, button, goal, decoration."""
    obj_id: str
    obj_type: str               # "button" | "piece" | "goal" | "decoration" | "trigger" | "unknown"
    position: Dict[str, int]    # {"r": row, "c": col}
    color: str
    size_w: int = 0
    size_h: int = 0
    behavior: str = ""          # Human-readable description of what this object does
    click_count: int = 0        # Total times clicked across all sessions
    effect_count: int = 0       # Times clicking caused any grid change
    level_up_count: int = 0     # Times clicking caused a level-up
    avg_cells_changed: float = 0.0
    affected_region: str = ""   # Typical region description: "r17-45 c0-36"
    cyclic: Optional[bool] = None  # Does repeated clicking cycle through states?
    active: Optional[bool] = None  # Is this currently interactive?
    notes: str = ""
    last_seen_level: int = 0
    first_discovered: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class EffectRecord:
    """Observed effect of clicking a specific position."""
    position_key: str          # "r{r}c{c}"
    r: int
    c: int
    total_clicks: int = 0
    total_changes: int = 0     # sum of all cells changed across all clicks
    max_change: int = 0        # single largest change observed
    zero_change_count: int = 0 # how many clicks had zero effect
    level_up_count: int = 0
    affected_region: str = ""
    notes: str = ""
    last_updated: float = field(default_factory=time.time)

    @property
    def effectiveness(self) -> float:
        """Fraction of clicks that caused any change."""
        return self.total_changes / max(self.total_clicks, 1)

    @property
    def avg_change(self) -> float:
        return self.total_changes / max(self.total_clicks - self.zero_change_count, 1)


@dataclass
class FailedApproach:
    """An explicitly documented approach that didn't work."""
    approach: str           # What was tried
    click_positions: List   # Positions attempted
    total_clicks: int       # How many times
    result: str             # What happened (or didn't)
    inference: str          # What this tells us
    sessions: int           # Across how many sessions
    timestamp: float = field(default_factory=time.time)


@dataclass
class LevelSolution:
    """Known solution for a specific level."""
    level: int
    sequence: List[Dict]    # [{"r": r, "c": c, "repeats": n, "notes": ""}, ...]
    preconditions: str      # What state is needed before this works
    confidence: float       # 0-1, how confident we are this works
    successes: int          # Times this solution succeeded
    attempts: int           # Times attempted
    notes: str = ""
    discovered_session: float = field(default_factory=time.time)


class GameKnowledgeBase:
    """
    Persistent accumulated lived experience for one game family.

    Loaded at session start, updated throughout, saved at session end.
    This is the cartridge that grows smarter each session.

    Usage:
        kb = GameKnowledgeBase("lp85")
        kb.load()

        # During play:
        kb.record_click_effect(r=32, c=4, cells_changed=1172, level_up=False)
        kb.mark_failed("clicked gray row-1 buttons", [(1,11),(1,16)], 12, "0 changes",
                       "decorative labels, not interactive")
        kb.record_level_solution(1, [{"r":32,"c":4,"repeats":4}], confidence=0.9)

        # For prompts:
        text = kb.to_prompt_text()

        kb.save()
    """

    CARTRIDGE_DIR = Path("arc-agi-3/experiments/cartridges")

    def __init__(self, game_family: str):
        self.game_family = game_family.split("-")[0]  # strip variant suffix
        self.objects: Dict[str, ObjectRecord] = {}
        self.effects: Dict[str, EffectRecord] = {}
        self.failed_approaches: List[FailedApproach] = []
        self.level_solutions: Dict[int, LevelSolution] = {}
        self.mechanics: List[str] = []          # High-confidence game rules
        self.open_questions: List[str] = []     # Unexplained phenomena
        self.strategic_insights: List[str] = [] # External facts (from source analysis etc.)
        self.session_count: int = 0
        self.best_level: int = 0
        self.understanding_confidence: float = 0.0
        self.last_updated: float = time.time()

    # ─── Persistence ──────────────────────────────────────────

    def _path(self) -> Path:
        self.CARTRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        return self.CARTRIDGE_DIR / f"{self.game_family}.knowledge.json"

    def load(self) -> bool:
        """Load from disk. Returns True if file existed.

        On first load (no .knowledge.json), automatically bootstraps from any
        prior session logs (v3/v4) to seed the KB with observed click effects.
        """
        path = self._path()
        if not path.exists():
            log.info(f"GameKnowledgeBase: no prior knowledge for {self.game_family} — bootstrapping from logs")
            self._bootstrap_from_logs()
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.objects = {k: ObjectRecord(**v) for k, v in data.get("objects", {}).items()}
            self.effects = {k: EffectRecord(**v) for k, v in data.get("effects", {}).items()}
            self.failed_approaches = [FailedApproach(**fa) for fa in data.get("failed_approaches", [])]
            self.level_solutions = {
                int(k): LevelSolution(**v)
                for k, v in data.get("level_solutions", {}).items()
            }
            self.mechanics = data.get("mechanics", [])
            self.open_questions = data.get("open_questions", [])
            self.strategic_insights = data.get("strategic_insights", [])
            self.session_count = data.get("session_count", 0)
            self.best_level = data.get("best_level", 0)
            self.understanding_confidence = data.get("understanding_confidence", 0.0)
            log.info(f"GameKnowledgeBase loaded: {len(self.objects)} objects, "
                     f"{len(self.effects)} effects, {len(self.level_solutions)} solutions, "
                     f"{len(self.failed_approaches)} failures")
            return True
        except Exception as e:
            log.warning(f"GameKnowledgeBase load failed: {e}")
            return False

    def _bootstrap_from_logs(self):
        """Seed KB from prior session logs (v3/v4) when first created.

        Aggregates click effects across all past sessions so we don't start blind.
        """
        import glob
        log_dir = self.CARTRIDGE_DIR.parent / "logs"
        pattern = str(log_dir / f"{self.game_family}*.json")
        logs = sorted(glob.glob(pattern))
        if not logs:
            return

        pos_data: dict = {}  # (r,c) -> {total, changed_total, level_ups, color}
        n_logs = 0
        for log_path in logs:
            try:
                with open(log_path) as f:
                    data = json.load(f)
                clicks = data.get("session_clicks", [])
                if not clicks:
                    continue
                n_logs += 1
                for click in clicks:
                    r, c = click.get("r"), click.get("c")
                    if r is None or c is None:
                        continue
                    key = (r, c)
                    if key not in pos_data:
                        pos_data[key] = {
                            "total": 0, "changed_total": 0,
                            "level_ups": 0, "color": click.get("color", "unknown")
                        }
                    pos_data[key]["total"] += 1
                    changed = click.get("changed", 0)
                    pos_data[key]["changed_total"] += changed
                    if click.get("level_up"):
                        pos_data[key]["level_ups"] += 1
            except Exception:
                continue

        if not pos_data:
            return

        log.info(f"Bootstrapping KB from {n_logs} session logs, {len(pos_data)} positions")
        for (r, c), d in pos_data.items():
            avg_change = d["changed_total"] / max(d["total"], 1)
            self.record_click_effect(
                r=r, c=c,
                color=d["color"],
                cells_changed=int(avg_change),
                level_up=d["level_ups"] > 0,
                level=0,
            )
            # Correct the effect counts
            obj_key = f"r{r}c{c}"
            if obj_key in self.effects:
                eff = self.effects[obj_key]
                eff.total_clicks = d["total"]
                eff.total_changes = d["changed_total"]
                eff.level_up_count = d["level_ups"]
                eff.zero_change_count = sum(
                    1 for _ in range(d["total"]) if avg_change == 0
                )
            if obj_key in self.objects:
                obj = self.objects[obj_key]
                obj.click_count = d["total"]
                obj.effect_count = 1 if d["changed_total"] > 0 else 0
                obj.level_up_count = d["level_ups"]
                if d["level_ups"] > 0:
                    obj.obj_type = "trigger"
                elif avg_change > 50:
                    obj.obj_type = "button"
                elif d["total"] >= 3 and d["changed_total"] == 0:
                    obj.obj_type = "decoration"

        log.info(f"Bootstrap complete: {len(self.objects)} objects pre-populated")

    def save(self):
        """Save to disk."""
        path = self._path()
        try:
            data = {
                "game_family": self.game_family,
                "session_count": self.session_count,
                "best_level": self.best_level,
                "understanding_confidence": self.understanding_confidence,
                "last_updated": time.time(),
                "objects": {k: asdict(v) for k, v in self.objects.items()},
                "effects": {k: asdict(v) for k, v in self.effects.items()},
                "failed_approaches": [asdict(fa) for fa in self.failed_approaches],
                "level_solutions": {str(k): asdict(v) for k, v in self.level_solutions.items()},
                "mechanics": self.mechanics,
                "open_questions": self.open_questions,
                "strategic_insights": self.strategic_insights,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            log.info(f"GameKnowledgeBase saved to {path}")
        except Exception as e:
            log.warning(f"GameKnowledgeBase save failed: {e}")

    # ─── Recording ────────────────────────────────────────────

    def record_click_effect(self, r: int, c: int, color: str, cells_changed: int,
                            level_up: bool, affected_region: str = "",
                            level: int = 0):
        """Record what happened when we clicked position (r, c)."""
        key = f"r{r}c{c}"
        if key not in self.effects:
            self.effects[key] = EffectRecord(
                position_key=key, r=r, c=c,
            )
        eff = self.effects[key]
        eff.total_clicks += 1
        eff.total_changes += cells_changed
        if cells_changed == 0:
            eff.zero_change_count += 1
        if cells_changed > eff.max_change:
            eff.max_change = cells_changed
        if level_up:
            eff.level_up_count += 1
        if affected_region and not eff.affected_region:
            eff.affected_region = affected_region
        eff.last_updated = time.time()

        # Update or create object record
        if key not in self.objects:
            self.objects[key] = ObjectRecord(
                obj_id=key,
                obj_type="unknown",
                position={"r": r, "c": c},
                color=color,
            )
        obj = self.objects[key]
        obj.click_count += 1
        if cells_changed > 0:
            obj.effect_count += 1
            obj.avg_cells_changed = eff.total_changes / max(eff.total_clicks - eff.zero_change_count, 1)
            obj.affected_region = affected_region
        if level_up:
            obj.level_up_count += 1
            obj.obj_type = "trigger"
        elif cells_changed > 50 and obj.obj_type == "unknown":
            obj.obj_type = "button"
        elif cells_changed == 0 and obj.click_count >= 3 and obj.effect_count == 0:
            obj.obj_type = "decoration"
            obj.active = False
        obj.last_seen_level = level
        obj.last_updated = time.time()

    def record_object_understanding(self, r: int, c: int, obj_type: str,
                                    behavior: str, notes: str = ""):
        """LLM-driven: update our understanding of what an object is."""
        key = f"r{r}c{c}"
        if key not in self.objects:
            self.objects[key] = ObjectRecord(
                obj_id=key, obj_type=obj_type,
                position={"r": r, "c": c}, color="unknown",
            )
        obj = self.objects[key]
        obj.obj_type = obj_type
        obj.behavior = behavior
        if notes:
            obj.notes = notes
        obj.last_updated = time.time()

    def mark_failed(self, approach: str, positions: list, total_clicks: int,
                    result: str, inference: str, sessions: int = 1):
        """Record an approach that provably doesn't work."""
        # Check if we already have this failure
        for fa in self.failed_approaches:
            if fa.approach == approach:
                fa.total_clicks += total_clicks
                fa.sessions = max(fa.sessions, sessions)
                return
        self.failed_approaches.append(FailedApproach(
            approach=approach,
            click_positions=[list(p) for p in positions],
            total_clicks=total_clicks,
            result=result,
            inference=inference,
            sessions=sessions,
        ))

    def record_level_solution(self, level: int, sequence: list,
                              preconditions: str = "", confidence: float = 0.9,
                              notes: str = ""):
        """Record that we solved a level with a specific sequence."""
        if level not in self.level_solutions:
            self.level_solutions[level] = LevelSolution(
                level=level, sequence=sequence,
                preconditions=preconditions,
                confidence=confidence,
                successes=1, attempts=1, notes=notes,
            )
        else:
            sol = self.level_solutions[level]
            sol.attempts += 1
            sol.successes += 1
            sol.confidence = min(0.99, sol.confidence + 0.05)

        if level > self.best_level:
            self.best_level = level

    def add_mechanic(self, mechanic: str):
        """Record a high-confidence game rule."""
        if mechanic not in self.mechanics:
            self.mechanics.append(mechanic)

    def add_question(self, question: str):
        """Record something we don't yet understand."""
        if question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_question(self, question_fragment: str, answer: str):
        """Mark an open question as answered, replace with mechanic."""
        self.open_questions = [q for q in self.open_questions
                               if question_fragment.lower() not in q.lower()]
        self.add_mechanic(answer)

    # ─── Prompt text ──────────────────────────────────────────

    def to_prompt_text(self, current_level: int = 0) -> str:
        """Format accumulated knowledge as LLM-consumable text."""
        lines = [f"GAME KNOWLEDGE BASE (session {self.session_count + 1}, "
                 f"best level reached: {self.best_level})"]

        # Known solutions — most actionable
        if self.level_solutions:
            lines.append("\nKNOWN LEVEL SOLUTIONS:")
            for lvl, sol in sorted(self.level_solutions.items()):
                seq_text = ", ".join(
                    f"click @({s.get('c','?')},{s.get('r','?')}) ×{s.get('repeats',1)}"
                    for s in sol.sequence
                )
                marker = " ← CURRENT LEVEL" if lvl == current_level + 1 else ""
                lines.append(f"  Level {lvl}: {seq_text} (confidence: {sol.confidence:.0%}){marker}")

        # Active objects (interactive, confirmed behavior)
        active_objs = [o for o in self.objects.values()
                       if o.obj_type not in ("decoration",) and o.effect_count > 0]
        if active_objs:
            lines.append("\nCONFIRMED INTERACTIVE OBJECTS:")
            for obj in sorted(active_objs, key=lambda o: -o.level_up_count * 100 - o.effect_count):
                eff_rate = obj.effect_count / max(obj.click_count, 1)
                lvl_marker = f" ★{obj.level_up_count} level-ups" if obj.level_up_count > 0 else ""
                lines.append(
                    f"  {obj.obj_type} {obj.color} @({obj.position['c']},{obj.position['r']}) "
                    f"[{eff_rate:.0%} effective, avg {obj.avg_cells_changed:.0f} cells]{lvl_marker}"
                )
                if obj.behavior:
                    lines.append(f"    → {obj.behavior}")
                if obj.affected_region:
                    lines.append(f"    → affects {obj.affected_region}")

        # Confirmed decorations / dead ends
        dead = [o for o in self.objects.values()
                if o.obj_type == "decoration" or (o.click_count >= 3 and o.effect_count == 0)]
        if dead:
            lines.append("\nCONFIRMED NON-INTERACTIVE (do not click):")
            for obj in dead[:8]:
                lines.append(
                    f"  {obj.color} @({obj.position['c']},{obj.position['r']}) "
                    f"[{obj.click_count} clicks, 0 effect] — {obj.notes or 'no changes ever observed'}"
                )

        # Known failures
        if self.failed_approaches:
            lines.append("\nFAILED APPROACHES (do not repeat):")
            for fa in self.failed_approaches[:5]:
                lines.append(f"  • {fa.approach}")
                lines.append(f"    Result: {fa.result} ({fa.total_clicks} clicks, {fa.sessions} sessions)")
                lines.append(f"    Inference: {fa.inference}")

        # Game mechanics
        if self.mechanics:
            lines.append("\nKNOWN GAME MECHANICS:")
            for m in self.mechanics:
                lines.append(f"  ✓ {m}")

        # Open questions — drive exploration
        untested = [o for o in self.objects.values() if o.click_count == 0]
        if self.open_questions or untested:
            lines.append("\nOPEN QUESTIONS (what to explore):")
            for q in self.open_questions[:4]:
                lines.append(f"  ? {q}")
            if untested:
                lines.append(f"  ? {len(untested)} objects identified but never clicked")

        # Strategic insights from source analysis
        if self.strategic_insights:
            lines.append("\nVERIFIED FACTS (from game source analysis):")
            for s in self.strategic_insights[:4]:
                lines.append(f"  !! {s}")

        return "\n".join(lines)

    def to_exploration_targets(self) -> list:
        """
        Return positions to prioritize, sorted by exploration value.

        Priority:
        1. Objects with level_up history (high reward)
        2. Unknown objects (never clicked)
        3. Objects with some effectiveness but no full understanding
        4. Skip confirmed decorations
        """
        candidates = []
        for key, obj in self.objects.items():
            if obj.obj_type == "decoration":
                continue
            score = 0
            if obj.level_up_count > 0:
                score += 100 * obj.level_up_count
            if obj.click_count == 0:
                score += 50  # unexplored
            elif obj.effect_count > 0:
                score += 20 * (obj.effect_count / obj.click_count)
            candidates.append((score, obj.position["r"], obj.position["c"]))
        return [(r, c) for _, r, c in sorted(candidates, reverse=True)]

    def get_level_solution(self, level: int) -> Optional[LevelSolution]:
        """Return known solution for this level, if any."""
        return self.level_solutions.get(level)

    @property
    def stats(self) -> dict:
        return {
            "session_count": self.session_count,
            "best_level": self.best_level,
            "n_objects": len(self.objects),
            "n_effects": len(self.effects),
            "n_solutions": len(self.level_solutions),
            "n_failures": len(self.failed_approaches),
            "n_mechanics": len(self.mechanics),
            "confidence": self.understanding_confidence,
        }
