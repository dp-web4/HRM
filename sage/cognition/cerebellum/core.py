"""
Cerebellum core — habit detection, compilation, and caching.

The cerebellum is a fast-path cache for the consciousness loop.
It does NOT reason — it recognizes known patterns and replays
proven action sequences without deliberation.

Integration points in the 12-step loop:
  Step 5 (Select):  router calls lookup() to check for matching habits
  Step 7 (Execute): if router chose a habit, execute() replays it
  Step 8 (Learn):   observe() records (state, actions, outcome) triples
  Step 9 (Remember): link_episode() connects habits to Thor's episodic index
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StateSignature:
    """Compact representation of a state for habit matching.

    Each domain provides its own feature extractor. The cerebellum
    doesn't know what features mean — it just checks overlap.
    """
    domain: str
    features: dict
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            # Deterministic hash from domain + sorted features
            canonical = json.dumps(
                {"domain": self.domain, "features": self.features},
                sort_keys=True, default=str
            )
            self.hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_wm(cls, wm, domain: str, goal_id: Optional[str] = None) -> 'StateSignature':
        """Build a state signature from WorkingMemory contents.

        Uses WM's stable_key() for the hash and extracts slot types/content
        as features. This is the primary integration point with CBP's WM spec.

        Args:
            wm: WorkingMemory instance (must have stable_key() and get_by_type())
            domain: Domain string (e.g., "arc-game:cd82", "conversation")
            goal_id: Optional goal to scope the signature to
        """
        features = {}
        wm_hash = wm.stable_key(goal_id)

        # Extract structural features from WM slots
        try:
            for slot_type in ("goal", "plan_step", "binding", "hypothesis", "constraint"):
                slots = wm.get_by_type(slot_type, goal_id=goal_id)
                if slots:
                    features[f"wm_{slot_type}_count"] = len(slots)
                    # Use first slot's content for identity (not all — too specific)
                    if hasattr(slots[0], "content"):
                        features[f"wm_{slot_type}_key"] = str(slots[0].content)[:64]
        except (AttributeError, TypeError):
            pass

        return cls(domain=domain, features=features, hash=wm_hash)

    def similarity(self, other: 'StateSignature') -> float:
        """Feature-overlap similarity. Simple and fast."""
        if self.domain != other.domain:
            return 0.0
        shared_keys = set(self.features) & set(other.features)
        if not shared_keys:
            return 0.0
        matches = sum(
            1 for k in shared_keys
            if self.features[k] == other.features[k]
        )
        return matches / len(shared_keys)


@dataclass
class Habit:
    """A compiled action sequence that produced a good outcome."""
    habit_id: str
    state_sig: StateSignature
    action_sequence: list  # [{plugin, args, effector}, ...]
    outcome_summary: str

    # Trust metrics (T3)
    training_count: int = 0
    success_count: int = 0
    last_fired: float = 0.0

    # Provenance
    source_episodes: list = field(default_factory=list)
    compile_time: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        if self.training_count == 0:
            return 0.0
        return self.success_count / self.training_count

    @property
    def is_mature(self) -> bool:
        """Mature = observed 3+ times with >80% reliability."""
        return self.training_count >= 3 and self.reliability > 0.8

    def to_dict(self) -> dict:
        return {
            "habit_id": self.habit_id,
            "state_sig": {
                "domain": self.state_sig.domain,
                "features": self.state_sig.features,
                "hash": self.state_sig.hash,
            },
            "action_sequence": self.action_sequence,
            "outcome_summary": self.outcome_summary,
            "training_count": self.training_count,
            "success_count": self.success_count,
            "last_fired": self.last_fired,
            "source_episodes": self.source_episodes,
            "compile_time": self.compile_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Habit':
        sig_data = d["state_sig"]
        return cls(
            habit_id=d["habit_id"],
            state_sig=StateSignature(
                domain=sig_data["domain"],
                features=sig_data["features"],
                hash=sig_data.get("hash", ""),
            ),
            action_sequence=d["action_sequence"],
            outcome_summary=d["outcome_summary"],
            training_count=d.get("training_count", 0),
            success_count=d.get("success_count", 0),
            last_fired=d.get("last_fired", 0.0),
            source_episodes=d.get("source_episodes", []),
            compile_time=d.get("compile_time", 0.0),
        )


@dataclass
class HabitMatch:
    """Result of checking whether a habit applies to current state."""
    habit: Habit
    similarity: float
    confidence: float  # reliability * similarity
    estimated_atp: float = 1.0  # habit execution is cheap


class Cerebellum:
    """Habit compiler and cache for the SAGE consciousness loop.

    Usage:
        cb = Cerebellum()

        # During loop step 5 (Select):
        matches = cb.lookup(current_state_sig)

        # During loop step 7 (Execute):
        if matches:
            success, result = cb.execute(matches[0].habit, env)

        # During loop step 8 (Learn):
        cb.observe(state_sig, actions, outcome)
    """

    def __init__(
        self,
        max_habits: int = 256,
        similarity_threshold: float = 0.85,
        maturity_threshold: int = 3,
        consensus_threshold: Optional[float] = None,
    ):
        if consensus_threshold is not None and not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0.0, 1.0] or None")
        self.max_habits = max_habits
        self.similarity_threshold = similarity_threshold
        self.maturity_threshold = maturity_threshold
        self.consensus_threshold = consensus_threshold
        self._habits: dict[str, Habit] = {}  # habit_id → Habit
        self._last_compiled: Optional[Habit] = None

    @property
    def last_compiled(self) -> Optional[Habit]:
        """The most recently compiled or updated habit."""
        return self._last_compiled

    # === Step 5: Select — called by router ===

    def lookup(self, state: StateSignature) -> list[HabitMatch]:
        """Check for cached habits matching the current state.

        Returns ranked matches above similarity_threshold.
        Only mature habits are returned.
        """
        matches = []
        for habit in self._habits.values():
            if not habit.is_mature:
                continue
            sim = state.similarity(habit.state_sig)
            if sim >= self.similarity_threshold:
                matches.append(HabitMatch(
                    habit=habit,
                    similarity=sim,
                    confidence=habit.reliability * sim,
                    estimated_atp=len(habit.action_sequence) * 0.1,
                ))
        matches.sort(key=lambda m: -m.confidence)
        return matches

    # === Step 7: Execute ===

    def execute(self, habit: Habit, env: Any) -> tuple[bool, Any]:
        """Execute a cached habit sequence.

        The env parameter is domain-specific — for ARC games it's the
        arcade environment, for conversations it's the message context.

        Returns (success, result). Updates habit trust metrics.
        """
        habit.last_fired = time.time()
        habit.training_count += 1

        try:
            result = self._replay_actions(habit.action_sequence, env)
            success = result.get("success", False) if isinstance(result, dict) else bool(result)
        except Exception as e:
            # Habit failed — don't increment success
            return False, {"error": str(e)}

        if success:
            habit.success_count += 1

        return success, result

    def _replay_actions(self, actions: list, env: Any) -> Any:
        """Replay an action sequence on the environment.

        Override this in domain-specific subclasses for real execution.
        Base implementation returns the action list (for testing).
        """
        # Domain-specific subclasses override this
        return {"success": True, "actions_replayed": len(actions)}

    # === Step 8: Learn ===

    def observe(
        self,
        state: StateSignature,
        action_sequence: list,
        outcome: dict,
    ) -> Optional[Habit]:
        """Observe a (state, actions, outcome) triple.

        If an existing habit matches this state, updates its counts.
        Otherwise creates a new candidate habit.
        """
        success = outcome.get("success", False)

        # Check for existing habit with matching state
        best_match = None
        best_sim = 0.0
        for habit in self._habits.values():
            sim = state.similarity(habit.state_sig)
            if sim > best_sim and sim >= self.similarity_threshold:
                # Also check action sequence similarity
                if self._actions_match(habit.action_sequence, action_sequence):
                    best_match = habit
                    best_sim = sim

        if best_match:
            best_match.training_count += 1
            if success:
                best_match.success_count += 1
            self._last_compiled = best_match
            return best_match

        # New candidate habit
        habit_id = f"h-{state.domain}-{state.hash}-{int(time.time())}"
        habit = Habit(
            habit_id=habit_id,
            state_sig=state,
            action_sequence=action_sequence,
            outcome_summary=outcome.get("summary", str(outcome)),
            training_count=1,
            success_count=1 if success else 0,
        )

        self._add_habit(habit)
        self._last_compiled = habit
        return habit

    def _actions_match(self, seq_a: list, seq_b: list) -> bool:
        """Check if two action sequences are structurally similar."""
        if len(seq_a) != len(seq_b):
            return False
        # Simple: same length and same action types
        for a, b in zip(seq_a, seq_b):
            if isinstance(a, dict) and isinstance(b, dict):
                if a.get("plugin") != b.get("plugin"):
                    return False
                if a.get("action") != b.get("action"):
                    return False
            elif a != b:
                return False
        return True

    def _add_habit(self, habit: Habit):
        """Add habit, evicting lowest-reliability if at capacity."""
        if len(self._habits) >= self.max_habits:
            # Evict lowest reliability
            worst = min(
                self._habits.values(),
                key=lambda h: (h.reliability, h.last_fired)
            )
            del self._habits[worst.habit_id]

        self._habits[habit.habit_id] = habit

    # === Step 9: Remember (Thor coordination) ===

    def link_episode(self, habit_id: str, episode_id: str):
        """Link a habit to an episodic memory entry."""
        if habit_id in self._habits:
            if episode_id not in self._habits[habit_id].source_episodes:
                self._habits[habit_id].source_episodes.append(episode_id)

    # === Batch compilation (overnight/consolidation) ===

    def compile_from_episodes(self, episodes: list[dict]) -> list[Habit]:
        """Compile habits from episodic memory.

        Scans episodes for repeated (state → action_seq → outcome)
        patterns. Called during consolidation (sleep phase).

        Each episode dict should have:
          state: StateSignature or dict with domain+features
          actions: list of action dicts
          outcome: dict with 'success' key
        """
        # Group by state hash
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)

        for ep in episodes:
            state = ep.get("state")
            if isinstance(state, dict):
                state = StateSignature(
                    domain=state["domain"],
                    features=state["features"],
                )
            if state is None:
                continue
            groups[state.hash].append(ep)

        compiled = []
        for state_hash, group in groups.items():
            if len(group) < self.maturity_threshold:
                continue

            # Check success rate
            successes = sum(
                1 for ep in group
                if ep.get("outcome", {}).get("success", False)
            )
            if successes / len(group) < 0.8:
                continue

            # Use the most common action sequence
            action_seqs = [
                json.dumps(ep.get("actions", []), sort_keys=True, default=str)
                for ep in group
            ]
            from collections import Counter
            most_common_seq, consensus_count = Counter(action_seqs).most_common(1)[0]
            consensus_ratio = consensus_count / len(group)

            # Consensus gate: skip when the dominant arc is below the floor.
            # A state seeing maturity_threshold episodes may still have no
            # clear preferred action sequence; without this gate, the
            # cerebellum would compile a plurality-winner habit that isn't
            # actually representative of agent behavior at that state.
            if (
                self.consensus_threshold is not None
                and consensus_ratio < self.consensus_threshold
            ):
                continue

            ref_ep = group[0]
            state = ref_ep.get("state")
            if isinstance(state, dict):
                state = StateSignature(
                    domain=state["domain"],
                    features=state["features"],
                )

            habit = Habit(
                habit_id=f"h-batch-{state.hash}-{int(time.time())}",
                state_sig=state,
                action_sequence=json.loads(most_common_seq),
                outcome_summary=(
                    f"Compiled from {len(group)} episodes "
                    f"({successes} successes, "
                    f"consensus {consensus_count}/{len(group)})"
                ),
                training_count=len(group),
                success_count=successes,
                source_episodes=[
                    ep.get("episode_id", "") for ep in group if ep.get("episode_id")
                ],
            )
            self._add_habit(habit)
            compiled.append(habit)

        return compiled

    # === Maintenance ===

    def prune(
        self,
        min_reliability: float = 0.3,
        max_age_seconds: float = 86400 * 30,
    ):
        """Remove unreliable or stale habits. Called during consolidation."""
        now = time.time()
        to_remove = []
        for hid, habit in self._habits.items():
            if habit.training_count >= 3 and habit.reliability < min_reliability:
                to_remove.append(hid)
            elif habit.last_fired > 0 and (now - habit.last_fired) > max_age_seconds:
                to_remove.append(hid)
        for hid in to_remove:
            del self._habits[hid]
        return len(to_remove)

    # === Persistence ===

    def export(self) -> dict:
        return {
            "version": 1,
            "max_habits": self.max_habits,
            "similarity_threshold": self.similarity_threshold,
            "habits": [h.to_dict() for h in self._habits.values()],
        }

    @classmethod
    def load(cls, data: dict) -> 'Cerebellum':
        cb = cls(
            max_habits=data.get("max_habits", 256),
            similarity_threshold=data.get("similarity_threshold", 0.85),
        )
        for hd in data.get("habits", []):
            habit = Habit.from_dict(hd)
            cb._habits[habit.habit_id] = habit
        return cb

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.export(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> 'Cerebellum':
        with open(path) as f:
            return cls.load(json.load(f))

    # === Stats ===

    @property
    def habit_count(self) -> int:
        return len(self._habits)

    @property
    def mature_count(self) -> int:
        return sum(1 for h in self._habits.values() if h.is_mature)

    def stats(self) -> dict:
        return {
            "total_habits": self.habit_count,
            "mature_habits": self.mature_count,
            "avg_reliability": (
                sum(h.reliability for h in self._habits.values()) / max(1, self.habit_count)
            ),
            "domains": list(set(h.state_sig.domain for h in self._habits.values())),
        }
