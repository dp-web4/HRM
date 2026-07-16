#!/usr/bin/env python3
"""
Working Memory for SAGE — dlPFC-analog scratchpad
==================================================

Typed, capacity-limited (Miller 4±3), ttl-decaying slot buffer.
The interface every brain-arch component reads/writes through.

Spec: phase2/brain-arch/working-memory.md
"""

import json
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable

# Torch is optional — used only when callers hand us tensor content.
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────
# Registered slot types
# ──────────────────────────────────────────────────────────────────────

DEFAULT_TYPES = {
    "goal": None,
    "plan_step": None,
    "intermediate_result": None,
    "binding": None,
    "hypothesis": None,
    "constraint": None,
    "context_handle": None,
    "other": None,
}


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WorkingMemorySlot:
    """Single active item. Displaceable, decays without refresh."""
    slot_id: str
    content_type: str
    content: Any
    priority: float
    timestamp: float
    goal_id: Optional[str] = None
    access_count: int = 0
    ttl_ticks: Optional[int] = None  # loop ticks remaining; None = no expiry

    def __post_init__(self):
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be 0-1, got {self.priority}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "content_type": self.content_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "goal_id": self.goal_id,
            "access_count": self.access_count,
            "ttl_ticks": self.ttl_ticks,
        }


@dataclass
class PlanStep:
    """Step in a multi-step plan."""
    step_id: int
    action: str
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    status: str = "pending"  # pending | active | complete | failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SensorGoalBinding:
    """Binding between sensor observation and goal."""
    sensor_id: str
    observation: Any
    goal_id: str
    relevance: float
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"relevance must be 0-1, got {self.relevance}")


@dataclass
class Event:
    """Mutation event — observable by metacog (Nomad)."""
    kind: str                    # write | update | evict | clear | tick | consolidate | warn
    slot_id: Optional[str]
    slot_type: Optional[str]
    timestamp: float
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# WorkingMemory
# ──────────────────────────────────────────────────────────────────────

class WorkingMemory:
    """Typed, capacity-limited, ttl-decaying scratchpad.

    Invariants:
      1. Capacity is a hard limit (default 7, Miller 4±3 upper).
      2. Slot types are registered; unknown types become `other` with a warn event.
      3. Payloads are JSON-serializable. Enforced at write.
      4. All mutations emit Events. Reads don't.
      5. Eviction: lowest (priority*(1-rw) + recency*rw).
      6. TTL expiry happens on tick(), not on read.
    """

    _TYPE_REGISTRY: Dict[str, Optional[Dict]] = dict(DEFAULT_TYPES)

    def __init__(
        self,
        capacity: int = 7,
        memory_retrieval: Optional[Any] = None,
        on_event: Optional[Callable[[Event], None]] = None,
        recency_weight: float = 0.3,
    ):
        """
        Args:
            capacity: Max active slots. Miller 4±3 → default 7.
            memory_retrieval: Optional long-term memory handle (Track 2).
            on_event: Optional callback, fired for every mutation.
            recency_weight: Blend between priority and age in eviction score.
        """
        self.capacity = capacity
        self.memory = memory_retrieval
        self.on_event = on_event
        self.recency_weight = recency_weight

        self.slots: Dict[str, WorkingMemorySlot] = {}
        self.active_plan: Optional[List[PlanStep]] = None
        self.current_step_index: Optional[int] = None
        self.bindings: List[SensorGoalBinding] = []

        # Stats
        self.total_adds: int = 0
        self.total_evictions: int = 0
        self.total_accesses: int = 0
        self.total_ticks: int = 0
        self.total_ttl_expiries: int = 0

    # ── type registry ──────────────────────────────────────────────

    @classmethod
    def register_type(cls, name: str, schema: Optional[Dict] = None) -> None:
        """Register a slot content_type. Schema is advisory for v0.1."""
        cls._TYPE_REGISTRY[name] = schema

    @classmethod
    def known_types(cls) -> List[str]:
        return sorted(cls._TYPE_REGISTRY.keys())

    # ── writes ──────────────────────────────────────────────────────

    def add_item(
        self,
        content_type: str,
        content: Any,
        priority: float,
        goal_id: Optional[str] = None,
        ttl_ticks: Optional[int] = None,
    ) -> str:
        """Add a slot. Evicts lowest-score item if at capacity."""
        # JSON-serializable guard
        try:
            json.dumps(content, default=self._default_serializer)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"content must be JSON-serializable (got {type(content).__name__}): {e}"
            )

        # Type registry
        if content_type not in self._TYPE_REGISTRY:
            self._emit(Event(
                kind="warn", slot_id=None, slot_type=content_type,
                timestamp=time.time(),
                reason=f"unknown content_type '{content_type}' → coerced to 'other'",
            ))
            content_type = "other"

        # Evict if full
        if len(self.slots) >= self.capacity:
            self._evict_lowest_priority()

        slot_id = f"wm_{uuid.uuid4().hex[:8]}"
        slot = WorkingMemorySlot(
            slot_id=slot_id,
            content_type=content_type,
            content=content,
            priority=priority,
            timestamp=time.time(),
            goal_id=goal_id,
            ttl_ticks=ttl_ticks,
        )
        self.slots[slot_id] = slot
        self.total_adds += 1
        self._emit(Event(kind="write", slot_id=slot_id,
                         slot_type=content_type, timestamp=slot.timestamp))
        return slot_id

    def update(self, slot_id: str, content: Any) -> bool:
        """Replace payload without changing slot_id or type."""
        slot = self.slots.get(slot_id)
        if slot is None:
            return False
        # JSON-serializable guard
        try:
            json.dumps(content, default=self._default_serializer)
        except (TypeError, ValueError) as e:
            raise ValueError(f"content not JSON-serializable: {e}")
        slot.content = content
        slot.timestamp = time.time()
        self._emit(Event(kind="update", slot_id=slot_id,
                         slot_type=slot.content_type, timestamp=slot.timestamp))
        return True

    def refresh(self, slot_id: str) -> bool:
        """Reset timestamp; re-arm ttl if set. No event (treated as read-like)."""
        slot = self.slots.get(slot_id)
        if slot is None:
            return False
        slot.timestamp = time.time()
        return True

    # ── reads (no events) ──────────────────────────────────────────

    def get_item(self, slot_id: str) -> Optional[WorkingMemorySlot]:
        slot = self.slots.get(slot_id)
        if slot is not None:
            slot.access_count += 1
            self.total_accesses += 1
        return slot

    def get_by_type(
        self, content_type: str, goal_id: Optional[str] = None
    ) -> List[WorkingMemorySlot]:
        out = [s for s in self.slots.values() if s.content_type == content_type]
        if goal_id is not None:
            out = [s for s in out if s.goal_id == goal_id]
        return out

    def get_context(self, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """Grouped view of current slots. Intended for router input."""
        if goal_id is not None:
            relevant = [s for s in self.slots.values() if s.goal_id == goal_id]
        else:
            relevant = list(self.slots.values())

        grouped: Dict[str, List[Any]] = {t: [] for t in self._TYPE_REGISTRY}
        for s in relevant:
            grouped.setdefault(s.content_type, []).append(s.content)

        grouped["active_plan"] = self.active_plan
        grouped["current_step"] = self.current_step_index
        grouped["sensor_bindings"] = (
            [b for b in self.bindings if b.goal_id == goal_id]
            if goal_id else list(self.bindings)
        )
        return grouped

    # ── plan flow (convenience layered on plan_step slots) ─────────

    def load_plan(self, plan: List[PlanStep]) -> None:
        self.active_plan = plan
        self.current_step_index = 0
        for i, step in enumerate(plan):
            self.add_item(
                content_type="plan_step",
                content=step.to_dict(),
                priority=0.9 if i == 0 else 0.7,
            )

    def advance_plan(self) -> Optional[PlanStep]:
        if not self.active_plan or self.current_step_index is None:
            return None
        if self.current_step_index < len(self.active_plan):
            self.active_plan[self.current_step_index].status = "complete"
        self.current_step_index += 1
        if self.current_step_index >= len(self.active_plan):
            return None
        next_step = self.active_plan[self.current_step_index]
        next_step.status = "active"
        return next_step

    def get_current_plan_step(self) -> Optional[PlanStep]:
        if self.active_plan and self.current_step_index is not None:
            if self.current_step_index < len(self.active_plan):
                return self.active_plan[self.current_step_index]
        return None

    # ── sensor bindings ────────────────────────────────────────────

    def bind_sensor_to_goal(
        self, sensor_id: str, observation: Any, goal_id: str, relevance: float
    ) -> None:
        self.bindings.append(SensorGoalBinding(
            sensor_id=sensor_id, observation=observation,
            goal_id=goal_id, relevance=relevance,
        ))
        if len(self.bindings) > 50:
            self.bindings = self.bindings[-50:]

    def get_sensor_bindings(
        self, goal_id: Optional[str] = None
    ) -> List[SensorGoalBinding]:
        if goal_id is not None:
            return [b for b in self.bindings if b.goal_id == goal_id]
        return list(self.bindings)

    # ── lifecycle ──────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance one consciousness-loop tick. Decays ttl, evicts expired."""
        self.total_ticks += 1
        expired: List[str] = []
        for slot_id, slot in self.slots.items():
            if slot.ttl_ticks is not None:
                slot.ttl_ticks -= 1
                if slot.ttl_ticks <= 0:
                    expired.append(slot_id)
        for slot_id in expired:
            slot = self.slots.pop(slot_id)
            self.total_evictions += 1
            self.total_ttl_expiries += 1
            self._emit(Event(
                kind="evict", slot_id=slot_id, slot_type=slot.content_type,
                timestamp=time.time(), reason="ttl_expired",
            ))
        self._emit(Event(
            kind="tick", slot_id=None, slot_type=None,
            timestamp=time.time(),
            reason=f"expired={len(expired)}",
        ))

    def clear(self, goal_id: Optional[str] = None) -> None:
        if goal_id is None:
            n = len(self.slots)
            self.slots.clear()
            self.active_plan = None
            self.current_step_index = None
            self.bindings.clear()
            self._emit(Event(
                kind="clear", slot_id=None, slot_type=None,
                timestamp=time.time(), reason=f"all ({n} slots)",
            ))
        else:
            to_remove = [sid for sid, s in self.slots.items() if s.goal_id == goal_id]
            for sid in to_remove:
                del self.slots[sid]
            self.bindings = [b for b in self.bindings if b.goal_id != goal_id]
            self._emit(Event(
                kind="clear", slot_id=None, slot_type=None,
                timestamp=time.time(),
                reason=f"goal={goal_id} ({len(to_remove)} slots)",
            ))

    def consolidate_to_ltm(self, goal_id: Optional[str] = None,
                           episodic_index=None,
                           session_id: str = '',
                           action_taken: Optional[str] = None,
                           outcome: Optional[str] = None,
                           reward: float = 0.0,
                           success: Optional[bool] = None,
                           snarc_scores: Optional[Dict] = None,
                           tags: Optional[list] = None) -> int:
        """Ship high-priority slots to episodic index.

        When episodic_index is provided (Thor #4), creates an Episode from
        the current WM dump and binds it to the hippocampal index.
        """
        high = [
            s for s in self.slots.values()
            if s.priority >= 0.7 and (goal_id is None or s.goal_id == goal_id)
        ]

        # Bind to episodic index if available
        if episodic_index is not None:
            try:
                from sage.cognition.episodic.data import Episode
                ep = Episode.from_wm_dump(
                    wm_dump=self.dump(),
                    session_id=session_id,
                    action_taken=action_taken,
                    outcome=outcome,
                    reward=reward,
                    success=success,
                    snarc_scores=snarc_scores,
                    tags=tags,
                )
                episodic_index.bind(ep)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Episodic bind failed: {e}")

        self._emit(Event(
            kind="consolidate", slot_id=None, slot_type=None,
            timestamp=time.time(),
            reason=f"count={len(high)} goal={goal_id}",
        ))
        return len(high)

    # ── snapshot / hash ────────────────────────────────────────────

    def dump(self) -> Dict[str, Any]:
        """JSON-serializable snapshot for episodic index + debugging."""
        return {
            "capacity": self.capacity,
            "slots": [s.to_dict() for s in self.slots.values()],
            "active_plan": [p.to_dict() for p in self.active_plan] if self.active_plan else None,
            "current_step_index": self.current_step_index,
            "bindings": [
                {
                    "sensor_id": b.sensor_id,
                    "observation": b.observation,
                    "goal_id": b.goal_id,
                    "relevance": b.relevance,
                    "timestamp": b.timestamp,
                }
                for b in self.bindings
            ],
            "stats": self.get_stats(),
            "snapshot_at": time.time(),
        }

    # `snapshot` is an alias for `dump` — Thor's episodic spec uses this name.
    snapshot = dump

    def stable_key(self, goal_id: Optional[str] = None) -> str:
        """Canonical hash of slot subset for habit-compiler matching (McNugget #3).

        Deterministic across runs: sorts by content_type + handle; ignores
        timestamps, access counts, ttl.
        """
        import hashlib
        items = sorted(self.slots.values(), key=lambda s: (s.content_type, s.slot_id))
        if goal_id is not None:
            items = [s for s in items if s.goal_id == goal_id]
        canonical = json.dumps(
            [{"type": s.content_type, "content": s.content} for s in items],
            sort_keys=True, default=self._default_serializer,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # ── stats / observability ──────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "current_size": len(self.slots),
            "utilization": len(self.slots) / self.capacity if self.capacity else 0.0,
            "total_adds": self.total_adds,
            "total_evictions": self.total_evictions,
            "total_ttl_expiries": self.total_ttl_expiries,
            "total_accesses": self.total_accesses,
            "total_ticks": self.total_ticks,
            "active_plan_steps": len(self.active_plan) if self.active_plan else 0,
            "current_step": self.current_step_index,
            "sensor_bindings": len(self.bindings),
        }

    # ── internals ──────────────────────────────────────────────────

    def _evict_lowest_priority(self) -> None:
        if not self.slots:
            return
        now = time.time()
        scores: Dict[str, float] = {}
        for slot_id, slot in self.slots.items():
            age = now - slot.timestamp
            recency = 1.0 / (1.0 + age)
            scores[slot_id] = (
                slot.priority * (1.0 - self.recency_weight)
                + recency * self.recency_weight
            )
        lowest = min(scores, key=scores.get)
        slot = self.slots.pop(lowest)
        self.total_evictions += 1
        self._emit(Event(
            kind="evict", slot_id=lowest, slot_type=slot.content_type,
            timestamp=now, reason="capacity_exceeded",
        ))

    def _emit(self, event: Event) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception as e:
            # Observer failures must not break WM. Log and continue.
            warnings.warn(f"WM on_event callback raised: {e}")

    @staticmethod
    def _default_serializer(obj: Any) -> Any:
        """Last-resort serializer for json.dumps guard.

        Accepts dataclasses and objects with .to_dict(). Rejects everything
        else, so e.g. raw tensors raise cleanly at write time.
        """
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def _test_basic_add_and_read():
    wm = WorkingMemory(capacity=5)
    sid = wm.add_item("goal", {"game": "toy_a"}, priority=1.0, goal_id="g1")
    s = wm.get_item(sid)
    assert s is not None and s.content == {"game": "toy_a"}
    assert wm.get_stats()["current_size"] == 1


def _test_capacity_hard_limit():
    events: List[Event] = []
    wm = WorkingMemory(capacity=3, on_event=events.append)
    for i in range(6):
        wm.add_item("other", f"item_{i}", priority=0.5)
    assert len(wm.slots) == 3, f"expected 3, got {len(wm.slots)}"
    evicts = [e for e in events if e.kind == "evict"]
    assert len(evicts) == 3, f"expected 3 evict events, got {len(evicts)}"
    assert all(e.reason == "capacity_exceeded" for e in evicts)


def _test_eviction_order():
    wm = WorkingMemory(capacity=2, recency_weight=0.3)
    wm.add_item("other", "old_high", priority=0.9)
    time.sleep(0.01)
    wm.add_item("other", "new_low", priority=0.3)
    wm.add_item("other", "trigger", priority=0.5)
    remaining = {s.content for s in wm.slots.values()}
    assert "old_high" in remaining, f"old high-priority should survive, got {remaining}"


def _test_ttl_decay():
    events: List[Event] = []
    wm = WorkingMemory(capacity=5, on_event=events.append)
    sid = wm.add_item("intermediate_result", {"x": 1}, priority=0.5, ttl_ticks=3)
    for _ in range(3):
        wm.tick()
    assert sid not in wm.slots, "slot should have expired after 3 ticks"
    ttl_evicts = [e for e in events if e.kind == "evict" and e.reason == "ttl_expired"]
    assert len(ttl_evicts) == 1
    assert wm.get_stats()["total_ttl_expiries"] == 1


def _test_unknown_type_coerced():
    events: List[Event] = []
    wm = WorkingMemory(capacity=5, on_event=events.append)
    sid = wm.add_item("made_up_type", "payload", priority=0.5)
    assert wm.get_item(sid).content_type == "other"
    warns = [e for e in events if e.kind == "warn"]
    assert len(warns) == 1


def _test_dump_roundtrip():
    wm = WorkingMemory(capacity=5)
    wm.add_item("goal", {"game": "toy_c", "level": 1}, priority=0.9, goal_id="g1")
    wm.add_item("hypothesis", {"claim": "button A moves left"}, priority=0.6)
    snap = wm.dump()
    # Roundtrip
    s = json.dumps(snap)
    back = json.loads(s)
    assert back["capacity"] == 5
    assert len(back["slots"]) == 2
    types = {sl["content_type"] for sl in back["slots"]}
    assert types == {"goal", "hypothesis"}


def _test_event_stream_completeness():
    events: List[Event] = []
    wm = WorkingMemory(capacity=5, on_event=events.append)
    sid = wm.add_item("goal", {}, priority=0.5)                 # write
    wm.update(sid, {"updated": True})                           # update
    wm.get_item(sid)                                            # read (no event)
    wm.tick()                                                   # tick
    wm.clear()                                                  # clear
    kinds = [e.kind for e in events]
    assert "write" in kinds
    assert "update" in kinds
    assert "tick" in kinds
    assert "clear" in kinds


def _test_goal_filtering():
    wm = WorkingMemory(capacity=10)
    wm.add_item("goal", "A-goal", priority=0.9, goal_id="A")
    wm.add_item("plan_step", "A-step", priority=0.8, goal_id="A")
    wm.add_item("goal", "B-goal", priority=0.9, goal_id="B")
    wm.clear(goal_id="A")
    remaining_goals = {s.goal_id for s in wm.slots.values()}
    assert remaining_goals == {"B"}, f"got {remaining_goals}"


def _test_stable_key_deterministic():
    wm1 = WorkingMemory(capacity=5)
    wm1.add_item("goal", {"g": 1}, priority=0.9, goal_id="A")
    wm1.add_item("hypothesis", {"h": 2}, priority=0.5, goal_id="A")
    k1 = wm1.stable_key(goal_id="A")

    wm2 = WorkingMemory(capacity=5)
    # Add in DIFFERENT order
    wm2.add_item("hypothesis", {"h": 2}, priority=0.5, goal_id="A")
    wm2.add_item("goal", {"g": 1}, priority=0.9, goal_id="A")
    k2 = wm2.stable_key(goal_id="A")

    assert k1 == k2, f"stable_key should be order-invariant: {k1} vs {k2}"


def _test_json_guard_rejects_tensor_like():
    class NotSerializable:
        pass
    wm = WorkingMemory(capacity=5)
    try:
        wm.add_item("other", NotSerializable(), priority=0.5)
    except ValueError:
        return
    raise AssertionError("should have rejected non-JSON-serializable content")


def _run_all_tests():
    tests = [
        _test_basic_add_and_read,
        _test_capacity_hard_limit,
        _test_eviction_order,
        _test_ttl_decay,
        _test_unknown_type_coerced,
        _test_dump_roundtrip,
        _test_event_stream_completeness,
        _test_goal_filtering,
        _test_stable_key_deterministic,
        _test_json_guard_rejects_tensor_like,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    print("Working Memory v0.2 tests")
    print("─" * 50)
    _run_all_tests()
