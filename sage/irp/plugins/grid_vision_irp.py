"""
GridVisionIRP — ARC-AGI-3 game grid perception plugin.

Bridges Andy's perception layer (grid → objects → embedding) into SAGE's
consciousness loop as a vision sensor. Push model: Andy's layer pushes
GridObservation into a buffer, SAGE polls at each cycle.

Transport-agnostic: works with direct push (embedded/competition), MCP
tools (orchestration), or REST (development).
"""

import base64
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import IRPPlugin, IRPState


@dataclass
class GridObservation:
    """What the perception layer hands to SAGE each frame.

    Produced by Andy's connected component analysis + embedding pipeline.
    Consumed by GridVisionIRP as a vision sensor observation.
    """

    # Raw grid
    frame_raw: np.ndarray           # (64, 64) uint8, values 0-15

    # Structured perception (connected component analysis)
    objects: List[Dict]             # [{"id": int, "color": int, "cells": [[r,c],...], "bbox": [r1,c1,r2,c2]}, ...]

    # Frame diff
    changes: List[Dict]             # [{"cell": [r,c], "was": int, "now": int}, ...]
    moved: List[Dict]               # [{"id": int, "from_bbox": [...], "to_bbox": [...], "delta": [dr,dc]}, ...]

    # Embedding (for cartridge search)
    embedding: Optional[np.ndarray] = None  # Andy's CLIP or custom encoding

    # Context
    step_number: int = 0
    action_taken: int = 0           # 0 = no action yet (first frame), 1-5 = last action
    level_id: str = ""

    # Optional perception notes
    perception_notes: Optional[str] = None

    # Cognitive classification — set by SAGE when writing step_record back to cartridge.
    # Andy's cartridge layer uses this to tag the h-row for prioritized cross-level search.
    # Values: None (routine step), "reflection" (hypothesis+strategy+key_insight),
    #         "discovery" (novel pattern first seen), "goal_update" (objective revision)
    cognitive_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict. Compatible with Andy's GridObservation.to_dict().

        embedding → base64-encoded float32 bytes (matches membot/vision/grid_observation.py)
        frame_raw → nested lists of ints
        """
        result: Dict[str, Any] = {
            "frame_raw": self.frame_raw.tolist(),
            "objects": self.objects,
            "changes": self.changes,
            "moved": self.moved,
            "step_number": self.step_number,
            "action_taken": self.action_taken,
            "level_id": self.level_id,
            "perception_notes": self.perception_notes,
            "cognitive_type": self.cognitive_type,
        }
        if self.embedding is not None:
            result["embedding"] = base64.b64encode(
                self.embedding.astype(np.float32).tobytes()
            ).decode("ascii")
            result["embedding_shape"] = list(self.embedding.shape)
        else:
            result["embedding"] = None
            result["embedding_shape"] = None
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GridObservation":
        """Deserialize from JSON-safe dict. Inverse of to_dict().

        Restores numpy arrays from serialized forms.
        """
        embedding = None
        if d.get("embedding") is not None:
            raw = base64.b64decode(d["embedding"])
            embedding = np.frombuffer(raw, dtype=np.float32).copy()
            shape = d.get("embedding_shape")
            if shape and len(shape) > 1:
                embedding = embedding.reshape(shape)
        return cls(
            frame_raw=np.array(d["frame_raw"], dtype=np.uint8),
            objects=d.get("objects", []),
            changes=d.get("changes", []),
            moved=d.get("moved", []),
            embedding=embedding,
            step_number=d.get("step_number", 0),
            action_taken=d.get("action_taken", 0),
            level_id=d.get("level_id", ""),
            perception_notes=d.get("perception_notes"),
            cognitive_type=d.get("cognitive_type"),
        )

    @property
    def change_magnitude(self) -> float:
        """Normalized count of cells that changed. [0, 1]."""
        if not self.changes:
            return 0.0
        return min(len(self.changes) / (64 * 64), 1.0)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_moved(self) -> int:
        return len(self.moved)


@dataclass
class StepRecord:
    """What SAGE writes back to Andy's cartridge (odd-pattern rows).

    Andy's perception layer writes even-pattern rows (frame + embedding + objects).
    SAGE writes odd-pattern rows (this struct) after each action decision.

    cognitive_type drives Andy's h-row tagging for prioritized cross-level search:
      None        → routine step (most common)
      "reflection" → deep reflection (hypothesis + strategy + key_insight) — HIGH PRIORITY
      "discovery"  → novel pattern first observed in this session
      "goal_update" → SAGE revised its objective mid-level
    """
    step: int
    action_taken: int                   # 1-5
    action_rationale: str               # human-readable reason
    salience: Dict[str, float]          # {surprise, novelty, arousal, reward, conflict}
    metabolic_state: str                # FOCUS / EXPLORE / CONSERVE / etc.
    atp_spent: float
    trust_posture: Dict[str, Any]       # {confidence, dominant_modality, label}
    policy_gate: str                    # "approved" / "blocked: <reason>"
    timestamp: float

    # Reasoning (set when reasoning plugin fires — not every step)
    reasoning_text: Optional[str] = None
    hypothesis: Optional[str] = None
    strategy: Optional[str] = None
    key_insight: Optional[str] = None

    # Cognitive flag for Andy's cartridge h-row
    cognitive_type: Optional[str] = None  # see class docstring

    level_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict for cartridge odd-pattern write."""
        return {
            "step": self.step,
            "action_taken": self.action_taken,
            "action_rationale": self.action_rationale,
            "salience": self.salience,
            "metabolic_state": self.metabolic_state,
            "atp_spent": self.atp_spent,
            "trust_posture": self.trust_posture,
            "policy_gate": self.policy_gate,
            "timestamp": self.timestamp,
            "reasoning_text": self.reasoning_text,
            "hypothesis": self.hypothesis,
            "strategy": self.strategy,
            "key_insight": self.key_insight,
            "cognitive_type": self.cognitive_type,
            "level_id": self.level_id,
        }

    @property
    def is_deep_reflection(self) -> bool:
        """True if this record contains a full hypothesis+strategy+insight triple."""
        return bool(self.hypothesis and self.strategy and self.key_insight)

    def auto_tag_cognitive_type(self) -> None:
        """Set cognitive_type based on content if not already set."""
        if self.cognitive_type is not None:
            return
        if self.is_deep_reflection:
            self.cognitive_type = "reflection"


class GridVisionIRP(IRPPlugin):
    """ARC-AGI-3 game grid perception as an IRP sensor plugin.

    Receives GridObservation from the perception layer (push model),
    presents it to the consciousness loop as a vision observation.

    Energy = inverse of frame-change magnitude. Big changes are salient
    (low energy = high priority in attention selection).

    Config keys:
        entity_id: str — plugin identity (default: 'grid_vision_irp')
        buffer_size: int — max observations to buffer (default: 10)
        max_iterations: int — always 1 (perception is immediate)
    """

    def __init__(self, config: Dict[str, Any]):
        config.setdefault('max_iterations', 1)
        config.setdefault('entity_id', 'grid_vision_irp')
        super().__init__(config)

        self.buffer_size = config.get('buffer_size', 10)
        self._buffer: List[GridObservation] = []
        self._lock = threading.Lock()
        self._frame_count = 0
        self._prev_frame: Optional[np.ndarray] = None

        print(f"✓ GridVisionIRP initialized (entity_id={self.entity_id})")

    # --- Push interface (perception layer calls this) ---

    def submit_observation(self, obs: GridObservation) -> None:
        """Push a new observation from the perception layer."""
        with self._lock:
            self._buffer.append(obs)
            if len(self._buffer) > self.buffer_size:
                self._buffer = self._buffer[-self.buffer_size:]
            self._frame_count += 1

    def get_latest(self) -> Optional[GridObservation]:
        """Non-blocking read of most recent observation. Returns None if empty."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    # --- Direct push from raw grid (no external perception layer) ---

    def push_raw_frame(self, frame: np.ndarray, step_number: int = 0,
                       action_taken: int = 0, level_id: str = "") -> GridObservation:
        """Create a minimal GridObservation from a raw grid frame.

        For use when Andy's perception layer isn't available (testing,
        competition fallback). Provides basic frame diff but no object
        detection or embedding.

        Accepts 2D (H, W) or 3D (1, H, W) grids — squeezes to 2D internally.
        """
        # Normalize to 2D: SDK returns (N, 64, 64) where N is animation frames.
        # Take the last frame (current state after animation completes).
        if frame.ndim == 3:
            frame = frame[-1]

        # Guard against empty or degenerate frames
        if frame.size == 0 or frame.ndim != 2:
            if self._prev_frame is not None:
                frame = self._prev_frame.copy()
            else:
                frame = np.zeros((64, 64), dtype=np.int8)

        changes = []
        moved = []
        if self._prev_frame is not None:
            diff_mask = frame != self._prev_frame
            diff_coords = np.argwhere(diff_mask)
            for r, c in diff_coords:
                changes.append({
                    "cell": [int(r), int(c)],
                    "was": int(self._prev_frame[r, c]),
                    "now": int(frame[r, c]),
                })

        self._prev_frame = frame.copy()

        obs = GridObservation(
            frame_raw=frame,
            objects=[],         # no object detection without perception layer
            changes=changes,
            moved=moved,
            step_number=step_number,
            action_taken=action_taken,
            level_id=level_id,
        )
        self.submit_observation(obs)
        return obs

    # --- IRP contract ---

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """Initialize grid vision state.

        x0: GridObservation or None (will poll buffer)
        task_ctx: {"level_id": str, "max_steps": int, "actions": [1,2,3,4,5]}
        """
        obs = x0 if isinstance(x0, GridObservation) else self.get_latest()

        return IRPState(
            x={
                "observation": obs,
                "frame_count": self._frame_count,
                "has_data": obs is not None,
            },
            step_idx=0,
            energy_val=self._compute_energy(obs),
            meta={
                "level_id": task_ctx.get("level_id", ""),
                "n_objects": obs.n_objects if obs else 0,
                "n_changes": len(obs.changes) if obs else 0,
                "n_moved": obs.n_moved if obs else 0,
                "change_magnitude": obs.change_magnitude if obs else 0.0,
                "step_number": obs.step_number if obs else 0,
                "action_taken": obs.action_taken if obs else 0,
                "perception_notes": obs.perception_notes if obs else None,
            },
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """Read latest observation from buffer.

        Perception is immediate — one step, no iterative refinement.
        """
        obs = self.get_latest()

        return IRPState(
            x={
                "observation": obs,
                "frame_count": self._frame_count,
                "has_data": obs is not None,
            },
            step_idx=state.step_idx + 1,
            energy_val=self._compute_energy(obs),
            meta={
                "level_id": state.meta.get("level_id", ""),
                "n_objects": obs.n_objects if obs else 0,
                "n_changes": len(obs.changes) if obs else 0,
                "n_moved": obs.n_moved if obs else 0,
                "change_magnitude": obs.change_magnitude if obs else 0.0,
                "step_number": obs.step_number if obs else 0,
                "action_taken": obs.action_taken if obs else 0,
                "perception_notes": obs.perception_notes if obs else None,
            },
        )

    def energy(self, state: IRPState) -> float:
        """Energy = 1 - change_magnitude.

        Big frame change → low energy → high salience (surprising).
        No change → high energy → low salience (boring).
        """
        return state.energy_val if state.energy_val is not None else 1.0

    def halt(self, history: List[IRPState]) -> bool:
        """Always halt after one step. Perception is immediate."""
        return len(history) >= 1

    # --- Helpers ---

    def _compute_energy(self, obs: Optional[GridObservation]) -> float:
        """Compute energy from observation. Lower = more salient."""
        if obs is None:
            return 1.0  # No data = boring
        return 1.0 - obs.change_magnitude

    def to_sensor_observation(self, obs: Optional[GridObservation] = None):
        """Convert to SensorObservation format for consciousness loop.

        Returns a dict matching the SensorObservation constructor kwargs.
        Used by _gather_observations() in sage_consciousness.py.
        """
        if obs is None:
            obs = self.get_latest()

        if obs is None:
            return None

        return {
            "sensor_id": "grid_vision_0",
            "modality": "vision",
            "data": {
                "type": "grid_frame",
                "frame_raw": obs.frame_raw,
                "objects": obs.objects,
                "changes": obs.changes,
                "moved": obs.moved,
                "embedding": obs.embedding,
                "step_number": obs.step_number,
                "action_taken": obs.action_taken,
                "level_id": obs.level_id,
                "n_objects": obs.n_objects,
                "n_changes": len(obs.changes),
                "n_moved": obs.n_moved,
                "change_magnitude": obs.change_magnitude,
                "perception_notes": obs.perception_notes,
            },
            "timestamp": time.time(),
            "trust": self.trust_weight,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Current plugin stats for logging."""
        latest = self.get_latest()
        return {
            "frames_received": self._frame_count,
            "buffer_depth": len(self._buffer),
            "latest_step": latest.step_number if latest else None,
            "latest_objects": latest.n_objects if latest else 0,
            "latest_changes": len(latest.changes) if latest else 0,
        }
