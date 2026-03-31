"""
GridVisionIRP — ARC-AGI-3 game grid perception plugin.

Bridges Andy's perception layer (grid → objects → embedding) into SAGE's
consciousness loop as a vision sensor. Push model: Andy's layer pushes
GridObservation into a buffer, SAGE polls at each cycle.

Transport-agnostic: works with direct push (embedded/competition), MCP
tools (orchestration), or REST (development).
"""

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
        """
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
