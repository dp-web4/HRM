"""
GameActionEffector — ARC-AGI-3 action dispatch.

Converts SAGE's action decisions into ARC-AGI-3 game API calls.
Actions 1-5 (4 movement + 1 select). Tracks action history for
efficiency scoring (RHAE).

Transport-agnostic: works with direct SDK call (competition),
REST API (development), or callback (testing).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

from sage.interfaces.base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


# ARC-AGI-3 action space
ACTION_NAMES = {
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
    5: "select",
}


@dataclass
class ActionRecord:
    """Record of a single game action for efficiency tracking."""
    step: int
    action: int
    action_name: str
    timestamp: float
    frame_changed: bool = False        # did the frame change after this action?
    cells_changed: int = 0             # how many cells changed
    rationale: str = ""                # from consciousness loop reasoning


class GameActionEffector(BaseEffector):
    """Dispatches actions to ARC-AGI-3 game environments.

    Wraps the ARC-AGI-3 SDK's action API. Tracks action history for
    RHAE efficiency scoring.

    Config keys:
        effector_id: str — identity (default: 'game_action')
        effector_type: str — 'game' (default)
        action_callback: Callable — direct function to call (testing/embedded)
        sdk_env: object — ARC-AGI-3 SDK environment instance
        max_actions_per_level: int — safety limit (default: 10000)
    """

    def __init__(self, config: Dict[str, Any]):
        config.setdefault('effector_id', 'game_action')
        config.setdefault('effector_type', 'game')
        config.setdefault('enabled', True)
        super().__init__(config)

        self.action_callback: Optional[Callable] = config.get('action_callback')
        self.sdk_env = config.get('sdk_env')
        self.max_actions = config.get('max_actions_per_level', 10000)

        # Action history for efficiency tracking
        self.history: List[ActionRecord] = []
        self.level_id: str = ""
        self._action_count = 0

        # Stats
        self.total_actions = 0
        self.effective_actions = 0  # actions that caused frame change
        self.wasted_actions = 0     # actions with no effect

        print(f"✓ GameActionEffector initialized (entity_id={self.effector_id})")

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Execute a game action.

        command.action should be 'game_action'
        command.parameters should contain:
            action: int (1-5)
            rationale: str (optional, from reasoning)
        """
        start = time.time()
        self.execute_count += 1

        # Validate
        valid, msg = self.validate_command(command)
        if not valid:
            self.error_count += 1
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=msg,
            )

        action = command.parameters.get('action', 0)
        rationale = command.parameters.get('rationale', '')

        # Dispatch via available backend
        result_data = self._dispatch_action(action)

        elapsed = time.time() - start

        # Record
        record = ActionRecord(
            step=self._action_count,
            action=action,
            action_name=ACTION_NAMES.get(action, f"unknown_{action}"),
            timestamp=time.time(),
            frame_changed=result_data.get('frame_changed', False),
            cells_changed=result_data.get('cells_changed', 0),
            rationale=rationale,
        )
        self.history.append(record)
        self._action_count += 1
        self.total_actions += 1

        if record.frame_changed:
            self.effective_actions += 1
        else:
            self.wasted_actions += 1

        self.success_count += 1
        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Action {action} ({ACTION_NAMES.get(action, '?')}): {record.cells_changed} cells changed",
            execution_time=elapsed,
            metadata={
                'action': action,
                'action_name': ACTION_NAMES.get(action, '?'),
                'frame_changed': record.frame_changed,
                'cells_changed': record.cells_changed,
                'total_actions': self.total_actions,
                'efficiency': self.efficiency_ratio,
            },
        )
        self.last_result = result
        return result

    def _dispatch_action(self, action: int) -> Dict[str, Any]:
        """Send action to game environment via available backend."""

        # Backend 1: Direct callback (testing, embedded)
        if self.action_callback is not None:
            return self.action_callback(action)

        # Backend 2: ARC-AGI-3 SDK environment
        if self.sdk_env is not None:
            try:
                obs = self.sdk_env.step(action)
                # SDK returns observation after action — caller can check frame diff
                return {
                    'frame_changed': True,  # assume change; GridVisionIRP will verify
                    'cells_changed': 0,     # calculated by perception layer
                    'raw_obs': obs,
                }
            except Exception as e:
                return {'frame_changed': False, 'cells_changed': 0, 'error': str(e)}

        # Backend 3: No backend configured (dry run)
        return {'frame_changed': False, 'cells_changed': 0, 'dry_run': True}

    def is_available(self) -> bool:
        """Check if game environment is accessible."""
        return self.enabled and (
            self.action_callback is not None
            or self.sdk_env is not None
        )

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """Validate game action command."""
        action = command.parameters.get('action', 0)

        if action not in ACTION_NAMES:
            return False, f"Invalid action {action}. Must be 1-5."

        if self._action_count >= self.max_actions:
            return False, f"Action limit reached ({self.max_actions}). Level may be stuck."

        return True, "ok"

    def get_info(self) -> Dict[str, Any]:
        """Return effector capabilities."""
        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'actions': ACTION_NAMES,
            'max_actions': self.max_actions,
            'current_count': self._action_count,
            'available': self.is_available(),
        }

    # --- Efficiency tracking (RHAE scoring) ---

    @property
    def efficiency_ratio(self) -> float:
        """Fraction of actions that caused frame changes. Higher = better RHAE."""
        if self.total_actions == 0:
            return 0.0
        return self.effective_actions / self.total_actions

    @property
    def rhae_estimate(self) -> float:
        """Rough RHAE estimate. Actual scoring uses human baseline comparison.

        RHAE = min(1, human_actions / agent_actions) ** 2
        We can't know human_actions, but efficiency_ratio approximates
        how many actions were wasted. Lower waste = higher RHAE.
        """
        return self.efficiency_ratio ** 2

    def reset_level(self, level_id: str = "") -> None:
        """Reset tracking for a new level."""
        self.history = []
        self._action_count = 0
        self.level_id = level_id
        self.total_actions = 0
        self.effective_actions = 0
        self.wasted_actions = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Current effector stats for logging."""
        return {
            'level_id': self.level_id,
            'total_actions': self.total_actions,
            'effective_actions': self.effective_actions,
            'wasted_actions': self.wasted_actions,
            'efficiency': self.efficiency_ratio,
            'rhae_estimate': self.rhae_estimate,
        }
