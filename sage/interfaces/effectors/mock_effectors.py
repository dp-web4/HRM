"""
Mock effectors for testing the effect system without real I/O.

Each mock logs operations to an operation_log list for test assertions.
Follows the same pattern as MockMotorEffector in mock_sensors.py.
"""

import time
from typing import Dict, Any, Tuple, List

from ..base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


class MockFileSystemEffector(BaseEffector):
    """Logs file operations without touching the filesystem."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operation_log: List[Dict[str, Any]] = []

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        self.operation_log.append({
            'action': command.action,
            'target': command.metadata.get('target', ''),
            'params': command.parameters,
            'timestamp': time.time(),
        })

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Mock {command.action} on {command.metadata.get('target', '')}",
            execution_time=time.time() - start_time,
            metadata={'action': command.action},
        )
        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('read', 'write', 'append', 'delete', 'list'):
            return False, f"Invalid action: {command.action}"
        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'file_io',
            'supported_actions': ['read', 'write', 'append', 'delete', 'list'],
            'mock': True,
        }


class MockToolUseEffector(BaseEffector):
    """Records tool invocations without executing them."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operation_log: List[Dict[str, Any]] = []

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        self.operation_log.append({
            'action': command.action,
            'tool': command.metadata.get('target', ''),
            'params': command.parameters,
            'timestamp': time.time(),
        })

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Mock {command.action}: {command.metadata.get('target', '')}",
            execution_time=time.time() - start_time,
            metadata={'tool': command.metadata.get('target', '')},
        )
        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('invoke', 'query'):
            return False, f"Invalid action: {command.action}"
        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'tool_use',
            'supported_actions': ['invoke', 'query'],
            'mock': True,
        }


class MockWebEffector(BaseEffector):
    """Records HTTP requests without making them."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operation_log: List[Dict[str, Any]] = []

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        self.operation_log.append({
            'method': command.action.upper(),
            'url': command.metadata.get('target', ''),
            'params': command.parameters,
            'timestamp': time.time(),
        })

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Mock {command.action.upper()} {command.metadata.get('target', '')}",
            execution_time=time.time() - start_time,
            metadata={
                'status_code': 200,
                'url': command.metadata.get('target', ''),
            },
        )
        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('get', 'post', 'put', 'delete', 'head'):
            return False, f"Invalid action: {command.action}"
        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'web',
            'supported_actions': ['get', 'post', 'put', 'delete', 'head'],
            'mock': True,
        }


class MockCognitiveEffector(BaseEffector):
    """Records memory writes, trust updates, and state changes."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operation_log: List[Dict[str, Any]] = []

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        self.operation_log.append({
            'action': command.action,
            'effector_type': command.effector_type,
            'target': command.metadata.get('target', ''),
            'params': command.parameters,
            'timestamp': time.time(),
        })

        result = EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Mock cognitive: {command.effector_type}/{command.action}",
            execution_time=time.time() - start_time,
            metadata={'cognitive_type': command.effector_type},
        )
        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        return True, ""  # Cognitive effects always valid

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'cognitive',
            'supported_actions': ['consolidate', 'update', 'transition'],
            'mock': True,
        }
