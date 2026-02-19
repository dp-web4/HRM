"""
ToolUseEffector — generic tool invocation as Effects.

Handles EffectType.TOOL_USE with a registry of callables.
"""

import time
import signal
from typing import Dict, Any, Tuple, Callable, Optional

from ..base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


class ToolUseEffector(BaseEffector):
    """
    Handles EffectType.TOOL_USE.

    Actions: invoke, query
    Target: tool name (via command.metadata['target'])
    Tools are registered callables with optional descriptions.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.allowed_tools: list = config.get('allowed_tools', [])
        self.max_execution_time: float = config.get('max_execution_time', 30.0)

    def register_tool(self, name: str, fn: Callable,
                      description: str = ""):
        """Register a callable tool."""
        self.tools[name] = {'fn': fn, 'description': description}

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        is_valid, message = self.validate_command(command)
        if not is_valid:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=message,
            )
            self._update_stats(result)
            return result

        tool_name = command.metadata.get('target', command.parameters.get('tool', ''))

        try:
            tool = self.tools[tool_name]
            fn = tool['fn']
            tool_result = fn(**command.parameters)

            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.SUCCESS,
                message=f"Tool '{tool_name}' executed",
                execution_time=time.time() - start_time,
                metadata={'tool': tool_name, 'result': tool_result},
            )

        except Exception as e:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Tool '{tool_name}' error: {str(e)}",
                execution_time=time.time() - start_time,
            )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('invoke', 'query'):
            return False, f"Invalid action: {command.action}"

        tool_name = command.metadata.get('target', command.parameters.get('tool', ''))
        if not tool_name:
            return False, "Missing tool name"

        if tool_name not in self.tools:
            return False, f"Unknown tool: {tool_name}"

        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' not in allowed list"

        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'tool_use',
            'supported_actions': ['invoke', 'query'],
            'registered_tools': {
                name: info['description'] for name, info in self.tools.items()
            },
            'allowed_tools': self.allowed_tools,
        }
