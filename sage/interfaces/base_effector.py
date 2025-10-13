"""
Base Effector Interface
Version: 1.0 (2025-10-12)

Abstract base class for all effectors. Effectors execute actions in the world
by writing to hardware devices (motors, displays, speakers) or virtual outputs.

Design Principles:
    - Safe execution: Never crash if hardware missing
    - Async-capable: Non-blocking writes with execute_async()
    - Validation: Check commands before execution
    - Graceful degradation: Return error status instead of crashing
    - Configuration-driven: All parameters in config dict
"""

import torch
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import time


class EffectorStatus(Enum):
    """Status codes for effector execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_COMMAND = "invalid_command"
    HARDWARE_UNAVAILABLE = "hardware_unavailable"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class EffectorCommand:
    """
    Container for effector commands.

    Attributes:
        effector_id: Target effector identifier
        effector_type: Type of effector (motor, display, etc.)
        action: Action name or command type
        parameters: Command-specific parameters
        data: Optional tensor data (for displays, audio, etc.)
        timeout: Maximum execution time in seconds
        priority: Command priority (higher = more important)
        metadata: Additional command information
    """
    effector_id: str
    effector_type: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    data: Optional[torch.Tensor] = None
    timeout: float = 5.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'action': self.action,
            'parameters': self.parameters,
            'timeout': self.timeout,
            'priority': self.priority,
            'metadata': self.metadata
        }

        if self.data is not None:
            result['data_shape'] = list(self.data.shape)
            result['data_dtype'] = str(self.data.dtype)

        return result


@dataclass
class EffectorResult:
    """
    Result of effector execution.

    Attributes:
        effector_id: Effector that executed the command
        status: Execution status
        message: Human-readable status message
        execution_time: Time taken to execute (seconds)
        timestamp: When execution completed
        metadata: Additional result information
    """
    effector_id: str
    status: EffectorStatus
    message: str = ""
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == EffectorStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'effector_id': self.effector_id,
            'status': self.status.value,
            'message': self.message,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class BaseEffector(ABC):
    """
    Abstract base class for all effectors.

    All effectors must implement:
        - execute(): Synchronous command execution
        - execute_async(): Asynchronous command execution
        - is_available(): Check if effector is accessible
        - validate_command(): Check command validity
        - get_info(): Return effector capabilities

    Configuration:
        config = {
            'effector_id': 'unique_id',
            'effector_type': 'motor',
            'device': 'cuda',  # or 'cpu'
            'enabled': True,
            'safety_limits': {...},
            # ... effector-specific params
        }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize effector with configuration.

        Args:
            config: Configuration dictionary with effector parameters
        """
        self.config = config
        self.effector_id = config.get('effector_id', self.__class__.__name__)
        self.effector_type = config.get('effector_type', 'generic')
        self.device = torch.device(config.get('device', 'cpu'))
        self.enabled = config.get('enabled', True)

        # Safety
        self.safety_limits = config.get('safety_limits', {})
        self.enable_safety_checks = config.get('enable_safety_checks', True)

        # Statistics
        self.execute_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        self.last_result: Optional[EffectorResult] = None

    @abstractmethod
    def execute(self, command: EffectorCommand) -> EffectorResult:
        """
        Synchronous command execution.

        This should:
        1. Validate command
        2. Check safety limits
        3. Execute action
        4. Return result (never raise exceptions for hardware failures)

        Args:
            command: Command to execute

        Returns:
            EffectorResult with execution status
        """
        pass

    async def execute_async(self, command: EffectorCommand) -> EffectorResult:
        """
        Asynchronous command execution.

        Default implementation wraps execute() in executor.
        Override for true async implementations.

        Args:
            command: Command to execute

        Returns:
            EffectorResult with execution status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, command)

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if effector is accessible and operational.

        Returns:
            True if effector can execute commands, False otherwise
        """
        pass

    @abstractmethod
    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """
        Validate command before execution.

        Check:
        - Command structure is valid
        - Parameters are within acceptable ranges
        - Action is supported
        - Safety limits are satisfied

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid: (True, "")
            If invalid: (False, "reason why invalid")
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get effector capabilities and metadata.

        Returns:
            Dictionary with effector information:
                - effector_id: str
                - effector_type: str
                - supported_actions: List[str]
                - parameter_schema: dict
                - safety_limits: dict
                - capabilities: dict
        """
        pass

    def enable(self):
        """Enable effector execution."""
        self.enabled = True

    def disable(self):
        """Disable effector execution."""
        self.enabled = False

    def _check_enabled(self) -> Optional[EffectorResult]:
        """
        Check if effector is enabled.

        Returns:
            EffectorResult with error if disabled, None if enabled
        """
        if not self.enabled:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message="Effector is disabled"
            )
        return None

    def _check_safety(self, command: EffectorCommand) -> Optional[EffectorResult]:
        """
        Check safety limits.

        Args:
            command: Command to check

        Returns:
            EffectorResult with error if unsafe, None if safe
        """
        if not self.enable_safety_checks:
            return None

        # Check if safety limits are defined
        if not self.safety_limits:
            return None

        # Validate against safety limits
        for param, value in command.parameters.items():
            if param in self.safety_limits:
                limits = self.safety_limits[param]

                # Check min/max bounds
                if 'min' in limits and value < limits['min']:
                    return EffectorResult(
                        effector_id=self.effector_id,
                        status=EffectorStatus.SAFETY_VIOLATION,
                        message=f"Parameter {param}={value} below minimum {limits['min']}"
                    )

                if 'max' in limits and value > limits['max']:
                    return EffectorResult(
                        effector_id=self.effector_id,
                        status=EffectorStatus.SAFETY_VIOLATION,
                        message=f"Parameter {param}={value} above maximum {limits['max']}"
                    )

        return None

    def _update_stats(self, result: EffectorResult):
        """Update execution statistics."""
        self.execute_count += 1
        self.total_execution_time += result.execution_time
        self.last_result = result

        if result.is_success():
            self.success_count += 1
        else:
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get effector statistics.

        Returns:
            Dictionary with:
                - execute_count: int
                - success_count: int
                - error_count: int
                - success_rate: float
                - avg_execution_time: float
                - last_status: str
        """
        success_rate = self.success_count / max(self.execute_count, 1)
        avg_time = self.total_execution_time / max(self.execute_count, 1)

        return {
            'effector_id': self.effector_id,
            'effector_type': self.effector_type,
            'execute_count': self.execute_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'avg_execution_time_ms': avg_time * 1000,
            'total_execution_time': self.total_execution_time,
            'last_status': self.last_result.status.value if self.last_result else None,
            'enabled': self.enabled,
            'available': self.is_available()
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.execute_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0

    def shutdown(self):
        """
        Cleanup effector resources.

        Override to implement cleanup logic (stop motors, close files, etc.)
        """
        self.enabled = False

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.effector_id}, "
                f"type={self.effector_type}, enabled={self.enabled}, "
                f"available={self.is_available()})")
