"""
Effector Hub - Unified Action Execution Interface
Version: 1.0 (2025-10-12)

Central hub for managing all effectors. Provides unified interface for executing
commands safely with proper validation and error handling.

Design Principles:
    - Safe execution: Never crash if hardware missing
    - Unified interface: Single execute() method for all effectors
    - Configuration-driven: Load effectors from config files
    - Async-capable: Support concurrent command execution
    - Priority queuing: Execute high-priority commands first
    - Hot-reload: Add/remove effectors at runtime
"""

import torch
import asyncio
import yaml
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from queue import PriorityQueue
import time

from .base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


class EffectorHub:
    """
    Central hub for managing multiple effectors.

    The hub provides:
    - Unified command execution interface
    - Safe execution with validation and error handling
    - Async execution for concurrent commands
    - Priority-based command queuing
    - Configuration-driven effector registration
    - Runtime effector management
    - Statistics and monitoring

    Example:
        hub = EffectorHub(config_path="effectors.yaml")

        # Execute single command
        cmd = EffectorCommand(
            effector_id='motor_0',
            effector_type='motor',
            action='move',
            parameters={'speed': 0.5, 'direction': 'forward'}
        )
        result = hub.execute(cmd)

        # Execute multiple commands
        results = hub.execute_batch([cmd1, cmd2, cmd3])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize effector hub.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML/JSON config file
            device: Default torch device for effectors
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.effectors: Dict[str, BaseEffector] = {}

        # Command queue for priority-based execution
        self.command_queue: PriorityQueue = PriorityQueue()

        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = {}

        # Statistics
        self.execute_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_execute_time = 0.0
        self.execute_times: List[float] = []

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def register_effector(self, effector: BaseEffector, effector_id: Optional[str] = None):
        """
        Register an effector with the hub.

        Args:
            effector: Effector instance to register
            effector_id: Optional custom ID (uses effector.effector_id if None)
        """
        effector_id = effector_id or effector.effector_id

        if effector_id in self.effectors:
            print(f"Warning: Overwriting existing effector '{effector_id}'")

        self.effectors[effector_id] = effector

    def unregister_effector(self, effector_id: str):
        """
        Unregister and cleanup an effector.

        Args:
            effector_id: ID of effector to remove
        """
        if effector_id in self.effectors:
            effector = self.effectors[effector_id]
            effector.shutdown()
            del self.effectors[effector_id]

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """
        Execute a single command synchronously.

        This is the primary interface for executing effector commands.
        Handles validation, safety checks, and error handling.

        Args:
            command: Command to execute

        Returns:
            EffectorResult with execution status
        """
        start_time = time.time()

        # Check if effector exists
        if command.effector_id not in self.effectors:
            return EffectorResult(
                effector_id=command.effector_id,
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message=f"Effector '{command.effector_id}' not registered",
                execution_time=time.time() - start_time
            )

        effector = self.effectors[command.effector_id]

        # Execute command (effector handles validation and safety)
        try:
            result = effector.execute(command)
            result.execution_time = time.time() - start_time

        except Exception as e:
            # Safety net: catch any unexpected exceptions
            result = EffectorResult(
                effector_id=command.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time
            )

        # Update statistics
        self._update_stats(result)

        return result

    async def execute_async(self, command: EffectorCommand) -> EffectorResult:
        """
        Execute a single command asynchronously.

        Args:
            command: Command to execute

        Returns:
            EffectorResult with execution status
        """
        start_time = time.time()

        # Check if effector exists
        if command.effector_id not in self.effectors:
            return EffectorResult(
                effector_id=command.effector_id,
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message=f"Effector '{command.effector_id}' not registered",
                execution_time=time.time() - start_time
            )

        effector = self.effectors[command.effector_id]

        # Execute command asynchronously
        try:
            result = await effector.execute_async(command)
            result.execution_time = time.time() - start_time

        except Exception as e:
            result = EffectorResult(
                effector_id=command.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time
            )

        # Update statistics
        self._update_stats(result)

        return result

    def execute_batch(self, commands: List[EffectorCommand]) -> List[EffectorResult]:
        """
        Execute multiple commands synchronously.

        Args:
            commands: List of commands to execute

        Returns:
            List of EffectorResults in same order as commands
        """
        return [self.execute(cmd) for cmd in commands]

    async def execute_batch_async(self, commands: List[EffectorCommand]) -> List[EffectorResult]:
        """
        Execute multiple commands asynchronously (concurrent execution).

        Args:
            commands: List of commands to execute

        Returns:
            List of EffectorResults in same order as commands
        """
        tasks = [self.execute_async(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)

    def queue_command(self, command: EffectorCommand):
        """
        Add command to priority queue for later execution.

        Commands with higher priority values are executed first.

        Args:
            command: Command to queue
        """
        # PriorityQueue sorts by priority (lower first), so negate for higher=first
        self.command_queue.put((-command.priority, time.time(), command))

    def execute_queued(self, max_commands: Optional[int] = None) -> List[EffectorResult]:
        """
        Execute queued commands in priority order.

        Args:
            max_commands: Maximum number of commands to execute
                         (executes all if None)

        Returns:
            List of EffectorResults
        """
        results = []
        count = 0

        while not self.command_queue.empty():
            if max_commands and count >= max_commands:
                break

            _, _, command = self.command_queue.get()
            result = self.execute(command)
            results.append(result)
            count += 1

        return results

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """
        Validate command without executing.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if command.effector_id not in self.effectors:
            return False, f"Effector '{command.effector_id}' not registered"

        effector = self.effectors[command.effector_id]
        return effector.validate_command(command)

    def enable_effector(self, effector_id: str):
        """Enable a specific effector."""
        if effector_id in self.effectors:
            self.effectors[effector_id].enable()

    def disable_effector(self, effector_id: str):
        """Disable a specific effector."""
        if effector_id in self.effectors:
            self.effectors[effector_id].disable()

    def list_effectors(self) -> List[str]:
        """Get list of registered effector IDs."""
        return list(self.effectors.keys())

    def get_effector_info(self, effector_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific effector."""
        if effector_id not in self.effectors:
            return None

        effector = self.effectors[effector_id]
        return effector.get_info()

    def get_all_effector_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all effectors."""
        return {eid: effector.get_info() for eid, effector in self.effectors.items()}

    def _update_stats(self, result: EffectorResult):
        """Update execution statistics."""
        self.execute_count += 1
        self.last_execute_time = time.time()
        self.execute_times.append(result.execution_time)

        if result.is_success():
            self.success_count += 1
        else:
            self.error_count += 1

        # Keep only recent execution times (last 100)
        if len(self.execute_times) > 100:
            self.execute_times = self.execute_times[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hub statistics.

        Returns:
            Dictionary with:
                - num_effectors: int
                - num_active: int
                - execute_count: int
                - success_count: int
                - error_count: int
                - success_rate: float
                - avg_execution_time_ms: float
                - effector_stats: dict
        """
        active_effectors = sum(1 for e in self.effectors.values() if e.enabled)
        success_rate = self.success_count / max(self.execute_count, 1)
        avg_exec_time = sum(self.execute_times) / len(self.execute_times) if self.execute_times else 0.0

        return {
            'num_effectors': len(self.effectors),
            'num_active': active_effectors,
            'execute_count': self.execute_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'avg_execution_time_ms': avg_exec_time * 1000,
            'queued_commands': self.command_queue.qsize(),
            'last_execute_time': self.last_execute_time,
            'effector_stats': {eid: effector.get_stats()
                             for eid, effector in self.effectors.items()}
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.execute_count = 0
        self.success_count = 0
        self.error_count = 0
        self.execute_times = []
        for effector in self.effectors.values():
            effector.reset_stats()

    def shutdown(self):
        """Shutdown all effectors and cleanup resources."""
        for effector_id in list(self.effectors.keys()):
            self.unregister_effector(effector_id)

    def __repr__(self) -> str:
        active = sum(1 for e in self.effectors.values() if e.enabled)
        return f"EffectorHub(effectors={len(self.effectors)}, active={active})"

    def __len__(self) -> int:
        """Return number of registered effectors."""
        return len(self.effectors)

    def __contains__(self, effector_id: str) -> bool:
        """Check if effector is registered."""
        return effector_id in self.effectors

    def __getitem__(self, effector_id: str) -> BaseEffector:
        """Get effector by ID."""
        return self.effectors[effector_id]


# Convenience function for creating hub from config
def create_effector_hub(config_path: Optional[Path] = None,
                       effector_configs: Optional[List[Dict[str, Any]]] = None,
                       device: Optional[torch.device] = None) -> EffectorHub:
    """
    Create effector hub from configuration.

    Args:
        config_path: Path to config file
        effector_configs: List of effector configurations
        device: Default torch device

    Returns:
        Configured EffectorHub instance

    Example:
        # From config file
        hub = create_effector_hub(config_path="effectors.yaml")

        # From dictionaries
        configs = [
            {'effector_id': 'motor', 'effector_type': 'motor', ...},
            {'effector_id': 'speaker', 'effector_type': 'audio', ...}
        ]
        hub = create_effector_hub(effector_configs=configs)
    """
    hub = EffectorHub(config_path=config_path, device=device)

    # If effector configs provided, they would be registered here
    # This requires effector factory functions (implemented in mock_sensors.py)
    if effector_configs:
        # Import effector factories
        from .mock_sensors import create_effector_from_config

        for effector_config in effector_configs:
            try:
                effector = create_effector_from_config(effector_config)
                hub.register_effector(effector)
            except Exception as e:
                print(f"Failed to create effector from config {effector_config}: {e}")

    return hub
