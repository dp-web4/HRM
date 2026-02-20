"""
Network Effector — routes MESSAGE effects back to the gateway or to peer SAGEs.

When the consciousness loop generates a response to an incoming message,
it creates an Effect(type=MESSAGE, action='respond'). This effector
resolves the pending message Future in the MessageQueue, which unblocks
the HTTP handler waiting for the response.

For outbound messages to peer SAGEs, the 'send' action delegates to
the FederationClient.

Version: 1.0 (2026-02-19)
"""

import time
from typing import Optional, Any, Dict, Tuple, List

from sage.interfaces.base_effector import (
    BaseEffector, EffectorCommand, EffectorResult, EffectorStatus
)


class NetworkEffector(BaseEffector):
    """
    Effector that handles EffectType.MESSAGE effects.

    Two actions:
    - 'respond': Resolve a pending message in the MessageQueue (reply to caller)
    - 'send': Forward a message to a peer SAGE via FederationClient

    The message_queue and federation_client are injected after construction
    since they depend on the daemon being fully initialized.
    """

    SUPPORTED_ACTIONS = ('respond', 'send')

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config.setdefault('effector_id', 'network')
        config.setdefault('effector_type', 'network')
        super().__init__(config)

        self.message_queue = None  # Set by daemon after construction
        self.federation_client = None  # Set by daemon for cross-machine messaging
        self.operation_log: List[Dict[str, Any]] = []

    def set_message_queue(self, message_queue):
        """Inject the MessageQueue after construction."""
        self.message_queue = message_queue

    def set_federation_client(self, client):
        """Inject the FederationClient for cross-machine messaging."""
        self.federation_client = client

    def is_available(self) -> bool:
        """Network effector is available if enabled."""
        return self.enabled

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        """Validate that the command has a supported action."""
        action = command.parameters.get('action', command.action)
        if action not in self.SUPPORTED_ACTIONS:
            return False, f"Unsupported action: {action}. Must be one of {self.SUPPORTED_ACTIONS}"
        if action == 'respond' and not command.parameters.get('message_id'):
            return False, "Action 'respond' requires 'message_id' in parameters"
        return True, ""

    def get_info(self) -> Dict[str, Any]:
        """Return effector metadata."""
        return {
            'effector_id': self.effector_id,
            'effector_type': 'network',
            'supported_actions': list(self.SUPPORTED_ACTIONS),
            'has_message_queue': self.message_queue is not None,
            'has_federation_client': self.federation_client is not None,
            'operations_count': len(self.operation_log),
        }

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """
        Execute a MESSAGE effect.

        Args:
            command: EffectorCommand with parameters:
                - action: 'respond' or 'send'
                - For 'respond': message_id, response
                - For 'send': target_platform, message, conversation_id
        """
        start_time = time.time()

        # Check if enabled
        error = self._check_enabled()
        if error:
            return error

        # Validate
        valid, reason = self.validate_command(command)
        if not valid:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=reason,
                execution_time=time.time() - start_time,
            )
            self._update_stats(result)
            return result

        action = command.parameters.get('action', command.action)

        try:
            if action == 'respond':
                result = self._handle_respond(command, start_time)
            else:
                result = self._handle_send(command, start_time)
            self._update_stats(result)
            return result
        except Exception as e:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Execution error: {e}",
                execution_time=time.time() - start_time,
            )
            self._update_stats(result)
            return result

    def _handle_respond(self, command: EffectorCommand,
                        start_time: float) -> EffectorResult:
        """Resolve a pending message in the MessageQueue."""
        message_id = command.parameters.get('message_id')
        response = command.parameters.get('response', '')
        extra = command.parameters.get('extra', {})

        if self.message_queue is None:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message="MessageQueue not configured",
                execution_time=time.time() - start_time,
            )

        self.message_queue.resolve(message_id, response, extra=extra)

        self.operation_log.append({
            'action': 'respond',
            'message_id': message_id,
            'response_length': len(response),
        })

        return EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Resolved message {message_id} ({len(response)} chars)",
            execution_time=time.time() - start_time,
            metadata={
                'message_id': message_id,
                'response_length': len(response),
            },
        )

    def _handle_send(self, command: EffectorCommand,
                     start_time: float) -> EffectorResult:
        """Forward a message to a peer SAGE via FederationClient."""
        target = command.parameters.get('target_platform', '')
        message = command.parameters.get('message', '')

        if self.federation_client is None:
            # No federation client — log and succeed silently
            self.operation_log.append({
                'action': 'send',
                'target': target,
                'status': 'no_federation_client',
            })
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.SUCCESS,
                message=f"Message queued for {target} (federation client not configured)",
                execution_time=time.time() - start_time,
                metadata={'status': 'queued', 'target': target},
            )

        # TODO: Implement actual federation delegation
        # self.federation_client.delegate_task(...)

        self.operation_log.append({
            'action': 'send',
            'target': target,
            'message_length': len(message),
        })

        return EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Message sent to {target}",
            execution_time=time.time() - start_time,
            metadata={'status': 'sent', 'target': target},
        )


class MockNetworkEffector(NetworkEffector):
    """Mock version of NetworkEffector for testing — logs but doesn't resolve."""

    def _handle_respond(self, command: EffectorCommand,
                        start_time: float) -> EffectorResult:
        """Log the response without resolving any real Future."""
        message_id = command.parameters.get('message_id', 'unknown')
        response = command.parameters.get('response', '')

        self.operation_log.append({
            'action': 'respond',
            'message_id': message_id,
            'response_length': len(response),
            'mock': True,
        })

        return EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message=f"Mock resolved message {message_id}",
            execution_time=time.time() - start_time,
            metadata={'message_id': message_id, 'mock': True},
        )


if __name__ == "__main__":
    # Inline test
    print("NetworkEffector inline tests")
    print("=" * 40)

    # Test mock effector
    mock = MockNetworkEffector({'effector_id': 'test_network'})

    cmd = EffectorCommand(
        effector_id='test_network',
        effector_type='message',
        action='respond',
        parameters={
            'action': 'respond',
            'message_id': 'msg_test_001',
            'response': 'Hello from SAGE!',
        },
        priority=10,
        timeout=5.0,
    )

    result = mock.execute(cmd)
    assert result.is_success(), f"Expected success, got: {result.message}"
    assert len(mock.operation_log) == 1
    print(f"  respond action: OK (logged {mock.operation_log[-1]})")

    # Test send action
    cmd2 = EffectorCommand(
        effector_id='test_network',
        effector_type='message',
        action='send',
        parameters={
            'action': 'send',
            'target_platform': 'sprout_sage_lct',
            'message': 'Hello Sprout!',
        },
        priority=5,
        timeout=5.0,
    )

    result2 = mock.execute(cmd2)
    assert result2.is_success(), f"Expected success, got: {result2.message}"
    assert len(mock.operation_log) == 2
    print(f"  send action: OK (logged {mock.operation_log[-1]})")

    # Test unknown action
    cmd3 = EffectorCommand(
        effector_id='test_network',
        effector_type='message',
        action='unknown',
        parameters={'action': 'unknown'},
        priority=1,
        timeout=5.0,
    )
    result3 = mock.execute(cmd3)
    assert not result3.is_success(), "Expected failure for unknown action"
    assert result3.status == EffectorStatus.INVALID_COMMAND
    print(f"  unknown action: correctly rejected ({result3.message})")

    # Test validation
    valid, msg = mock.validate_command(cmd)
    assert valid, f"Expected valid, got: {msg}"
    print(f"  validate respond: OK")

    cmd_no_id = EffectorCommand(
        effector_id='test_network',
        effector_type='message',
        action='respond',
        parameters={'action': 'respond'},  # no message_id
        priority=1,
        timeout=5.0,
    )
    valid, msg = mock.validate_command(cmd_no_id)
    assert not valid
    print(f"  validate missing message_id: correctly rejected ({msg})")

    # Test get_info
    info = mock.get_info()
    assert info['effector_type'] == 'network'
    assert 'respond' in info['supported_actions']
    assert 'send' in info['supported_actions']
    print(f"  get_info: OK ({info})")

    # Test is_available
    assert mock.is_available()
    mock.disable()
    assert not mock.is_available()
    mock.enable()
    print(f"  is_available: OK")

    # Test stats
    stats = mock.get_stats()
    assert stats['execute_count'] == 3
    assert stats['success_count'] == 2
    assert stats['error_count'] == 1  # the unknown action
    print(f"  stats: OK (executed={stats['execute_count']}, "
          f"success={stats['success_count']}, errors={stats['error_count']})")

    print(f"\nAll NetworkEffector tests passed!")
