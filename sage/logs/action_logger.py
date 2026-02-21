"""
SAGE Action Logger — structured JSONL logging for the collective.

Each machine logs its actions to HRM/sage/logs/machines/{machine}/.
Logs are append-only JSONL, committed to git for collective visibility.

Auto-detects machine name. Buffers writes for efficiency.
Supports both action logging and interaction (cross-machine) logging.

Usage:
    from sage.logs.action_logger import ActionLogger

    logger = ActionLogger()  # auto-detects machine
    logger.log_action('inference', 'Responded to greeting',
                      model='qwen2.5-14b',
                      metabolic={'state': 'wake', 'atp_before': 85, 'atp_after': 82})
    logger.log_interaction('inbound', from_lct='lct://sage:sprout:agent@resident',
                           message_id='msg_001', latency_ms=2100)
    logger.flush()

Version: 1.0 (2026-02-20)
"""

import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _detect_machine() -> str:
    """Detect machine name from environment or hostname."""
    # Explicit override
    env = os.environ.get('SAGE_MACHINE', '').lower()
    if env:
        return env

    hostname = socket.gethostname().lower()

    if 'thor' in hostname:
        return 'thor'
    if hostname == 'ubuntu':
        # Sprout's Jetson Orin Nano defaults to 'ubuntu'
        if Path('/home/sprout').exists():
            return 'sprout'
    if 'mcnugget' in hostname:
        return 'mcnugget'
    if 'legion' in hostname:
        return 'legion'

    # CBP / WSL2 detection
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return 'cbp'
    except (FileNotFoundError, PermissionError):
        pass

    # macOS detection
    if hostname == 'mcnugget' or 'mcnugget' in hostname:
        return 'mcnugget'

    return hostname


def _find_logs_dir(machine: str) -> Path:
    """Find the machine's log directory in HRM."""
    # Try common workspace locations
    candidates = [
        Path(os.environ.get('HRM_ROOT', '')) / 'sage' / 'logs' / 'machines' / machine,
        Path.home() / 'ai-workspace' / 'HRM' / 'sage' / 'logs' / 'machines' / machine,
        Path.home() / 'repos' / 'HRM' / 'sage' / 'logs' / 'machines' / machine,
        Path('/mnt/c/exe/projects/ai-agents/HRM/sage/logs/machines') / machine,
    ]

    for candidate in candidates:
        if candidate.parent.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

    # Fallback: create in current working directory
    fallback = Path.cwd() / 'sage' / 'logs' / 'machines' / machine
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _now_iso() -> str:
    """UTC ISO 8601 timestamp."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _content_hash(content: str) -> str:
    """SHA-256 hash of content for verification without exposing content."""
    return 'sha256:' + hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class ActionLogger:
    """
    Structured action logger for SAGE machines.

    Appends JSONL entries to actions.jsonl and interactions.jsonl
    in the machine's log directory within HRM.
    """

    def __init__(self, machine: Optional[str] = None,
                 logs_dir: Optional[Path] = None,
                 lct: Optional[str] = None,
                 buffer_size: int = 10):
        self.machine = machine or _detect_machine()
        self.logs_dir = logs_dir or _find_logs_dir(self.machine)
        self.lct = lct or f'lct://sage:{self.machine}:agent@resident'
        self.buffer_size = buffer_size

        self._action_buffer: List[str] = []
        self._interaction_buffer: List[str] = []

        # Ensure snapshot directory exists
        (self.logs_dir / 'state-snapshots').mkdir(exist_ok=True)

    def log_action(self, action: str, description: str,
                   target: Optional[str] = None,
                   model: Optional[str] = None,
                   metabolic: Optional[Dict[str, Any]] = None,
                   details: Optional[Dict[str, Any]] = None,
                   session: Optional[str] = None,
                   salience: Optional[Dict[str, float]] = None,
                   **extra):
        """
        Log a machine action.

        Args:
            action: Action type (boot, inference, interaction, etc.)
            description: Human-readable summary
            target: Target LCT URI or resource
            model: Model name that performed the action
            metabolic: {state, atp_before, atp_after, cycle}
            details: Action-specific data
            session: Session identifier
            salience: SNARC scores if computed
        """
        entry = {
            'ts': _now_iso(),
            'machine': self.machine,
            'lct': self.lct,
            'action': action,
            'description': description,
        }

        if target:
            entry['target'] = target
        if model:
            entry['model'] = model
        if metabolic:
            entry['metabolic'] = metabolic
        if details:
            entry['details'] = details
        if session:
            entry['session'] = session
        if salience:
            entry['salience'] = salience
        if extra:
            entry.update(extra)

        self._action_buffer.append(json.dumps(entry, separators=(',', ':')))

        if len(self._action_buffer) >= self.buffer_size:
            self._flush_actions()

    def log_interaction(self, direction: str,
                        from_lct: Optional[str] = None,
                        to_lct: Optional[str] = None,
                        message_id: str = '',
                        conversation_id: Optional[str] = None,
                        content: Optional[str] = None,
                        response: Optional[str] = None,
                        metabolic_state: Optional[str] = None,
                        atp_cost: Optional[float] = None,
                        latency_ms: Optional[float] = None,
                        auth: Optional[str] = None):
        """
        Log a cross-machine network interaction.

        Content is hashed, not stored — privacy by design.
        """
        entry = {
            'ts': _now_iso(),
            'direction': direction,
            'from_lct': from_lct or (self.lct if direction == 'outbound' else ''),
            'to_lct': to_lct or (self.lct if direction == 'inbound' else ''),
            'message_id': message_id,
        }

        if conversation_id:
            entry['conversation_id'] = conversation_id
        if content:
            entry['content_hash'] = _content_hash(content)
        if response:
            entry['response_hash'] = _content_hash(response)
        if metabolic_state:
            entry['metabolic_state'] = metabolic_state
        if atp_cost is not None:
            entry['atp_cost'] = atp_cost
        if latency_ms is not None:
            entry['latency_ms'] = latency_ms
        if auth:
            entry['auth'] = auth

        self._interaction_buffer.append(json.dumps(entry, separators=(',', ':')))

        if len(self._interaction_buffer) >= self.buffer_size:
            self._flush_interactions()

    def log_state_snapshot(self, snapshot: Dict[str, Any]):
        """Write a daily state snapshot."""
        date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        path = self.logs_dir / 'state-snapshots' / f'{date}.json'

        snapshot.setdefault('date', date)
        snapshot.setdefault('machine', self.machine)

        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def flush(self):
        """Flush all buffered entries to disk."""
        self._flush_actions()
        self._flush_interactions()

    def _flush_actions(self):
        if not self._action_buffer:
            return
        path = self.logs_dir / 'actions.jsonl'
        with open(path, 'a') as f:
            for line in self._action_buffer:
                f.write(line + '\n')
        self._action_buffer.clear()

    def _flush_interactions(self):
        if not self._interaction_buffer:
            return
        path = self.logs_dir / 'interactions.jsonl'
        with open(path, 'a') as f:
            for line in self._interaction_buffer:
                f.write(line + '\n')
        self._interaction_buffer.clear()

    def __del__(self):
        """Flush on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass

    def __repr__(self):
        return (f"ActionLogger(machine='{self.machine}', "
                f"logs_dir='{self.logs_dir}', "
                f"buffered_actions={len(self._action_buffer)}, "
                f"buffered_interactions={len(self._interaction_buffer)})")


if __name__ == '__main__':
    print("ActionLogger self-test")
    print("=" * 40)

    logger = ActionLogger()
    print(f"  Machine: {logger.machine}")
    print(f"  Logs dir: {logger.logs_dir}")
    print(f"  LCT: {logger.lct}")

    # Log some test actions
    logger.log_action('boot', 'Daemon started for self-test',
                      model='test-model',
                      metabolic={'state': 'wake', 'atp_before': 100, 'atp_after': 100, 'cycle': 0})

    logger.log_action('inference', 'Generated test response',
                      model='test-model',
                      target='lct://test:user@cli',
                      details={'response_tokens': 42, 'latency_ms': 150},
                      session='test-001')

    logger.log_interaction('inbound',
                           from_lct='lct://sage:sprout:agent@resident',
                           message_id='msg_test_001',
                           content='Hello from Sprout!',
                           response='Hello back!',
                           metabolic_state='wake',
                           atp_cost=2.5,
                           latency_ms=1500)

    logger.log_action('shutdown', 'Self-test complete')

    logger.flush()

    # Verify files
    actions_path = logger.logs_dir / 'actions.jsonl'
    interactions_path = logger.logs_dir / 'interactions.jsonl'

    with open(actions_path) as f:
        actions = f.readlines()
    with open(interactions_path) as f:
        interactions = f.readlines()

    print(f"\n  Actions logged: {len(actions)}")
    for line in actions:
        entry = json.loads(line)
        print(f"    [{entry['action']}] {entry['description']}")

    print(f"  Interactions logged: {len(interactions)}")
    for line in interactions:
        entry = json.loads(line)
        print(f"    [{entry['direction']}] {entry['from_lct']} → {entry['to_lct']}")

    # Test state snapshot
    logger.log_state_snapshot({
        'model': 'test-model',
        'model_status': 'loaded',
        'uptime_hours': 0.01,
        'cycles_today': 0,
        'actions_today': len(actions),
        'interactions_today': len(interactions),
    })
    print(f"  State snapshot written")

    # Verify content hashing (privacy)
    interaction = json.loads(interactions[0])
    assert 'content_hash' in interaction
    assert interaction['content_hash'].startswith('sha256:')
    assert 'Hello from Sprout!' not in interactions[0]  # Content NOT in log
    print(f"  Privacy check: content hashed, not stored")

    print(f"\nAll ActionLogger tests passed!")

    # Clean up test data
    import os
    os.remove(actions_path)
    os.remove(interactions_path)
    snapshot_dir = logger.logs_dir / 'state-snapshots'
    for f in snapshot_dir.glob('*.json'):
        os.remove(f)
    print("  Test data cleaned up")
