"""
Canonical Effect Schema for SAGE Effector System

An Effect describes a desired state change — physical or informational.
An Effector is the mechanism that manifests that change.

PolicyGate evaluates Effects (the intention).
EffectorRegistry dispatches Effects to Effectors (the mechanism).
Plugins produce Effects from their refinement results.

This unifies three prior schemas:
  - PolicyGate actions (action_id, action_type, role, trust_score)
  - EffectorCommand (effector_id, effector_type, action, parameters)
  - Phase 2 plan proposal (type, target, parameters, reversible)

The Effect carries enough information for all three consumers via
adapter methods: to_policy_action(), to_effector_command(), to_dict().

Version: 1.0 (2026-02-19)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class EffectType(Enum):
    """Taxonomy of effect types spanning the physical-to-informational spectrum."""

    # Physical
    MOTOR = "motor"                  # Joint commands, gripper, robot arm, navigation
    AUDIO = "audio"                  # TTS, sound playback
    VISUAL = "visual"                # Display, LED, monitor

    # Informational
    FILE_IO = "file_io"              # Read/write/delete files
    DATABASE = "database"            # Database queries/mutations
    API_CALL = "api_call"            # HTTP requests, external APIs

    # Communicative
    TOOL_USE = "tool_use"            # Generic tool invocation (MCP, CLI, etc.)
    MESSAGE = "message"              # Inter-agent or human communication
    WEB = "web"                      # Web browsing, scraping

    # Cognitive (internal effects)
    MEMORY_WRITE = "memory_write"    # Write to experience buffer
    TRUST_UPDATE = "trust_update"    # Modify trust weights
    STATE_CHANGE = "state_change"    # Request metabolic state transition

    # Meta
    COMPOSITE = "composite"          # An effect containing sub-effects


class EffectStatus(Enum):
    """Lifecycle status of an effect."""
    PROPOSED = "proposed"        # Created by plugin, not yet evaluated
    APPROVED = "approved"        # Passed PolicyGate
    WARNED = "warned"            # PolicyGate warned, proceeding
    DENIED = "denied"            # PolicyGate denied
    DISPATCHED = "dispatched"    # Sent to effector
    EXECUTING = "executing"      # Effector working on it
    COMPLETED = "completed"      # Successfully executed
    FAILED = "failed"            # Execution failed
    TIMEOUT = "timeout"          # Execution timed out


@dataclass
class Effect:
    """
    Canonical description of a desired state change.

    Carries enough information for:
    - PolicyGate to evaluate (action_type, target, role, trust_score)
    - EffectorHub to dispatch (effect_type, action, parameters)
    - Audit trail to record (effect_id, source_plugin, timestamps, status)
    - ATP budgeting to cost (atp_cost, priority)
    """

    # === Identity ===
    effect_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    effect_type: EffectType = EffectType.TOOL_USE
    action: str = ""                            # Verb: 'write', 'speak', 'move', 'query'

    # === What to do ===
    target: str = ""                            # Target resource: filepath, URL, device, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Any] = None                  # Payload (tensor, bytes, text, etc.)

    # === Who is asking ===
    source_plugin: str = ""                     # Plugin that proposed this effect
    role: str = ""                              # Actor's role in trust context
    trust_score: float = 0.5                    # Actor's current trust level

    # === Resource constraints ===
    atp_cost: float = 1.0                       # Estimated ATP cost
    priority: int = 0                           # Higher = more urgent
    timeout: float = 5.0                        # Max execution time (seconds)
    reversible: bool = True                     # Can this effect be undone?

    # === Lifecycle ===
    status: EffectStatus = EffectStatus.PROPOSED
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # === Policy evaluation results (filled by PolicyGate) ===
    policy_decision: Optional[str] = None       # 'allow', 'warn', 'deny'
    policy_rule_id: Optional[str] = None
    policy_reason: str = ""
    accountability_frame: str = "normal"
    metabolic_state: str = "wake"

    # === Execution results (filled by Effector) ===
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[str] = None

    # === Extensibility ===
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === Sub-effects for COMPOSITE type ===
    children: List['Effect'] = field(default_factory=list)

    def to_policy_action(self) -> Dict[str, Any]:
        """
        Convert to PolicyGate action format for backward compatibility.

        Produces the dict format expected by PolicyGateIRP._evaluate_action():
        {action_id, action_type, target, role, trust_score, parameters}
        """
        return {
            'action_id': self.effect_id,
            'action_type': self.action,
            'target': self.target,
            'role': self.role,
            'trust_score': self.trust_score,
            'parameters': self.parameters,
        }

    def to_effector_command(self, effector_id: str):
        """
        Convert to EffectorCommand for backward compatibility with existing effectors.

        Args:
            effector_id: Target effector to dispatch to

        Returns:
            EffectorCommand instance
        """
        try:
            from .base_effector import EffectorCommand
        except ImportError:
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from base_effector import EffectorCommand
            except (ImportError, ModuleNotFoundError):
                # Fallback: create compatible dataclass when torch unavailable
                from dataclasses import dataclass as _dc, field as _f
                @_dc
                class EffectorCommand:
                    effector_id: str = ""
                    effector_type: str = ""
                    action: str = ""
                    parameters: dict = _f(default_factory=dict)
                    data: object = None
                    timeout: float = 5.0
                    priority: int = 0
                    metadata: dict = _f(default_factory=dict)
        try:
            import torch
            tensor_data = self.data if isinstance(self.data, torch.Tensor) else None
        except ImportError:
            tensor_data = None

        return EffectorCommand(
            effector_id=effector_id,
            effector_type=self.effect_type.value,
            action=self.action,
            parameters=self.parameters,
            data=tensor_data,
            timeout=self.timeout,
            priority=self.priority,
            metadata={
                'effect_id': self.effect_id,
                'source_plugin': self.source_plugin,
                'target': self.target,
                **self.metadata,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for audit trail, SNARC buffer, etc."""
        result = {
            'effect_id': self.effect_id,
            'effect_type': self.effect_type.value,
            'action': self.action,
            'target': self.target,
            'source_plugin': self.source_plugin,
            'role': self.role,
            'trust_score': self.trust_score,
            'status': self.status.value,
            'policy_decision': self.policy_decision,
            'atp_cost': self.atp_cost,
            'priority': self.priority,
            'reversible': self.reversible,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'accountability_frame': self.accountability_frame,
            'metabolic_state': self.metabolic_state,
            'metadata': self.metadata,
        }
        if self.children:
            result['children'] = [c.to_dict() for c in self.children]
        return result

    def approve(self, rule_id: str = "", reason: str = ""):
        """Mark this effect as approved by PolicyGate."""
        self.status = EffectStatus.APPROVED
        self.policy_decision = 'allow'
        self.policy_rule_id = rule_id
        self.policy_reason = reason

    def warn(self, rule_id: str = "", reason: str = ""):
        """Mark this effect as warned by PolicyGate."""
        self.status = EffectStatus.WARNED
        self.policy_decision = 'warn'
        self.policy_rule_id = rule_id
        self.policy_reason = reason

    def deny(self, rule_id: str = "", reason: str = ""):
        """Mark this effect as denied by PolicyGate."""
        self.status = EffectStatus.DENIED
        self.policy_decision = 'deny'
        self.policy_rule_id = rule_id
        self.policy_reason = reason

    def complete(self, result: Optional[Dict[str, Any]] = None):
        """Mark this effect as successfully completed."""
        self.status = EffectStatus.COMPLETED
        self.completed_at = time.time()
        self.execution_result = result

    def fail(self, error: str = ""):
        """Mark this effect as failed."""
        self.status = EffectStatus.FAILED
        self.completed_at = time.time()
        self.execution_error = error


# ============================================================================
# Inline tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Effect Schema — Inline Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # Test 1: Effect creation with defaults
    e = Effect()
    check("default effect_type is TOOL_USE", e.effect_type == EffectType.TOOL_USE)
    check("default status is PROPOSED", e.status == EffectStatus.PROPOSED)
    check("effect_id is generated", len(e.effect_id) == 12)
    check("children is empty list", e.children == [])

    # Test 2: Effect creation with explicit values
    e2 = Effect(
        effect_type=EffectType.FILE_IO,
        action='write',
        target='/tmp/test.txt',
        source_plugin='language',
        role='developer',
        trust_score=0.8,
        atp_cost=0.5,
        priority=5,
        reversible=True,
        parameters={'content': 'hello', 'encoding': 'utf-8'},
    )
    check("explicit effect_type", e2.effect_type == EffectType.FILE_IO)
    check("explicit action", e2.action == 'write')
    check("explicit target", e2.target == '/tmp/test.txt')
    check("explicit trust_score", e2.trust_score == 0.8)

    # Test 3: to_policy_action adapter
    pa = e2.to_policy_action()
    check("policy action_id matches effect_id", pa['action_id'] == e2.effect_id)
    check("policy action_type matches action", pa['action_type'] == 'write')
    check("policy target matches", pa['target'] == '/tmp/test.txt')
    check("policy role matches", pa['role'] == 'developer')
    check("policy trust_score matches", pa['trust_score'] == 0.8)
    check("policy parameters present", 'content' in pa['parameters'])

    # Test 4: to_effector_command adapter
    cmd = e2.to_effector_command('filesystem_0')
    check("command effector_id", cmd.effector_id == 'filesystem_0')
    check("command effector_type", cmd.effector_type == 'file_io')
    check("command action", cmd.action == 'write')
    check("command parameters", cmd.parameters == e2.parameters)
    check("command metadata has effect_id", cmd.metadata['effect_id'] == e2.effect_id)
    check("command metadata has target", cmd.metadata['target'] == '/tmp/test.txt')
    check("command priority", cmd.priority == 5)
    check("command timeout", cmd.timeout == 5.0)

    # Test 5: to_dict serialization
    d = e2.to_dict()
    check("dict has effect_type as string", d['effect_type'] == 'file_io')
    check("dict has status as string", d['status'] == 'proposed')
    check("dict has source_plugin", d['source_plugin'] == 'language')
    check("dict has no children key when empty", 'children' not in d)

    # Test 6: Lifecycle mutations
    e2.approve(rule_id='R1', reason='trusted developer')
    check("approve sets status", e2.status == EffectStatus.APPROVED)
    check("approve sets decision", e2.policy_decision == 'allow')
    check("approve sets rule_id", e2.policy_rule_id == 'R1')

    e3 = Effect(effect_type=EffectType.MOTOR, action='move')
    e3.deny(reason='untrusted')
    check("deny sets status", e3.status == EffectStatus.DENIED)
    check("deny sets decision", e3.policy_decision == 'deny')

    e4 = Effect(effect_type=EffectType.API_CALL, action='post')
    e4.warn(reason='rate limit approaching')
    check("warn sets status", e4.status == EffectStatus.WARNED)

    e5 = Effect(effect_type=EffectType.FILE_IO, action='write')
    e5.complete(result={'bytes_written': 42})
    check("complete sets status", e5.status == EffectStatus.COMPLETED)
    check("complete sets completed_at", e5.completed_at is not None)
    check("complete sets result", e5.execution_result['bytes_written'] == 42)

    e6 = Effect(effect_type=EffectType.WEB, action='get')
    e6.fail(error='timeout')
    check("fail sets status", e6.status == EffectStatus.FAILED)
    check("fail sets error", e6.execution_error == 'timeout')

    # Test 7: COMPOSITE effect with children
    parent = Effect(
        effect_type=EffectType.COMPOSITE,
        action='deploy',
        target='staging',
        children=[
            Effect(effect_type=EffectType.FILE_IO, action='write', target='app.py'),
            Effect(effect_type=EffectType.API_CALL, action='post', target='https://api.example.com/deploy'),
            Effect(effect_type=EffectType.MESSAGE, action='send', target='#deployments'),
        ]
    )
    check("composite has 3 children", len(parent.children) == 3)
    check("child 0 is FILE_IO", parent.children[0].effect_type == EffectType.FILE_IO)
    check("child 1 is API_CALL", parent.children[1].effect_type == EffectType.API_CALL)
    check("child 2 is MESSAGE", parent.children[2].effect_type == EffectType.MESSAGE)

    d_parent = parent.to_dict()
    check("composite dict has children", 'children' in d_parent)
    check("composite dict children count", len(d_parent['children']) == 3)

    # Test 8: All EffectType values are distinct
    values = [et.value for et in EffectType]
    check("all EffectType values unique", len(values) == len(set(values)))
    check("13 effect types total", len(values) == 13)

    # Test 9: All EffectStatus values are distinct
    statuses = [es.value for es in EffectStatus]
    check("all EffectStatus values unique", len(statuses) == len(set(statuses)))
    check("9 status values total", len(statuses) == 9)

    print()
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {failed}")
        exit(1)
